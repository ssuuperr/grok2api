"""Grok Imagine WebSocket client - image generation channel."""

from __future__ import annotations

import asyncio
import base64
import json
import re
import ssl
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Callable, Awaitable, AsyncGenerator, Tuple

import aiohttp

try:
    from aiohttp_socks import ProxyConnector
    AIOHTTP_SOCKS_AVAILABLE = True
except ImportError:
    ProxyConnector = None
    AIOHTTP_SOCKS_AVAILABLE = False

try:
    from curl_cffi import requests as curl_requests
    CURL_CFFI_AVAILABLE = True
except ImportError:
    curl_requests = None
    CURL_CFFI_AVAILABLE = False

from app.core.config import setting
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.services.grok.cache import MIME_TYPES, DEFAULT_MIME, image_cache_service
from app.services.grok.token import token_manager


StreamCallback = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass
class ImageProgress:
    image_id: str
    stage: str = "preview"
    blob: str = ""
    blob_size: int = 0
    url: str = ""
    is_final: bool = False


@dataclass
class GenerationProgress:
    total: int = 4
    images: Dict[str, ImageProgress] = field(default_factory=dict)
    completed: int = 0

    def check_blocked(self) -> bool:
        has_medium = any(img.stage == "medium" for img in self.images.values())
        has_final = any(img.is_final for img in self.images.values())
        return has_medium and not has_final


class GrokImagineWsClient:
    def __init__(self) -> None:
        self._ssl_context = ssl.create_default_context()
        self._url_pattern = re.compile(r"/images/([a-f0-9-]+)\.(png|jpg)")

    def _get_ws_headers(self, sso: str) -> Dict[str, str]:
        return {
            "Cookie": f"sso={sso}; sso-rw={sso}",
            "Origin": "https://grok.com",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        }

    def _extract_image_id(self, url: str) -> Optional[str]:
        match = self._url_pattern.search(url)
        return match.group(1) if match else None

    @staticmethod
    def _is_final_image(url: str, blob_size: int) -> bool:
        return url.endswith(".jpg") and blob_size > 100000

    async def _verify_age(self, sso: str) -> bool:
        if not CURL_CFFI_AVAILABLE:
            logger.warning("[ImagineWS] curl_cffi 未安装，跳过年龄验证")
            return False

        cf_clearance = setting.grok_config.get("cf_clearance", "")
        if not cf_clearance:
            logger.warning("[ImagineWS] cf_clearance 未配置，跳过年龄验证")
            return False

        cookie_parts = [f"sso={sso}", f"sso-rw={sso}"]
        cf_cookie = cf_clearance
        if cf_cookie and not cf_cookie.startswith("cf_clearance="):
            cf_cookie = f"cf_clearance={cf_cookie}"
        cookie_parts.append(cf_cookie)
        cookie_str = "; ".join(cookie_parts)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/133.0.0.0 Safari/537.36"
            ),
            "Origin": "https://grok.com",
            "Referer": "https://grok.com/",
            "Accept": "*/*",
            "Cookie": cookie_str,
            "Content-Type": "application/json",
        }

        proxy = setting.get_proxy("service")

        logger.info("[ImagineWS] 正在进行年龄验证...")

        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: curl_requests.post(
                    "https://grok.com/rest/auth/set-birth-date",
                    headers=headers,
                    json={"birthDate": "2001-01-01T16:00:00.000Z"},
                    impersonate="chrome133a",
                    proxy=proxy,
                    verify=False,
                ),
            )
        except Exception as e:
            logger.error(f"[ImagineWS] 年龄验证失败: {e}")
            return False

        if resp.status_code == 200:
            logger.info("[ImagineWS] 年龄验证成功")
            return True

        logger.warning(f"[ImagineWS] 年龄验证响应: {resp.status_code} - {resp.text[:200]}")
        return False

    async def _get_proxy(self) -> Tuple[aiohttp.BaseConnector, Optional[str]]:
        proxy_url = await setting.get_proxy_async("service")
        if proxy_url and proxy_url.startswith("socks"):
            if not AIOHTTP_SOCKS_AVAILABLE:
                raise GrokApiException("缺少 aiohttp_socks，无法使用 socks 代理", "PROXY_ERROR")
            return ProxyConnector.from_url(proxy_url, ssl=self._ssl_context), None
        return aiohttp.TCPConnector(ssl=self._ssl_context), proxy_url or None

    def _build_local_url(self, image_id: str, ext: str, image_bytes: bytes) -> str:
        raw_path = f"imagine/{image_id}.{ext}"
        cache_path = image_cache_service.cache_dir / raw_path.replace("/", "-")
        cache_path.write_bytes(image_bytes)

        base_url = setting.global_config.get("base_url", "")
        img_path = raw_path.replace("/", "-")
        return f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"

    def _build_data_url(self, image_bytes: bytes, ext: str) -> str:
        mime = MIME_TYPES.get(f".{ext}", DEFAULT_MIME)
        encoded = base64.b64encode(image_bytes).decode()
        return f"data:{mime};base64,{encoded}"

    async def _save_final_images(self, progress: GenerationProgress, n: int) -> Tuple[List[str], List[str]]:
        result_urls: List[str] = []
        result_b64: List[str] = []
        image_mode = setting.global_config.get("image_mode", "url")

        saved_ids: set[str] = set()
        for img in sorted(
            progress.images.values(),
            key=lambda x: (x.is_final, x.blob_size),
            reverse=True,
        ):
            if img.image_id in saved_ids:
                continue
            if len(saved_ids) >= n:
                break

            try:
                image_bytes = base64.b64decode(img.blob)
                ext = "jpg" if img.is_final else "png"
                if image_mode == "base64":
                    result_urls.append(self._build_data_url(image_bytes, ext))
                    result_b64.append(img.blob)
                else:
                    result_urls.append(self._build_local_url(img.image_id, ext, image_bytes))
                    result_b64.append(img.blob)
                saved_ids.add(img.image_id)
            except Exception as e:
                logger.error(f"[ImagineWS] 保存图片失败: {e}")

        return result_urls, result_b64

    async def generate(
        self,
        prompt: str,
        aspect_ratio: str,
        n: int,
        enable_nsfw: bool = True,
        max_retries: int = 5,
        stream_callback: Optional[StreamCallback] = None,
    ) -> Dict[str, Any]:
        last_error = None
        blocked_retries = 0
        max_blocked_retries = 3

        for attempt in range(max_retries):
            sso = await token_manager.select_token("grok-2-image")

            age_verified = await token_manager.get_age_verified(sso)
            if age_verified == 0:
                logger.info(f"[ImagineWS] SSO {sso[:20]}... 未验证年龄，开始验证...")
                if await self._verify_age(sso):
                    await token_manager.set_age_verified(sso, 1)
                else:
                    logger.warning(f"[ImagineWS] SSO {sso[:20]}... 年龄验证失败")

            result = await self._do_generate(
                sso=sso,
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                n=n,
                enable_nsfw=enable_nsfw,
                stream_callback=stream_callback,
            )

            if result.get("success"):
                return result

            error_code = result.get("error_code", "")
            auth_token = f"sso-rw={sso};sso={sso}"

            if error_code == "blocked":
                blocked_retries += 1
                await token_manager.record_failure(auth_token, 403, "imagine blocked")
                if blocked_retries >= max_blocked_retries:
                    return {
                        "success": False,
                        "error_code": "blocked",
                        "error": f"连续 {max_blocked_retries} 次被 blocked，请稍后重试",
                    }
                continue

            if error_code in ["rate_limit_exceeded", "unauthorized"]:
                status = 429 if error_code == "rate_limit_exceeded" else 401
                await token_manager.record_failure(auth_token, status, result.get("error", ""))
                await token_manager.apply_cooldown(auth_token, status)
                last_error = result
                logger.info(f"[ImagineWS] 尝试 {attempt + 1}/{max_retries} 失败，切换 SSO...")
                continue

            return result

        return last_error or {"success": False, "error": "所有重试都失败了"}

    async def _do_generate(
        self,
        sso: str,
        prompt: str,
        aspect_ratio: str,
        n: int,
        enable_nsfw: bool,
        stream_callback: Optional[StreamCallback] = None,
    ) -> Dict[str, Any]:
        request_id = str(uuid.uuid4())
        headers = self._get_ws_headers(sso)
        ws_url = setting.grok_config.get("imagine_ws_url", "wss://grok.com/ws/imagine/listen")
        timeout = setting.grok_config.get("imagine_generation_timeout", 120)

        logger.info(f"[ImagineWS] 连接 WebSocket: {ws_url}")

        connector, proxy = await self._get_proxy()

        try:
            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.ws_connect(
                    ws_url,
                    headers=headers,
                    heartbeat=20,
                    receive_timeout=timeout,
                    proxy=proxy,
                ) as ws:
                    reset_msg = {
                        "type": "conversation.item.create",
                        "timestamp": int(time.time() * 1000),
                        "item": {"type": "message", "content": [{"type": "reset"}]},
                    }
                    await ws.send_json(reset_msg)

                    message = {
                        "type": "conversation.item.create",
                        "timestamp": int(time.time() * 1000),
                        "item": {
                            "type": "message",
                            "content": [{
                                "requestId": request_id,
                                "text": prompt,
                                "type": "input_text",
                                "properties": {
                                    "section_count": 0,
                                    "is_kids_mode": False,
                                    "enable_nsfw": enable_nsfw,
                                    "skip_upsampler": False,
                                    "is_initial": False,
                                    "aspect_ratio": aspect_ratio,
                                },
                            }],
                        },
                    }

                    await ws.send_json(message)
                    logger.info(f"[ImagineWS] 已发送请求: {prompt[:50]}...")

                    progress = GenerationProgress(total=n)
                    error_info = None
                    start_time = time.time()
                    last_activity = time.time()
                    medium_received_time = None

                    while time.time() - start_time < timeout:
                        try:
                            ws_msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                        except asyncio.TimeoutError:
                            if medium_received_time and progress.completed == 0:
                                if time.time() - medium_received_time > 10:
                                    return {
                                        "success": False,
                                        "error_code": "blocked",
                                        "error": "生成被阻止，无法获取最终图片",
                                    }
                            if progress.completed > 0 and time.time() - last_activity > 10:
                                break
                            continue

                        if ws_msg.type == aiohttp.WSMsgType.TEXT:
                            last_activity = time.time()
                            msg = json.loads(ws_msg.data)
                            msg_type = msg.get("type")

                            if msg_type == "image":
                                blob = msg.get("blob", "")
                                url = msg.get("url", "")

                                if blob and url:
                                    image_id = self._extract_image_id(url)
                                    if not image_id:
                                        continue

                                    blob_size = len(blob)
                                    is_final = self._is_final_image(url, blob_size)
                                    if is_final:
                                        stage = "final"
                                    elif blob_size > 30000:
                                        stage = "medium"
                                        if medium_received_time is None:
                                            medium_received_time = time.time()
                                    else:
                                        stage = "preview"

                                    img_progress = ImageProgress(
                                        image_id=image_id,
                                        stage=stage,
                                        blob=blob,
                                        blob_size=blob_size,
                                        url=url,
                                        is_final=is_final,
                                    )

                                    existing = progress.images.get(image_id)
                                    if not existing or not existing.is_final:
                                        progress.images[image_id] = img_progress
                                        progress.completed = len(
                                            [img for img in progress.images.values() if img.is_final]
                                        )

                                        if stream_callback:
                                            await stream_callback({
                                                "type": "progress",
                                                "image_id": image_id,
                                                "stage": stage,
                                                "completed": progress.completed,
                                                "total": n,
                                            })

                            elif msg_type == "error":
                                error_info = {
                                    "error_code": msg.get("err_code", ""),
                                    "error": msg.get("err_msg", ""),
                                }
                                if error_info["error_code"] == "rate_limit_exceeded":
                                    return {"success": False, **error_info}

                            if progress.completed >= n:
                                break

                            if medium_received_time and progress.completed == 0:
                                if time.time() - medium_received_time > 15:
                                    return {
                                        "success": False,
                                        "error_code": "blocked",
                                        "error": "生成被阻止，无法获取最终图片",
                                    }

                        elif ws_msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break

                    result_urls, result_b64 = await self._save_final_images(progress, n)

                    if result_urls:
                        return {
                            "success": True,
                            "urls": result_urls,
                            "b64_list": result_b64,
                            "count": len(result_urls),
                        }

                    if error_info:
                        return {"success": False, **error_info}

                    if progress.check_blocked():
                        return {
                            "success": False,
                            "error_code": "blocked",
                            "error": "生成被阻止，无法获取最终图片",
                        }
                    return {"success": False, "error": "未收到图片数据"}

        except aiohttp.ClientError as e:
            logger.error(f"[ImagineWS] 连接错误: {e}")
            return {"success": False, "error": f"连接失败: {e}"}

    async def generate_stream(
        self,
        prompt: str,
        aspect_ratio: str,
        n: int,
        enable_nsfw: bool = True,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        queue: asyncio.Queue = asyncio.Queue()
        done = asyncio.Event()

        async def callback(item: Dict[str, Any]):
            await queue.put(item)

        async def generate_task():
            result = await self.generate(
                prompt=prompt,
                aspect_ratio=aspect_ratio,
                n=n,
                enable_nsfw=enable_nsfw,
                stream_callback=callback,
            )
            await queue.put({"type": "result", **result})
            done.set()

        task = asyncio.create_task(generate_task())

        try:
            while not done.is_set() or not queue.empty():
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=1.0)
                    yield item
                    if item.get("type") == "result":
                        break
                except asyncio.TimeoutError:
                    continue
        finally:
            if not task.done():
                task.cancel()


imagine_ws_client = GrokImagineWsClient()
