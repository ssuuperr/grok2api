"""Grok API 响应处理器 - 处理流式和非流式响应"""

import re
import orjson
import uuid
import time
import asyncio
from typing import AsyncGenerator, Tuple, Any

from app.core.config import setting
from app.core.exception import GrokApiException
from app.core.logger import logger
from app.models.openai_schema import (
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionMessage,
    OpenAIChatCompletionChunkResponse,
    OpenAIChatCompletionChunkChoice,
    OpenAIChatCompletionChunkMessage
)
from app.services.grok.cache import image_cache_service, video_cache_service


def extract_tool_text(raw: str, rollout_id: str = "") -> str:
    """从 xai:tool_usage_card XML 中提取工具调用的可读文本（移植自源项目）"""
    if not raw:
        return ""
    name_match = re.search(
        r"<xai:tool_name>(.*?)</xai:tool_name>", raw, flags=re.DOTALL
    )
    args_match = re.search(
        r"<xai:tool_args>(.*?)</xai:tool_args>", raw, flags=re.DOTALL
    )

    name = name_match.group(1) if name_match else ""
    if name:
        name = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", name, flags=re.DOTALL).strip()

    args = args_match.group(1) if args_match else ""
    if args:
        args = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", args, flags=re.DOTALL).strip()

    payload = None
    if args:
        try:
            payload = orjson.loads(args)
        except orjson.JSONDecodeError:
            payload = None

    label = name
    text = args
    prefix = f"[{rollout_id}]" if rollout_id else ""

    if name == "web_search":
        label = f"{prefix}[WebSearch]"
        if isinstance(payload, dict):
            text = payload.get("query") or payload.get("q") or ""
    elif name == "search_images":
        label = f"{prefix}[SearchImage]"
        if isinstance(payload, dict):
            text = (
                payload.get("image_description")
                or payload.get("description")
                or payload.get("query")
                or ""
            )
    elif name == "chatroom_send":
        label = f"{prefix}[AgentThink]"
        if isinstance(payload, dict):
            text = payload.get("message") or ""

    if label and text:
        return f"{label} {text}".strip()
    if label:
        return label
    if text:
        return text
    # 回退：去掉所有标签，保留纯文本
    return re.sub(r"<[^>]+>", "", raw, flags=re.DOTALL).strip()


class StreamTimeoutManager:
    """流式响应超时管理"""
    
    def __init__(self, chunk_timeout: int = 120, first_timeout: int = 30, total_timeout: int = 600):
        self.chunk_timeout = chunk_timeout
        self.first_timeout = first_timeout
        self.total_timeout = total_timeout
        self.start_time = asyncio.get_event_loop().time()
        self.last_chunk_time = self.start_time
        self.first_received = False
    
    def check_timeout(self) -> Tuple[bool, str]:
        """检查超时"""
        now = asyncio.get_event_loop().time()
        
        if not self.first_received and now - self.start_time > self.first_timeout:
            return True, f"首次响应超时({self.first_timeout}秒)"
        
        if self.total_timeout > 0 and now - self.start_time > self.total_timeout:
            return True, f"总超时({self.total_timeout}秒)"
        
        if self.first_received and now - self.last_chunk_time > self.chunk_timeout:
            return True, f"数据块超时({self.chunk_timeout}秒)"
        
        return False, ""
    
    def mark_received(self):
        """标记收到数据"""
        self.last_chunk_time = asyncio.get_event_loop().time()
        self.first_received = True
    
    def duration(self) -> float:
        """获取总耗时"""
        return asyncio.get_event_loop().time() - self.start_time


class GrokResponseProcessor:
    """Grok响应处理器"""

    @staticmethod
    async def process_normal(response, auth_token: str, model: str = None) -> OpenAIChatCompletionResponse:
        """处理非流式响应"""
        response_closed = False
        try:
            async for chunk in response.aiter_lines():
                if not chunk:
                    continue

                data = orjson.loads(chunk)

                # 错误检查
                if error := data.get("error"):
                    raise GrokApiException(
                        f"API错误: {error.get('message', '未知错误')}",
                        "API_ERROR",
                        {"code": error.get("code")}
                    )

                grok_resp = data.get("result", {}).get("response", {})
                
                # 视频响应
                if video_resp := grok_resp.get("streamingVideoGenerationResponse"):
                    if video_url := video_resp.get("videoUrl"):
                        content = await GrokResponseProcessor._build_video_content(video_url, auth_token)
                        result = GrokResponseProcessor._build_response(content, model or "grok-imagine-1.0-video")
                        response_closed = True
                        response.close()
                        return result

                # 图片生成响应（通过 modelResponse 收集）
                if mr := grok_resp.get("modelResponse"):
                    # 递归收集所有图片 URL
                    image_urls = GrokResponseProcessor._collect_image_urls(mr)
                    if image_urls:
                        content = mr.get("message", "")
                        content = await GrokResponseProcessor._append_images(content, image_urls, auth_token)
                        result = GrokResponseProcessor._build_response(content, model or mr.get("model"))
                        response_closed = True
                        response.close()
                        return result
                    
                    # 普通文本响应
                    if error_msg := mr.get("error"):
                        raise GrokApiException(f"模型错误: {error_msg}", "MODEL_ERROR")
                    content = mr.get("message", "")
                    if content:
                        result = GrokResponseProcessor._build_response(content, model or mr.get("model"))
                        response_closed = True
                        response.close()
                        return result

            raise GrokApiException("无响应数据", "NO_RESPONSE")

        except orjson.JSONDecodeError as e:
            logger.error(f"[Processor] JSON解析失败: {e}")
            raise GrokApiException(f"JSON解析失败: {e}", "JSON_ERROR") from e
        except Exception as e:
            logger.error(f"[Processor] 处理错误: {type(e).__name__}: {e}")
            raise GrokApiException(f"响应处理错误: {e}", "PROCESS_ERROR") from e
        finally:
            if not response_closed and hasattr(response, 'close'):
                try:
                    response.close()
                except Exception as e:
                    logger.warning(f"[Processor] 关闭响应失败: {e}")

    @staticmethod
    async def process_stream(response, auth_token: str, session: Any = None) -> AsyncGenerator[str, None]:
        """处理流式响应（对齐源项目 StreamProcessor.process）"""
        # 状态变量 — 对齐源项目，用 think_opened 统一管理 <think> 标签
        is_image = False
        think_opened = False          # 是否已发送 <think>（替代原 is_thinking + thinking_finished）
        image_think_active = False    # 图片生成进度期间的 thinking 状态
        model = None
        rollout_id = ""               # 当前 rolloutId（用于工具卡片格式化）
        video_progress_started = False
        last_video_progress = -1
        response_closed = False
        show_thinking = setting.grok_config.get("show_thinking", True)

        # 标签过滤配置（对齐源项目 filter_tags 机制）
        filter_tags_raw = setting.grok_config.get("filtered_tags", "")
        filter_tags = [t.strip() for t in filter_tags_raw.split(",") if t.strip()]
        tool_usage_enabled = "xai:tool_usage_card" in filter_tags
        _tool_usage_opened = False    # xai:tool_usage_card 标签是否处于打开状态（跨 chunk）
        _tool_usage_buffer = ""       # 跨 chunk 的 tool_usage_card 缓冲区

        def _filter_tool_card(token_text: str) -> str:
            """解析 xai:tool_usage_card 标签，提取结构化文本（移植自源项目）"""
            nonlocal _tool_usage_opened, _tool_usage_buffer

            if not token_text or not tool_usage_enabled:
                return token_text

            output_parts = []
            rest = token_text
            start_tag = "<xai:tool_usage_card"
            end_tag = "</xai:tool_usage_card>"

            while rest:
                if _tool_usage_opened:
                    # 正在一个打开的标签内，寻找结束标签
                    end_idx = rest.find(end_tag)
                    if end_idx == -1:
                        # 整个 rest 都在标签内，继续缓冲
                        _tool_usage_buffer += rest
                        return "".join(output_parts)
                    end_pos = end_idx + len(end_tag)
                    _tool_usage_buffer += rest[:end_pos]
                    line = extract_tool_text(_tool_usage_buffer, rollout_id)
                    if line:
                        if output_parts and not output_parts[-1].endswith("\n"):
                            output_parts[-1] += "\n"
                        output_parts.append(f"{line}\n")
                    _tool_usage_buffer = ""
                    _tool_usage_opened = False
                    rest = rest[end_pos:]
                    continue

                # 寻找开始标签
                start_idx = rest.find(start_tag)
                if start_idx == -1:
                    # 没有标签，保留原文
                    output_parts.append(rest)
                    break

                # 标签前的内容保留
                if start_idx > 0:
                    output_parts.append(rest[:start_idx])

                # 寻找结束标签
                end_idx = rest.find(end_tag, start_idx)
                if end_idx == -1:
                    # 标签未闭合，缓冲起来等下一个 chunk
                    _tool_usage_opened = True
                    _tool_usage_buffer = rest[start_idx:]
                    break

                # 完整标签在同一个 chunk 内
                end_pos = end_idx + len(end_tag)
                raw_card = rest[start_idx:end_pos]
                line = extract_tool_text(raw_card, rollout_id)
                if line:
                    if output_parts and not output_parts[-1].endswith("\n"):
                        output_parts[-1] += "\n"
                    output_parts.append(f"{line}\n")
                rest = rest[end_pos:]

            return "".join(output_parts)

        def _filter_token(token_text: str) -> str:
            """过滤 token 中的特殊标签（对齐源项目 _filter_token）"""
            if not token_text:
                return token_text

            # 先处理 tool_usage_card 解析
            if tool_usage_enabled:
                token_text = _filter_tool_card(token_text)
                if not token_text:
                    return ""

            if not filter_tags:
                return token_text

            # 对其他标签做精确的 <tag / </tag> 匹配，而非子串包含
            for tag in filter_tags:
                if tag == "xai:tool_usage_card":
                    continue  # 已在上面处理
                if f"<{tag}" in token_text or f"</{tag}" in token_text:
                    return ""

            return token_text

        # 超时管理
        timeout_mgr = StreamTimeoutManager(
            chunk_timeout=setting.grok_config.get("stream_chunk_timeout", 120),
            first_timeout=setting.grok_config.get("stream_first_response_timeout", 30),
            total_timeout=setting.grok_config.get("stream_total_timeout", 600)
        )

        def make_chunk(content: str, finish: str = None):
            """生成响应块"""
            chunk_data = OpenAIChatCompletionChunkResponse(
                id=f"chatcmpl-{uuid.uuid4()}",
                created=int(time.time()),
                model=model or "grok-4-mini-thinking-tahoe",
                choices=[OpenAIChatCompletionChunkChoice(
                    index=0,
                    delta=OpenAIChatCompletionChunkMessage(
                        role="assistant",
                        content=content
                    ) if content else {},
                    finish_reason=finish
                )]
            )
            return f"data: {chunk_data.model_dump_json()}\n\n"

        try:
            async for chunk in response.aiter_lines():
                # 超时检查
                is_timeout, timeout_msg = timeout_mgr.check_timeout()
                if is_timeout:
                    logger.warning(f"[Processor] {timeout_msg}")
                    yield make_chunk("", "stop")
                    yield "data: [DONE]\n\n"
                    return

                logger.debug(f"[Processor] 收到数据块: {len(chunk)} bytes")
                if not chunk:
                    continue

                try:
                    data = orjson.loads(chunk)

                    # 错误检查
                    if error := data.get("error"):
                        error_msg = error.get('message', '未知错误')
                        logger.error(f"[Processor] API错误: {error_msg}")
                        yield make_chunk(f"Error: {error_msg}", "stop")
                        yield "data: [DONE]\n\n"
                        return

                    grok_resp = data.get("result", {}).get("response", {})
                    logger.debug(f"[Processor] 解析响应: {len(grok_resp)} bytes")
                    if not grok_resp:
                        continue
                    
                    timeout_mgr.mark_received()

                    # 获取当前 thinking 状态
                    is_thinking = bool(grok_resp.get("isThinking"))

                    # 更新 rolloutId（用于工具卡片格式化）
                    if rid := grok_resp.get("rolloutId"):
                        rollout_id = str(rid)

                    # 更新模型
                    if user_resp := grok_resp.get("userResponse"):
                        if m := user_resp.get("model"):
                            model = m

                    # 视频处理
                    if video_resp := grok_resp.get("streamingVideoGenerationResponse"):
                        progress = video_resp.get("progress", 0)
                        v_url = video_resp.get("videoUrl")
                        
                        # 进度更新
                        if progress > last_video_progress:
                            last_video_progress = progress
                            if show_thinking:
                                if not video_progress_started:
                                    content = f"<think>视频已生成{progress}%\n"
                                    video_progress_started = True
                                elif progress < 100:
                                    content = f"视频已生成{progress}%\n"
                                else:
                                    content = f"视频已生成{progress}%</think>\n"
                                yield make_chunk(content)
                        
                        # 视频URL
                        if v_url:
                            logger.info("[Processor] 视频生成完成")
                            video_content = await GrokResponseProcessor._build_video_content(v_url, auth_token)
                            yield make_chunk(video_content)
                        
                        continue

                    # 图片生成进度（streamingImageGenerationResponse）— 对齐源项目
                    if img_gen := grok_resp.get("streamingImageGenerationResponse"):
                        if not show_thinking:
                            continue
                        image_think_active = True
                        if not think_opened:
                            yield make_chunk("<think>\n")
                            think_opened = True
                        idx = img_gen.get("imageIndex", 0) + 1
                        progress = img_gen.get("progress", 0)
                        yield make_chunk(f"正在生成第{idx}张图片中，当前进度{progress}%\n")
                        continue

                    # modelResponse 统一处理（对齐源项目：先于 is_image 和对话分支）
                    if mr := grok_resp.get("modelResponse"):
                        # 关闭图片生成进度的 think 标签
                        if image_think_active and think_opened:
                            yield make_chunk("\n</think>\n")
                            think_opened = False
                        image_think_active = False

                        # 收集 modelResponse 中的所有图片 URL
                        all_images = GrokResponseProcessor._collect_image_urls(mr)
                        logger.info(f"[Processor] modelResponse 图片: {all_images}")
                        if all_images:
                            # 规范化 URL（去掉完整域名前缀，只保留路径）
                            normalized = []
                            for u in all_images:
                                if u.startswith("https://assets.grok.com/"):
                                    u = u[len("https://assets.grok.com/"):]
                                elif u.startswith("https://grok.com/"):
                                    u = u[len("https://grok.com/"):]
                                normalized.append(u)
                            all_images = normalized
                            logger.info(f"[Processor] 规范化后图片路径: {all_images}")
                            # 有图片 — 渲染图片
                            image_mode = setting.global_config.get("image_mode", "url")
                            for idx, img in enumerate(all_images, 1):
                                try:
                                    if image_mode == "base64":
                                        base64_str = await image_cache_service.download_base64(f"/{img}", auth_token)
                                        if base64_str:
                                            yield make_chunk(f"![image_{idx}]({base64_str})\n")
                                        else:
                                            base_url = setting.global_config.get("base_url", "")
                                            fallback = f"{base_url}/images/{img.replace('/', '-')}" if base_url else f"https://assets.grok.com/{img}"
                                            yield make_chunk(f"![image_{idx}]({fallback})\n")
                                    else:
                                        await image_cache_service.download_image(f"/{img}", auth_token)
                                        img_path = img.replace('/', '-')
                                        base_url = setting.global_config.get("base_url", "")
                                        img_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                                        yield make_chunk(f"![image_{idx}]({img_url})\n")
                                except Exception as e:
                                    logger.warning(f"[Processor] 图片处理失败: {e}")
                                    yield make_chunk(f"![image_{idx}](https://assets.grok.com/{img})\n")
                        continue

                    # cardAttachment 处理（某些模型通过卡片返回图片）
                    if card := grok_resp.get("cardAttachment"):
                        json_data = card.get("jsonData")
                        if isinstance(json_data, str) and json_data.strip():
                            try:
                                card_data = orjson.loads(json_data)
                            except Exception:
                                card_data = None
                            if isinstance(card_data, dict):
                                image = card_data.get("image") or {}
                                original = image.get("original")
                                title = image.get("title") or ""
                                if original:
                                    title_safe = title.replace("\n", " ").strip() or "image"
                                    yield make_chunk(f"![{title_safe}]({original})\n")
                        continue

                    # ---- 以下为 token 文本处理 ----
                    token = grok_resp.get("token", "")

                    # 图片模式标记（仅传统图生图通道需要）
                    if grok_resp.get("imageAttachmentInfo"):
                        is_image = True

                    # is_image 模式下仅输出 token（最终图片由上方 modelResponse 处理输出）
                    if is_image:
                        if token:
                            yield make_chunk(token)
                        continue

                    # ---- 对话模式 token 处理 ----
                    if not token:
                        continue

                    if isinstance(token, list):
                        continue

                    # 使用对齐源项目的 _filter_token 过滤特殊标签
                    # （替代旧的 filtered_tags 子串匹配和 toolUsageCardId 分支）
                    filtered = _filter_token(token)
                    if not filtered:
                        continue

                    content = filtered

                    # Thinking 状态切换 — 对齐源项目 think_opened 模式
                    in_think = is_thinking or image_think_active
                    if in_think:
                        if not show_thinking:
                            continue
                        if not think_opened:
                            logger.debug("[Processor] 开始 thinking 输出")
                            content = f"<think>\n{content}"
                            think_opened = True
                    else:
                        if think_opened:
                            logger.debug("[Processor] 结束 thinking 输出")
                            content = f"\n</think>\n{content}"
                            think_opened = False

                    yield make_chunk(content)

                except (orjson.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"[Processor] 解析失败: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"[Processor] 处理出错: {e}")
                    continue

            # 流结束时，如果 think 标签未关闭，补发关闭
            if think_opened:
                yield make_chunk("</think>\n")
            yield make_chunk("", "stop")
            yield "data: [DONE]\n\n"
            logger.info(f"[Processor] 流式完成，耗时: {timeout_mgr.duration():.2f}秒")

        except Exception as e:
            logger.error(f"[Processor] 严重错误: {e}")
            yield make_chunk(f"处理错误: {e}", "error")
            yield "data: [DONE]\n\n"
        finally:
            if not response_closed and hasattr(response, 'close'):
                try:
                    response.close()
                    logger.debug("[Processor] 响应已关闭")
                except Exception as e:
                    logger.warning(f"[Processor] 关闭失败: {e}")
            
            if session:
                try:
                    await session.close()
                    logger.debug("[Processor] 会话已关闭")
                except Exception as e:
                    logger.warning(f"[Processor] 关闭会话失败: {e}")

    @staticmethod
    def _collect_image_urls(obj) -> list:
        """递归收集响应中的所有图片 URL（对齐源项目 _collect_images）
        
        支持 generatedImageUrls、imageUrls、imageURLs 三种字段名
        """
        urls = []
        seen = set()
        
        def add(url):
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
        
        def walk(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"generatedImageUrls", "imageUrls", "imageURLs"}:
                        if isinstance(item, list):
                            for url in item:
                                if isinstance(url, str):
                                    add(url)
                        elif isinstance(item, str):
                            add(item)
                        continue
                    walk(item)
            elif isinstance(value, list):
                for item in value:
                    walk(item)
        
        walk(obj)
        return urls

    @staticmethod
    async def _build_video_content(video_url: str, auth_token: str) -> str:
        """构建视频内容
        
        优先尝试缓存下载并返回本地代理URL，
        失败时使用 base_url 构建可通过反向代理访问的 fallback URL
        """
        logger.info(f"[Processor] 检测到视频URL: {video_url}")
        base_url = setting.global_config.get("base_url", "")
        
        try:
            cache_path = await video_cache_service.download_video(f"/{video_url}", auth_token)
            if cache_path:
                video_path = video_url.replace('/', '-')
                local_url = f"{base_url}/images/{video_path}" if base_url else f"/images/{video_path}"
                logger.info(f"[Processor] 视频缓存成功，本地URL: {local_url}")
                return f'<video src="{local_url}" controls="controls" width="500" height="300"></video>\n'
            else:
                logger.warning(f"[Processor] 视频缓存下载返回空，使用 fallback")
        except Exception as e:
            logger.warning(f"[Processor] 缓存视频失败: {e}，使用 fallback")
        
        # fallback：优先使用 base_url 构建可通过反向代理访问的 URL
        if base_url:
            # 尝试通过服务端代理返回（添加到视频缓存路径）
            video_path = video_url.replace('/', '-')
            fallback_url = f"{base_url}/images/{video_path}"
            logger.info(f"[Processor] 视频 fallback（反向代理）: {fallback_url}")
        else:
            fallback_url = f"https://assets.grok.com/{video_url}"
            logger.info(f"[Processor] 视频 fallback（直连）: {fallback_url}")
        
        return f'<video src="{fallback_url}" controls="controls" width="500" height="300"></video>\n'

    @staticmethod
    async def _append_images(content: str, images: list, auth_token: str) -> str:
        """追加图片到内容"""
        image_mode = setting.global_config.get("image_mode", "url")
        
        for img in images:
            try:
                if image_mode == "base64":
                    base64_str = await image_cache_service.download_base64(f"/{img}", auth_token)
                    if base64_str:
                        content += f"\n![Generated Image]({base64_str})"
                    else:
                        content += f"\n![Generated Image](https://assets.grok.com/{img})"
                else:
                    cache_path = await image_cache_service.download_image(f"/{img}", auth_token)
                    if cache_path:
                        img_path = img.replace('/', '-')
                        base_url = setting.global_config.get("base_url", "")
                        img_url = f"{base_url}/images/{img_path}" if base_url else f"/images/{img_path}"
                        content += f"\n![Generated Image]({img_url})"
                    else:
                        content += f"\n![Generated Image](https://assets.grok.com/{img})"
            except Exception as e:
                logger.warning(f"[Processor] 处理图片失败: {e}")
                content += f"\n![Generated Image](https://assets.grok.com/{img})"
        
        return content

    @staticmethod
    def _build_response(content: str, model: str) -> OpenAIChatCompletionResponse:
        """构建响应对象"""
        return OpenAIChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=model,
            choices=[OpenAIChatCompletionChoice(
                index=0,
                message=OpenAIChatCompletionMessage(
                    role="assistant",
                    content=content
                ),
                finish_reason="stop"
            )],
            usage=None
        )
