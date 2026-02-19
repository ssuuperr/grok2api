"""语音服务 - 精简版 LiveKit Token 获取"""

import orjson
from typing import Dict, Any
from curl_cffi.requests import AsyncSession as curl_AsyncSession

from app.core.config import setting
from app.core.logger import logger
from app.services.grok.token import token_manager
from app.services.grok.statsig import get_dynamic_headers
from app.core.exception import GrokApiException


# LiveKit Token API 端点
LIVEKIT_TOKEN_URL = "https://grok.com/rest/app-chat/token"
BROWSER = "chrome136"


class VoiceService:
    """语音服务 - 获取 LiveKit Token"""

    @staticmethod
    async def get_token(
        voice: str = "cove",
        personality: str = "assistant",
        speed: float = 1.0,
    ) -> Dict[str, Any]:
        """获取 LiveKit Token
        
        Args:
            voice: 语音角色（cove, sage, ember 等）
            personality: 交互人格
            speed: 语速倍率

        Returns:
            包含 LiveKit Token 和连接信息的字典
        """
        # 获取 SSO Token
        sso = await token_manager.get_token("grok-4")
        if not sso:
            raise GrokApiException("没有可用的 Token", "NO_AVAILABLE_TOKEN")

        # 构建请求头
        headers = get_dynamic_headers("/rest/app-chat/token")
        cf = setting.grok_config.get("cf_clearance", "")
        headers["Cookie"] = f"{sso};{cf}" if cf else sso

        # 构建请求体
        payload = {
            "voice": voice,
            "personality": personality,
            "speed": speed,
        }

        # 获取代理
        proxy = await setting.get_proxy_async("service")
        proxies = {"http": proxy, "https": proxy} if proxy else None

        # 发送请求
        async with curl_AsyncSession(impersonate=BROWSER) as session:
            try:
                response = await session.post(
                    LIVEKIT_TOKEN_URL,
                    headers=headers,
                    data=orjson.dumps(payload),
                    timeout=30,
                    proxies=proxies,
                )

                if response.status_code != 200:
                    error_text = response.text[:200] if response.text else "未知错误"
                    logger.error(f"[Voice] Token 获取失败: {response.status_code} - {error_text}")
                    raise GrokApiException(
                        f"语音 Token 获取失败: {response.status_code}",
                        "VOICE_TOKEN_ERROR",
                        {"status": response.status_code}
                    )

                result = response.json()
                logger.info(f"[Voice] Token 获取成功")
                return result

            except GrokApiException:
                raise
            except Exception as e:
                logger.error(f"[Voice] 请求异常: {e}")
                raise GrokApiException(
                    f"语音服务请求失败: {e}",
                    "VOICE_REQUEST_ERROR"
                ) from e


# 全局实例
voice_service = VoiceService()
