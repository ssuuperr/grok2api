"""语音 API 路由 - OpenAI TTS 兼容"""

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from typing import Optional

from app.services.grok.voice import voice_service
from app.services.auth import auth_manager
from app.core.logger import logger

router = APIRouter(tags=["Audio"])


class SpeechRequest(BaseModel):
    """语音请求 - OpenAI TTS 兼容格式"""
    model: str = Field(default="tts-1", description="模型名称（兼容用，实际使用 LiveKit）")
    input: str = Field(default="", description="文本内容（预留，当前不使用）")
    voice: str = Field(default="cove", description="语音角色: cove, sage, ember 等")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="语速倍率")


@router.post("/audio/speech")
async def create_speech(request: Request, body: SpeechRequest):
    """获取 LiveKit Token 以建立语音连接

    返回 LiveKit Token 和 WebSocket 连接信息，
    客户端使用该信息建立实时语音通信。
    """
    # 鉴权
    auth_header = request.headers.get("Authorization", "")
    api_key = auth_header.replace("Bearer ", "") if auth_header.startswith("Bearer ") else ""
    auth_manager.verify(api_key)

    logger.info(f"[Audio] 请求语音 Token: voice={body.voice}, speed={body.speed}")

    result = await voice_service.get_token(
        voice=body.voice,
        speed=body.speed,
    )

    return result
