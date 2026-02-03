"""
NSFW (Unhinged) 模式服务

使用 gRPC-Web 协议开启账号的 NSFW 功能。
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from curl_cffi.requests import AsyncSession

from app.core.config import setting
from app.core.logger import logger
from app.services.grok.grpc_web import (
    encode_grpc_web_payload,
    parse_grpc_web_response,
    get_grpc_status,
)


# 常量
NSFW_API = "https://grok.com/auth_mgmt.AuthManagement/UpdateUserFeatureControls"
BROWSER = "chrome133a"
TIMEOUT = 30


@dataclass
class NSFWResult:
    """NSFW 操作结果"""
    success: bool
    http_status: int
    grpc_status: Optional[int] = None
    grpc_message: Optional[str] = None
    error: Optional[str] = None


class NSFWService:
    """NSFW 模式服务"""

    def __init__(self, proxy: str = None):
        self.proxy = proxy

    async def _get_proxy(self) -> Optional[str]:
        """获取代理配置"""
        if self.proxy:
            return self.proxy
        return await setting.get_proxy_async("service")

    def _build_headers(self, token: str) -> dict:
        """构造 gRPC-Web 请求头"""
        # 去除 sso= 前缀
        token = token[4:] if token.startswith("sso=") else token
        cookie = f"sso={token}; sso-rw={token}"

        return {
            "accept": "*/*",
            "content-type": "application/grpc-web+proto",
            "origin": "https://grok.com",
            "referer": "https://grok.com/",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "x-grpc-web": "1",
            "x-user-agent": "connect-es/2.1.1",
            "cookie": cookie,
        }

    @staticmethod
    def _build_payload() -> bytes:
        """构造请求 payload"""
        # protobuf: enable_unhinged=true (field1=1, field2=1)
        protobuf = bytes([0x08, 0x01, 0x10, 0x01])
        return encode_grpc_web_payload(protobuf)

    async def enable(self, token: str) -> NSFWResult:
        """为单个 token 开启 NSFW 模式"""
        headers = self._build_headers(token)
        payload = self._build_payload()
        
        proxy = await self._get_proxy()
        proxies = {"http": proxy, "https": proxy} if proxy else None

        try:
            async with AsyncSession(impersonate=BROWSER) as session:
                response = await session.post(
                    NSFW_API,
                    data=payload,
                    headers=headers,
                    timeout=TIMEOUT,
                    proxies=proxies,
                )

                if response.status_code != 200:
                    logger.warning(f"[NSFW] HTTP 失败: {token[:20]}... - 状态码 {response.status_code}")
                    return NSFWResult(
                        success=False,
                        http_status=response.status_code,
                        error=f"HTTP {response.status_code}",
                    )

                # 解析 gRPC-Web 响应
                content_type = response.headers.get("content-type")
                _, trailers = parse_grpc_web_response(
                    response.content, content_type=content_type
                )

                grpc_status = get_grpc_status(trailers)

                # HTTP 200 且无 grpc-status（空响应）或 grpc-status=0 都算成功
                success = grpc_status.code == -1 or grpc_status.ok

                if success:
                    logger.info(f"[NSFW] 开启成功: {token[:20]}...")
                else:
                    logger.warning(f"[NSFW] gRPC 失败: {token[:20]}... - code={grpc_status.code}, msg={grpc_status.message}")

                return NSFWResult(
                    success=success,
                    http_status=response.status_code,
                    grpc_status=grpc_status.code,
                    grpc_message=grpc_status.message or None,
                )

        except Exception as e:
            logger.error(f"[NSFW] 异常: {token[:20]}... - {e}")
            return NSFWResult(success=False, http_status=0, error=str(e)[:100])


# 保持向后兼容的函数式 API
async def enable_nsfw(sso_token: str) -> tuple[bool, str]:
    """
    为指定 Token 开启 NSFW (Unhinged) 模式
    
    Args:
        sso_token: SSO Token（不含 sso= 前缀）
    
    Returns:
        Tuple[bool, str]: (是否成功, 消息)
    """
    service = NSFWService()
    result = await service.enable(sso_token)
    
    if result.success:
        return True, "成功开启 NSFW 模式"
    elif result.grpc_message:
        return False, f"gRPC 错误: {result.grpc_message}"
    elif result.error:
        return False, result.error
    else:
        return False, f"状态码: {result.http_status}"


async def batch_enable_nsfw(tokens: List[str]) -> Dict[str, Any]:
    """
    批量开启 NSFW 模式
    
    Args:
        tokens: SSO Token 列表
    
    Returns:
        Dict: 包含成功/失败统计和详细结果
    """
    service = NSFWService()
    success_count = 0
    fail_count = 0
    results = []
    
    for token in tokens:
        result = await service.enable(token)
        results.append({
            "token": token[:20] + "...",
            "success": result.success,
            "http_status": result.http_status,
            "grpc_status": result.grpc_status,
            "grpc_message": result.grpc_message,
            "error": result.error,
        })
        
        if result.success:
            success_count += 1
        else:
            fail_count += 1
    
    return {
        "total": len(tokens),
        "success": success_count,
        "failed": fail_count,
        "results": results
    }


__all__ = ["NSFWService", "NSFWResult", "enable_nsfw", "batch_enable_nsfw"]
