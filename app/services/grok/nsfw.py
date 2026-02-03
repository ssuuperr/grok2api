"""NSFW/Unhinged 模式开启服务

使用 gRPC-Web 协议调用 Grok 的 UpdateUserFeatureControls 接口
"""

import struct
from typing import Tuple, List, Dict, Any
from curl_cffi.requests import AsyncSession

from app.core.config import setting
from app.core.logger import logger


# 常量
NSFW_ENDPOINT = "https://grok.com/auth_mgmt.AuthManagement/UpdateUserFeatureControls"
TIMEOUT = 30
BROWSER = "chrome133a"


def _encode_grpc_unhinged() -> bytes:
    """
    编码 gRPC-Web 启用 NSFW 的 payload
    
    Protobuf: field 1 (varint) = 1, field 2 (varint) = 1
    表示启用 unhinged/NSFW 模式
    """
    # Protobuf 编码：0x08 0x01 = field 1 varint 1, 0x10 0x01 = field 2 varint 1
    payload = bytes([0x08, 0x01, 0x10, 0x01])
    # gRPC-Web 格式：[1 byte 压缩标志(0)] + [4 bytes 大端序长度] + [N bytes 数据]
    return b'\x00' + struct.pack('>I', len(payload)) + payload


async def enable_nsfw(sso_token: str) -> Tuple[bool, str]:
    """
    为指定 Token 开启 NSFW (Unhinged) 模式
    
    Args:
        sso_token: SSO Token（不含 sso= 前缀）
    
    Returns:
        Tuple[bool, str]: (是否成功, 消息)
    """
    headers = {
        "accept": "*/*",
        "content-type": "application/grpc-web+proto",
        "origin": "https://grok.com",
        "referer": "https://grok.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
        "x-grpc-web": "1",
        "x-user-agent": "connect-es/2.1.1",
        "cookie": f"sso={sso_token}; sso-rw={sso_token}"
    }
    
    data = _encode_grpc_unhinged()
    
    # 获取代理配置
    proxy = await setting.get_proxy_async("service")
    proxies = {"http": proxy, "https": proxy} if proxy else None
    
    try:
        async with AsyncSession(impersonate=BROWSER) as session:
            response = await session.post(
                NSFW_ENDPOINT,
                data=data,
                headers=headers,
                timeout=TIMEOUT,
                proxies=proxies
            )
            
            if response.status_code == 200:
                logger.info(f"[NSFW] Token 开启 NSFW 成功: {sso_token[:20]}...")
                return True, "成功开启 NSFW 模式"
            else:
                msg = f"状态码: {response.status_code}"
                logger.warning(f"[NSFW] Token 开启 NSFW 失败: {sso_token[:20]}... - {msg}")
                return False, msg
                
    except Exception as e:
        msg = str(e)[:100]
        logger.error(f"[NSFW] Token 开启 NSFW 异常: {sso_token[:20]}... - {msg}")
        return False, msg


async def batch_enable_nsfw(tokens: List[str]) -> Dict[str, Any]:
    """
    批量开启 NSFW 模式
    
    Args:
        tokens: SSO Token 列表
    
    Returns:
        Dict: 包含成功/失败统计和详细结果
    """
    success_count = 0
    fail_count = 0
    results = []
    
    for token in tokens:
        success, msg = await enable_nsfw(token)
        results.append({
            "token": token[:20] + "...",
            "success": success,
            "message": msg
        })
        
        if success:
            success_count += 1
        else:
            fail_count += 1
    
    return {
        "total": len(tokens),
        "success": success_count,
        "failed": fail_count,
        "results": results
    }
