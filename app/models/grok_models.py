"""Grok 模型配置和枚举定义"""

from enum import Enum
from typing import Dict, Any, Tuple


# 模型配置
_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    # ==================== Grok 3 系列 ====================
    "grok-3": {
        "grok_model": ("grok-3", "MODEL_MODE_GROK_3"),
        "rate_limit_model": "grok-3",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 3",
        "description": "Standard Grok 3 model",
        "raw_model_path": "xai/grok-3",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-3-fast": {
        "grok_model": ("grok-3", "MODEL_MODE_FAST"),
        "rate_limit_model": "grok-3",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 3 Fast",
        "description": "Fast and efficient Grok 3 model",
        "raw_model_path": "xai/grok-3",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-3-mini": {
        "grok_model": ("grok-3", "MODEL_MODE_GROK_3_MINI_THINKING"),
        "rate_limit_model": "grok-3",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 3 Mini",
        "description": "Lightweight Grok 3 model with mini thinking capabilities",
        "raw_model_path": "xai/grok-3",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-3-thinking": {
        "grok_model": ("grok-3", "MODEL_MODE_GROK_3_THINKING"),
        "rate_limit_model": "grok-3",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 3 Thinking",
        "description": "Grok 3 model with deep thinking capabilities",
        "raw_model_path": "xai/grok-3",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    # ==================== Grok 4 系列 ====================
    "grok-4": {
        "grok_model": ("grok-4", "MODEL_MODE_GROK_4"),
        "rate_limit_model": "grok-4",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4",
        "description": "Standard Grok 4 model",
        "raw_model_path": "xai/grok-4",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4-mini": {
        "grok_model": ("grok-4-mini", "MODEL_MODE_GROK_4_MINI_THINKING"),
        "rate_limit_model": "grok-4-mini",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4 Mini",
        "description": "Lightweight Grok 4 model with mini thinking",
        "raw_model_path": "xai/grok-4-mini",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4-fast": {
        "grok_model": ("grok-4-mini-thinking-tahoe", "MODEL_MODE_GROK_4_MINI_THINKING"),
        "rate_limit_model": "grok-4-mini-thinking-tahoe",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4 Fast",
        "description": "Fast version of Grok 4 with mini thinking capabilities",
        "raw_model_path": "xai/grok-4-mini-thinking-tahoe",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4-fast-expert": {
        "grok_model": ("grok-4-mini-thinking-tahoe", "MODEL_MODE_EXPERT"),
        "rate_limit_model": "grok-4-mini-thinking-tahoe",
        "cost": {"type": "high_cost", "multiplier": 4, "description": "计4次调用"},
        "requires_super": False,
        "display_name": "Grok 4 Fast Expert",
        "description": "Expert mode of Grok 4 Fast with enhanced reasoning",
        "raw_model_path": "xai/grok-4-mini-thinking-tahoe",
        "default_temperature": 1.0,
        "default_max_output_tokens": 32768,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4-thinking": {
        "grok_model": ("grok-4", "MODEL_MODE_GROK_4_THINKING"),
        "rate_limit_model": "grok-4",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4 Thinking",
        "description": "Grok 4 model with deep thinking capabilities",
        "raw_model_path": "xai/grok-4",
        "default_temperature": 1.0,
        "default_max_output_tokens": 32768,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4-expert": {
        "grok_model": ("grok-4", "MODEL_MODE_EXPERT"),
        "rate_limit_model": "grok-4",
        "cost": {"type": "high_cost", "multiplier": 4, "description": "计4次调用"},
        "requires_super": False,
        "display_name": "Grok 4 Expert",
        "description": "Full Grok 4 model with expert mode capabilities",
        "raw_model_path": "xai/grok-4",
        "default_temperature": 1.0,
        "default_max_output_tokens": 32768,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4-heavy": {
        "grok_model": ("grok-4-heavy", "MODEL_MODE_HEAVY"),
        "rate_limit_model": "grok-4-heavy",
        "cost": {"type": "independent", "multiplier": 1, "description": "独立计费，只有Super用户可用"},
        "requires_super": True,
        "display_name": "Grok 4 Heavy",
        "description": "Most powerful Grok 4 model with heavy computational capabilities. Requires Super Token for access.",
        "raw_model_path": "xai/grok-4-heavy",
        "default_temperature": 1.0,
        "default_max_output_tokens": 65536,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    # ==================== Grok 4.1 系列 ====================
    "grok-4.1-mini": {
        "grok_model": ("grok-4-1-thinking-1129", "MODEL_MODE_GROK_4_1_MINI_THINKING"),
        "rate_limit_model": "grok-4-1-thinking-1129",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4.1 Mini",
        "description": "Lightweight Grok 4.1 model with mini thinking",
        "raw_model_path": "xai/grok-4-1-thinking-1129",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4.1-fast": {
        "grok_model": ("grok-4-1-thinking-1129", "MODEL_MODE_FAST"),
        "rate_limit_model": "grok-4-1-thinking-1129",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4.1 Fast",
        "description": "Fast version of Grok 4.1",
        "raw_model_path": "xai/grok-4-1-thinking-1129",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4.1-expert": {
        "grok_model": ("grok-4-1-thinking-1129", "MODEL_MODE_EXPERT"),
        "rate_limit_model": "grok-4-1-thinking-1129",
        "cost": {"type": "high_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4.1 Expert",
        "description": "Expert mode of Grok 4.1 with enhanced reasoning",
        "raw_model_path": "xai/grok-4-1-thinking-1129",
        "default_temperature": 1.0,
        "default_max_output_tokens": 32768,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    "grok-4.1-thinking": {
        "grok_model": ("grok-4-1-thinking-1129", "MODEL_MODE_GROK_4_1_THINKING"),
        "rate_limit_model": "grok-4-1-thinking-1129",
        "cost": {"type": "high_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4.1 Thinking",
        "description": "Grok 4.1 model with advanced thinking and tool capabilities",
        "raw_model_path": "xai/grok-4-1-thinking-1129",
        "default_temperature": 1.0,
        "default_max_output_tokens": 32768,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    # ==================== Grok 4.20 ====================
    "grok-4.20-beta": {
        "grok_model": ("grok-420", "MODEL_MODE_GROK_420"),
        "rate_limit_model": "grok-420",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 4.20 Beta",
        "description": "Grok 4.20 Beta model",
        "raw_model_path": "xai/grok-420",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95
    },
    # ==================== 图像模型 ====================
    "grok-2-image": {
        "grok_model": ("grok-2-image", "MODEL_MODE_FAST"),
        "rate_limit_model": "grok-2-image",
        "cost": {"type": "low_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok 2 Image",
        "description": "Image generation (WS) + image editing (REST). Auto-dispatches by input.",
        "raw_model_path": "xai/grok-2-image",
        "default_temperature": 1.0,
        "default_max_output_tokens": 4096,
        "supported_max_output_tokens": 8192,
        "default_top_p": 0.95,
        "image_generation_count": 4,
        "prompt_style": "imagine",
        "channel": "imagine_ws_smart",
        "image_edit_model": "imagine-image-edit"
    },
    "grok-imagine-1.0": {
        "grok_model": ("grok-3", "MODEL_MODE_FAST"),
        "rate_limit_model": "grok-3",
        "cost": {"type": "high_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok Imagine 1.0",
        "description": "Unified image model: text-to-image (WebSocket) + image-to-image edit (REST API).",
        "raw_model_path": "xai/grok-imagine-1.0",
        "default_temperature": 1.0,
        "default_max_output_tokens": 4096,
        "supported_max_output_tokens": 8192,
        "default_top_p": 0.95,
        "image_generation_count": 4,
        "prompt_style": "imagine",
        "channel": "imagine_1.0",
        "image_edit_model": "imagine-image-edit"
    },
    "grok-imagine-1.0-video": {
        "grok_model": ("grok-3", "MODEL_MODE_FAST"),
        "rate_limit_model": "grok-3",
        "cost": {"type": "high_cost", "multiplier": 1, "description": "计1次调用"},
        "requires_super": False,
        "display_name": "Grok Imagine 1.0 Video",
        "description": "Video generation model. Supports text-to-video and image-to-video.",
        "raw_model_path": "xai/grok-imagine-1.0-video",
        "default_temperature": 1.0,
        "default_max_output_tokens": 8192,
        "supported_max_output_tokens": 131072,
        "default_top_p": 0.95,
        "is_video_model": True
    },
}


class TokenType(Enum):
    """Token类型"""
    NORMAL = "ssoNormal"
    SUPER = "ssoSuper"


class Models(Enum):
    """支持的模型"""
    # Grok 3 系列
    GROK_3 = "grok-3"
    GROK_3_FAST = "grok-3-fast"
    GROK_3_MINI = "grok-3-mini"
    GROK_3_THINKING = "grok-3-thinking"
    # Grok 4 系列
    GROK_4 = "grok-4"
    GROK_4_MINI = "grok-4-mini"
    GROK_4_FAST = "grok-4-fast"
    GROK_4_FAST_EXPERT = "grok-4-fast-expert"
    GROK_4_THINKING = "grok-4-thinking"
    GROK_4_EXPERT = "grok-4-expert"
    GROK_4_HEAVY = "grok-4-heavy"
    # Grok 4.1 系列
    GROK_4_1_MINI = "grok-4.1-mini"
    GROK_4_1_FAST = "grok-4.1-fast"
    GROK_4_1_EXPERT = "grok-4.1-expert"
    GROK_4_1_THINKING = "grok-4.1-thinking"
    # Grok 4.20
    GROK_4_20_BETA = "grok-4.20-beta"
    # 图像/视频模型
    GROK_2_IMAGE = "grok-2-image"
    GROK_IMAGINE_1_0 = "grok-imagine-1.0"
    GROK_IMAGINE_1_0_VIDEO = "grok-imagine-1.0-video"

    @classmethod
    def get_model_info(cls, model: str) -> Dict[str, Any]:
        """获取模型配置"""
        return _MODEL_CONFIG.get(model, {})

    @classmethod
    def is_valid_model(cls, model: str) -> bool:
        """检查模型是否有效"""
        return model in _MODEL_CONFIG
     
    @classmethod
    def to_grok(cls, model: str) -> Tuple[str, str]:
        """转换为Grok内部模型名和模式
        
        Returns:
            (模型名, 模式类型) 元组
        """
        config = _MODEL_CONFIG.get(model)
        return config["grok_model"] if config else (model, "MODEL_MODE_FAST")
    
    @classmethod
    def to_rate_limit(cls, model: str) -> str:
        """转换为速率限制模型名"""
        config = _MODEL_CONFIG.get(model)
        return config["rate_limit_model"] if config else model
    
    @classmethod
    def get_all_model_names(cls) -> list[str]:
        """获取所有模型名称"""
        return list(_MODEL_CONFIG.keys())
