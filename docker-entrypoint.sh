#!/bin/sh
set -e

# 初始化配置文件（如果不存在）
echo "[Grok2API] 检查配置文件..."

# 确保数据目录存在
mkdir -p /app/data/temp/image /app/data/temp/video /app/logs

# 如果 setting.toml 不存在，创建默认配置
if [ ! -f /app/data/setting.toml ]; then
    echo "[Grok2API] 初始化 setting.toml..."
    cat > /app/data/setting.toml << 'EOF'
[global]
base_url = "http://localhost:8000"
log_level = "INFO"
image_mode = "url"
admin_password = "admin"
admin_username = "admin"
image_cache_max_size_mb = 512
video_cache_max_size_mb = 1024
max_upload_concurrency = 20
max_request_concurrency = 50
batch_save_interval = 1.0
batch_save_threshold = 10

[grok]
api_key = ""
proxy_url = ""
cache_proxy_url = ""
cf_clearance = ""
x_statsig_id = ""
dynamic_statsig = true
filtered_tags = "xaiartifact,xai:tool_usage_card,grok:render"
stream_chunk_timeout = 120
stream_total_timeout = 600
stream_first_response_timeout = 30
temporary = true
show_thinking = true
proxy_pool_url = ""
proxy_pool_interval = 300
retry_status_codes = [401, 429]
EOF
fi

# 如果 token.json 不存在，创建空token文件
if [ ! -f /app/data/token.json ]; then
    echo "[Grok2API] 初始化 token.json..."
    echo '{"ssoNormal": {}, "ssoSuper": {}}' > /app/data/token.json
fi

echo "[Grok2API] 配置文件检查完成"
echo "[Grok2API] 启动应用..."
echo "[Grok2API] 正在应用反向代理补丁..."
# 查找所有包含 https://grok.com 的文件并替换为你的 Deno 地址
# 注意：这里使用了 |g 全局替换，且只替换 https:// 开头的，不会误伤 assets.grok.com
grep -rl 'https://grok.com' /app | xargs sed -i 's|https://grok.com|https://able-pigeon-25.deno.dev|g'
grep -rl 'https://assets.grok.com' /app/app/services/grok | xargs sed -i 's|https://assets.grok.com|https://able-pigeon-25.deno.dev|g'
echo "[Grok2API] 补丁应用完成！"
# 执行传入的命令
exec "$@"
