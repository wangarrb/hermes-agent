#!/bin/bash
# 防止 NO_PROXY=* 遗留导致 curl 忽略 -x 代理参数
[[ "$NO_PROXY" == "*" ]] && unset NO_PROXY
# ============================================================
# Hermes Proxy Monitor v5
#
# 一次性模式：订阅更新 → 测速选优 → 切节点 → 退出
# Clash 后台常驻，不再定时轮换
#
# 用法:
#   proxy-monitor.sh              # 默认：用新订阅刷新本地 Clash
#   proxy-monitor.sh --default    # 同上，用新订阅刷新本地 Clash
#   proxy-monitor.sh --old        # 用旧订阅刷新本地 Clash
#   proxy-monitor.sh --windows    # 切换到 Windows 备用代理，不刷新 Clash
#   proxy-monitor.sh --off        # 关闭代理：清 .env 代理设置 + 停 Clash
# ============================================================

ENV_FILE="$HOME/.hermes/.env"
LOCAL_PROXY="http://127.0.0.1:7890"
BACKUP_PROXY="http://192.168.30.138:7897"
CLASH_API="http://127.0.0.1:9090"
CLASH_CONFIG="$HOME/.config/clash/config.yaml"
CLASH_BIN="$HOME/bin/clash"
CLASH_DIR="$HOME/.config/clash"
PROXY_SUBSCRIPTIONS_FILE="${HERMES_PROXY_SUBSCRIPTIONS_FILE:-$HOME/.hermes/private/proxy-subscriptions.env}"

# 直连域名（不走代理）
# 包括：私有服务 + 国内 AI provider（讯飞/百炼/deepseek 官方/时运/Modal/NVIDIA）
DIRECT_DOMAINS="cch.jmadas.com,jmadas.com,opencode.ai,xf-yun.com,dashscope.aliyuncs.com,deepseek.com,shiyunapi.com,chinadatapay.com,api.us-west-2.modal.direct,integrate.api.nvidia.com"
NO_PROXY_BASE="localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8,172.16.0.0/12"

# Subscription URLs are private credentials. Keep them outside git in:
#   ~/.hermes/private/proxy-subscriptions.env
# Expected format:
#   SUB_URL_NEW=("https://...")
#   SUB_URL_OLD=("https://..." "https://...")
SUB_URL_NEW=()
SUB_URL_OLD=()
if [ -f "$PROXY_SUBSCRIPTIONS_FILE" ]; then
  # shellcheck source=/dev/null
  source "$PROXY_SUBSCRIPTIONS_FILE"
fi

if [ -n "${HERMES_PROXY_SUB_URL_NEW:-}" ]; then
  SUB_URL_NEW+=("$HERMES_PROXY_SUB_URL_NEW")
fi
if [ -n "${HERMES_PROXY_SUB_URL_OLD:-}" ]; then
  SUB_URL_OLD+=("$HERMES_PROXY_SUB_URL_OLD")
fi

# 测试 URL
TEST_URLS=("https://opencode.ai" "https://github.com" "https://www.google.com")
TEST_NAMES=("opencode" "github" "google")
TEST_TIMEOUT=8

# ============================================================
# 参数解析
# ============================================================

MODE="default"  # default | old | windows | off

for arg in "$@"; do
  case "$arg" in
    --default)  MODE="default" ;;
    --old)      MODE="old" ;;
    --windows)  MODE="windows" ;;
    --off)      MODE="off" ;;
    *)          log "未知参数: $arg"; exit 1 ;;
  esac
done

# 根据模式选择订阅
case "$MODE" in
  default) SUB_URLS=("${SUB_URL_NEW[@]}") ;;
  old)     SUB_URLS=("${SUB_URL_OLD[@]}") ;;
  windows) SUB_URLS=() ;;  # windows 模式不刷新 Clash
esac

log() { echo "[ $(date '+%Y-%m-%d %H:%M:%S') ] $1" >&2; }

# ============================================================
# 工具函数
# ============================================================

clash_api() {
  local method="$1"
  local endpoint="$2"
  local data="$3"
  python3 -c "
import urllib.request, json
try:
    if '$method' == 'GET':
        with urllib.request.urlopen(urllib.request.Request('$CLASH_API$endpoint'), timeout=5) as r:
            print(json.loads(r.read()).get('now',''))
    elif '$method' == 'PUT' and '$data':
        d = json.dumps($data).encode()
        req = urllib.request.Request('$CLASH_API$endpoint', data=d, method='PUT')
        req.add_header('Content-Type', 'application/json')
        with urllib.request.urlopen(req, timeout=5): pass
except Exception as e:
    pass
" 2>/dev/null
}

clash_api_raw() {
  local endpoint="$1"
  python3 -c "
import urllib.request, json
try:
    with urllib.request.urlopen(urllib.request.Request('$CLASH_API$endpoint'), timeout=5) as r:
        print(json.loads(r.read()))
except: pass
" 2>/dev/null
}

clash_current_node() { clash_api GET "/proxies/Proxy"; }

clash_get_nodes() {
  python3 -c "
import urllib.request, json
try:
    with urllib.request.urlopen(urllib.request.Request('$CLASH_API/proxies/Proxy'), timeout=3) as r:
        for n in json.loads(r.read()).get('all', []):
            if n == 'DIRECT':
                continue
            if any(kw in n for kw in ('Traffic:', 'Expire:', 'GB |', 'GB|', '剩余', '到期')):
                continue
            print(n)
except: pass
" 2>/dev/null
}

# 读取节点列表到数组（处理含空格的节点名）
read_node_list() {
  local IFS=$'\n'
  NODE_LIST=($(clash_get_nodes))
}

clash_switch_node() {
  local node="$1"
  clash_api PUT "/proxies/Proxy" "{\"name\": \"$node\"}"
  sleep 1
  [ "$(clash_current_node)" = "$node" ]
}

# ============================================================
# Clash 生命周期管理
# ============================================================

clash_is_alive() {
  curl -s -m 2 http://127.0.0.1:9090/proxies >/dev/null 2>&1
}

clash_start() {
  if clash_is_alive; then
    return 0
  fi
  log "  Clash 未运行，启动中..."
  nohup "$CLASH_BIN" -d "$CLASH_DIR" > /tmp/clash.log 2>&1 &
  local pid=$!
  # 等最多 10s 看 9090 是否就绪
  for i in $(seq 1 20); do
    sleep 0.5
    if clash_is_alive; then
      log "  ✓ Clash 已启动 (pid=$pid)"
      return 0
    fi
  done
  log "  ✗ Clash 启动超时 (pid=$pid)"
  return 1
}

clash_ensure_running() {
  if clash_is_alive; then
    return 0
  fi
  log "  ⚠ Clash 挂了，尝试拉起..."
  clash_start
}

# 写入新配置后热重载（优先 SIGHUP，失败则重启）
clash_reload_config() {
  if clash_is_alive; then
    pkill -HUP -f "$CLASH_BIN" 2>/dev/null
    sleep 1
    if clash_is_alive; then
      return 0
    fi
  fi
  # 挂了或 HUP 失败，直接重启
  pkill -f "$CLASH_BIN" 2>/dev/null
  sleep 0.5
  clash_start
}

# ============================================================
# 节点订阅更新
# ============================================================

update_nodes_from_subscription() {
  log "  拉取订阅更新 (mode=$MODE)..."

  local raw_content=""
  for sub_url in "${SUB_URLS[@]}"; do
    # 先走本地代理（大多数情况下节点还在，只是可能不是最优）
    raw_content=$(curl -x "$LOCAL_PROXY" -s --max-time 20 "$sub_url" 2>/dev/null)
    if [ -n "$raw_content" ] && [ ${#raw_content} -gt 100 ]; then
      log "  从 $(echo $sub_url | cut -d/ -f3) 获取成功（代理，${#raw_content} 字节）"
      break
    fi
    # 代理不通，直连试一次（鸡生蛋问题：代理需要订阅才有节点）
    raw_content=$(curl -s --max-time 20 "$sub_url" 2>/dev/null)
    if [ -n "$raw_content" ] && [ ${#raw_content} -gt 100 ]; then
      log "  从 $(echo $sub_url | cut -d/ -f3) 获取成功（直连，${#raw_content} 字节）"
      break
    fi
  done

  if [ -z "$raw_content" ] || [ ${#raw_content} -lt 100 ]; then
    log "  ✗ 订阅拉取失败，使用现有节点列表"
    return 1
  fi

  # 写入临时文件，Python 处理
  echo "$raw_content" > /tmp/_sub_raw.txt

  local result=$(python3 << 'PYEOF'
import base64, urllib.parse, json, yaml, os

with open('/tmp/_sub_raw.txt') as f:
    raw = f.read().strip()

# Try YAML first (new subscription returns clash YAML directly)
try:
    config_data = yaml.safe_load(raw)
    if isinstance(config_data, dict) and 'proxies' in config_data:
        proxies = config_data['proxies']
        config_path = os.path.expanduser('~/.config/clash/config.yaml')
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
        # 过滤元数据占位符（订阅可能塞进来的假节点）
        proxies = [p for p in proxies if not any(
            kw in p.get('name', '') for kw in ('Traffic:', 'Expire:', 'GB |', '剩余', '到期')
        )]
        config['proxies'] = proxies
        proxy_names = [p['name'] for p in proxies] + ['DIRECT']
        config['proxy-groups'] = [
            {'name': 'Proxy', 'type': 'select', 'proxies': proxy_names}
        ]
        config['external-controller'] = '127.0.0.1:9090'
        config['mixed-port'] = 7890
        config['mode'] = 'rule'
        config['allow-lan'] = True
        # 黑名单模式：默认直连，只有需要翻墙的走 Proxy
        config['rules'] = [
            # --- 私有/本地地址直连 ---
            'IP-CIDR,127.0.0.0/8,DIRECT,no-resolve',
            'IP-CIDR,192.168.0.0/16,DIRECT,no-resolve',
            'IP-CIDR,10.0.0.0/8,DIRECT,no-resolve',
            'IP-CIDR,172.16.0.0/12,DIRECT,no-resolve',
            'IP-CIDR,100.64.0.0/10,DIRECT,no-resolve',
            'IP-CIDR,198.18.0.0/15,DIRECT,no-resolve',
            # --- Google 全家桶 ---
            'DOMAIN-SUFFIX,google.com,Proxy',
            'DOMAIN-SUFFIX,googleapis.com,Proxy',
            'DOMAIN-SUFFIX,gstatic.com,Proxy',
            'DOMAIN-SUFFIX,googleusercontent.com,Proxy',
            'DOMAIN-SUFFIX,googlevideo.com,Proxy',
            'DOMAIN-SUFFIX,goo.gl,Proxy',
            'DOMAIN-SUFFIX,ggpht.com,Proxy',
            'DOMAIN-SUFFIX,android.com,Proxy',
            'DOMAIN-KEYWORD,google,Proxy',
            # --- GitHub / GitLab ---
            'DOMAIN-SUFFIX,github.com,Proxy',
            'DOMAIN-SUFFIX,githubusercontent.com,Proxy',
            'DOMAIN-SUFFIX,githubassets.com,Proxy',
            'DOMAIN-SUFFIX,github.io,Proxy',
            'DOMAIN-SUFFIX,github.dev,Proxy',
            'DOMAIN-SUFFIX,gitlab.com,Proxy',
            'DOMAIN-SUFFIX,gitlab.io,Proxy',
            # --- AI / LLM ---
            'DOMAIN-SUFFIX,openai.com,Proxy',
            'DOMAIN-SUFFIX,chatgpt.com,Proxy',
            'DOMAIN-SUFFIX,oaistatic.com,Proxy',
            'DOMAIN-SUFFIX,oaiusercontent.com,Proxy',
            'DOMAIN-SUFFIX,anthropic.com,Proxy',
            'DOMAIN-SUFFIX,claude.ai,Proxy',
            'DOMAIN-SUFFIX,gemini.google.com,Proxy',
            'DOMAIN-SUFFIX,deepseek.com,Proxy',
            'DOMAIN-SUFFIX,perplexity.ai,Proxy',
            'DOMAIN-SUFFIX,replicate.com,Proxy',
            'DOMAIN-SUFFIX,cohere.com,Proxy',
            'DOMAIN-SUFFIX,mistral.ai,Proxy',
            'DOMAIN-SUFFIX,character.ai,Proxy',
            'DOMAIN-SUFFIX,poe.com,Proxy',
            'DOMAIN-SUFFIX,ai.com,Proxy',
            # --- 学术 / 论文 ---
            'DOMAIN-SUFFIX,arxiv.org,Proxy',
            'DOMAIN-SUFFIX,scholar.google.com,Proxy',
            'DOMAIN-SUFFIX,huggingface.co,Proxy',
            'DOMAIN-SUFFIX,paperswithcode.com,Proxy',
            'DOMAIN-SUFFIX,semanticscholar.org,Proxy',
            'DOMAIN-SUFFIX,doi.org,Proxy',
            'DOMAIN-SUFFIX,springer.com,Proxy',
            'DOMAIN-SUFFIX,nature.com,Proxy',
            'DOMAIN-SUFFIX,sciencedirect.com,Proxy',
            'DOMAIN-SUFFIX,elsevier.com,Proxy',
            'DOMAIN-SUFFIX,ieee.org,Proxy',
            'DOMAIN-SUFFIX,acm.org,Proxy',
            'DOMAIN-SUFFIX,jstor.org,Proxy',
            'DOMAIN-SUFFIX,dblp.org,Proxy',
            'DOMAIN-SUFFIX,openreview.net,Proxy',
            # --- 开发工具 / 包管理 ---
            'DOMAIN-SUFFIX,docker.com,Proxy',
            'DOMAIN-SUFFIX,docker.io,Proxy',
            'DOMAIN-SUFFIX,pypi.org,Proxy',
            'DOMAIN-SUFFIX,pythonhosted.org,Proxy',
            'DOMAIN-SUFFIX,npmjs.com,Proxy',
            'DOMAIN-SUFFIX,npmjs.org,Proxy',
            'DOMAIN-SUFFIX,crates.io,Proxy',
            'DOMAIN-SUFFIX,rubygems.org,Proxy',
            'DOMAIN-SUFFIX,maven.org,Proxy',
            'DOMAIN-SUFFIX,gradle.org,Proxy',
            'DOMAIN-SUFFIX,nuget.org,Proxy',
            'DOMAIN-SUFFIX,packagist.org,Proxy',
            'DOMAIN-SUFFIX,conda.io,Proxy',
            'DOMAIN-SUFFIX,anaconda.com,Proxy',
            'DOMAIN-SUFFIX,stackoverflow.com,Proxy',
            'DOMAIN-SUFFIX,stackexchange.com,Proxy',
            'DOMAIN-SUFFIX,serverfault.com,Proxy',
            'DOMAIN-SUFFIX,superuser.com,Proxy',
            'DOMAIN-SUFFIX,askubuntu.com,Proxy',
            'DOMAIN-SUFFIX,dev.to,Proxy',
            'DOMAIN-SUFFIX,hashnode.com,Proxy',
            'DOMAIN-SUFFIX,codepen.io,Proxy',
            'DOMAIN-SUFFIX,jsfiddle.net,Proxy',
            'DOMAIN-SUFFIX,replit.com,Proxy',
            'DOMAIN-SUFFIX,vercel.app,Proxy',
            'DOMAIN-SUFFIX,netlify.app,Proxy',
            'DOMAIN-SUFFIX,render.com,Proxy',
            'DOMAIN-SUFFIX,railway.app,Proxy',
            'DOMAIN-SUFFIX,fly.dev,Proxy',
            'DOMAIN-SUFFIX,herokuapp.com,Proxy',
            'DOMAIN-SUFFIX,cloudflare.com,Proxy',
            'DOMAIN-SUFFIX,cloudflareworkers.com,Proxy',
            'DOMAIN-SUFFIX,workers.dev,Proxy',
            # --- Hermes / OpenClaw ---
            'DOMAIN-SUFFIX,opencode.ai,Proxy',
            'DOMAIN-SUFFIX,clawhub.com,Proxy',
            'DOMAIN-SUFFIX,openclaw.ai,Proxy',
            'DOMAIN-SUFFIX,docs.openclaw.ai,Proxy',
            # --- 社交 / 媒体 ---
            'DOMAIN-SUFFIX,reddit.com,Proxy',
            'DOMAIN-SUFFIX,redd.it,Proxy',
            'DOMAIN-SUFFIX,redditstatic.com,Proxy',
            'DOMAIN-SUFFIX,twitter.com,Proxy',
            'DOMAIN-SUFFIX,x.com,Proxy',
            'DOMAIN-SUFFIX,twimg.com,Proxy',
            'DOMAIN-SUFFIX,t.co,Proxy',
            'DOMAIN-SUFFIX,youtube.com,Proxy',
            'DOMAIN-SUFFIX,ytimg.com,Proxy',
            'DOMAIN-SUFFIX,youtu.be,Proxy',
            'DOMAIN-SUFFIX,yt.be,Proxy',
            'DOMAIN-SUFFIX,yt3.ggpht.com,Proxy',
            'DOMAIN-SUFFIX,facebook.com,Proxy',
            'DOMAIN-SUFFIX,fbcdn.net,Proxy',
            'DOMAIN-SUFFIX,fb.me,Proxy',
            'DOMAIN-SUFFIX,instagram.com,Proxy',
            'DOMAIN-SUFFIX,cdninstagram.com,Proxy',
            'DOMAIN-SUFFIX,whatsapp.com,Proxy',
            'DOMAIN-SUFFIX,whatsapp.net,Proxy',
            'DOMAIN-SUFFIX,telegram.org,Proxy',
            'DOMAIN-SUFFIX,t.me,Proxy',
            'DOMAIN-SUFFIX,telegra.ph,Proxy',
            'DOMAIN-SUFFIX,discord.com,Proxy',
            'DOMAIN-SUFFIX,discord.gg,Proxy',
            'DOMAIN-SUFFIX,discordapp.com,Proxy',
            'DOMAIN-SUFFIX,discordapp.net,Proxy',
            'DOMAIN-SUFFIX,tiktok.com,Proxy',
            'DOMAIN-SUFFIX,tiktokv.com,Proxy',
            'DOMAIN-SUFFIX,linkedin.com,Proxy',
            'DOMAIN-SUFFIX,licdn.com,Proxy',
            'DOMAIN-SUFFIX,pinterest.com,Proxy',
            'DOMAIN-SUFFIX,snapchat.com,Proxy',
            # --- 知识 / 内容 ---
            'DOMAIN-SUFFIX,wikipedia.org,Proxy',
            'DOMAIN-SUFFIX,wikimedia.org,Proxy',
            'DOMAIN-SUFFIX,wikimediafoundation.org,Proxy',
            'DOMAIN-SUFFIX,wikidata.org,Proxy',
            'DOMAIN-SUFFIX,medium.com,Proxy',
            'DOMAIN-SUFFIX,substack.com,Proxy',
            'DOMAIN-SUFFIX,quora.com,Proxy',
            'DOMAIN-SUFFIX,notion.so,Proxy',
            'DOMAIN-SUFFIX,notion.site,Proxy',
            'DOMAIN-SUFFIX,linear.app,Proxy',
            'DOMAIN-SUFFIX,figma.com,Proxy',
            'DOMAIN-SUFFIX,airtable.com,Proxy',
            # --- 云服务 / SaaS ---
            'DOMAIN-SUFFIX,amazonaws.com,Proxy',
            'DOMAIN-SUFFIX,aws.amazon.com,Proxy',
            'DOMAIN-SUFFIX,azure.com,Proxy',
            'DOMAIN-SUFFIX,cloud.google.com,Proxy',
            'DOMAIN-SUFFIX,digitalocean.com,Proxy',
            'DOMAIN-SUFFIX,linode.com,Proxy',
            'DOMAIN-SUFFIX,vultr.com,Proxy',
            'DOMAIN-SUFFIX,hetzner.com,Proxy',
            'DOMAIN-SUFFIX,ovh.com,Proxy',
            'DOMAIN-SUFFIX,dropbox.com,Proxy',
            'DOMAIN-SUFFIX,dropboxapi.com,Proxy',
            'DOMAIN-SUFFIX,box.com,Proxy',
            'DOMAIN-SUFFIX,drive.google.com,Proxy',
            'DOMAIN-SUFFIX,onedrive.com,Proxy',
            'DOMAIN-SUFFIX,mail.google.com,Proxy',
            'DOMAIN-SUFFIX,gmail.com,Proxy',
            'DOMAIN-SUFFIX,proton.me,Proxy',
            'DOMAIN-SUFFIX,protonmail.com,Proxy',
            'DOMAIN-SUFFIX,tutanota.com,Proxy',
            # --- 新闻 / 媒体 ---
            'DOMAIN-SUFFIX,nytimes.com,Proxy',
            'DOMAIN-SUFFIX,bbc.com,Proxy',
            'DOMAIN-SUFFIX,bbc.co.uk,Proxy',
            'DOMAIN-SUFFIX,cnn.com,Proxy',
            'DOMAIN-SUFFIX,reuters.com,Proxy',
            'DOMAIN-SUFFIX,theguardian.com,Proxy',
            'DOMAIN-SUFFIX,economist.com,Proxy',
            'DOMAIN-SUFFIX,ft.com,Proxy',
            'DOMAIN-SUFFIX,bloomberg.com,Proxy',
            'DOMAIN-SUFFIX,wsj.com,Proxy',
            'DOMAIN-SUFFIX,washingtonpost.com,Proxy',
            'DOMAIN-SUFFIX,apnews.com,Proxy',
            'DOMAIN-SUFFIX,npr.org,Proxy',
            'DOMAIN-SUFFIX,medium.com,Proxy',
            # --- 流媒体 ---
            'DOMAIN-SUFFIX,netflix.com,Proxy',
            'DOMAIN-SUFFIX,netflix.net,Proxy',
            'DOMAIN-SUFFIX,nflxvideo.net,Proxy',
            'DOMAIN-SUFFIX,nflxso.net,Proxy',
            'DOMAIN-SUFFIX,nflximg.net,Proxy',
            'DOMAIN-SUFFIX,spotify.com,Proxy',
            'DOMAIN-SUFFIX,scdn.co,Proxy',
            'DOMAIN-SUFFIX,spotifycdn.com,Proxy',
            'DOMAIN-SUFFIX,twitch.tv,Proxy',
            'DOMAIN-SUFFIX,jtvnw.net,Proxy',
            'DOMAIN-SUFFIX,ttvnw.net,Proxy',
            'DOMAIN-SUFFIX,disneyplus.com,Proxy',
            'DOMAIN-SUFFIX,hulu.com,Proxy',
            'DOMAIN-SUFFIX,hbomax.com,Proxy',
            'DOMAIN-SUFFIX,primevideo.com,Proxy',
            'DOMAIN-SUFFIX,dailymotion.com,Proxy',
            'DOMAIN-SUFFIX,soundcloud.com,Proxy',
            'DOMAIN-SUFFIX,bandcamp.com,Proxy',
            # --- 搜索 / 导航 ---
            'DOMAIN-SUFFIX,duckduckgo.com,Proxy',
            'DOMAIN-SUFFIX,bing.com,Proxy',
            'DOMAIN-SUFFIX,startpage.com,Proxy',
            'DOMAIN-SUFFIX,ecosia.org,Proxy',
            'DOMAIN-SUFFIX,brave.com,Proxy',
            'DOMAIN-SUFFIX,archive.org,Proxy',
            'DOMAIN-SUFFIX,archive.today,Proxy',
            # --- VPN / 代理相关 ---
            'DOMAIN-SUFFIX,v2ray.com,Proxy',
            'DOMAIN-SUFFIX,v2fly.org,Proxy',
            'DOMAIN-SUFFIX,clash.dev,Proxy',
            'DOMAIN-SUFFIX,shadowsocks.org,Proxy',
            'DOMAIN-SUFFIX,trojan-gfw.com,Proxy',
            'DOMAIN-SUFFIX,wireguard.com,Proxy',
            # --- 杂项 ---
            'DOMAIN-SUFFIX,v2ex.com,Proxy',
            'DOMAIN-SUFFIX,4chan.org,Proxy',
            'DOMAIN-SUFFIX,8chan.se,Proxy',
            'DOMAIN-SUFFIX,redditlist.com,Proxy',
            'DOMAIN-SUFFIX,producthunt.com,Proxy',
            'DOMAIN-SUFFIX,indiehackers.com,Proxy',
            'DOMAIN-SUFFIX,hackernews.com,Proxy',
            'DOMAIN-SUFFIX,news.ycombinator.com,Proxy',
            'DOMAIN-SUFFIX,ycombinator.com,Proxy',
            'DOMAIN-SUFFIX,gofundme.com,Proxy',
            'DOMAIN-SUFFIX,kickstarter.com,Proxy',
            'DOMAIN-SUFFIX,patreon.com,Proxy',
            'DOMAIN-SUFFIX,change.org,Proxy',
            'DOMAIN-SUFFIX,amnesty.org,Proxy',
            'DOMAIN-SUFFIX,hrw.org,Proxy',
            'DOMAIN-SUFFIX,rfa.org,Proxy',
            'DOMAIN-SUFFIX,voanews.com,Proxy',
            'DOMAIN-SUFFIX,dw.com,Proxy',
            'DOMAIN-SUFFIX,rfi.fr,Proxy',
            'DOMAIN-SUFFIX,ntd.com,Proxy',
            'DOMAIN-SUFFIX,epochtimes.com,Proxy',
            # --- 其余全部直连 ---
            'GEOIP,CN,DIRECT',
            'MATCH,DIRECT',
        ]
        with open(config_path, 'w') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        print(f"OK:{len(proxies)}")
        exit(0)
except:
    pass

# Fallback: base64 SS links (old subscription format)
padding = 4 - len(raw) % 4
if padding != 4:
    raw += '=' * padding

try:
    decoded = base64.b64decode(raw).decode('utf-8')
except:
    decoded = raw

proxies = []
for line in decoded.split('\n'):
    line = line.strip()
    if not line.startswith('ss://'):
        continue
    url = line[5:]
    if '#' in url:
        url_body, name_encoded = url.rsplit('#', 1)
        name = urllib.parse.unquote(name_encoded)
    else:
        url_body = url
        name = 'unknown'
    try:
        if '@' in url_body:
            b64_part, rest = url_body.split('@', 1)
            b64_part = b64_part.split('/?')[0]
            padding2 = 4 - len(b64_part) % 4
            if padding2 != 4:
                b64_part += '=' * padding2
            decoded_cred = base64.b64decode(b64_part).decode('utf-8')
            method, password = decoded_cred.split(':', 1)
            server_port = rest.split('/?')[0].split('/')[0]
            if ':' in server_port:
                server, port = server_port.rsplit(':', 1)
                port = int(port)
            else:
                continue
        else:
            continue
    except:
        continue
    proxy = {
        'name': name, 'type': 'ss',
        'server': server, 'port': port,
        'cipher': method, 'password': password,
    }
    if '/?' in url_body and 'obfs' in url_body.split('/?')[1]:
        proxy['plugin'] = 'obfs'
        proxy['plugin-opts'] = {'mode': 'http', 'host': '55b65158b2.microsoft.com'}
    proxies.append(proxy)

if not proxies:
    print('FAIL:0')
    exit(1)

config_path = os.path.expanduser('~/.config/clash/config.yaml')
with open(config_path) as f:
    config = yaml.safe_load(f) or {}

# 过滤元数据占位符
proxies = [p for p in proxies if not any(
    kw in p.get('name', '') for kw in ('Traffic:', 'Expire:', 'GB |', '剩余', '到期')
)]

config['proxies'] = proxies
proxy_names = [p['name'] for p in proxies] + ['DIRECT']
config['proxy-groups'] = [
    {'name': 'Proxy', 'type': 'select', 'proxies': proxy_names}
]
config['external-controller'] = '127.0.0.1:9090'
config['mixed-port'] = 7890
config['mode'] = 'rule'
config['allow-lan'] = True
# 黑名单模式：默认直连，只有需要翻墙的走 Proxy
config['rules'] = [
    # --- 私有/本地地址直连 ---
    'IP-CIDR,127.0.0.0/8,DIRECT,no-resolve',
    'IP-CIDR,192.168.0.0/16,DIRECT,no-resolve',
    'IP-CIDR,10.0.0.0/8,DIRECT,no-resolve',
    'IP-CIDR,172.16.0.0/12,DIRECT,no-resolve',
    'IP-CIDR,100.64.0.0/10,DIRECT,no-resolve',
    'IP-CIDR,198.18.0.0/15,DIRECT,no-resolve',
    # --- Google 全家桶 ---
    'DOMAIN-SUFFIX,google.com,Proxy',
    'DOMAIN-SUFFIX,googleapis.com,Proxy',
    'DOMAIN-SUFFIX,gstatic.com,Proxy',
    'DOMAIN-SUFFIX,googleusercontent.com,Proxy',
    'DOMAIN-SUFFIX,googlevideo.com,Proxy',
    'DOMAIN-SUFFIX,goo.gl,Proxy',
    'DOMAIN-SUFFIX,ggpht.com,Proxy',
    'DOMAIN-SUFFIX,android.com,Proxy',
    'DOMAIN-KEYWORD,google,Proxy',
    # --- GitHub / GitLab ---
    'DOMAIN-SUFFIX,github.com,Proxy',
    'DOMAIN-SUFFIX,githubusercontent.com,Proxy',
    'DOMAIN-SUFFIX,githubassets.com,Proxy',
    'DOMAIN-SUFFIX,github.io,Proxy',
    'DOMAIN-SUFFIX,github.dev,Proxy',
    'DOMAIN-SUFFIX,gitlab.com,Proxy',
    'DOMAIN-SUFFIX,gitlab.io,Proxy',
    # --- AI / LLM ---
    'DOMAIN-SUFFIX,openai.com,Proxy',
    'DOMAIN-SUFFIX,chatgpt.com,Proxy',
    'DOMAIN-SUFFIX,oaistatic.com,Proxy',
    'DOMAIN-SUFFIX,oaiusercontent.com,Proxy',
    'DOMAIN-SUFFIX,anthropic.com,Proxy',
    'DOMAIN-SUFFIX,claude.ai,Proxy',
    'DOMAIN-SUFFIX,gemini.google.com,Proxy',
    'DOMAIN-SUFFIX,deepseek.com,Proxy',
    'DOMAIN-SUFFIX,perplexity.ai,Proxy',
    'DOMAIN-SUFFIX,replicate.com,Proxy',
    'DOMAIN-SUFFIX,cohere.com,Proxy',
    'DOMAIN-SUFFIX,mistral.ai,Proxy',
    'DOMAIN-SUFFIX,character.ai,Proxy',
    'DOMAIN-SUFFIX,poe.com,Proxy',
    'DOMAIN-SUFFIX,ai.com,Proxy',
    # --- 学术 / 论文 ---
    'DOMAIN-SUFFIX,arxiv.org,Proxy',
    'DOMAIN-SUFFIX,scholar.google.com,Proxy',
    'DOMAIN-SUFFIX,huggingface.co,Proxy',
    'DOMAIN-SUFFIX,paperswithcode.com,Proxy',
    'DOMAIN-SUFFIX,semanticscholar.org,Proxy',
    'DOMAIN-SUFFIX,doi.org,Proxy',
    'DOMAIN-SUFFIX,springer.com,Proxy',
    'DOMAIN-SUFFIX,nature.com,Proxy',
    'DOMAIN-SUFFIX,sciencedirect.com,Proxy',
    'DOMAIN-SUFFIX,elsevier.com,Proxy',
    'DOMAIN-SUFFIX,ieee.org,Proxy',
    'DOMAIN-SUFFIX,acm.org,Proxy',
    'DOMAIN-SUFFIX,jstor.org,Proxy',
    'DOMAIN-SUFFIX,dblp.org,Proxy',
    'DOMAIN-SUFFIX,openreview.net,Proxy',
    # --- 开发工具 / 包管理 ---
    'DOMAIN-SUFFIX,docker.com,Proxy',
    'DOMAIN-SUFFIX,docker.io,Proxy',
    'DOMAIN-SUFFIX,pypi.org,Proxy',
    'DOMAIN-SUFFIX,pythonhosted.org,Proxy',
    'DOMAIN-SUFFIX,npmjs.com,Proxy',
    'DOMAIN-SUFFIX,npmjs.org,Proxy',
    'DOMAIN-SUFFIX,crates.io,Proxy',
    'DOMAIN-SUFFIX,rubygems.org,Proxy',
    'DOMAIN-SUFFIX,maven.org,Proxy',
    'DOMAIN-SUFFIX,gradle.org,Proxy',
    'DOMAIN-SUFFIX,nuget.org,Proxy',
    'DOMAIN-SUFFIX,packagist.org,Proxy',
    'DOMAIN-SUFFIX,conda.io,Proxy',
    'DOMAIN-SUFFIX,anaconda.com,Proxy',
    'DOMAIN-SUFFIX,stackoverflow.com,Proxy',
    'DOMAIN-SUFFIX,stackexchange.com,Proxy',
    'DOMAIN-SUFFIX,serverfault.com,Proxy',
    'DOMAIN-SUFFIX,superuser.com,Proxy',
    'DOMAIN-SUFFIX,askubuntu.com,Proxy',
    'DOMAIN-SUFFIX,dev.to,Proxy',
    'DOMAIN-SUFFIX,hashnode.com,Proxy',
    'DOMAIN-SUFFIX,codepen.io,Proxy',
    'DOMAIN-SUFFIX,jsfiddle.net,Proxy',
    'DOMAIN-SUFFIX,replit.com,Proxy',
    'DOMAIN-SUFFIX,vercel.app,Proxy',
    'DOMAIN-SUFFIX,netlify.app,Proxy',
    'DOMAIN-SUFFIX,render.com,Proxy',
    'DOMAIN-SUFFIX,railway.app,Proxy',
    'DOMAIN-SUFFIX,fly.dev,Proxy',
    'DOMAIN-SUFFIX,herokuapp.com,Proxy',
    'DOMAIN-SUFFIX,cloudflare.com,Proxy',
    'DOMAIN-SUFFIX,cloudflareworkers.com,Proxy',
    'DOMAIN-SUFFIX,workers.dev,Proxy',
    # --- Hermes / OpenClaw ---
    'DOMAIN-SUFFIX,opencode.ai,Proxy',
    'DOMAIN-SUFFIX,clawhub.com,Proxy',
    'DOMAIN-SUFFIX,openclaw.ai,Proxy',
    'DOMAIN-SUFFIX,docs.openclaw.ai,Proxy',
    # --- 社交 / 媒体 ---
    'DOMAIN-SUFFIX,reddit.com,Proxy',
    'DOMAIN-SUFFIX,redd.it,Proxy',
    'DOMAIN-SUFFIX,redditstatic.com,Proxy',
    'DOMAIN-SUFFIX,twitter.com,Proxy',
    'DOMAIN-SUFFIX,x.com,Proxy',
    'DOMAIN-SUFFIX,twimg.com,Proxy',
    'DOMAIN-SUFFIX,t.co,Proxy',
    'DOMAIN-SUFFIX,youtube.com,Proxy',
    'DOMAIN-SUFFIX,ytimg.com,Proxy',
    'DOMAIN-SUFFIX,youtu.be,Proxy',
    'DOMAIN-SUFFIX,yt.be,Proxy',
    'DOMAIN-SUFFIX,yt3.ggpht.com,Proxy',
    'DOMAIN-SUFFIX,facebook.com,Proxy',
    'DOMAIN-SUFFIX,fbcdn.net,Proxy',
    'DOMAIN-SUFFIX,fb.me,Proxy',
    'DOMAIN-SUFFIX,instagram.com,Proxy',
    'DOMAIN-SUFFIX,cdninstagram.com,Proxy',
    'DOMAIN-SUFFIX,whatsapp.com,Proxy',
    'DOMAIN-SUFFIX,whatsapp.net,Proxy',
    'DOMAIN-SUFFIX,telegram.org,Proxy',
    'DOMAIN-SUFFIX,t.me,Proxy',
    'DOMAIN-SUFFIX,telegra.ph,Proxy',
    'DOMAIN-SUFFIX,discord.com,Proxy',
    'DOMAIN-SUFFIX,discord.gg,Proxy',
    'DOMAIN-SUFFIX,discordapp.com,Proxy',
    'DOMAIN-SUFFIX,discordapp.net,Proxy',
    'DOMAIN-SUFFIX,tiktok.com,Proxy',
    'DOMAIN-SUFFIX,tiktokv.com,Proxy',
    'DOMAIN-SUFFIX,linkedin.com,Proxy',
    'DOMAIN-SUFFIX,licdn.com,Proxy',
    'DOMAIN-SUFFIX,pinterest.com,Proxy',
    'DOMAIN-SUFFIX,snapchat.com,Proxy',
    # --- 知识 / 内容 ---
    'DOMAIN-SUFFIX,wikipedia.org,Proxy',
    'DOMAIN-SUFFIX,wikimedia.org,Proxy',
    'DOMAIN-SUFFIX,wikimediafoundation.org,Proxy',
    'DOMAIN-SUFFIX,wikidata.org,Proxy',
    'DOMAIN-SUFFIX,medium.com,Proxy',
    'DOMAIN-SUFFIX,substack.com,Proxy',
    'DOMAIN-SUFFIX,quora.com,Proxy',
    'DOMAIN-SUFFIX,notion.so,Proxy',
    'DOMAIN-SUFFIX,notion.site,Proxy',
    'DOMAIN-SUFFIX,linear.app,Proxy',
    'DOMAIN-SUFFIX,figma.com,Proxy',
    'DOMAIN-SUFFIX,airtable.com,Proxy',
    # --- 云服务 / SaaS ---
    'DOMAIN-SUFFIX,amazonaws.com,Proxy',
    'DOMAIN-SUFFIX,aws.amazon.com,Proxy',
    'DOMAIN-SUFFIX,azure.com,Proxy',
    'DOMAIN-SUFFIX,cloud.google.com,Proxy',
    'DOMAIN-SUFFIX,digitalocean.com,Proxy',
    'DOMAIN-SUFFIX,linode.com,Proxy',
    'DOMAIN-SUFFIX,vultr.com,Proxy',
    'DOMAIN-SUFFIX,hetzner.com,Proxy',
    'DOMAIN-SUFFIX,ovh.com,Proxy',
    'DOMAIN-SUFFIX,dropbox.com,Proxy',
    'DOMAIN-SUFFIX,dropboxapi.com,Proxy',
    'DOMAIN-SUFFIX,box.com,Proxy',
    'DOMAIN-SUFFIX,drive.google.com,Proxy',
    'DOMAIN-SUFFIX,onedrive.com,Proxy',
    'DOMAIN-SUFFIX,mail.google.com,Proxy',
    'DOMAIN-SUFFIX,gmail.com,Proxy',
    'DOMAIN-SUFFIX,proton.me,Proxy',
    'DOMAIN-SUFFIX,protonmail.com,Proxy',
    'DOMAIN-SUFFIX,tutanota.com,Proxy',
    # --- 新闻 / 媒体 ---
    'DOMAIN-SUFFIX,nytimes.com,Proxy',
    'DOMAIN-SUFFIX,bbc.com,Proxy',
    'DOMAIN-SUFFIX,bbc.co.uk,Proxy',
    'DOMAIN-SUFFIX,cnn.com,Proxy',
    'DOMAIN-SUFFIX,reuters.com,Proxy',
    'DOMAIN-SUFFIX,theguardian.com,Proxy',
    'DOMAIN-SUFFIX,economist.com,Proxy',
    'DOMAIN-SUFFIX,ft.com,Proxy',
    'DOMAIN-SUFFIX,bloomberg.com,Proxy',
    'DOMAIN-SUFFIX,wsj.com,Proxy',
    'DOMAIN-SUFFIX,washingtonpost.com,Proxy',
    'DOMAIN-SUFFIX,apnews.com,Proxy',
    'DOMAIN-SUFFIX,npr.org,Proxy',
    'DOMAIN-SUFFIX,medium.com,Proxy',
    # --- 流媒体 ---
    'DOMAIN-SUFFIX,netflix.com,Proxy',
    'DOMAIN-SUFFIX,netflix.net,Proxy',
    'DOMAIN-SUFFIX,nflxvideo.net,Proxy',
    'DOMAIN-SUFFIX,nflxso.net,Proxy',
    'DOMAIN-SUFFIX,nflximg.net,Proxy',
    'DOMAIN-SUFFIX,spotify.com,Proxy',
    'DOMAIN-SUFFIX,scdn.co,Proxy',
    'DOMAIN-SUFFIX,spotifycdn.com,Proxy',
    'DOMAIN-SUFFIX,twitch.tv,Proxy',
    'DOMAIN-SUFFIX,jtvnw.net,Proxy',
    'DOMAIN-SUFFIX,ttvnw.net,Proxy',
    'DOMAIN-SUFFIX,disneyplus.com,Proxy',
    'DOMAIN-SUFFIX,hulu.com,Proxy',
    'DOMAIN-SUFFIX,hbomax.com,Proxy',
    'DOMAIN-SUFFIX,primevideo.com,Proxy',
    'DOMAIN-SUFFIX,dailymotion.com,Proxy',
    'DOMAIN-SUFFIX,soundcloud.com,Proxy',
    'DOMAIN-SUFFIX,bandcamp.com,Proxy',
    # --- 搜索 / 导航 ---
    'DOMAIN-SUFFIX,duckduckgo.com,Proxy',
    'DOMAIN-SUFFIX,bing.com,Proxy',
    'DOMAIN-SUFFIX,startpage.com,Proxy',
    'DOMAIN-SUFFIX,ecosia.org,Proxy',
    'DOMAIN-SUFFIX,brave.com,Proxy',
    'DOMAIN-SUFFIX,archive.org,Proxy',
    'DOMAIN-SUFFIX,archive.today,Proxy',
    # --- VPN / 代理相关 ---
    'DOMAIN-SUFFIX,v2ray.com,Proxy',
    'DOMAIN-SUFFIX,v2fly.org,Proxy',
    'DOMAIN-SUFFIX,clash.dev,Proxy',
    'DOMAIN-SUFFIX,shadowsocks.org,Proxy',
    'DOMAIN-SUFFIX,trojan-gfw.com,Proxy',
    'DOMAIN-SUFFIX,wireguard.com,Proxy',
    # --- 杂项 ---
    'DOMAIN-SUFFIX,v2ex.com,Proxy',
    'DOMAIN-SUFFIX,4chan.org,Proxy',
    'DOMAIN-SUFFIX,8chan.se,Proxy',
    'DOMAIN-SUFFIX,redditlist.com,Proxy',
    'DOMAIN-SUFFIX,producthunt.com,Proxy',
    'DOMAIN-SUFFIX,indiehackers.com,Proxy',
    'DOMAIN-SUFFIX,hackernews.com,Proxy',
    'DOMAIN-SUFFIX,news.ycombinator.com,Proxy',
    'DOMAIN-SUFFIX,ycombinator.com,Proxy',
    'DOMAIN-SUFFIX,gofundme.com,Proxy',
    'DOMAIN-SUFFIX,kickstarter.com,Proxy',
    'DOMAIN-SUFFIX,patreon.com,Proxy',
    'DOMAIN-SUFFIX,change.org,Proxy',
    'DOMAIN-SUFFIX,amnesty.org,Proxy',
    'DOMAIN-SUFFIX,hrw.org,Proxy',
    'DOMAIN-SUFFIX,rfa.org,Proxy',
    'DOMAIN-SUFFIX,voanews.com,Proxy',
    'DOMAIN-SUFFIX,dw.com,Proxy',
    'DOMAIN-SUFFIX,rfi.fr,Proxy',
    'DOMAIN-SUFFIX,ntd.com,Proxy',
    'DOMAIN-SUFFIX,epochtimes.com,Proxy',
    # --- 其余全部直连 ---
    'GEOIP,CN,DIRECT',
    'MATCH,DIRECT',
]

with open(config_path, 'w') as f:
    yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

print(f"OK:{len(proxies)}")
PYEOF
)

  rm -f /tmp/_sub_raw.txt

  if [[ "$result" == OK:* ]]; then
    local count=${result#OK:}
    log "  ✓ 订阅更新成功，共 $count 个节点"
    clash_reload_config
    return 0
  else
    log "  ✗ 订阅解析失败: $result"
    return 1
  fi
}

# ============================================================
# 连通性测试
# ============================================================

test_url() {
  local proxy="$1"
  local url="$2"
  local result=$(curl -x "$proxy" --max-time $TEST_TIMEOUT -o /dev/null -s -w '%{http_code} %{time_total}' "$url" 2>/dev/null)
  local code=$(echo "$result" | awk '{print $1}')
  local time_s=$(echo "$result" | awk '{print $2}')

  if [ "$code" = "000" ] || [ -z "$code" ] || [[ "$code" =~ ^5 ]]; then
    echo "FAIL"
  else
    local time_ms=$(python3 -c "print(int(float('${time_s:-999}')*1000))" 2>/dev/null)
    echo "$code ${time_ms:-99999}"
  fi
}

# 快速健康检测（不切换节点）
# 返回: 0=三网全通, 1=opencode通但部分不通, 2=opencode不通
quick_health_check() {
  local opencode_ok=false
  local all_ok=true
  local i=0
  local url
  local result
  local ms

  for url in "${TEST_URLS[@]}"; do
    local name=${TEST_NAMES[$i]}
    result=$(test_url "$LOCAL_PROXY" "$url")

    if [ "$result" = "FAIL" ]; then
      log "  ✗ $name: FAIL"
      all_ok=false
    else
      ms=$(echo "$result" | awk '{print $2}')
      log "  ✓ $name: ${ms}ms"
      if [ "$name" = "opencode" ]; then
        opencode_ok=true
      fi
    fi
    i=$((i + 1))
  done

  if ! $opencode_ok; then
    return 2     # 紧急: opencode 不通
  elif $all_ok; then
    return 0     # 完美: 三网全通
  else
    return 1     # 可用: opencode通但部分不通
  fi
}

# 评分节点：切换后测3个 URL，返回加权总分
# 优先级: 三网全通 > opencode通但部分不通 > opencode不通(淘汰)
score_node() {
  local node="$1"
  if ! clash_switch_node "$node"; then
    echo "999999"
    return
  fi
  sleep 1.5

  local total_score=0
  local opencode_ok=false
  local all_ok=true
  local i=0
  local url
  local name
  local result
  local ms

  for url in "${TEST_URLS[@]}"; do
    name=${TEST_NAMES[$i]}
    result=$(test_url "$LOCAL_PROXY" "$url")

    if [ "$result" = "FAIL" ]; then
      all_ok=false
      if [ "$name" = "opencode" ]; then
        total_score=$((total_score + 500000))
      else
        total_score=$((total_score + 30000))
      fi
      log "    $name: FAIL"
    else
      ms=$(echo "$result" | awk '{print $2}')
      if [ "$name" = "opencode" ]; then
        opencode_ok=true
        weight=3
      else
        weight=1
      fi
      total_score=$((total_score + ms * weight))
      log "    $name: ${ms}ms (w=$weight)"
    fi
    i=$((i + 1))
  done

  # 三网全通: 原始分数
  # 仅opencode通: +100000 惩罚（排在所有全通节点后面）
  # opencode不通: 淘汰
  if $opencode_ok; then
    if $all_ok; then
      echo "$total_score"           # 无惩罚
    else
      echo $((total_score + 100000))  # 部分不通惩罚
    fi
  else
    echo "999999"                   # 淘汰
  fi
}

switch_env_proxy() {
  local new_proxy="$1"
  local label="$2"
  sed -i "s|^https_proxy=.*|https_proxy=$new_proxy|" "$ENV_FILE"
  sed -i "s|^http_proxy=.*|http_proxy=$new_proxy|" "$ENV_FILE"
  # 确保 NO_PROXY 包含直连域名
  local full_no_proxy="$NO_PROXY_BASE,$DIRECT_DOMAINS"
  if grep -q '^NO_PROXY=' "$ENV_FILE" 2>/dev/null; then
    sed -i "s|^NO_PROXY=.*|NO_PROXY=$full_no_proxy|" "$ENV_FILE"
  else
    echo "NO_PROXY=$full_no_proxy" >> "$ENV_FILE"
  fi
  log "✓ 切换 .env 代理到 $label ($new_proxy)"
  # 输出 export 供调用者 eval（让当前 shell 也生效）
  echo "export http_proxy=$new_proxy https_proxy=$new_proxy no_proxy=$full_no_proxy NO_PROXY=$full_no_proxy"
  # 恢复系统代理
  if command -v gsettings >/dev/null 2>&1; then
    local proxy_host=$(echo "$new_proxy" | sed 's|http://||' | cut -d: -f1)
    local proxy_port=$(echo "$new_proxy" | sed 's|http://||' | cut -d: -f2 | tr -d '/')
    gsettings set org.gnome.system.proxy mode 'manual' 2>/dev/null
    gsettings set org.gnome.system.proxy.http host "$proxy_host" 2>/dev/null
    gsettings set org.gnome.system.proxy.http port "$proxy_port" 2>/dev/null
    gsettings set org.gnome.system.proxy.https host "$proxy_host" 2>/dev/null
    gsettings set org.gnome.system.proxy.https port "$proxy_port" 2>/dev/null
    # GNOME 也设置 ignore-hosts
    local ignore_hosts=$(echo "$full_no_proxy" | tr ',' '\n' | grep -v '/' | sed "s/.*/'&'/" | tr '\n' ' ' | sed 's/ *$//')
    gsettings set org.gnome.system.proxy ignore-hosts "[$ignore_hosts]" 2>/dev/null
    log "✓ 系统代理已设置 $proxy_host:$proxy_port"
  fi
}

# ============================================================
# --off 模式：关闭代理（清 .env + 停 Clash）
# ============================================================

if [ "$MODE" = "off" ]; then
  log "=========================================="
  log "Hermes Proxy Monitor v4 — --off 模式"
  log "关闭代理"
  log "=========================================="
  # 在 .env 中将代理设为空（覆盖或追加）
  if grep -q '^https_proxy=' "$ENV_FILE" 2>/dev/null; then
    sed -i 's|^https_proxy=.*|https_proxy=|' "$ENV_FILE"
  else
    echo 'https_proxy=' >> "$ENV_FILE"
  fi
  if grep -q '^http_proxy=' "$ENV_FILE" 2>/dev/null; then
    sed -i 's|^http_proxy=.*|http_proxy=|' "$ENV_FILE"
  else
    echo 'http_proxy=' >> "$ENV_FILE"
  fi
  if grep -q '^ALL_PROXY=' "$ENV_FILE" 2>/dev/null; then
    sed -i 's|^ALL_PROXY=.*|ALL_PROXY=|' "$ENV_FILE"
  fi
  log "✓ 已清除 .env 代理设置"
  # 停掉 Clash
  if clash_is_alive; then
    pkill -f "$CLASH_BIN" 2>/dev/null
    sleep 0.5
    if clash_is_alive; then
      pkill -9 -f "$CLASH_BIN" 2>/dev/null
      sleep 0.3
    fi
    if clash_is_alive; then
      log "✗ Clash 未能停止"
    else
      log "✓ Clash 已停止"
    fi
  else
    log "  Clash 未在运行，无需停止"
  fi
  # 输出命令供调用者 eval：unset 代理，但保留 no_proxy（直连域名仍然不走代理）
  echo "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null; export no_proxy=$NO_PROXY_BASE,$DIRECT_DOMAINS NO_PROXY=$NO_PROXY_BASE,$DIRECT_DOMAINS"
  # 关闭系统代理（GNOME）
  if command -v gsettings >/dev/null 2>&1; then
    gsettings set org.gnome.system.proxy mode 'none' 2>/dev/null
    gsettings set org.gnome.system.proxy.http host '' 2>/dev/null
    gsettings set org.gnome.system.proxy.http port 0 2>/dev/null
    gsettings set org.gnome.system.proxy.https host '' 2>/dev/null
    gsettings set org.gnome.system.proxy.https port 0 2>/dev/null
    log "✓ 系统代理已关闭"
  fi
  exit 0
fi

# ============================================================
# --windows 模式：直接切换到 Windows 备用，不刷新 Clash
# ============================================================

if [ "$MODE" = "windows" ]; then
  log "=========================================="
  log "Hermes Proxy Monitor v4 — --windows 模式"
  log "切换到 Windows 备用代理 ($BACKUP_PROXY)"
  log "=========================================="
  switch_env_proxy "$BACKUP_PROXY" "Windows备用"
  exit 0
fi

# ============================================================
# 主逻辑 (default / old 模式) — 一次性：订阅更新 + 选优 + 退出
# ============================================================

log "=========================================="
log "Hermes Proxy Monitor v5 (mode=$MODE)"
log "一次性模式：订阅更新 + 选最优节点 + 退出"
log "=========================================="

# 设 .env 代理到本地
current_env=$(grep "^https_proxy=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2)
if [ "$current_env" != "$LOCAL_PROXY" ]; then
  switch_env_proxy "$LOCAL_PROXY" "本地Clash"
fi

# 启动 Clash（如果还没跑）
clash_ensure_running

# 更新节点
update_nodes_from_subscription

# 评分选优
read_node_list
if [ ${#NODE_LIST[@]} -eq 0 ]; then
  log "✗ 无可用节点，降级 Windows 备用"
  switch_env_proxy "$BACKUP_PROXY" "Windows备用"
  exit 1
fi

best_node=""
best_score=999999

for node in "${NODE_LIST[@]}"; do
  score=$(score_node "$node")
  log "  $node: score=$score"
  if [ "$score" -lt "$best_score" ]; then
    best_score=$score
    best_node="$node"
  fi
done

if [ -n "$best_node" ] && [ "$best_score" -lt 500000 ]; then
  clash_switch_node "$best_node"
  log "✓ 最优节点: $best_node (score=$best_score)"
else
  log "✗ 所有节点不通，降级 Windows 备用"
  switch_env_proxy "$BACKUP_PROXY" "Windows备用"
fi

log "完成。Clash 后台运行中，节点已选定。"
