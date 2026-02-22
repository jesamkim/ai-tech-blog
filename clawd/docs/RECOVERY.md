# 🔧 Yan Recovery Guide

EC2가 날아갔을 때 Yan(OpenClaw 에이전트)을 복구하기 위한 가이드.

## 1. 인프라 정보

| 항목 | 값 |
|------|-----|
| OS | Amazon Linux 2023 |
| Instance type | t3.xlarge (4 vCPU, 16GB RAM) |
| Region | us-west-2 |
| Node.js | v22.22.0 |
| Python | 3.9.25 |
| OpenClaw | 2026.2.9 |

## 2. 복구 순서

### Step 1: EC2 인스턴스 준비
```bash
# Amazon Linux 2023 인스턴스 생성 후
sudo yum update -y
```

### Step 2: Node.js 설치
```bash
curl -fsSL https://rpm.nodesource.com/setup_22.x | sudo bash -
sudo yum install -y nodejs
```

### Step 3: OpenClaw 설치
```bash
sudo npm install -g openclaw
```

### Step 4: Python 패키지 설치
```bash
pip3 install google-api-python-client google-auth google-auth-oauthlib google-auth-httplib2
pip3 install yt-dlp youtube-transcript-api
```

### Step 5: GitHub CLI 설치
```bash
sudo yum install -y gh
gh auth login
```

### Step 6: Workspace 복구
```bash
cd ~
git clone https://github.com/jesamkim/clawd.git
```

### Step 7: gog CLI 설치 (Google Workspace)
```bash
# gog 설치 (버전 확인: v0.9.0)
# https://github.com/aandrew-me/gog 참고
```

### Step 8: OpenClaw 설정
```bash
# OpenClaw 초기 설정
openclaw init
# 또는 설정 파일 복원 (아래 "수동 복원 필요" 섹션 참고)
```

### Step 9: systemd 서비스 등록
```bash
sudo tee /etc/systemd/system/openclaw-gateway.service << 'EOF'
[Unit]
Description=OpenClaw Gateway Service
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=/home/ec2-user
Environment="PATH=/usr/bin:/usr/local/bin"
Environment="NODE_ENV=production"
Environment="AWS_REGION=us-west-2"
Environment="OPENCLAW_GATEWAY_TOKEN=<새로 생성할 토큰>"
ExecStart=/usr/bin/openclaw gateway run --bind loopback --port 18789 --token <새로 생성할 토큰>
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=openclaw-gateway

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable openclaw-gateway
sudo systemctl start openclaw-gateway
```

## 3. 수동 복원 필요 (Git에 저장 불가)

⚠️ **보안상 Git에 저장하면 안 되는 것들:**

| 항목 | 위치 | 복원 방법 |
|------|------|----------|
| OpenClaw config | `~/.config/openclaw/openclaw.json` | `openclaw init`으로 재설정 |
| Gateway token | systemd service 파일 | 새로 생성 |
| Google OAuth credentials | `~/.gog/credentials/jesamkim@gmail.com.json` | `gog auth` 재인증 |
| GitHub auth | `~/.config/gh/hosts.yml` | `gh auth login` |
| Telegram Bot token | OpenClaw config 내 | BotFather에서 확인 |
| WhatsApp session | OpenClaw config 내 | QR 재인증 |
| Bedrock API (IAM) | EC2 IAM Role | 인스턴스에 IAM Role 할당 |

### memory_search 로컬 임베딩 설정
OpenClaw config에 아래 추가 필요:
```json
{
  "agents": {
    "defaults": {
      "memorySearch": {
        "enabled": true,
        "provider": "local",
        "local": {
          "modelPath": "hf:nomic-ai/nomic-embed-text-v1.5-GGUF"
        }
      }
    }
  }
}
```
- 모델은 첫 `memory_search` 호출 시 자동 다운로드 (~48MB)
- 캐시 위치: `~/.node-llama-cpp/models/`
- 벡터 DB: SQLite 기반, 자동 생성

### 중요 설정 메모 (토큰/키 제외)
- Telegram bot username: (USER.md 참고)
- OpenClaw workspace: `/home/ec2-user/clawd`
- OpenClaw port: 18789
- Model: `amazon-bedrock/global.anthropic.claude-opus-4-6-v1`
- Channel: Telegram (primary), WhatsApp (secondary)

## 4. Git에 저장된 것들 (이 레포)

✅ **자동 복구 가능:**
- 모든 스킬 (`skills/`)
- 에이전트 설정 (`SOUL.md`, `IDENTITY.md`, `USER.md`, `AGENTS.md`, `TOOLS.md`)
- 메모리 파일 (`memory/`)
- 스크립트 (`scripts/`)
- 문서 (`docs/`)

## 5. 복구 확인 체크리스트

- [ ] OpenClaw gateway 실행 중 (`systemctl status openclaw-gateway`)
- [ ] Telegram 연결 확인
- [ ] WhatsApp 연결 확인 (QR 재인증)
- [ ] Gmail 읽기/쓰기 (`python3 ~/google-helper/gmail_check.py`)
- [ ] GitHub push 가능 (`cd ~/clawd && git push`)
- [ ] Bedrock API 호출 가능 (IAM Role 확인)
- [ ] memory_search 동작 확인 (검색 테스트)
- [ ] 스킬 정상 동작 (`/paper test`, `/eng test`)
