# 🐰 Yan — AI Personal Assistant

OpenClaw 기반 AI 개인 비서. AWS EC2에서 24/7 운영.

## 구조

```
clawd/
├── SOUL.md              # Yan의 성격과 행동 방식
├── IDENTITY.md          # 이름, 유래, 이모지
├── USER.md              # 사용자 프로필
├── AGENTS.md            # 행동 규칙
├── TOOLS.md             # 도구 설정 노트
├── MEMORY.md            # 장기 기억
├── HEARTBEAT.md         # 주기적 체크 설정
├── memory/              # 날짜별 대화 기록/메모
├── skills/              # 커스텀 스킬
│   ├── paper-finder/    # 📄 학술 논문 검색
│   ├── youtube-analyzer/# 🎬 YouTube 트랜스크립트 분석
│   ├── english/         # 🇺🇸 영한/한영 번역
│   ├── humanizer/       # ✍️ AI 글 자연스럽게 교정
│   ├── brainstorming/   # 💡 아이디어 → 설계
│   └── ...              # TDD, 디버깅, 코드리뷰 등
├── docs/
│   ├── RECOVERY.md      # 🔧 EC2 재해 복구 가이드
│   ├── CLAUDE-CODE-COMPAT.md  # Claude Code 호환 가이드
│   ├── plans/           # 설계 문서
│   └── papers/          # 논문 검색 결과
└── scripts/             # 유틸리티 스크립트
```

## 커스텀 스킬

| 스킬 | 트리거 | 설명 |
|------|--------|------|
| paper-finder | `/paper <키워드>` | Semantic Scholar + Papers with Code 기반 논문 검색 |
| youtube-analyzer | `/yt <URL>` | YouTube 트랜스크립트 추출 + 내용 분석 |
| english | `/eng <텍스트>` | 비즈니스 영어 번역 연습 |

## 인프라

- **Runtime:** [OpenClaw](https://github.com/openclaw/openclaw) 2026.2.9
- **Host:** AWS EC2 t3.xlarge (Amazon Linux 2023)
- **Model:** Claude Opus 4.6 (Amazon Bedrock)
- **Memory Search:** Local embedding (nomic-embed-text-v1.5 GGUF)
- **Channels:** Telegram (primary), WhatsApp

## 백업

- 수동: "백업 해" 명령
- 자동: 2일마다 자동 commit + push

## 재해 복구

EC2가 날아가면 → [docs/RECOVERY.md](docs/RECOVERY.md)

## Claude Code 호환

이 스킬들을 Claude Code에서 사용하려면 → [docs/CLAUDE-CODE-COMPAT.md](docs/CLAUDE-CODE-COMPAT.md)
