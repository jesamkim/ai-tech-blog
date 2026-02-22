# TOOLS.md - Local Notes

Skills define *how* tools work. This file is for *your* specifics — the stuff that's unique to your setup.

## What Goes Here

Things like:
- Camera names and locations
- SSH hosts and aliases  
- Preferred voices for TTS
- Speaker/room names
- Device nicknames
- Anything environment-specific

## Examples

```markdown
### Cameras
- living-room → Main area, 180° wide angle
- front-door → Entrance, motion-triggered

### SSH
- home-server → 192.168.1.100, user: admin

### TTS
- Preferred voice: "Nova" (warm, slightly British)
- Default speaker: Kitchen HomePod
```

## Why Separate?

Skills are shared. Your setup is yours. Keeping them apart means you can update skills without losing your notes, and share skills without leaking your infrastructure.

---

## Notion

### 페이지 생성 시 필수 절차 ⚠️

**절대 규칙:**
1. 페이지 생성 (`POST /pages`)
2. 내용 추가 (`PATCH /blocks/{id}/children`)
3. **검증 확인** (`GET /blocks/{id}/children`) ← 필수!
   - 블록 개수 확인
   - 주요 섹션 확인
   - 내용 누락 여부 확인

**검증 없이 완료하지 말 것!**

API 호출이 실패하거나 중간에 끊길 수 있음. 반드시 최종 확인 후 사용자에게 보고.

### 설정

- API Key: `~/.config/notion/api_key`
- 기본 부모 페이지: `yan` (ID: 2f79b4bf-d0d2-807a-b51f-caa964c91969)
- API Version: 2025-09-03

---

## 웹 검색 우선 원칙

**답변하기 전에 웹 검색이 필요한 경우:**
- 📅 날짜/시간 민감한 정보
- 📊 정책/제도/금리 (학자금 대출, 세금 등)
- 🆕 최신 뉴스/이슈
- 📈 통계/수치/가격
- 🔧 최신 기술/제품 정보
- ⚖️ 법률/규제

**검색 후 답변!** 최신 정보 확인 필수.

---

Add whatever helps you do your job. This is your cheat sheet.
