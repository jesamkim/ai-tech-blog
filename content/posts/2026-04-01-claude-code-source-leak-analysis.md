---
title: "Claude Code 소스코드 유출 분석 – npm source map 하나가 512,000줄을 열었다"
date: 2026-04-01T10:00:00+09:00
categories: ["AI/ML 기술 심층분석"]
tags: ["Claude Code", "Anthropic", "Source Code Leak", "AI Agent", "npm", "Supply Chain Security"]
author: "Jesam Kim"
description: "2026년 3월 31일 npm source map 파일을 통해 유출된 Claude Code 전체 소스코드를 분석한다. 메모리 아키텍처, 멀티에이전트 오케스트레이션, anti-distillation 메커니즘까지, AI 코딩 에이전트 내부 구조를 실사용자 관점에서 살펴본다."
cover:
  image: "/ai-tech-blog/images/cover-claude-code-leak.png"
---

## 무슨 일이 있었나

2026년 3월 31일, 보안 연구자 [Chaofan Shou](https://x.com/Fried_rice/status/2038894956459290963)가 npm에 배포된 Claude Code v2.1.88 패키지에서 59.8MB짜리 source map 파일(cli.js.map)을 발견했다. 이 파일 안에 Claude Code의 전체 원본 TypeScript 소스가 들어 있었다.

몇 시간 만에 약 1,900개 TypeScript 파일, 512,000줄 이상의 코드가 [GitHub에 미러링](https://github.com/Kuberwastaken/claude-code)되었고, 41,500회 이상 fork되었다. [Hacker News에서는 1,800포인트, 900개 이상의 댓글](https://news.ycombinator.com/item?id=47584540)이 달렸다.

Anthropic은 [The Register에 보낸 공식 성명](https://www.theregister.com/2026/03/31/anthropic_claude_code_source_code/)에서 사실을 인정했다:

> "Earlier today, a Claude Code release included some internal source code. No sensitive customer data or credentials were involved or exposed. This was a release packaging issue caused by human error, not a security breach."

고객 데이터나 인증 정보 유출은 없었고, 빌드 패키징 과정의 휴먼 에러라는 설명이다.

## 어떻게 유출되었나

유출 경위 자체가 개발자라면 공감할 만한 실수다.

Claude Code는 [Bun](https://bun.sh/) 번들러로 빌드된다. Bun은 기본 설정에서 source map을 생성하며, source map의 `sourcesContent` 배열에는 원본 소스 전체가 JSON 문자열로 포함된다. `.npmignore`에 `*.map` 패턴이 빠진 채 npm에 배포된 것이 원인이다.

```json
{
  "version": 3,
  "sources": ["../src/main.tsx", "../src/tools/BashTool.ts", "..."],
  "sourcesContent": ["// 원본 소스코드 전체가 여기에", "..."],
  "mappings": "AAAA,SAAS,OAAO..."
}
```

Bun에 관련 버그([oven-sh/bun#28001](https://github.com/oven-sh/bun/issues/28001))가 3월 11일에 이미 보고되어 있었다. Production 모드에서도 source map이 서빙된다는 내용이다. Anthropic은 [작년 말 Bun을 인수](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone)했으니, 자사 툴체인의 알려진 버그가 자사 제품 소스코드를 유출시킨 셈이다.

[Alex Kim의 분석](https://alex000kim.com/posts/2026-03-31-claude-code-source-leak/)이 지적했듯이, `.npmignore`나 `package.json`의 `files` 필드 한 줄이면 막을 수 있었다. CI/CD 파이프라인에 배포 전 아티팩트 크기 검증이 없었다는 뜻이기도 하다.

## 유출된 코드에서 드러난 것들

공개된 분석 자료들을 종합하면, Claude Code는 단순한 API 래퍼가 아니라 상당한 규모의 소프트웨어 시스템이다. [DEV.to의 Gabriel Anhaia 분석](https://dev.to/gabrielanhaia/claude-codes-entire-source-code-was-just-leaked-via-npm-source-maps-heres-whats-inside-cjo)에 따르면 플러그인 시스템에 약 40,000줄, 쿼리 시스템에 46,000줄이 투입되어 있다.

### 3계층 메모리 아키텍처

[VentureBeat의 분석](https://venturebeat.com/technology/claude-codes-source-code-appears-to-have-leaked-heres-what-we-know)에 따르면, Claude Code는 긴 세션에서 모델이 혼란에 빠지는 "context entropy" 문제를 3계층 메모리 구조로 해결한다:

- <strong>MEMORY.md</strong>: 줄당 약 150자의 경량 포인터 인덱스. 항상 컨텍스트에 로드된다
- <strong>Topic 파일</strong>: 프로젝트별 지식을 분산 저장. 필요할 때만 로드한다
- <strong>Raw 트랜스크립트</strong>: 전체를 다시 읽지 않고, 특정 식별자로 grep 검색만 수행한다

"모든 것을 저장하고 전부 읽어오는" 방식 대신, 포인터 인덱스로 필요한 것만 가져오는 구조다. 에이전트가 파일 쓰기에 성공한 후에만 인덱스를 갱신할 수 있고("Strict Write Discipline"), 자신의 메모리를 "hint"로 취급해 실제 코드베이스와 대조 검증하도록 설계되어 있다.

### autoDream: 백그라운드 메모리 정리

[Kuberwastaken의 분석](https://github.com/Kuberwastaken/claude-code)이 상세히 다룬 autoDream은 사용자가 비활성 상태일 때 메모리를 정리하는 백그라운드 프로세스다. 이름 그대로 Claude가 "꿈"을 꾸는 것이다.

실행 조건이 3중으로 걸려 있다. 마지막 dream 이후 24시간 경과, 5세션 이상, consolidation lock 획득. 세 조건을 모두 만족해야 동작한다.

실행되면 4단계를 거친다:
1. <strong>Orient</strong>: 메모리 디렉토리 탐색, MEMORY.md 읽기
2. <strong>Gather Recent Signal</strong>: daily logs, drifted memories, transcript 검색
3. <strong>Consolidate</strong>: 메모리 파일 갱신, 상대 날짜를 절대 날짜로 변환
4. <strong>Prune and Index</strong>: MEMORY.md를 200줄/25KB 이내로 유지

dream 서브에이전트에는 read-only bash만 허용된다. 프로젝트 파일을 수정할 수 없고, 메인 에이전트의 작업 흐름과 분리되어 동작한다.

### Multi-Agent Coordinator

`CLAUDE_CODE_COORDINATOR_MODE=1` 환경변수로 활성화되는 멀티에이전트 시스템도 드러났다. Claude Code가 단일 에이전트에서 코디네이터로 전환되어 여러 워커를 병렬로 관리한다.

흐름은 4단계다. 워커들이 병렬로 코드베이스를 조사하고(Research), 코디네이터가 결과를 종합해 스펙을 작성하고(Synthesis), 워커들이 스펙에 따라 구현하고(Implementation), 다시 워커들이 테스트한다(Verification).

오케스트레이션 로직이 코드가 아니라 프롬프트로 구현되어 있다. "Do not rubber-stamp weak work", "You must understand findings before directing follow-up work" 같은 자연어 지시로 품질을 관리한다.

### KAIROS: 미출시 자율 에이전트 데몬

소스 전반에서 150회 이상 참조되는 KAIROS(고대 그리스어로 "적절한 때")는 아직 출시되지 않은 상시 구동 에이전트 모드다. 현재 AI 도구 대부분이 반응형(사용자 입력이 있어야 응답)인 것과 달리, KAIROS는 백그라운드에서 스스로 판단하고 동작한다.

[VentureBeat에 따르면](https://venturebeat.com/technology/claude-codes-source-code-appears-to-have-leaked-heres-what-we-know), GitHub webhook 구독으로 PR을 모니터링하고, 푸시 알림을 보내고, cron 스케줄링으로 자동 실행된다. 사용자 워크플로우를 방해하지 않도록 15초 blocking budget이 걸려 있어, 그 이상 걸리는 작업은 자동 지연된다.

### Anti-Distillation: 모델 증류 방어

경쟁사가 Claude Code의 API 트래픽을 녹화해 자체 모델을 학습시키는 것을 막는 메커니즘이 두 가지 발견되었다.

첫째, <strong>Fake Tools Injection</strong>. `ANTI_DISTILLATION_CC` 플래그가 켜지면 API 요청에 가짜 도구 정의를 섞는다. 이 트래픽으로 모델을 학습시키면 존재하지 않는 도구를 호출하게 된다.

둘째, <strong>Connector-text Summarization</strong>. 도구 호출 사이의 추론 텍스트를 요약본과 암호화 서명으로 대체한다. 트래픽을 녹화해도 전체 reasoning chain은 얻지 못한다.

[Alex Kim의 분석](https://alex000kim.com/posts/2026-03-31-claude-code-source-leak/)에 따르면, 기술적 우회는 가능하다. MITM 프록시로 `anti_distillation` 필드를 제거하거나, `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS` 환경변수를 설정하면 된다. 실질적 보호는 법적 수단이고, 기술적 장벽은 "의도적으로 우회했다"는 증거를 만드는 역할에 가까워 보인다.

### Undercover Mode: 오픈소스 기여 시 AI 은폐

Anthropic 직원(`USER_TYPE === 'ant'`)이 외부 오픈소스에 기여할 때, 커밋 메시지와 PR에서 내부 정보를 제거하는 모드다. 시스템 프롬프트에 이런 지시가 포함된다:

> "You are operating UNDERCOVER in a PUBLIC/OPEN-SOURCE repository. Your commit messages, PR titles, and PR bodies MUST NOT contain ANY Anthropic-internal information. Do not blow your cover."

내부 모델 코드명(Capybara, Tengu 등), "Claude Code" 문구, `Co-Authored-By` 헤더가 전부 제거된다. `CLAUDE_CODE_UNDERCOVER=1`로 강제 활성화는 가능하지만, 강제 비활성화는 불가능하다. 소스에 "There is NO force-OFF"라고 적혀 있다.

"AI가 사람인 척하는 것"이라는 비판과, "내부 코드명 보안을 위한 합리적 조치"라는 반론이 동시에 나왔다. 어느 쪽이든, Anthropic 직원들이 Claude Code로 오픈소스에 기여하고 있다는 사실 자체는 확인된 셈이다.

### 내부 모델 코드명

Undercover Mode 프롬프트를 통해 내부 코드명도 드러났다:

| 코드명 | 대상 |
|--------|------|
| <strong>Tengu</strong> | Claude Code 프로젝트 자체 (수백 번 참조) |
| <strong>Capybara</strong> | Claude 4.6 변종 |
| <strong>Fennec</strong> | Opus 4.6 |
| <strong>Numbat</strong> | 미출시 테스트 모델 |

모두 동물 이름이다. [VentureBeat에 따르면](https://venturebeat.com/technology/claude-codes-source-code-appears-to-have-leaked-heres-what-we-know), Capybara v8의 false claims rate가 29–30%로, v4의 16.7%에서 오히려 후퇴했다는 내부 코멘트도 발견되었다.

### Native Client Attestation: API 호출 인증

Claude Code 바이너리가 정품인지 확인하는 메커니즘도 있다. API 요청 헤더에 `cch=00000` placeholder를 넣고, Bun의 Zig HTTP 레이어에서 해시값으로 교체한다. JavaScript 런타임 아래에서 동작하므로 JS 코드에서는 보이지 않는다.

이것이 최근 [OpenCode와의 법적 분쟁](https://github.com/anomalyco/opencode/pull/18186)의 기술적 배경이다. 서드파티 도구가 Claude Code의 내부 API로 구독 가격에 Opus를 쓰는 것을 차단한다.

### 기타 발견 사항

<strong>Frustration Detection</strong>: 사용자 좌절을 regex로 감지한다. "wtf", "this sucks", "fucking broken" 같은 패턴이다. LLM 회사가 감정 분석에 정규표현식을 쓰는 이유는 단순하다. 빠르고 싸다.

<strong>250K API 호출/일 낭비</strong>: autoCompact 기능에서 연쇄 실패가 발생해 하루 25만 건의 API 호출이 허비되고 있었다([autoCompact.ts](https://github.com/alex000kim/claude-code/blob/main/src/services/compact/autoCompact.ts#L68-L70)의 코멘트). 수정은 `MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3` 한 줄이었다.

<strong>Bash 보안</strong>: 23개 보안 체크, 18개 Zsh 빌트인 차단, unicode zero-width space 주입 방어, HackerOne 리뷰에서 발견된 malformed token bypass까지 구현되어 있다.

<strong>Buddy System</strong>: 타마고치 스타일 ASCII 펫 시스템. 18종의 생물, 6단계 레어리티(Common 60%부터 Legendary 1%까지), 1% shiny 확률. 4/1–7 티저 기간이 코드에 명시되어 있어, 만우절 이벤트로 보인다.

<strong>Prompt Cache Economics</strong>: 14개의 cache-break 벡터를 추적하고, 모드 토글이 캐시를 깨지 않도록 "sticky latch"를 적용한다. `DANGEROUS_uncachedSystemPromptSection()`이라는 함수명이 인상적이다.

## 비즈니스 맥락

기술적 내용 못지않게 타이밍이 나쁘다.

[Anthropic의 Series G 발표](https://www.anthropic.com/news/anthropic-raises-30-billion-series-g-funding-380-billion-post-money-valuation)(2026년 2월)에 따르면, Claude Code의 ARR은 $2.5B이며 2026년 초 대비 2배 이상 성장했다. Anthropic 전체로는 [Bloomberg 보도 기준](https://www.bloomberg.com/news/articles/2026-03-03/anthropic-nears-20-billion-revenue-run-rate-amid-pentagon-feud) $20B에 근접하는 ARR을 기록 중이고, enterprise가 80%를 차지한다.

[Gizmodo의 지적](https://gizmodo.com/source-code-for-anthropics-claude-code-leaks-at-the-exact-wrong-time-2000740379)처럼, IPO를 준비하는 시점에서 며칠 전 미발표 모델 스펙이 유출(Mythos 사건)되고, 10일 전에는 OpenCode에 법적 경고를 보냈으며, OpenAI가 Codex 무제한 접근을 제공하며 경쟁을 강화하는 상황이다.

한 가지 더. Anthropic의 Claude Code 책임자 Boris Cherny는 작년 12월에 "[지난 30일간 내 기여의 100%를 Claude Code가 작성했다](https://x.com/bcherny/status/2004897269674639461)"고 말했다. 이번 빌드 패키징 실수도 AI가 만들었을 가능성을 배제하기 어렵다.

## Enterprise 보안 관점

소스코드 유출 자체보다 실질적으로 주의할 부분은 npm 공급망이다.

같은 날(3월 31일), npm의 axios 패키지에 공급망 공격이 있었다. [VentureBeat 보도](https://venturebeat.com/technology/claude-codes-source-code-appears-to-have-leaked-heres-what-we-know)에 따르면, 악성 버전(1.14.1, 0.30.4)에 Remote Access Trojan(RAT)이 포함되었다. Claude Code가 HTTP 클라이언트로 axios를 사용하기 때문에, 3/31 00:21–03:29 UTC 사이에 npm으로 설치한 사용자는 영향받을 수 있다.

확인해야 할 것:
- `package-lock.json`에서 axios 1.14.1 또는 0.30.4, 의존성 `plain-crypto-js` 존재 여부
- 해당 버전 발견 시 호스트 격리 + 시크릿 로테이션
- npm 설치 대신 native installer(`curl -fsSL https://claude.ai/install.sh | bash`) 사용

enterprise 환경에서 AI 코딩 도구의 의존성 체인이 CI/CD 보안 스캔에 포함되어 있는지 점검할 필요가 있다. AI 도구가 개발 워크플로우에 깊이 통합될수록, 공격 표면도 그만큼 넓어진다.

## 44개 feature flag 뒤의 완성된 기능들

[The AI Corner의 정리](https://www.the-ai-corner.com/p/claude-code-source-code-leaked-2026)에 따르면, 44개 feature flag 뒤에 이미 구현된 기능들이 있다:

- 24/7 백그라운드 에이전트 (GitHub webhook + 푸시 알림)
- 멀티에이전트 오케스트레이션
- cron 스케줄링 (생성, 삭제, 외부 webhook)
- 음성 명령 모드 (별도 CLI entrypoint)
- Playwright 기반 브라우저 제어
- 자동 wake/sleep 에이전트
- 세션 간 persistent memory

The AI Corner의 표현을 빌리면, "2주마다 새 기능을 출시하는 이유는 이미 다 만들어져 있기 때문"이다. compile-time flag로 외부 빌드에서 제거되지만, source map은 dead-code elimination을 거치지 않는다.

## 정리

`.npmignore` 한 줄이 빠져서 $2.5B ARR 제품의 전체 소스가 유출되었다. CI에서 아티팩트 크기 체크 한 단계만 있었어도 막을 수 있었을 것이다.

유출된 내용 중 기술적으로 가장 관심이 가는 부분은 3계층 메모리 아키텍처와 autoDream이다. AI 에이전트가 자기 컨텍스트를 어떻게 관리하고, 세션 사이에 지식을 어떻게 유지하는지에 대한 구체적인 설계를 볼 수 있다. Anti-distillation이나 Undercover Mode 같은 것들은 AI 제품의 상업적/전략적 측면을 잘 보여준다.

경쟁사 입장에서 이 소스는 R&D 비용을 크게 절약해 주는 참고 자료다. 하지만 AI 코딩 에이전트 시장 자체가 빠르게 움직이고 있어서, 몇 달 후에 이 코드가 얼마나 유효할지는 미지수다. [Ars Technica의 지적](https://arstechnica.com/ai/2026/03/entire-claude-code-cli-source-code-leaks-thanks-to-exposed-map-file/)처럼, 가장 큰 손실은 코드 자체보다 feature flag로 드러난 제품 로드맵일 수 있다.

---

## References

1. Chaofan Shou, X post announcing the discovery (2026-03-31) — [x.com/Fried_rice](https://x.com/Fried_rice/status/2038894956459290963)
2. The Register, "Anthropic goes nude, exposes Claude Code source by accident" (2026-03-31) — [theregister.com](https://www.theregister.com/2026/03/31/anthropic_claude_code_source_code/)
3. VentureBeat, "Claude Code's source code appears to have leaked: here's what we know" (2026-03-31) — [venturebeat.com](https://venturebeat.com/technology/claude-codes-source-code-appears-to-have-leaked-heres-what-we-know)
4. Alex Kim, "The Claude Code Source Leak: fake tools, frustration regexes, undercover mode, and more" (2026-03-31) — [alex000kim.com](https://alex000kim.com/posts/2026-03-31-claude-code-source-leak/)
5. Kuberwastaken, "Claude Code in Rust & a Breakdown of How it Works" (2026-03-31) — [github.com/Kuberwastaken/claude-code](https://github.com/Kuberwastaken/claude-code)
6. Gabriel Anhaia, DEV.to analysis (2026-03-31) — [dev.to/gabrielanhaia](https://dev.to/gabrielanhaia/claude-codes-entire-source-code-was-just-leaked-via-npm-source-maps-heres-whats-inside-cjo)
7. The AI Corner, "Claude Code Source Code Leaked: What's Inside" (2026-03-31) — [the-ai-corner.com](https://www.the-ai-corner.com/p/claude-code-source-code-leaked-2026)
8. Gizmodo, "Source Code for Anthropic's Claude Code Leaks at the Exact Wrong Time" (2026-03-31) — [gizmodo.com](https://gizmodo.com/source-code-for-anthropics-claude-code-leaks-at-the-exact-wrong-time-2000740379)
9. Ars Technica, "Entire Claude Code CLI source code leaks thanks to exposed map file" (2026-03-31) — [arstechnica.com](https://arstechnica.com/ai/2026/03/entire-claude-code-cli-source-code-leaks-thanks-to-exposed-map-file/)
10. Hacker News discussion, 1,800+ points, 900+ comments — [news.ycombinator.com](https://news.ycombinator.com/item?id=47584540)
11. Bun issue #28001, "Source maps served in production mode" (2026-03-11) — [github.com/oven-sh/bun](https://github.com/oven-sh/bun/issues/28001)
12. Anthropic, "Anthropic acquires Bun as Claude Code reaches $1B milestone" — [anthropic.com](https://www.anthropic.com/news/anthropic-acquires-bun-as-claude-code-reaches-usd1b-milestone)
13. Anthropic, Series G funding announcement, Claude Code $2.5B ARR (2026-02-12) — [anthropic.com](https://www.anthropic.com/news/anthropic-raises-30-billion-series-g-funding-380-billion-post-money-valuation)
14. Bloomberg, "Anthropic Nears $20 Billion Revenue Run Rate" (2026-03-03) — [bloomberg.com](https://www.bloomberg.com/news/articles/2026-03-03/anthropic-nears-20-billion-revenue-run-rate-amid-pentagon-feud)
