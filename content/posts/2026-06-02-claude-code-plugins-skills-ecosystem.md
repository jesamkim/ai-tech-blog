---
title: "Claude Code 생태계 정리 — 플러그인, 스킬, 그리고 누가 plan을 들고 있나"
date: 2026-06-02T19:00:00+09:00
draft: false
cover:
  image: "/ai-tech-blog/images/claude-code-plugins-skills-ecosystem/cover.png"
  alt: "Claude Code 플러그인과 스킬 생태계"
  relative: false
categories: ["GenAI"]
tags: ["Claude Code", "Plugins", "Agent Skills", "Superpowers", "AI-DLC", "Dynamic Workflows", "AWS"]
author: "Jesam Kim"
summary: "Claude Code 플러그인 마켓플레이스와 Agent Skills 생태계에서 GitHub star가 높은 도구들을 언제 쓰는지 정리하고, Superpowers/AI-DLC/Dynamic Workflows가 plan을 각각 어디에 두는지를 기준으로 분류합니다."
description: "Claude Code 플러그인 마켓플레이스와 Agent Skills 생태계에서 GitHub star가 높은 도구들을 언제 쓰는지 정리하고, Superpowers/AI-DLC/Dynamic Workflows가 plan을 각각 어디에 두는지를 기준으로 분류합니다."
---

Claude Code에 플러그인 마켓플레이스와 Agent Skills가 들어오면서, 한동안 README 한 줄짜리였던 확장들이 2026년 들어 GitHub star 수만~수십만 단위로 올라왔습니다. star가 많다고 다 좋은 도구는 아니지만, 적어도 "사람들이 실제로 설치해서 쓰고 있다"는 신호는 됩니다. 이 글은 그중 개발 작업에 직접 쓰이는 것들을 골라, 무엇을 잘하고 언제 켜는지 정리한 것입니다.

핵심 질문 하나를 먼저 깔아두겠습니다. **plan을 누가 들고 있는가.** 어떤 도구는 방법론(문서)이 plan을 쥐고 Claude를 가이드하고, 어떤 도구는 Claude가 매 턴 직접 오케스트레이션하고, 또 어떤 기능은 plan을 아예 코드로 옮겨버립니다. 이 차이가 도구를 고르는 기준이 됩니다.

## Superpowers — 코드 짜기 전에 한 발 물러서게 만드는 프레임워크

[obra/superpowers](https://github.com/obra/superpowers)는 현재 이 분야 star 1위입니다(<strong>215,499 stars</strong>). 제작자는 Jesse Vincent(obra)이고, Anthropic 공식 플러그인 마켓플레이스에 등재돼 있어 `/plugin install superpowers@claude-plugins-official`로 바로 설치됩니다. 설명 문구 그대로 "an agentic skills framework & software development methodology that works"입니다.

동작 방식이 특징적입니다. 코딩 에이전트를 켜는 순간부터 작동하는데, 요청을 받자마자 코드를 짜지 않습니다. 한 발 물러서서 "진짜 뭘 하려는 건지" 되묻고, 거기서 spec을 뽑아낸 뒤 읽기 쉬운 단위로 설계를 제시합니다. 승인을 받으면 그때 구현 계획으로 넘어가는데, red/green TDD와 YAGNI, DRY를 강조합니다. 실제 태스크 수행은 subagent-driven-development로 각 단위를 처리하고 리뷰까지 붙입니다. 이 과정에서 스킬이 자동으로 트리거됩니다.

대표 스킬을 보면 성격이 드러납니다. `brainstorming`은 코드 작성 전에 활성화되어 요구사항을 정제하고, `using-git-worktrees`는 설계 승인 후 격리된 워크스페이스를 만듭니다. 그 외 `writing-plans`, `subagent-driven-development`, 그리고 `code-reviewer` 에이전트가 묶여 있습니다. Claude Code 외에 Codex CLI/App, Factory Droid, Gemini CLI, OpenCode, Cursor, GitHub Copilot CLI에서도 동작합니다.

여기서 plan은 **방법론이 들고 있습니다.** Superpowers는 "어떻게 진행할지"의 절차를 스킬 묶음으로 정의해두고, Claude가 그 절차를 따라가도록 가이드합니다.

## AWS AI-DLC — 엔터프라이즈 제약을 차단으로 강제하는 방법론

[awslabs/aidlc-workflows](https://github.com/awslabs/aidlc-workflows)는 AWS Labs 공식 저장소입니다(<strong>2,636 stars</strong>). 설명은 "AI-Driven Life Cycle (AI-DLC) adaptive workflow steering rules for AI coding agents". 이름 그대로 플러그인이라기보다 **steering rules 기반의 방법론**입니다. Kiro Steering Files를 기반으로 프로젝트 워크스페이스 안에서 동작하고, Claude Code에서는 CLAUDE.md project memory로 워크플로우를 구현합니다.

3단계 적응형 워크플로우로 구성됩니다.

| 단계 | 하는 일 |
|------|---------|
| Inception | 요구 분석과 설계 |
| Construction | per-unit 구현/테스트 루프 |
| Operations | 운영 단계 |

Construction의 per-unit loop가 실무적으로 쓸모 있습니다. 복잡한 프로젝트를 병렬화 가능한 work package로 분해하고, 각 package마다 functional design → NFR → code generation → test cycle을 돈다는 구조입니다. 큰 작업을 한 번에 밀어붙이지 않고 단위로 쪼개 검증한다는 점에서 Superpowers의 subagent-driven 접근과 결이 비슷합니다.

엔터프라이즈 환경에서 의미가 큰 부분은 Extension system입니다. HIPAA, 내부 SDK 규칙, 보안 baseline 같은 blocking constraint를 레이어링할 수 있고, 위반 시 경고에 그치지 않고 실제로 차단(blocker)합니다. 코딩 에이전트가 만든 결과물이 조직 정책을 우회하지 못하게 막는 장치입니다. 세션 연속성도 워크스페이스 파일(`aidlc-docs/aidlc-state.md`)로 관리되어, Cursor에서 시작한 작업을 Claude Code로 이어받는 크로스 하니스가 가능합니다. 지원 대상은 Kiro(구 Amazon Q Developer), Cursor, Cline, Claude Code, GitHub Copilot입니다.

방법론 정의 문서는 [AWS DevOps 블로그](https://aws.amazon.com/blogs/devops/ai-driven-development-life-cycle/)에 있고, 비공식 Claude Code 플러그인 래퍼인 [ijin/aidlc-cc-plugin](https://github.com/ijin/aidlc-cc-plugin)(17 stars)도 참고용으로 존재합니다.

AI-DLC에서 plan은 **워크스페이스의 steering 문서가 들고 있습니다.** 상태와 제약이 파일로 박혀 있고, 어떤 하니스로 들어와도 그 문서를 읽어 같은 절차를 잇습니다.

## Dynamic Workflows — plan을 코드로 옮긴 네이티브 기능

여기서 분류를 분명히 해야 합니다. **Dynamic Workflows는 플러그인도 스킬도 아닙니다.** 2026년 5월 28일 Claude Opus 4.8과 함께 출시된 Claude Code 네이티브 기능이고, 현재 research preview입니다([공식 발표](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code), [공식 문서](https://code.claude.com/docs/en/workflows)).

정의는 이렇습니다. dynamic workflow는 서브에이전트를 대규모로 오케스트레이션하는 JavaScript 스크립트입니다. Claude가 작업 설명을 받아 스크립트를 직접 작성하면, 런타임이 그 스크립트를 백그라운드에서 실행하고, 세션은 응답 가능한 상태를 유지합니다. 최대 <strong>1,000개</strong>의 subagent를 병렬로 돌릴 수 있습니다.

핵심 차이가 여기 있습니다. subagent든 skill이든 agent team이든, 보통은 Claude가 매 턴마다 "다음에 뭘 시킬지"를 판단하며 오케스트레이션합니다. 반면 workflow는 **그 plan을 코드로 옮깁니다.** 루프, 조건 분기, fan-out이 스크립트 안에 결정론적으로 적혀 있고, 런타임이 그대로 집행합니다. Claude의 턴마다 판단이 끼어들지 않으니 대규모 반복 작업에서 흔들림이 줄어듭니다.

활성화는 두 가지입니다. 프롬프트에 "workflow" 키워드를 넣거나, effort 메뉴의 'ultracode' 설정(effort=xhigh에 자동 workflow 판단이 붙음)을 켜는 방식입니다.

## OpenAI 공식 스킬 카탈로그 — 표준이 만든 크로스 에이전트 호환

흥미로운 지점은 스킬 포맷 자체가 벤더를 넘어 표준화되고 있다는 점입니다. [openai/skills](https://github.com/openai/skills)(<strong>21,119 stars</strong>)는 OpenAI가 직접 운영하는 "Skills Catalog for Codex"입니다. 이름 그대로 자사 Codex를 1차 대상으로 한 스킬 모음입니다.

다만 여기서 분류를 정확히 해둘 필요가 있습니다. 이 카탈로그는 Claude Code 전용으로 만들어진 게 아니라 **Codex용**입니다. 그런데 Agent Skills가 "folders of instructions, scripts, and resources"라는 공통 구조에 "write once, use everywhere"를 표방하는 [개방 표준](https://agentskills.io)을 따르기 때문에, 같은 포맷의 스킬이 Claude Code를 포함한 다른 에이전트에서도 동작합니다. 즉 OpenAI가 만든 스킬을 Claude Code 쪽으로 가져다 쓰는 게 가능한 구조이지, OpenAI가 Claude Code를 위해 만든 것은 아닙니다.

레포 구조를 보면 `.system`(Codex 최신 버전에 자동 설치), `.curated`(`$skill-installer <name>`으로 설치), `.experimental`(폴더 지정 또는 GitHub URL로 설치)로 나뉩니다. 카탈로그에는 Skill Creator(스킬 생성 도우미), Concise Planning(간결한 계획 수립) 같은 항목이 포함됩니다. 같은 표준을 쓰는 다른 공식 카탈로그로는 [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)(React/웹 디자인), [remotion-dev/skills](https://github.com/remotion-dev/skills)(React 영상 생성) 등이 있습니다.

plan 축으로 보면 이건 "plan을 누가 드느냐"보다 한 단계 위의 이야기입니다. 스킬이라는 단위 자체가 에이전트 벤더를 넘어 이식 가능해지면서, 한쪽에서 만든 절차를 다른 쪽으로 옮기는 비용이 낮아졌습니다.

## 세 가지를 한 축에 놓으면

같은 "에이전트가 큰 작업을 단위로 쪼개 처리한다"는 목표를 두고도, plan을 두는 위치가 다릅니다.

| 도구 | 분류 | plan을 누가 들고 있나 |
|------|------|----------------------|
| Superpowers | 플러그인(스킬 묶음) | 방법론이 절차로 가이드 |
| AI-DLC | steering rules 방법론 | 워크스페이스 문서가 상태/제약 보유 |
| Dynamic Workflows | Claude Code 네이티브 기능 | 코드(스크립트)가 결정론적으로 보유 |

방법론에 plan을 두면 사람이 읽고 고치기 쉽고, 문서에 두면 하니스를 넘나들며 상태가 유지되고, 코드에 두면 실행이 결정론적이고 대규모로 확장됩니다. 셋은 경쟁 관계라기보다 plan을 어디에 두고 싶은가에 따라 갈리는 선택지입니다.

## 그 외 알아둘 만한 도구

- [hesreallyhim/awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code) (<strong>45,454 stars</strong>) — 스킬, 훅, 슬래시 커맨드, 오케스트레이터, 플러그인을 모은 큐레이션 리스트입니다. 뭐가 있는지부터 훑을 때 출발점으로 둘 만합니다.
- [SuperClaude-Org/SuperClaude_Framework](https://github.com/SuperClaude-Org/SuperClaude_Framework) (<strong>23,133 stars</strong>) — "configuration framework that enhances Claude Code with specialized commands, cognitive personas, and development methodologies". Superpowers처럼 커맨드와 방법론을 묶은 프레임워크 계열입니다.
- [ryoppippi/ccusage](https://github.com/ryoppippi/ccusage) (<strong>15,409 stars</strong>) — "Analyze coding (agent) CLI token usage and costs from local data." 로컬 데이터에서 토큰/비용을 분석하는 유틸 CLI입니다. 워크플로우 도구가 아니라 사용량 가시성을 주는 쪽입니다.

## 선택 가이드

정리하면 이렇게 고르면 됩니다.

- 코드부터 짜는 습관을 막고 spec → 설계 → TDD 구현으로 강제하고 싶다면 **Superpowers**.
- 조직 보안/규정 제약을 실제 차단으로 박아두고, 하니스를 넘나들며 상태를 잇고 싶다면 **AI-DLC**.
- 수백 단위의 반복 작업을 결정론적 스크립트로 대규모 병렬 실행하고 싶다면 **Dynamic Workflows**(플러그인 설치 아님, 네이티브 기능).
- 뭐가 있는지부터 보고 싶으면 **awesome-claude-code**, 토큰 비용이 궁금하면 **ccusage**.
- OpenAI 쪽에서 만든 스킬을 표준 포맷으로 가져다 쓰고 싶으면 **openai/skills**(Codex 카탈로그, Claude Code 호환).

star 순위는 인기의 지표일 뿐이고, 본인 워크플로우에서 plan을 어디에 두고 싶은지가 선택의 축입니다.

## References

- [obra/superpowers](https://github.com/obra/superpowers)
- [awslabs/aidlc-workflows](https://github.com/awslabs/aidlc-workflows)
- [AI-Driven Development Life Cycle — AWS DevOps Blog](https://aws.amazon.com/blogs/devops/ai-driven-development-life-cycle/)
- [ijin/aidlc-cc-plugin](https://github.com/ijin/aidlc-cc-plugin)
- [Introducing Dynamic Workflows in Claude Code](https://claude.com/blog/introducing-dynamic-workflows-in-claude-code)
- [Dynamic Workflows — Claude Code Docs](https://code.claude.com/docs/en/workflows)
- [openai/skills — Skills Catalog for Codex](https://github.com/openai/skills)
- [Agent Skills open standard](https://agentskills.io)
- [vercel-labs/agent-skills](https://github.com/vercel-labs/agent-skills)
- [remotion-dev/skills](https://github.com/remotion-dev/skills)
- [hesreallyhim/awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code)
- [SuperClaude-Org/SuperClaude_Framework](https://github.com/SuperClaude-Org/SuperClaude_Framework)
- [ryoppippi/ccusage](https://github.com/ryoppippi/ccusage)
