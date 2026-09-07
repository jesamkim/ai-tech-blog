---
title: "에이전트 디버깅을 재설계하다: OrcaReplay 아키텍처와 실전 활용"
date: 2026-09-06T09:00:00+09:00
draft: false
cover:
  image: "/ai-tech-blog/images/orcareplay-agent-record-replay/cover.png"
  alt: "기록된 에이전트 실행을 재생하고 분기하는 디버깅 타임라인"
  relative: false
categories: ["MLOps & Platform"]
tags: ["OrcaReplay", "AI Agent", "Debugging", "Record-Replay", "LangChain", "Claude Code", "OrcaRouter", "AgentOps"]
author: "Jesam Kim"
description: "2026년 9월 출시된 OrcaReplay는 AI 에이전트 실행을 그대로 기록해 오프라인에서 재현하는 오픈소스 CLI 도구입니다. 에이전트가 왜 그 파일을 삭제했는지 관찰 대시보드는 답하지 못하지만, 기록된 트레이스는 답합니다. 아키텍처와 핵심 명령, LangChain 생태계와의 연결성, 현재 한계를 확인합니다."
---

에이전트가 실행을 마치면 터미널 창은 닫힙니다. 그 안에서 어떤 파일이 왜 바뀌었는지 알고 싶을 때 관찰 대시보드는 비용, 토큰 수, 지연 시간을 보여 줍니다. "왜 그 파일을 삭제했나"라는 질문에는 침묵합니다.

2026년 9월 2일, OrcaRouter(Continuum AI Corp)가 이 문제를 겨냥한 오픈소스 도구를 발표했습니다. [OrcaReplay](https://github.com/Continuum-AI-Corp/OrcaReplay)는 AI 에이전트 실행을 전체 이벤트 스트림으로 기록하고, 기록된 그대로 네트워크를 차단한 채 재현하며, 특정 지점부터 다른 모델로 분기(fork)하는 CLI 도구입니다. 라이선스는 Apache-2.0(코드), CC BY 4.0(트레이스 포맷)입니다.

이 글에서는 OrcaReplay의 캡처 아키텍처, 주요 명령어와 출력 형식, LangChain/LangGraph 에코시스템과의 연결성, 그리고 현재 상태와 한계를 정리합니다. 설치와 명령어 예시는 [공식 README](https://github.com/Continuum-AI-Corp/OrcaReplay)(2026-09-04 기준)에서 직접 확인한 내용을 바탕으로 합니다.

---

## 에이전트 디버깅이 기존 도구와 맞지 않는 이유

기존 소프트웨어는 실패를 재현하기 어렵지 않습니다. 같은 입력을 주면 같은 결과가 나옵니다. 에이전트는 다릅니다. 모델 샘플링, 리포지토리 상태, 실행 타이밍이 모두 결과에 영향을 미칩니다. 실패했던 그 실행을 다시 돌리면 다른 결과가 나올 수 있고, 심지어 아무 문제 없이 통과할 수도 있습니다.

관찰 대시보드는 이 간극을 채우지 못합니다. 대시보드는 "이 실행에 얼마가 들었는가"를 잘 답하지만, "이 실행에서 왜 그 파일이 바뀌었는가"는 답하지 못합니다. 비용은 집계된 숫자이고, 원인은 순서의 속성이기 때문입니다.

OrcaReplay의 README는 이 문제를 간결하게 표현합니다.

> *A dashboard is a bill. A recorded trace is the run.*

---

## 5가지 캡처 레이어 아키텍처

OrcaReplay가 에이전트를 수정하지 않고도 전체 실행을 기록할 수 있는 핵심은 에이전트와 모델 API 사이에 로컬 프록시를 세우는 구조입니다. 모델 API는 무상태(stateless)이기 때문에 에이전트는 매 턴마다 전체 대화 기록을 다시 전송합니다. 프록시가 이 지점에 서면 매 요청, 스트리밍 응답, tool call, tool result를 순서대로 볼 수 있습니다.

프록시 하나만으로 잡을 수 없는 이벤트를 위해 네 개의 레이어가 추가됩니다.

{{< figure src="/ai-tech-blog/images/orcareplay-agent-record-replay/diagram-capture-layers.png" alt="OrcaReplay의 5가지 캡처 레이어 아키텍처 - 프록시, PATH 심, JSON-RPC Tee, 섀도 Git 인덱스, fetch 훅" caption="OrcaReplay 5개 캡처 레이어. 에이전트를 수정하지 않고 환경변수 2개만으로 대부분의 경우를 커버합니다." >}}

<strong>레이어 1: 프록시 (base-URL 환경변수)</strong>

가장 일반적인 경로입니다. `ANTHROPIC_BASE_URL` 또는 `OPENAI_BASE_URL` 환경변수를 레코더로 가리키면 에이전트는 자신이 원래 API를 쓰는 것처럼 동작하면서 모든 요청/응답이 기록됩니다. Claude Code는 `ANTHROPIC_BASE_URL`, Codex CLI(API key 로그인)는 `OPENAI_BASE_URL`로 캡처합니다. 자식 프로세스는 부모의 환경변수를 상속하므로, 에이전트를 스폰하는 게이트웨이나 오케스트레이터를 기록하면 그 아래 에이전트도 함께 캡처됩니다.

<strong>레이어 2: PATH 심 (셸 명령)</strong>

셸 명령의 exit code, 실행 시간, stdout/stderr 분리를 캡처합니다. 에이전트가 exit 0으로 종료했더라도 그 안에서 실행한 검사 명령이 exit 1을 냈다면 트레이스에 기록됩니다. "그린 빌드 안에 숨어 있던 실패"를 찾는 레이어입니다.

<strong>레이어 3: JSON-RPC Tee (MCP)</strong>

MCP(Model Context Protocol) 설정 파일을 자동으로 재기록해 MCP tool call도 캡처합니다. `--mcp-config` 플래그로 활성화하며, 포크 재실행 시에도 같은 설정으로 재계측합니다.

<strong>레이어 4: 섀도 Git 인덱스 (파일시스템)</strong>

대화 턴마다 워크스페이스 스냅샷을 찍어 어떤 파일이 언제 바뀌었는지 기록합니다. 도구 호출마다 스냅샷을 찍지 않고 대화 턴 단위로 찍어 노이즈를 줄입니다.

<strong>레이어 5: fetch 훅 (URL 하드코딩 에이전트)</strong>

Node.js의 `globalThis.fetch`를 가로채는 방식으로, provider URL이 소스코드에 하드코딩된 에이전트도 캡처합니다. Vercel AI SDK 에이전트가 이 경로로 기록되며, Bun도 지원합니다. URL 하드코딩 에이전트나 컨테이너에서 실행 중인 에이전트에는 opt-in TLS 인터셉트(`--tls-intercept`)와 `orca attach` 명령도 제공됩니다.

---

## 기록된 타임라인 읽기: orca show

`orca record claude`로 Claude Code 세션을 기록하면 `.orca/runs/` 에 트레이스가 저장됩니다. `orca show last`는 이 트레이스를 타임라인으로 출력합니다.

```
orca show last
run_6473f858b59e  generic-openai@0.1.0  14 events  exit 0

SEQ  KIND   WHAT            DETAIL
  0  RUN    run started     generic-openai
  1  SNAP   tree            919d32ba...  0 changed
  2  MODEL  claude-opus-4-6   1 messages
  3  MODEL  claude-opus-4-6   stop: tool_use  100 in / 20 out
  4  TOOL   edit_file       {"path":"auth.ts",…}
  5  SNAP   tree            c6af62b7...  1 changed
  6  FILE   auth.ts         modified  +1 -3
  7  TOOL   edit_file ok
  8  MODEL  claude-opus-4-6   3 messages
  9  MODEL  claude-opus-4-6   stop: end_turn  101 in / 5 out
 10  SNAP   tree            c6af62b7...  0 changed
 11  SHELL  ["sh","-c","node --check nonexistent-file.ts"]  /tmp/hunt
 12  SHELL  shell result    exit 1  43ms
 13  RUN    run ended       exit 0
     usage  input=201  output=25  cost=$0.004890
```

이 출력에서 세 가지 사실이 드러납니다. 에이전트의 자체 트랜스크립트나 실행 exit code만으로는 알 수 없는 정보들입니다.

- seq 6: `auth.ts` 파일이 실제로 바뀌었습니다 (`+1 -3`)
- seq 12: 에이전트가 실행한 검사 명령이 exit 1로 실패했습니다
- seq 13: 그럼에도 에이전트 프로세스는 exit 0으로 종료했습니다

---

## 인과 그래프: orca graph

`orca graph last`는 이벤트 간 관계를 보여 줍니다. 엣지에는 두 종류가 있습니다.

```
orca graph last
FROM              TO             KIND    WHY
3 model.response  4 tool.call    recorded  tool_use block in the response
4 tool.call       6 fs.change    inferred  changed path appears in tool input
4 tool.call       7 tool.result  recorded  tool result answers its call
7 tool.result     8 model.request recorded  tool_result block in the request
11 shell.exec    12 shell.result  recorded  shell result answers its exec
                                  1 inferred
```

<strong>recorded 엣지</strong>는 실행 당시 프로토콜에 의해 기록된 사실입니다. `tool_use` 블록이 응답 안에 있었다는 것은 사실입니다. <strong>inferred 엣지</strong>는 OrcaReplay가 규칙을 적용해 추론한 것입니다. 파일 변경이 직전 tool call에서 비롯되었다는 추론은 좋은 추측이지만 사실과 동일하지 않습니다. inferred 엣지는 절대 트레이스 파일에 다시 기록되지 않습니다. 추론은 트레이스 위에 올린 뷰이고, 트레이스 자체는 실제 기록 그대로입니다.

---

## 오프라인 재현과 포크

{{< figure src="/ai-tech-blog/images/orcareplay-agent-record-replay/diagram-record-replay-fork.png" alt="OrcaReplay의 Record-Replay-Fork 타임라인. 커서 위치로 디스크 재생과 네트워크 실행 경계를 제어합니다." caption="커서 위치 하나가 정확 재현, 포크, 모델 비교를 결정합니다. 커서 이전 턴은 디스크에서 재생(비용 0), 이후 턴은 다른 모델이 실행합니다." >}}

<strong>정확 재현</strong>

```bash
orca replay last
# info replay.done reused=2/2 exact=2 divergences=0 exit=0
```

네트워크를 차단하고 디스크에 저장된 응답을 그대로 돌려줍니다. 토큰을 소비하지 않고, 매 실행마다 동일한 결과를 보장합니다. 비결정적이던 에이전트 실패가 결정적인 재현 가능한 실패로 바뀝니다.

<strong>포크 (특정 단계부터 다른 모델)</strong>

```bash
orca replay last --from 4 --model claude-haiku-4-5 --ui
```

첫 세 턴은 디스크에서 재생하고, 네 번째 턴부터 `claude-haiku-4-5`가 네트워크로 실행합니다. 같은 파일, 같은 대화 접두사에서 모델만 다릅니다. 이 조건이 갖춰져야 "어느 모델이 더 잘 수행하는가"라는 질문에 신뢰 있는 답을 줄 수 있습니다.

<strong>모델 비교</strong>

```bash
orca compare last --from 5 \
  --models claude-opus-4-6,claude-haiku-4-5 \
  --verify "npm test"
```

```
MODEL              VERDICT  TOKENS     COST        WALL
claude-opus-4-6      pass     201/25     $0.004890   0.3s
claude-haiku-4-5   pass     201/25     $0.000326   0.3s
```

두 모델 모두 통과하고, Haiku 4.5가 15배 저렴합니다. 모델이 유일한 변수이므로 비교 결과에 의미가 생깁니다.

---

## LangChain/LangGraph 생태계와의 연결성

LangChain/LangGraph 사용자에게 중요한 질문입니다. OrcaReplay의 공식 README는 이렇게 명시합니다.

> **LangGraph / LangChain** | `OPENAI_BASE_URL`, `ANTHROPIC_BASE_URL` | should work — it goes through the official clients, **but nothing here tests it yet**

공식 클라이언트를 통과하는 구조이므로 이론적으로는 작동해야 하지만, 검증된 엔드-투-엔드 테스트가 아직 없습니다. README는 이를 "Prove LangGraph"라는 항목으로 컨트리뷰션 목록에 올려 두었습니다.

{{< figure src="/ai-tech-blog/images/orcareplay-agent-record-replay/diagram-comparison-table.png" alt="OrcaReplay와 LangSmith 기능 비교표. 오프라인 재현, 포크, 셸 exit code 캡처는 OrcaReplay만 지원합니다." caption="OrcaReplay와 LangSmith 기능 비교. 두 도구는 목적이 달라 단순 우열 비교는 적절하지 않습니다." >}}

한편, 이 글을 작성하는 과정에서 `langchain-replay`라는 이름의 독립적인 PyPI 패키지나 GitHub 저장소는 확인하지 못했습니다. LangChain 공식 생태계에서 트레이스 관찰과 재실행에 가장 가까운 도구는 <strong>LangSmith</strong>입니다. LangSmith는 LangGraph 에이전트의 단계별 트레이스, 프롬프트/모델/도구 비교, 그리고 2026년 9월 공개된 베타 기능 Messages View를 제공합니다. 그러나 LangSmith는 SaaS 관찰 플랫폼이고, 오프라인 바이트 단위 재현이나 특정 체크포인트에서 포크하는 기능은 현재 제공하지 않습니다.

Python 기반의 LangChain 사용자를 위한 리플레이 도구로는 [Kitaru](https://github.com/zenml-io/kitaru)(zenml-io, MIT)가 있습니다. Kitaru는 LangSmith, Langfuse, Braintrust 등에서 트레이스를 임포트해 리플레이 기반 평가(eval)를 실행하는 Python 패키지(`pip install kitaru`)입니다. OrcaReplay와 접근 방식이 다르며, 평가와 회귀 테스트에 중점을 둡니다.

---

## 현재 상태와 한계

OrcaReplay는 스스로 "Early. v0 is the walking skeleton"이라고 표현합니다. 2026년 9월 6일 기준으로 GitHub 스타는 11개이고, 139개 커밋이 있습니다. 1,393개 테스트가 Node 20과 22에서 통과합니다.

<strong>확인된 작동</strong>

- Claude Code (ANTHROPIC_BASE_URL, 실제 버그 수정 세션으로 엔드투엔드 검증)
- Codex CLI (API key 로그인, OPENAI_BASE_URL)
- OpenAI Agents SDK
- Vercel AI SDK (fetch 훅)
- grok-cli, OpenClaw, opencode

<strong>현재 한계와 주의사항</strong>

- <strong>LangGraph/LangChain</strong>: "작동해야 하지만 아직 테스트 없음" - 공식 지원 전 직접 검증 필요
- <strong>ChatGPT 구독 로그인 Codex CLI</strong>: 자체 백엔드를 사용하므로 `--tls-intercept`가 필요하고, 이는 의도적 opt-in 결정을 요구합니다
- <strong>트레이스는 민감 정보</strong>: README는 "셸 히스토리 + 힙 덤프 수준의 민감도"로 취급하라고 권고합니다. 인증 헤더와 환경변수는 기본적으로 제외되지만, 완전한 보호는 보장되지 않습니다
- <strong>CLI 중심</strong>: Python API(`from orcareplay import Orca`)가 있지만 주된 인터페이스는 CLI입니다
- <strong>Node.js 전용</strong>: `npm i -g orcareplay`, Node 20 이상 필요. Python 에이전트를 기록할 수는 있지만 OrcaReplay 자체는 Node.js 런타임에서 동작합니다
- <strong>MCP 캡처</strong>: `--mcp-config` 플래그가 있지만 기본 비활성화 상태입니다

---

## 정리

OrcaReplay가 해결하는 문제는 관찰 도구가 아니라 디버거의 문제입니다. 실패한 실행을 다시 실행하면 다른 결과가 나오고, 결과가 달라지면 원인 추론은 추측이 됩니다. 레코드/리플레이 방식은 이 비결정성을 제거합니다.

도구는 이른 단계에 있습니다. LangGraph 지원은 미검증, 스타 수는 두 자릿수입니다. Claude Code와 Codex CLI(API key)에서는 엔드투엔드 검증이 완료되었고, 아키텍처는 공개되어 있으며, 트레이스 포맷은 CC BY 4.0으로 누구나 재구현할 수 있습니다.

에이전트가 실패했을 때 "왜"라는 질문에 답할 수 있는 파일이 있는지 없는지는, 실행을 시작하기 전에 결정됩니다.

---

## References

- [GitHub - Continuum-AI-Corp/OrcaReplay](https://github.com/Continuum-AI-Corp/OrcaReplay) - 공식 저장소 (Apache-2.0)
- [OrcaRouter 공식 블로그: AI Agent Debugging: Replay the Run, Not the Dashboard](https://www.orcarouter.ai/blog/ai-agent-debugging-replay-the-run) - Alistair Wren (2026-09-04 확인)
- [OrcaRouter 공식 블로그: How to Record a Claude Code Session](https://www.orcarouter.ai/blog/record-a-claude-code-session) - Gideon Frost (2026-09-04)
- [PR Newswire: OrcaRouter Launches OrcaReplay](https://www.prnewswire.com/news-releases/orcarouter-launches-orcareplay-an-open-source-record-and-replay-engine-for-ai-agents-302867250.html) - 2026-09-02
- [GitHub - zenml-io/kitaru: Agent traces you can run, not just read](https://github.com/zenml-io/kitaru) - Python 기반 리플레이 eval 도구 (MIT)
- [LangSmith 공식 문서 (LangChain)](https://docs.smith.langchain.com/) - LangChain 에코시스템 관찰 플랫폼
- [GitHub Topics: agent-testing](https://github.com/topics/agent-testing?o=desc&s=stars) - 에이전트 테스트 오픈소스 생태계 개요
