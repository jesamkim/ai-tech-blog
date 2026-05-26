---
title: "GPT-5.5가 Bedrock에 온다 — Mantle 엔드포인트로 OpenAI SDK 그대로 쓰기"
date: 2026-05-26T21:30:00+09:00
draft: false
author: "Jesam Kim"
description: "Amazon Bedrock의 새 추론 엔진 Mantle. OpenAI 호환 엔드포인트로 OpenAI SDK 코드를 거의 그대로 AWS에서 돌리는 법, Invoke/Converse와의 차이, Responses vs Chat Completions 선택 기준을 공식 문서 기반으로 정리합니다."
tags:
  - Amazon Bedrock
  - Mantle
  - OpenAI
  - GPT-5.5
  - GenAI
  - AWS
categories:
  - AWS AI
cover:
  image: "/ai-tech-blog/images/bedrock-mantle-openai-migration/cover.png"
  alt: "Amazon Bedrock Mantle과 OpenAI 호환 API"
  relative: false
---

## 들어가며

2026년 4월 28일, AWS와 OpenAI는 [확장 파트너십을 발표](https://www.aboutamazon.com/news/aws/bedrock-openai-models)했다. 핵심은 세 가지로, 셋 다 limited preview 단계다. OpenAI 최신 모델이 Amazon Bedrock에서 사용 가능해졌고, OpenAI 코딩 에이전트 Codex가 Bedrock 위에 올라왔으며, OpenAI 모델로 동작하는 Amazon Bedrock Managed Agents가 추가됐다. 같은 주 OpenAI는 [GPT-5.5 Instant를 ChatGPT 기본 모델로 전환](https://techcrunch.com/2026/05/05/openai-releases-gpt-5-5-instant-a-new-default-model-for-chatgpt/)했다.

엔지니어 입장에서 의미는 단순하다. OpenAI SDK로 짜둔 코드를 거의 손대지 않고 AWS 인프라 위에서 돌릴 수 있는 통로가 열렸다는 점이다. 그 통로의 이름이 <strong>Mantle</strong>이다.

이 글은 Mantle을 처음 접하는 사람을 위한 정리다. Mantle이 무엇이고, 기존 Invoke·Converse API와 어떻게 다른지, 마이그레이션은 실제로 어떻게 하는지, Responses API와 Chat Completions API 중 무엇을 언제 써야 하는지를 공식 문서 기준으로 짚는다.

## 1. Mantle이 뭔가 — 엔진과 엔드포인트의 분리

[AWS 공식 문서](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html)는 Mantle을 "a distributed inference engine for large-scale machine learning model serving"으로 정의한다. 풀어쓰면 Bedrock 안에서 대규모 모델 서빙을 담당하는 분산 추론 엔진이다. 그리고 이 엔진은 `bedrock-mantle.{region}.api.aws` 라는 별도 엔드포인트로 OpenAI 호환 API를 노출한다.

엔진과 엔드포인트의 분리가 중요한 포인트다. Mantle은 내부 추론 엔진의 이름이고, 사용자가 만나는 인터페이스는 그 엔진이 말단에 노출하는 OpenAI 호환 HTTP API다.

공식 문서가 정리한 핵심 가치는 다섯 가지다.

- <strong>Asynchronous inference</strong> — Responses API를 통해 long-running 워크로드 지원
- <strong>Stateful conversation management</strong> — 대화 컨텍스트를 서버측에서 자동 재구성
- <strong>Simplified tool use</strong> — agentic 워크플로 통합 단순화
- <strong>Flexible response modes</strong> — streaming/non-streaming 양쪽 지원
- <strong>Easy migration</strong> — 기존 OpenAI SDK 코드 호환

![Bedrock의 두 엔드포인트 — bedrock-runtime과 bedrock-mantle](/ai-tech-blog/images/bedrock-mantle-openai-migration/diagram1-endpoints.svg)

*Bedrock의 두 엔드포인트 — 같은 모델을 다른 인터페이스로 노출. 출처: [AWS Bedrock 공식 문서 (apis.html)](https://docs.aws.amazon.com/bedrock/latest/userguide/apis.html)*

## 2. 이전 방식과 어떻게 다른가 — Invoke / Converse / Mantle

처음 접하는 사람이 가장 헷갈리는 지점이 여기다. Bedrock에는 이미 추론용 API가 여럿 있었고, Mantle은 그 위에 추가된 새 옵션이다. [AWS 공식 API 비교 문서](https://docs.aws.amazon.com/bedrock/latest/userguide/apis.html)를 따라 정리하면 이렇다.

| 엔드포인트 | API | 특징 |
|---|---|---|
| `bedrock-runtime` | <strong>Invoke</strong> | 가장 오래된 방식. 모델별 raw request/response 포맷. 풀 컨트롤이 가능하지만 모델을 바꾸면 코드도 바꿔야 한다 |
| `bedrock-runtime` | <strong>Converse</strong> | 모든 모델을 같은 인터페이스로 호출하는 AWS native 표준. 한 번 짠 코드로 모델을 교체할 수 있다 |
| `bedrock-runtime` | Messages API (via InvokeModel) | Anthropic native 포맷 직접 액세스 |
| `bedrock-runtime` | Chat Completions | OpenAI 호환 stateless |
| <strong>`bedrock-mantle`</strong> | <strong>Responses API</strong> | OpenAI Responses API 호환. 서버측 stateful 대화, agentic 워크플로 |
| <strong>`bedrock-mantle`</strong> | <strong>Chat Completions</strong> | OpenAI Chat Completions 호환. 모든 Bedrock 모델 지원 |
| <strong>`bedrock-mantle`</strong> | Messages API | Anthropic native 포맷 (Mantle에서도 사용 가능) |

같은 표 안의 권장 시나리오도 명시돼 있다.

- OpenAI API와 호환되는 엔드포인트에서 마이그레이션 → Responses API 또는 Chat Completions API
- OpenAI 호환 엔드포인트에서 다루지 않는 모델 → Converse 또는 Invoke
- 모델 간 일관된 인터페이스 필요 → Converse API
- 요청/응답 포맷 풀 컨트롤 필요 → Invoke API
- Bedrock을 처음 쓰는 경우 → "We recommend using open APIs such as Messages API, Chat Completions API, or Responses API. These APIs are available on both endpoints, but we recommend the bedrock-mantle endpoint."

OpenAI 자체 가이드 역시 [장기적으로는 Responses API를 권장](https://platform.openai.com/docs/guides/migrate-to-responses)한다. 정리하면 SA 관점에서 Mantle은 "OpenAI 표준에 합쳐 들어간 입구"다. Invoke나 Converse가 사라지는 흐름이 아니라, 마이그레이션·표준화·신규 워크로드를 위한 별도 옵션으로 자리 잡았다.

## 3. 3분 마이그레이션 — `OPENAI_BASE_URL` 한 줄

실제 코드 변화는 작다. 같은 OpenAI Python SDK로 두 가지 환경변수만 바꾼다.

<strong>Before — OpenAI 직접 호출:</strong>

```python
import os
from openai import OpenAI

# 기본값
# os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
# os.environ["OPENAI_API_KEY"] = "sk-..."

client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)
```

<strong>After — Bedrock Mantle 경유:</strong>

```python
import os
from openai import OpenAI

os.environ["OPENAI_BASE_URL"] = "https://bedrock-mantle.us-east-1.api.aws/v1"
os.environ["OPENAI_API_KEY"] = "<Bedrock API key>"  # AWS 콘솔에서 발급

client = OpenAI()
resp = client.chat.completions.create(
    model="openai.gpt-oss-120b",   # Bedrock 모델 식별자
    messages=[{"role": "user", "content": "Hello"}],
)
```

차이는 두 줄이다. `OPENAI_BASE_URL`을 `bedrock-mantle.{region}.api.aws/v1`로 바꾸고, API 키를 [Bedrock API key](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html)로 발급해 끼운다. 모델 식별자는 OpenAI가 아닌 Bedrock 모델 ID 체계(`openai.gpt-oss-120b`처럼)를 따른다.

인증은 두 가지 옵션이 있다. OpenAI SDK를 그대로 쓰는 경우 Bedrock API key가 필수이고, HTTP 직접 호출이라면 AWS credentials도 사용할 수 있다.

리전은 현재 13곳을 지원한다. us-east-1, us-east-2, us-west-2, ap-northeast-1(도쿄), ap-southeast-2, ap-southeast-3, ap-south-1, eu-central-1, eu-west-1, eu-west-2, eu-south-1, eu-north-1, sa-east-1.

⚠️ 한국 리전(ap-northeast-2 서울)은 현재 지원 목록에 없다. 한국에서 운용하는 워크로드라면 도쿄(ap-northeast-1)를 후보로 두는 게 자연스럽다. 데이터 거버넌스나 레이턴시 요건에 따라 달라질 수 있다.

## 4. Responses API vs Chat Completions — 무엇을 언제?

Mantle 안에는 두 개의 OpenAI 호환 API가 있고, 둘은 완전히 동일하지 않다. AWS Principal Developer Advocate Danilo Poccia가 공개한 [bedrock-mantle CLI 레포](https://github.com/danilop/bedrock-mantle) README에서 발췌한 비교가 가장 명확하다.

| Feature | Responses API | Chat Completions API |
|---|---|---|
| 모델 지원 | OpenAI OSS GPT 모델만 (현재) | 모든 Bedrock 모델 |
| State Management | Server-side (stateful) | Client-side (stateless) |
| Background Processing | ✓ | ✗ |
| ZDR (Zero Data Retention) 호환 | ✗ (~30일 저장) | ✓ |
| Conversation Context | 자동 (`previous_response_id`) | 수동 (history 직접 관리) |
| Cancel Request | ✓ | ✗ |

각 API가 적합한 상황을 정리하면 다음과 같다.

<strong>Responses API를 고를 만한 경우</strong>

- agentic 워크플로 — built-in tool use, code interpreter, web search 같은 기능 사용
- 멀티 턴 대화에서 토큰을 아끼고 싶을 때 — 서버가 컨텍스트를 유지하므로 history 재전송 불필요
- 백그라운드 long-running 작업 — 연결 타임아웃을 피하고 비동기로 처리
- 멀티모달 입력
- 단, 현재 <strong>OpenAI OSS GPT 모델만</strong> 지원한다는 점은 분명한 제약이다

<strong>Chat Completions API를 고를 만한 경우</strong>

- 모든 Bedrock 모델 — Anthropic, Meta, Mistral, Cohere, Amazon Nova, OpenAI까지 자유롭게 전환
- ZDR 컴플라이언스 — 금융·의료·공공처럼 데이터 미보관 요건이 있는 도메인
- stateless 워크로드 — 클라이언트가 대화 history를 풀 컨트롤
- 단발성 호출에서 latency를 줄이고 싶을 때

![Mantle 엔드포인트 API 선택 의사결정 플로우](/ai-tech-blog/images/bedrock-mantle-openai-migration/diagram2-decision.svg)

*Mantle 엔드포인트 API 선택 가이드. SA 관점 정리.*

엔지니어 관점의 권고는 단순하다. 새로 시작한다면 모델 전환이 자유로운 Chat Completions로 출발하고, agentic·stateful 요건이 분명해지는 시점에 Responses API를 평가한다. ZDR이 비즈니스 요건이라면 Chat Completions를 고정값으로 둔다.

## 5. GPT-5.5 limited preview와 앞으로의 그림

[2026-04-28 About Amazon 발표](https://www.aboutamazon.com/news/aws/bedrock-openai-models)는 세 가지 신규 항목을 공식화했고, 모두 limited preview 상태다.

- <strong>OpenAI Models on Amazon Bedrock</strong> — OpenAI 최신 모델을 기존 Bedrock API와 컨트롤로 사용
- <strong>Codex on Amazon Bedrock</strong> — OpenAI 코딩 에이전트. Bedrock API + Codex CLI / 데스크톱 앱 / VS Code 확장
- <strong>Amazon Bedrock Managed Agents, powered by OpenAI</strong> — OpenAI 모델 기반 production-ready 에이전트 빌딩 환경

발표문은 Codex 사용 규모를 "more than 4 million people use Codex every week"로 적었다. 같은 주 [TechCrunch](https://techcrunch.com/2026/05/05/openai-releases-gpt-5-5-instant-a-new-default-model-for-chatgpt/) 보도에 따르면 OpenAI는 5월 5일 GPT-5.5 Instant를 출시했고, ChatGPT의 새 기본 모델이 됐다.

엔터프라이즈 컨트롤은 그대로 상속된다. About Amazon 발표문 인용을 그대로 옮기면, OpenAI 모델은 IAM 기반 액세스 관리, AWS PrivateLink 연결, guardrails, 전송·저장 시 암호화, AWS CloudTrail 로깅, 기존 컴플라이언스 프레임워크와의 통합을 모두 받는다. 이 부분이 Bedrock으로 들어왔을 때의 운영적 가치다. OpenAI 키를 별도 보관·로테이션할 필요 없이 기존 AWS 자격증명·로깅·결제 체계를 그대로 쓴다.

⚠️ 단정 표현은 조심해야 한다. 현재 시점에 GPT-5.5는 Bedrock에서 GA가 아니라 limited preview다. 정식 GA 시점은 [공식 발표문](https://www.aboutamazon.com/news/aws/bedrock-openai-models)에 명시돼 있지 않다. 또한 Mantle은 OpenAI 전용 통로가 아니다. Anthropic Claude 역시 Mantle 엔드포인트의 [Messages API로 호출 가능](https://docs.aws.amazon.com/bedrock/latest/userguide/apis.html)하다고 공식 문서에 적혀 있다. AWS 블로그의 [Claude Opus 4.7 발표글](https://aws.amazon.com/blogs/aws/introducing-anthropics-claude-opus-4-7-model-in-amazon-bedrock/)도 `AnthropicBedrockMantle` SDK 클래스를 예시 코드에 사용한다.

## 마치며

Bedrock의 의미는 모델 선택지의 양이 아니라 그 선택지에 동일한 가드레일을 걸어둔다는 점에 있다. 단일 콘솔, 단일 IAM, 단일 빌링 위에서 OpenAI·Anthropic·Meta·Mistral·Cohere·Amazon 모델을 다룬다. Mantle은 이 그림에 OpenAI 호환 표준이라는 입구를 추가한 행보로 읽을 수 있다.

한계는 솔직하게 짚어두는 게 맞다. 서울 리전이 빠져 있고, Responses API는 현재 OpenAI OSS GPT 한정이며, ZDR 요건이 있다면 API 선택지가 좁아진다. GPT-5.5는 GA 시점이 정해지지 않은 limited preview다. 이 제약을 인지한 상태에서 OpenAI SDK 자산이 있는 팀이라면 마이그레이션 비용이 거의 없다는 점이 가장 실용적인 장점이다.

후속 글에서는 Mantle의 [Projects API](https://aws.amazon.com/about-aws/whats-new/2026/03/amazon-bedrock-projects-api-mantle-inference-engine/)로 다중 프로젝트 IAM 격리와 비용 가시성을 구성하는 방법, 그리고 Codex on Amazon Bedrock을 엔터프라이즈 IDE에 연결하는 워크플로를 다룰 계획이다.

## References

1. [Inference using Responses API — Amazon Bedrock User Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-mantle.html) (AWS 공식 문서)
2. [APIs supported by Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/apis.html) (AWS 공식 문서)
3. [AWS and OpenAI announce expanded partnership to bring frontier intelligence to the infrastructure you already trust](https://www.aboutamazon.com/news/aws/bedrock-openai-models) (About Amazon, 2026-04-28)
4. [Amazon Bedrock announces OpenAI-compatible Projects API](https://aws.amazon.com/about-aws/whats-new/2026/03/amazon-bedrock-projects-api-mantle-inference-engine/) (AWS What's New, 2026-02-26)
5. [danilop/bedrock-mantle — CLI for exploring Amazon Bedrock OpenAI-compatible APIs](https://github.com/danilop/bedrock-mantle) (GitHub, AWS Principal Developer Advocate Danilo Poccia)
6. [Introducing Anthropic's Claude Opus 4.7 model in Amazon Bedrock](https://aws.amazon.com/blogs/aws/introducing-anthropics-claude-opus-4-7-model-in-amazon-bedrock/) (AWS Blog)
7. [OpenAI to make its models available via Amazon's servers](https://www.axios.com/2026/04/28/amazon-cloud-deal-openai) (Axios, 2026-04-28)
8. [OpenAI releases GPT-5.5 Instant, a new default model for ChatGPT](https://techcrunch.com/2026/05/05/openai-releases-gpt-5-5-instant-a-new-default-model-for-chatgpt/) (TechCrunch, 2026-05-05)
