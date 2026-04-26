---
title: "Bedrock AgentCore Managed Harness 심층 해부: 3번의 API 호출로 끝나는 에이전트 배포"
date: 2026-04-26T10:00:00+09:00
draft: false
cover:
  image: "/ai-tech-blog/images/bedrock-agentcore-managed-harness-deep-dive/cover.png"
  alt: "Amazon Bedrock AgentCore Managed Harness"
  relative: false
categories: ["AWS AI/ML"]
tags: ["AgentCore", "Bedrock", "AI Agents", "Strands Agents", "Preview"]
author: "Jesam Kim"
description: "2026년 4월 22일 프리뷰로 공개된 Amazon Bedrock AgentCore Managed Harness를 실전 배포 관점에서 분석합니다. model, systemPrompt, tools 세 개의 선언만으로 에이전트를 배포하는 추상화의 의미, microVM 세션 격리 구조, 기존 Bedrock Agents와의 차이, 한국 개발자를 위한 프리뷰 제약사항을 정리했습니다."
---

AWS가 2026년 4월 22일 <strong>Amazon Bedrock AgentCore Managed Harness</strong>를 프리뷰로 공개했습니다. 같은 날 <strong>AgentCore CLI</strong>와 <strong>AgentCore Skills</strong>도 함께 발표되었고, 세 컴포넌트는 하나의 패키지로 움직입니다. 공식 발표는 [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/get-to-your-first-working-agent-in-minutes-announcing-new-features-in-amazon-bedrock-agentcore/)와 [What's New 공지](https://aws.amazon.com/about-aws/whats-new/2026/04/agentcore-new-features-to-build-agents-faster/)에서 확인할 수 있습니다.

핵심 주장은 단순합니다. 에이전트를 배포하려면 `model`, `systemPrompt`, `tools` 세 가지만 선언하면 된다는 것입니다. 나머지 세션 관리, 실행 환경 격리, 상태 영속화, 관측성, 인증은 AWS가 관리합니다.

이 글은 Solutions Architect 관점에서 Managed Harness를 해부합니다. 어떤 설계가 담겨 있는지, 어떤 경우에 선택해야 하는지, 프리뷰 단계에서 무엇을 조심해야 하는지를 다룹니다.

---

## 에이전트 개발의 90%가 인프라 plumbing이었다

2024년 이후 에이전트 프레임워크는 쏟아졌습니다. LangChain, LangGraph, CrewAI, AutoGen, Strands Agents, Pydantic AI, Google ADK. 각자 다른 추상화를 제공하지만, 실제로 프로덕션에 올려본 팀들은 비슷한 문제를 공유합니다.

핵심 로직은 몇백 줄이면 끝납니다. 그 주변의 인프라가 문제입니다. 세션 스토리지를 어디에 둘지, 장시간 태스크 중간에 프로세스가 죽으면 어떻게 복구할지, 도구 실행을 격리할 샌드박스를 어떻게 구성할지, 다중 사용자 환경에서 에이전트 메모리를 어떻게 분리할지. 여기에 IAM, 로깅, 모니터링, 비용 추적까지 얹으면 실제 에이전트 코드는 전체의 10% 정도로 줄어듭니다.

AgentCore는 2025년 AWS re:Invent 직전에 <strong>Runtime, Memory, Gateway, Browser, Code Interpreter, Identity, Observability</strong> 7개 서비스로 프리뷰가 공개되었습니다. 각 서비스는 각자 강력했지만, 사용하려면 Python SDK로 AgentCore Runtime을 감싸고 Gateway를 IAM으로 연결하고 Memory를 RUNTIME 전략으로 설정하는 식의 조립이 필요했습니다. 조립의 난이도는 여전히 팀 몫이었습니다.

Managed Harness는 이 조립 부담을 AWS 쪽으로 넘긴 추상화입니다. 에이전트 자체는 `harness.json`에 선언하고, 실행 환경은 `agentcore deploy` 한 줄로 배포합니다. Runtime, Memory, Gateway, Browser, Code Interpreter를 직접 조립하는 대신, 그 위에 한 층 더 올린 <strong>매니지드 런타임</strong>을 받는 셈입니다.

![전통적 에이전트 스택 vs Managed Harness 추상화 비교](/ai-tech-blog/images/bedrock-agentcore-managed-harness-deep-dive/stack-comparison.png)

*전통적 에이전트 인프라 구성과 Managed Harness의 추상화 계층 비교.*

## Managed Harness의 핵심 개념

Managed Harness의 철학은 <strong>선언형 에이전트(declarative agent)</strong>입니다. 코드로 에이전트를 "짠다"는 발상 자체를 걷어냅니다.

### 3개의 선언만 필요하다

[AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/get-to-your-first-working-agent-in-minutes-announcing-new-features-in-amazon-bedrock-agentcore/)가 강조하는 숫자는 <strong>3번의 API 호출</strong>입니다. 에이전트를 생성, 배포, 호출하는 데 총 3개의 API가 쓰입니다. 내부적으로 AWS가 생성하는 리소스 수는 훨씬 많지만, 개발자가 작성하는 선언은 세 가지로 고정됩니다.

| 선언 | 역할 | 예시 값 |
|---|---|---|
| `model` | 추론을 담당할 LLM | `global.anthropic.claude-sonnet-4-6` |
| `systemPrompt` | 에이전트 페르소나와 정책 | `system-prompt.md` 파일 참조 |
| `tools` | 에이전트가 호출할 도구 | AgentCore Browser, Code Interpreter, Gateway, MCP 서버 |

[ClassMethod의 분석](https://dev.classmethod.jp/en/articles/bedrock-agentcore-managed-harness-preview/)에 공개된 `harness.json` 예시는 다음처럼 간결합니다.

```json
{
  "name": "MyHarness",
  "model": {
    "provider": "bedrock",
    "modelId": "global.anthropic.claude-sonnet-4-6"
  },
  "tools": [
    {
      "type": "agentcore_code_interpreter",
      "name": "code-interpreter"
    }
  ],
  "skills": []
}
```

`tools` 배열에 `agentcore_browser`, `agentcore_gateway`, `remote_mcp_server`를 추가하면 각각 웹 브라우징, 엔터프라이즈 API 연결, 외부 MCP 서버 연결이 붙습니다. Code Interpreter는 샌드박스 안에서 파이썬 코드를 실행합니다.

### Strands Agents가 엔진이다

Managed Harness가 내부에서 돌리는 에이전트 엔진은 AWS 오픈소스 프레임워크인 <strong>Strands Agents</strong>입니다. [SiliconANGLE 보도](https://siliconangle.com/2026/04/22/aws-accelerates-ai-agent-development-amazon-bedrock-agentcore/)가 이 점을 분명히 합니다. 루프 제어, 도구 바인딩, 에러 복구, 스트리밍 출력이 Strands의 ReAct 스타일 루프 위에서 실행됩니다.

직접 Strands를 사용할 때와 다른 점은 <strong>인프라 결정권</strong>입니다. 직접 쓰면 모델 provider 선택, 세션 스토리지, 도구 등록 방식을 팀이 결정합니다. Managed Harness에서는 선언된 `provider`와 `tools` 값을 받아 AWS가 적절한 Strands 설정을 생성합니다.

Strands 자체가 궁금하다면 [AWS Developer Blog](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/)와 [GitHub 저장소](https://github.com/strands-agents/sdk-python)를 참고하면 됩니다. Managed Harness 사용자는 Strands API를 직접 다루지 않지만, 엔진의 동작 모델을 이해해두면 시스템 프롬프트 설계와 도구 반환값 형태를 잡을 때 도움이 됩니다.

### 지원 모델과 도구

모델 provider는 <strong>Amazon Bedrock, OpenAI, Google Gemini</strong> 세 곳입니다. Bedrock에서는 Claude Sonnet 4.6이 기본값이고 Opus 4.6도 선택할 수 있습니다. OpenAI는 GPT-5.4, Gemini는 Gemini 2.5 Pro 계열을 바라봅니다. provider 전환은 `harness.json`의 `model` 블록 교체만으로 끝납니다.

빌트인 도구는 네 종류입니다.

- <strong>AgentCore Browser</strong>: 격리된 Chromium 세션. 로그인, 폼 제출, 스크래핑을 처리합니다.
- <strong>Code Interpreter</strong>: 파이썬 샌드박스. 데이터 분석, 파일 변환, 계산 태스크에 씁니다.
- <strong>AgentCore Gateway</strong>: OpenAPI/Smithy 스펙을 받아 엔터프라이즈 API를 도구화합니다.
- <strong>Remote MCP Server</strong>: 외부 [Model Context Protocol](https://modelcontextprotocol.io/) 서버를 붙입니다.

도구 선언 시 주의할 점이 있습니다. `tools` 배열에 포함된 각 항목은 실제로 IAM 권한과 매핑됩니다. 예를 들어 `agentcore_browser`를 선언하면 AWS가 생성하는 실행 역할에 Browser 서비스 접근 권한이 자동으로 추가됩니다. 최소 권한 원칙을 지키고 싶다면 선언 목록을 짧게 유지하는 것이 유리합니다. 하니스 하나에 모든 도구를 넣기보다 용도별로 하니스를 나누는 편이 감사 관점에서도 편합니다.

## 실전 배포 가이드

CLI는 `npm install -g @aws/agentcore@preview`로 설치합니다. 현재 버전은 `1.0.0-preview.1`이고, 소스는 [aws/agentcore-cli GitHub](https://github.com/aws/agentcore-cli)에 공개되어 있습니다.

### 프로젝트 생성

`agentcore create` 명령은 인터랙티브 위저드입니다. 이름, 리전, provider, 기본 도구를 물어본 뒤 프로젝트 디렉터리를 생성합니다.

```
harnessSample/
├── agentcore/
│   ├── agentcore.json     # 프로젝트 전체 스펙
│   ├── aws-targets.json   # 배포 계정/리전
│   └── cdk/               # CDK 스택
└── app/
    └── MyHarness/
        ├── harness.json       # 하니스 선언
        └── system-prompt.md   # 시스템 프롬프트
```

`agentcore.json`은 프로젝트 메타데이터와 하니스 목록을 담습니다. `aws-targets.json`은 어느 계정의 어느 리전에 배포할지를 지정합니다. `cdk/`는 AWS가 생성한 CDK 스택으로, 실제 리소스 프로비저닝을 여기서 수행합니다. Terraform 지원은 이어서 제공될 예정이라고 [공식 블로그](https://aws.amazon.com/blogs/machine-learning/get-to-your-first-working-agent-in-minutes-announcing-new-features-in-amazon-bedrock-agentcore/)가 예고했습니다.

하니스 정의는 `app/<하니스명>/harness.json`에 놓습니다. 여러 하니스를 하나의 프로젝트에 두고 독립적으로 배포할 수 있습니다.

### 로컬 개발

`agentcore dev`는 로컬 웹 인스펙터를 띄웁니다. 시스템 프롬프트, 도구 호출, 응답을 브라우저 UI에서 확인하면서 반복 개선할 수 있습니다. 배포 전에 프롬프트를 흔들어보는 용도로 적합합니다.

로컬 개발 모드의 내부 동작은 클라우드와 완전히 같지 않습니다. microVM 대신 로컬 도커 컨테이너가 세션 격리를 담당하고, 파일시스템은 로컬 디스크에 마운트됩니다. Code Interpreter나 Browser 같은 도구는 AWS 측 샌드박스를 호출하도록 연결됩니다. 즉 에이전트 루프는 로컬에서, 도구 실행은 원격에서 이뤄지는 하이브리드 구조입니다. 이 때문에 로컬 개발에도 `profile`에 유효한 AWS 자격 증명이 필요합니다.

![AgentCore CLI 샘플 프로젝트 구조](/ai-tech-blog/images/bedrock-agentcore-managed-harness-deep-dive/cli-sample.jpg)

*AgentCore CLI가 생성하는 프로젝트 레이아웃. 출처: [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/get-to-your-first-working-agent-in-minutes-announcing-new-features-in-amazon-bedrock-agentcore/)*

### 배포

`agentcore deploy`는 내부적으로 CDK 스택을 합성해 대상 계정에 올립니다. AgentCore Runtime 엔드포인트, IAM 역할, CloudWatch 로그 그룹, 필요하다면 VPC 엔드포인트까지 자동 생성됩니다. 배포 결과로 Runtime ARN이 출력되고, 이 ARN이 API 호출 시 타겟이 됩니다.

배포된 하니스는 [InvokeAgentRuntime API](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-get-started-cli.html)로 호출합니다. 동기 호출, 스트리밍, 비동기 작업 모두 지원합니다. 동일한 `sessionId`로 여러 번 호출하면 세션이 이어집니다.

![Managed Harness invoke 데모](/ai-tech-blog/images/bedrock-agentcore-managed-harness-deep-dive/harness-invoke-demo.jpg)

*Managed Harness 호출 결과 예시. 출처: [AWS Machine Learning Blog](https://aws.amazon.com/blogs/machine-learning/get-to-your-first-working-agent-in-minutes-announcing-new-features-in-amazon-bedrock-agentcore/)*

### AgentCore Skills와의 연동

`harness.json`의 `skills` 필드는 비어 있었지만, 여기에는 <strong>AgentCore Skills</strong>를 등록할 수 있습니다. Skills는 코딩 어시스턴트용 플러그인 묶음으로, 첫 번째 타깃은 [Kiro](https://kiro.dev/)입니다. AWS가 Claude Code, OpenAI Codex, Cursor용 플러그인도 4월 말까지 제공한다고 밝혔습니다. 개발자가 IDE 안에서 AgentCore 리소스를 직접 조작할 수 있는 창구가 열리는 셈입니다.

## 내부 아키텍처: microVM 세션 격리와 영속 파일시스템

Managed Harness의 실행 모델은 <strong>세션당 microVM</strong>입니다. 이 선택이 전체 설계를 지배합니다.

![Managed Harness 세션 라이프사이클](/ai-tech-blog/images/bedrock-agentcore-managed-harness-deep-dive/session-lifecycle.png)

*microVM 세션 격리, 영속 파일시스템, suspend/resume 흐름.*

### 왜 microVM인가

에이전트가 Code Interpreter로 임의 파이썬 코드를 실행하고, Browser로 임의 웹사이트에 로그인합니다. 한 세션의 결과가 다른 세션으로 누출되면 보안 사고가 됩니다. 컨테이너 수준 격리로는 부족합니다.

AgentCore는 [Firecracker](https://firecracker-microvm.github.io/) 기반 microVM에서 세션을 돌립니다. AWS Lambda와 Fargate가 쓰는 바로 그 기술입니다. 수십 밀리초 단위로 뜨고, 커널 수준에서 격리되며, VM 단위로 수명을 관리합니다. 세션마다 VM 한 대를 잡아서, 다른 세션과 메모리, 파일시스템, 프로세스 네임스페이스를 모두 분리합니다.

같은 구조는 보안뿐 아니라 <strong>부작용 억제</strong>에도 기여합니다. 에이전트가 파일시스템 전체를 날려도, 시스템 패키지를 설치해도, 다른 사용자에게는 영향이 없습니다. 실험성 태스크를 맡길 때 심리적 허들이 낮아집니다. 팀에서 "이 프롬프트가 뭘 해버릴지 모르니 일단 돌려보자"는 식의 빠른 반복이 가능한 이유입니다.

### 영속 파일시스템과 suspend/resume

VM 생명주기 동안 세션은 전용 파일시스템을 가집니다. Code Interpreter가 만든 CSV, Browser가 받은 스크린샷, 에이전트가 작성한 중간 결과물이 그대로 남습니다. 사용자가 며칠 뒤 같은 `sessionId`로 돌아오면 해당 파일시스템이 복원됩니다.

이 영속성이 <strong>장시간 실행 태스크</strong>를 가능하게 합니다. 8시간짜리 데이터 처리 파이프라인을 에이전트에 맡긴다고 가정합니다. 중간에 네트워크가 끊기거나 사용자가 노트북을 덮어도 문제없습니다. 세션은 suspend되고, 다음 호출에서 마지막 체크포인트부터 resume됩니다. [ClassMethod 분석](https://dev.classmethod.jp/en/articles/bedrock-agentcore-managed-harness-preview/)은 이 suspend/resume 동작을 Managed Harness의 결정적 차별점으로 꼽습니다.

직접 Strands Agents를 돌릴 때는 이 기능을 구현하려면 별도 체크포인트 스토리지, 상태 직렬화, 재시작 트리거가 필요합니다. Managed Harness는 이걸 런타임 수준에서 제공합니다.

### Runtime과의 관계

AgentCore Runtime 자체는 2025년 re:Invent에서 공개된 에이전트 호스팅 서비스입니다. Managed Harness는 Runtime을 걷어낸 게 아니라 그 <strong>위</strong>에 얹힌 매니지드 레이어입니다. 배포된 하니스는 내부적으로 Runtime 엔드포인트로 노출되고, 호출 API도 Runtime과 동일한 `InvokeAgentRuntime`을 씁니다.

차이는 정의 방식입니다. Runtime을 직접 쓰면 파이썬 코드로 에이전트 루프를 작성해 컨테이너 이미지로 말아 올립니다. Managed Harness는 JSON 선언만 올리면 AWS가 대응하는 컨테이너를 합성합니다.

이 관계는 마이그레이션 관점에서도 중요합니다. Managed Harness로 프로토타입을 빠르게 만들어 프로덕션에 올린 뒤, 트래픽이 커지고 커스텀 로직이 필요해지면 같은 팀이 Strands 코드 기반으로 이식할 수 있습니다. 호출 API가 동일하기 때문에 클라이언트 쪽은 거의 그대로 유지됩니다. 반대 방향도 가능합니다. Strands로 구성한 에이전트가 표준 도구 조합으로 수렴한다면 `harness.json`으로 옮겨 운영 부담을 줄일 수 있습니다.

## 선택 기준: Managed Harness vs Code-defined vs Bedrock Agents

지금 AWS에서 에이전트를 만드는 경로는 세 가지입니다. 무엇을 고르느냐가 6개월 뒤 팀의 운영 부담을 결정합니다.

![에이전트 구축 옵션 선택 트리](/ai-tech-blog/images/bedrock-agentcore-managed-harness-deep-dive/decision-tree.png)

*유즈케이스별 Managed Harness / Code-defined / Bedrock Agents 선택 가이드.*

### Managed Harness가 맞는 경우

<strong>적합 조건</strong>: 표준 도구(Browser, Code Interpreter, Gateway, MCP)로 해결되고, 에이전트 로직이 시스템 프롬프트와 도구 조합으로 표현되며, 빠르게 프로덕션에 올려야 하는 경우.

대표 예시는 사내 운영 자동화 에이전트입니다. "S3 버킷 목록을 가져와 조건에 맞는 파일을 정리하고 결과를 Slack에 보내라"는 식의 태스크는 시스템 프롬프트로 정책을 기술하고 Gateway로 내부 API를 붙이는 것으로 충분합니다. 팀은 인프라가 아니라 프롬프트와 정책에 집중할 수 있습니다.

또 하나의 전형적 시나리오는 <strong>프로토타입</strong>입니다. 사업 팀이 "이런 에이전트 있으면 편할 텐데"라고 제안했을 때, `harness.json` 한 파일로 하루 안에 돌아가는 데모를 보여줄 수 있습니다. 이 속도가 아이디어 검증 사이클을 바꿉니다. 프로덕션 승격 시점에 Strands 코드로 이식할지, 아니면 Managed Harness로 그대로 운영할지는 그때 결정하면 됩니다.

### Code-defined (Strands 직접 사용)가 맞는 경우

<strong>적합 조건</strong>: 커스텀 루프 제어, 비표준 도구, 외부 프레임워크와의 깊은 연동이 필요한 경우.

에이전트가 자체 벡터 스토어를 참조하면서 여러 모델을 라우팅해야 한다거나, 사내 빌드 시스템과 직접 붙어야 하는 경우가 대표적입니다. Strands Agents SDK를 직접 쓰면 모든 의사결정 지점을 코드로 제어할 수 있고, [AgentCore Runtime 문서](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime.html)의 배포 가이드를 따라 컨테이너를 직접 올립니다. 운영 부담은 늘어나지만 자유도는 최대입니다.

### Bedrock Agents가 맞는 경우

<strong>적합 조건</strong>: Action Groups, Knowledge Base 기반 RAG, Lambda 중심 오케스트레이션이 주된 요구이고, 이미 팀이 Bedrock Agents 생태계에 익숙한 경우.

Bedrock Agents는 Action Group 스키마와 Knowledge Base 연동이 성숙합니다. 2024년부터 쌓인 운영 사례가 많고, 도메인 특화 에이전트에서 검증된 패턴이 있습니다. 반면 microVM 격리나 장시간 suspend/resume 같은 기능은 Managed Harness 쪽이 앞섭니다.

### 한 줄 요약

- 빠르게 표준 조합으로 올리고 싶다 → <strong>Managed Harness</strong>
- 루프와 도구를 코드로 제어하고 싶다 → <strong>Strands 직접</strong>
- Knowledge Base + Lambda 조합이 주력이다 → <strong>Bedrock Agents</strong>

## 프리뷰 제약사항과 프로덕션 고려점

프리뷰 단계의 가장 큰 제약은 <strong>리전 4곳</strong>입니다. us-west-2(오레곤), us-east-1(버지니아), eu-central-1(프랑크푸르트), ap-southeast-2(시드니)만 지원합니다. [What's New 공지](https://aws.amazon.com/about-aws/whats-new/2026/04/agentcore-new-features-to-build-agents-faster/)가 확인하는 범위입니다.

한국에서 프로덕션에 올리려면 몇 가지 현실적 결정이 필요합니다. 첫째, 지연시간 허용 범위가 관건입니다. 시드니가 물리적으로 가장 가깝지만 왕복 100ms 내외를 감당할 수 있어야 합니다. 초당 수십 건 이상 트래픽이면 오레곤이 용량 측면에서 유리합니다. 둘째, 데이터 거주 이슈를 봐야 합니다. 민감 데이터가 세션 파일시스템에 잠시라도 머문다면 프랑크푸르트나 오레곤의 컴플라이언스 요건을 따로 확인해야 합니다. 셋째, 서울 리전(ap-northeast-2) 확장을 기다리는 선택지가 있습니다. AgentCore Runtime은 이미 서울 리전 일부 기능을 지원하므로, Managed Harness도 GA 시점에 확장될 가능성이 높습니다.

인증은 <strong>IAM 기본</strong>입니다. 엔드 유저용 인증이 필요하다면 AgentCore Identity 통합을 고려해야 합니다. Identity는 OAuth 2.0과 기업 IdP를 받고, 발급된 토큰을 Runtime 호출에 주입하는 역할을 합니다.

관측성은 [AgentCore Observability](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html)가 기본 제공합니다. CloudWatch 로그, OpenTelemetry 트레이스, 세션별 지표가 콘솔에서 조회됩니다. 장시간 세션이 많아지면 Observability 대시보드가 디버깅의 1차 창구가 됩니다.

프로덕션 체크리스트 차원에서 짚어둘 항목이 몇 개 있습니다. 첫째, 세션 TTL입니다. 영속 파일시스템이 영원히 유지되지는 않으므로, 긴 워크플로의 경우 어느 시점에 세션이 회수되는지를 확인해야 합니다. 둘째, 레이트 리밋입니다. 기반 Bedrock 모델의 TPM/RPM 한도가 하니스 전체 처리량을 결정합니다. 셋째, 도구 호출 순환 고리입니다. 시스템 프롬프트가 느슨하면 에이전트가 같은 도구를 반복 호출하는 루프에 빠질 수 있고, microVM 단위 비용이 누적됩니다. Observability의 도구 호출 히스토그램을 알람으로 걸어두면 조기에 잡을 수 있습니다. 넷째, 재현성입니다. 동일한 입력으로도 LLM 응답이 달라질 수 있으므로, 배포 전 회귀 테스트는 확률적 평가 지표로 설계해야 합니다.

[공식 블로그](https://aws.amazon.com/blogs/machine-learning/get-to-your-first-working-agent-in-minutes-announcing-new-features-in-amazon-bedrock-agentcore/)에 따르면 CLI, Managed Harness, Skills에는 <strong>추가 과금이 없습니다</strong>. 사용자는 뒷단의 Bedrock 추론 비용, Runtime 실행 비용, 도구별 비용(Code Interpreter, Browser 등)만 부담합니다. 추상화 레이어 자체에 별도 비용이 붙지 않는 구조는 프리뷰 도입 장벽을 상당히 낮춥니다.

VTEX의 VP of Engineering [Rodrigo Moreira](https://siliconangle.com/2026/04/22/aws-accelerates-ai-agent-development-amazon-bedrock-agentcore/)는 "AgentCore Managed Harness와 CLI로 에이전트 기반 기능의 출시 시간을 단축할 수 있었다"고 공개 블로그에서 언급했습니다. 프리뷰 시점의 단일 고객 인용이지만, 엔터프라이즈 커머스 도메인에서 실전 배포 사례가 최소 하나는 존재한다는 신호입니다.

## 시스템 프롬프트 설계가 전부다

Managed Harness의 선언 세 가지 중 현실적으로 팀이 가장 많이 시간을 쓰는 영역은 `systemPrompt`입니다. 모델은 기본값으로 두고 도구는 표준 조합에서 고르면 되지만, 시스템 프롬프트의 품질이 에이전트의 신뢰도를 결정합니다.

프리뷰 기간 동안 몇 가지 패턴을 권장합니다. 첫째, <strong>에이전트의 경계</strong>를 명시합니다. "이 에이전트는 X를 할 수 있고 Y는 하지 않는다"는 문장을 선두에 둡니다. 둘째, <strong>도구 사용 정책</strong>을 구체화합니다. Code Interpreter를 쓸 때 어떤 입력을 기대하는지, Browser로 어떤 사이트만 허용하는지를 프롬프트 안에 명시하면 런타임 오용을 줄일 수 있습니다. 셋째, <strong>출력 형식</strong>을 고정합니다. JSON 응답이 필요한 태스크라면 스키마를 예시로 넣는 편이 후행 파이프라인 연결 비용을 낮춥니다. 넷째, <strong>실패 처리</strong>를 포함합니다. "도구가 실패하면 재시도하지 말고 에러 메시지를 그대로 보고하라" 같은 지침이 무한 루프를 막습니다.

`system-prompt.md`는 마크다운이므로 섹션을 나눠 관리하는 것이 실용적입니다. 버전 관리 관점에서도 Git 히스토리로 프롬프트 변경을 추적할 수 있다는 점이 코드와 일체화된 `harness.json` 구조의 장점입니다.

## GA를 기다리며

프리뷰에서 GA로 넘어가는 구간에 개발자가 준비할 항목도 정리해둘 만합니다. API 시그니처는 `1.0.0-preview.1` 기준으로 잡혀 있지만, GA 시점에 `harness.json` 스키마가 변경될 여지가 있습니다. CLI 릴리스 노트와 [AgentCore 개발자 가이드](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-get-started-cli.html)를 북마크해두면 변경 포인트를 빠르게 추적할 수 있습니다.

컴플라이언스 쪽은 특히 주의가 필요합니다. 프리뷰 서비스는 FedRAMP, PCI DSS, HIPAA 같은 표준 인증 범위에서 제외되는 경우가 많습니다. 민감 워크로드를 올리기 전에 [AWS Service Level Agreements](https://aws.amazon.com/legal/service-level-agreements/) 페이지에서 GA 전환 시점과 지원 인증 범위를 다시 확인해야 합니다. 내부 결재 라인에는 "현재는 프리뷰이며 프로덕션 SLA가 적용되지 않는다"는 명시가 들어가야 안전합니다.

한 가지 더. Managed Harness는 AWS가 정의한 추상화입니다. 에이전트 로직을 이 추상화에 맞춰 쓰면 생산성이 높지만, AWS 종속성이 따라붙습니다. 멀티클라우드를 고려하는 팀이라면 `systemPrompt`와 도구 호출 로직을 별도 파일로 분리해 Strands 코드로 이식 가능한 구조를 유지하는 편이 좋습니다. Strands 자체는 모델 agnostic 오픈소스이므로, AWS 밖에서도 재사용 가능한 자산이 됩니다.

## 마무리: 첫 하니스 올리기

정리하면 Managed Harness는 AgentCore 위에 선언형 레이어를 얹어 에이전트 개발의 인프라 부담을 줄이는 추상화입니다. 핵심은 <strong>세 가지 선언(model, systemPrompt, tools)</strong>, 엔진은 <strong>Strands Agents</strong>, 실행 모델은 <strong>microVM + 영속 파일시스템 + suspend/resume</strong>, 비용은 <strong>기반 리소스 사용분만</strong>. 2026년 4월 22일 프리뷰가 시작되었고, 4개 리전에서 사용할 수 있습니다.

지금 해볼 일은 단순합니다. us-west-2 프로필로 CLI를 설치하고 `agentcore create`로 첫 프로젝트를 만든 뒤 `agentcore dev`로 로컬 인스펙터에서 프롬프트를 흔들어보는 것입니다. 빌트인 도구 중 Code Interpreter 하나만 붙여도 "CSV를 받아서 요약하고 차트를 그려라" 수준의 태스크가 동작합니다. 거기서부터 Gateway, Browser, MCP로 확장하면 됩니다.

Managed Harness가 모든 에이전트를 대체하진 않습니다. 코드로 루프를 제어해야 하는 도메인은 여전히 Strands 직접 사용이 맞고, Knowledge Base 중심 워크플로는 Bedrock Agents가 익숙합니다. 다만 <strong>"JSON 선언과 CLI 한 줄"</strong>로 시작하는 옵션이 생겼다는 건, 2026년 하반기 이후 신규 프로젝트의 기본값이 바뀔 가능성이 크다는 뜻입니다.

---

## References

1. AWS Machine Learning Blog, "Get to your first working agent in minutes: Announcing new features in Amazon Bedrock AgentCore" (2026-04-22). https://aws.amazon.com/blogs/machine-learning/get-to-your-first-working-agent-in-minutes-announcing-new-features-in-amazon-bedrock-agentcore/
2. AWS What's New, "Amazon Bedrock AgentCore announces new features to build agents faster" (2026-04-22). https://aws.amazon.com/about-aws/whats-new/2026/04/agentcore-new-features-to-build-agents-faster/
3. Amazon Bedrock AgentCore Developer Guide, "Get started with AgentCore CLI". https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agentcore-get-started-cli.html
4. AgentCore CLI GitHub Repository. https://github.com/aws/agentcore-cli
5. ClassMethod DevelopersIO, "Bedrock AgentCore Managed Harness Preview Deep Dive" (2026-04-23). https://dev.classmethod.jp/en/articles/bedrock-agentcore-managed-harness-preview/
6. SiliconANGLE, "AWS accelerates AI agent development with Amazon Bedrock AgentCore updates" (2026-04-22). https://siliconangle.com/2026/04/22/aws-accelerates-ai-agent-development-amazon-bedrock-agentcore/
