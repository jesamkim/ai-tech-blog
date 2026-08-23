---
title: "에이전트 보안은 한 번의 Tool Call로 끝나지 않는다: Per-Action Check에서 Trajectory Assurance로"
date: 2026-08-23T09:00:00+09:00
draft: false
categories: ["논문 리뷰", "AI 에이전트"]
tags: ["Agentic AI", "AI Security", "Trajectory Assurance", "Bedrock AgentCore", "Cedar", "MCP", "A2A"]
author: "Jesam Kim"
cover:
  image: "/ai-tech-blog/images/trajectory-assurance-agent-security/cover.png"
  alt: "에이전트의 여러 실행 궤적이 정책 검증 지점으로 모이는 장면"
  relative: false
description: "개별 tool call이 모두 허용 범위 안에 있어도 그 순서가 쌓이면 규칙을 위반할 수 있습니다. 2026년 8월 공개된 vision paper 한 편을 근거로 per-action check의 한계를 정리하고, 세션 이력을 보는 정책을 실제로 어떻게 표현하는지 Amazon Bedrock AgentCore의 temporal policy 문서로 확인합니다."
---

사내 문서를 검색하고 요약해서 메일로 보내는 에이전트를 하나 상상해 보겠습니다. 이 에이전트에는 세 가지 도구가 붙어 있습니다. 문서 검색, 문서 본문 읽기, 메일 발송입니다. 세 도구는 각각 정당한 권한을 받았고, 호출 한 건씩 떼어 놓고 보면 어느 것도 규칙을 어기지 않습니다. 검색은 읽기 전용이고, 본문 읽기도 사용자가 접근 권한을 가진 문서만 반환하며, 메일 발송은 사용자가 명시적으로 요청한 기능입니다.

그런데 이 세 호출이 다음 순서로 일어나면 상황이 달라집니다. 에이전트가 인사 평가 문서를 검색하고, 그 본문을 읽어 컨텍스트에 담고, 곧바로 외부 도메인 주소로 요약 메일을 발송합니다. 호출 하나하나를 검사하는 게이트는 이 흐름을 통과시킵니다. 세 번째 호출을 심사하는 시점에 그 게이트가 보는 것은 "이 사용자가 메일을 보낼 권한이 있는가"뿐이고, 두 번째 호출에서 민감 문서가 컨텍스트로 들어왔다는 사실은 심사 대상에 들어 있지 않기 때문입니다.

2026년 8월에 arXiv에 올라온 [Securing Agentic AI: From Per-Action Checks to Trajectory Assurance](https://arxiv.org/abs/2608.01558)가 정확히 이 지점을 다룹니다. 논문의 표현으로는 "개별적으로 허용되는 행동의 연쇄가 종합적으로는 시스템 수준의 제약과 안전 불변식을 위반할 수 있다"는 문제입니다. 다만 이 논문의 성격을 먼저 분명히 해 둘 필요가 있습니다. Purdue University의 Alireza Lotfi, Subangkar Karmaker Shanto, Elisa Bertino와 University of Texas at Dallas의 Imtiaz Karim이 쓴 6페이지 분량의 문서이고, ACM AI Leadership Summit 2026의 Visionary Track에 채택되었습니다. 저자들이 본문에서 "vision paper로서 우리의 목표는 완전한 해법을 제안하는 것이 아니라 핵심 문제를 식별하는 것"이라고 직접 밝힙니다. 실험 결과나 벤치마크로 검증한 방법론은 여기에 없고, 저자들이 열한 개로 정리한 연구 방향이 본문의 골격입니다. 이 글에서 논문을 인용하는 부분은 모두 그 전제 위에서 읽어야 합니다.

그래서 이 글은 두 가지를 구분해서 다룹니다. 앞부분은 논문이 제기한 문제와 연구 방향이고, 뒷부분은 그 방향 중 일부가 실제 제품 문서에 어떤 형태로 나타나 있는지를 확인하는 작업입니다. 뒷부분은 검증된 해법의 소개로 읽으면 안 되고, 문제를 정책으로 표현할 수 있다는 것을 보여 주는 실물 사례로 읽어야 합니다.

## Per-action check가 놓치는 것

지금 대부분의 에이전트 보안 통제는 요청 한 건 단위로 작동합니다. 게이트웨이나 프록시가 tool call을 가로채고, 호출 주체의 신원과 대상 도구, 입력 파라미터를 정책과 맞춰 보고, 허용하거나 거부합니다. 이 구조는 stateless입니다. 판단에 필요한 정보가 현재 요청 안에 모두 들어 있다고 전제하기 때문입니다.

이 전제는 두 가지 조건이 성립할 때만 안전합니다. 첫째, 위반이 단일 행동으로 완성되어야 합니다. 둘째, 그 행동의 위험도가 이전에 무슨 일이 있었는지와 무관하게 결정되어야 합니다. 에이전트는 두 조건을 모두 깨뜨립니다. 여러 단계를 자율적으로 이어 붙이는 것이 에이전트의 작동 방식 자체이고, 각 단계의 위험도는 앞선 단계가 무엇을 컨텍스트에 넣었고 어떤 권한을 이미 소모했는지에 따라 달라집니다.

논문은 이 문제를 <strong>behavioral containment</strong>라는 이름으로 다루면서, 시스템 수준 격리와 행동 수준 억제를 나눕니다. 샌드박스를 씌우고 자격 증명을 분리하는 작업은 전자에 해당하고, 이미 상당히 정리된 영역입니다. 어려운 쪽은 후자입니다. 논문이 드는 예시 중 하나는 5G 스케줄링 에이전트가 공공안전 슬라이스의 우선순위를 조금씩 낮추는 상황이고, 다른 하나는 임상 에이전트가 개별적으로는 모두 정당한 퇴원 결정을 연달아 앞당기면서, FHIR 진료 경로가 정한 집계 관찰 구간을 위반하는 상황입니다. 저자들은 두 예시에 "적대적 입력이 개입하지 않는다"는 점을 명시합니다. 공격자가 없어도, 프롬프트 인젝션이 없어도, 조합만으로 위반이 성립합니다.

기존 도구가 왜 부족한지에 대해서도 논문은 세 가지를 지적합니다. 런타임 가드는 stateless이고, 형식 명세를 정책으로 컴파일하는 접근은 완전성을 보장하지 못하며, 실제 적용은 단일 에이전트 범위에 머물러 있습니다. 여기에 평가 쪽 문제를 하나 덧붙입니다. 에이전트 벤치마크의 대다수가 구체적인 정책 명세를 아예 제공하지 않는다는 것입니다. 위반을 판정할 기준이 데이터셋에 없으면 통제 기법의 성능을 비교하기도 어렵습니다.

## Trajectory assurance가 판단에 넣어야 하는 것

논문이 방향으로 제시하는 것은 개별 행동의 적법성 대신 궤적 전체의 적법성을 검증 가능한 속성으로 만드는 일입니다. 저자들의 표현으로는 "권고성 지침과 stateless 가드레일에서 검증 가능한 행동 불변식으로의 전환"입니다. 구체적인 알고리즘은 논문에 없습니다. 그래서 여기서는 논문의 문제 제기를 받아, 요청 한 건 단위 판단이 실제로 무엇을 보지 못하는지를 네 가지로 나눠 정리하겠습니다. 이 분류 자체는 논문의 것이 아니고, 문제를 구현 관점에서 옮겨 놓은 것입니다.

<strong>상태</strong>가 첫 번째입니다. 같은 도구를 같은 파라미터로 호출해도, 그 시점의 세션 상태에 따라 위험도가 달라집니다. 앞의 예시에서 메일 발송의 위험도를 결정한 것은 발송 권한이 아니라 컨텍스트에 이미 민감 문서가 들어와 있다는 상태였습니다. 요청 한 건만 보는 게이트는 이 상태를 관측하지 않습니다.

<strong>순서</strong>가 두 번째입니다. 어떤 행동은 특정 행동이 앞서 일어났을 때만 허용됩니다. 송금 전에 승인이 있어야 하고, 삭제 전에 백업 확인이 있어야 하고, 외부 전송 전에 검토 단계를 통과해야 합니다. 두 행동을 따로 심사하면 두 행동 모두 통과하지만, 순서가 뒤집힌 궤적은 규칙 위반입니다. 순서 제약은 시간축을 판단에 넣지 않으면 표현할 방법이 없습니다.

<strong>누적 권한</strong>이 세 번째입니다. 에이전트가 다른 에이전트에게 작업을 위임하고, 그 에이전트가 또 위임하는 구조에서 권한은 홉을 넘어가며 조금씩 넓어질 수 있습니다. 논문은 A2A 프로토콜에서 신원이 전송 계층에서만 확립되고 위임 홉을 따라 전파되지 않아 confused deputy 상황이 생긴다고 지적합니다. 각 홉을 개별 심사하면 각 홉은 정당한데, 체인 끝에서 최초 요청자가 가지지 않은 권한이 행사됩니다.

<strong>누적 비용과 데이터 범위</strong>가 네 번째입니다. 조회 한 건은 언제나 허용 범위 안에 있지만, 조회 만 건은 데이터베이스 전체를 반출하는 것과 다름없습니다. API 호출 한 건의 요금은 무시할 수준이지만, 재시도 루프에 빠진 에이전트는 예산을 소진합니다. 논문은 후자를 multi-agent 환경의 실패 양상으로 denial-of-wallet이라고 부릅니다. 두 경우 모두 임계값이 개별 행동에 있지 않고 합계에 있습니다.

## 여러 행동의 조합을 정책으로 제한하기

위의 네 가지를 실제 운영 규칙으로 옮기면 대체로 다음 형태가 됩니다. 아래는 특정 정책 언어의 문법이 아니라 규칙의 구조를 보여 주는 의사 코드입니다.

```text
# 1. 읽기 검증을 통과한 대상만 쓰기 허용
allow  write(resource=R)
  only if  earlier in session: read(resource=R) succeeded

# 2. 민감 데이터가 세션에 유입된 뒤에는 외부 전송 금지
deny   send_external(*)
  if    earlier in session: read(classification="sensitive")

# 3. 임계값을 넘는 행동은 사람의 승인 이후에만 허용
allow  transfer(amount=A)
  only if  A <= 1_000_000
       or  earlier in session: human_approval(amount >= A) within 1h

# 4a. 같은 도구의 반복 호출 상한
deny   query_customer_db(*)
  if    count(session, query_customer_db) > 200

# 4b. 위임 깊이와 권한 확대 차단
deny   delegate(to=B)
  if    depth(session) >= 3
       or  scope(B) not subset_of scope(current_principal)
```

네 규칙 모두 왼쪽 조건에는 현재 요청이 들어가고 오른쪽 조건에는 세션 이력이 들어갑니다. 이력을 참조하지 않으면 어느 규칙도 표현할 수 없습니다.

여기서 실무적으로 중요한 선택이 하나 생깁니다. 이 이력을 어디에서 관리할 것인지입니다. 에이전트 코드 안에서 직접 추적하면 구현은 빨라지지만, 통제가 에이전트와 같은 신뢰 경계 안에 놓입니다. 프롬프트 인젝션으로 에이전트의 판단이 흔들리면 그 판단에 얹힌 통제도 함께 흔들립니다. 논문이 지적한 "데이터와 지시 사항의 분리가 신뢰할 만한 수준으로 되어 있지 않다"는 문제가 여기에 그대로 적용됩니다. 그래서 이력 기록과 판정은 에이전트 코드 밖의 강제 지점에 두는 편이 낫습니다.

## AgentCore Policy 문서에 나타난 구현 형태

Amazon Bedrock AgentCore의 Policy가 위 문제를 다루는 실물 사례 하나입니다. 이것을 앞선 논의의 답으로 제시하려는 것은 아닙니다. 논문이 연구 방향으로 남겨 둔 문제를 정책 언어로 어떻게 표현하는지, 그리고 그 표현에 어떤 제약이 붙는지 확인하기 위한 예시입니다. 같은 문제에 대한 다른 접근도 존재하고, 아래 내용은 공식 문서가 명시한 범위까지입니다.

기본 구조는 요청 한 건 단위 심사입니다. [AWS 공식 블로그](https://aws.amazon.com/blogs/machine-learning/secure-ai-agents-with-policy-in-amazon-bedrock-agentcore/)가 2026년 3월에 설명한 내용에 따르면, AgentCore Gateway가 에이전트에서 도구로 향하는 모든 요청을 런타임에 가로채 정책 엔진에 넘기고, 정책은 [Cedar](https://www.cedarpolicy.com/en) 언어로 작성됩니다. 판정은 default-deny이고 forbid가 permit을 이깁니다. 정책을 자연어로 기술해 Cedar로 변환하는 경로와 Cedar를 직접 작성하는 경로가 함께 제공됩니다. 이 시점의 블로그에는 궤적이나 이력을 참조하는 판정에 대한 언급이 없습니다. 시간과 관련된 조건으로는 `context.time.hour` 같은 현재 시각 조건만 나옵니다. 이 조건은 현재 요청 하나에 포함된 컨텍스트만 참조합니다.

이력을 참조하는 부분은 [temporal policy](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-temporal.html)라는 이름으로 개발자 문서에 따로 정리되어 있습니다. 문서의 정의는 "현재 요청만이 아니라 세션 안에서 에이전트가 수행한 행동의 이력에 따라 판정이 달라지는 정책"입니다. 문법은 [Dogwood](https://dogwood-policy.github.io/dogwood/index.html)로 쓰는데, Dogwood는 Cedar 위에 만들어진 상위 집합이어서 유효한 Cedar 정책은 모두 유효한 Dogwood 정책이라고 문서가 밝힙니다. 기존 정책을 옮겨 쓸 필요는 없고, 이력을 봐야 하는 규칙에만 temporal 조건을 더하면 됩니다. 문서에 실린 예시는 다음과 같습니다.

```text
permit ( principal, action == AgentCore::Action::"SellShares", resource )
when temporal {
    formerly within 1h AgentCore::Action::"ApproveSale"::response{
        eventResource:   resource,
        input.stock:     context.input.stock,
        input.shares:    context.input.shares,
        output.approved: true
    }
};
```

앞 절의 세 번째 규칙과 구조가 같습니다. 매도를 허용하는 조건이 매도 요청 자체에 있지 않고, 같은 세션에서 한 시간 안에 승인 이벤트가 있었고 그 승인의 종목과 수량이 현재 요청과 일치하며 승인 결과가 참이었다는 이력에 있습니다. 문서가 나열하는 temporal 연산자는 이력에 일치하는 이벤트가 있었는지 보는 `formerly within`, 기준 이벤트 이후로 조건이 유지되었는지 보는 `since within`, 그리고 구간 내 집계인 `count`와 `sum`입니다. 앞 절의 반복 호출 상한과 누적 비용 상한이 뒤의 두 연산자에 대응합니다.

이력의 범위는 policy session입니다. 세션 ID는 호출자가 생성해서 `x-amzn-bedrock-agentcore-policy-session-id` 헤더로 매 요청에 실어 보내야 하고, Gateway가 대신 만들어 주지 않습니다. 헤더를 빼면 세션이 성립하지 않으며, 엔진에 temporal policy가 하나라도 있으면 세션 ID 없는 요청은 검증 오류로 실패합니다.

문서가 함께 명시한 제약이 여러 개 있는데, 설계 단계에서 미리 알아야 하는 것들입니다.

| 항목 | 문서에 기재된 내용 |
|---|---|
| 정책 개수 | 정책 엔진당 temporal policy 25개 |
| 연산자 개수 | 정책 하나당 temporal 연산자 3개 |
| 시간 구간 | temporal 조건 하나당 최대 24시간 |
| 계정과 리전 | 세션이 계정 간, 리전 간으로 전파되지 않아 Gateway와 모든 타깃이 같은 계정, 같은 리전에 있어야 함 |
| IAM | Gateway 역할에 `bedrock-agentcore:GetWorkloadAccessToken` 권한 필요 |
| 리전 | 서울을 포함한 다수 리전에서 사용 가능하고, 문서의 표에 제외된 리전도 함께 표시됨 |

동작 방식에서 오해하기 쉬운 지점도 문서가 짚어 둡니다. 판정 대상과 같은 행동을 참조하는 조건에서는 현재 요청의 이벤트도 집계에 포함됩니다. 그리고 이력에는 허용된 뒤 완료된 행동이 `response` 이벤트로, 정책이 거부한 행동이 `error` 이벤트로 기록되므로, `response`를 참조하는 조건은 거부된 선행 행동과 일치하지 않습니다. 선행 행동이 별도 정책으로 허용되어 있지 않으면 그 행동에 의존하는 규칙은 영원히 성립하지 않습니다.

보안 관점에서 문서가 직접 언급한 한계가 하나 더 있습니다. `count` 기반 상한은 세션 내부에서만 유효합니다. 세션 ID를 호출자가 공급하기 때문에, 새 세션을 시작하면 집계가 다시 0에서 출발합니다. 세션당 호출 200회 제한은 호출자 전체에 대한 유량 제어가 되지 못합니다. 세션 경계를 넘는 제한이 필요하면 다른 수단을 함께 두어야 합니다. 이 지점은 앞서 정리한 네 가지 중 누적 비용 항목이 세션 단위 정책만으로는 완결되지 않는다는 뜻이기도 합니다.

## MCP와 A2A가 검사 범위를 넓히는 이유

궤적 단위 통제가 필요해진 배경에는 도구와 에이전트를 연결하는 표준이 자리 잡았다는 사실이 있습니다. 논문은 MCP를 에이전트와 도구를 잇는 수직축으로, A2A를 에이전트와 에이전트를 잇는 수평축으로 놓고, A2A가 150개 이상의 조직이 채택한 지배적 표준이 되었다고 서술합니다.

두 축이 붙으면서 신뢰 경계의 수가 늘어납니다. 논문이 MCP 쪽에서 지적하는 것은 도구 정의의 무결성입니다. 악성 메타데이터를 심는 tool poisoning과, 이전까지 신뢰받던 서버가 정상 도구 정의를 조용히 교체하는 rug pull이 여기에 해당합니다. 공급망 쪽 예시로는 postmark-mcp 패키지가 열다섯 개 릴리스 동안 정상 동작한 뒤 데이터를 유출한 사례를 듭니다. 저자들은 이 위험이 코드 패키지에 그치지 않고 `SKILL.md` 같은 지시 사항 산출물까지 확장된다고 봅니다.

A2A 쪽에서는 세 가지를 짚습니다. 공유 컨텍스트 식별자에 소유권 의미가 없어 권한 없는 클라이언트가 대화에 붙을 수 있고, Agent Card의 능력 기술이 증명 없이 자기 주장으로 제출되며, 위임 권한이 홉을 따라 전파되지 않습니다. 프로토콜 명세가 보안 통제를 강제 요구가 아닌 권고 수준으로 두었다는 점도 함께 지적하면서, 저자들은 A2A와 MCP의 경계를 하나의 attack surface로 다루자고 제안합니다.

이 구조가 궤적 통제와 만나는 지점은 분명합니다. 위임이 여러 홉을 지나면 판정에 필요한 이력이 여러 시스템에 흩어집니다. AgentCore 문서가 계정과 리전을 넘는 세션 전파를 지원하지 않는다고 못 박고, 직접 운영하는 API 게이트웨이나 쿠버네티스 클러스터가 체인에 끼면 Workload Access Token을 전달하는 로직을 직접 넣어야 한다고 안내하는 것도 같은 문제입니다. 이력을 이어 붙일 수 없는 구간에서는 궤적 판정이 성립하지 않습니다.

## 관측, 중단, 감사 기록

정책만으로는 운영이 되지 않습니다. 궤적 단위 규칙은 요청 한 건 단위 규칙보다 오탐과 미탐의 양상이 복잡하기 때문에, 판정을 관측하고 필요할 때 멈추고 사후에 재구성할 수단이 함께 있어야 합니다.

관측에서 먼저 필요한 것은 강제하지 않고 관찰하는 단계입니다. AgentCore Policy에는 `LOG_ONLY` 모드가 있어서, 정책을 `ENFORCE`로 올리기 전에 그 정책이 무엇을 거부할지 먼저 볼 수 있습니다. 궤적 규칙은 정상 업무 흐름을 잘못 막을 여지가 요청 단위 규칙보다 큽니다. 앞의 두 번째 규칙을 그대로 켜면 민감 문서를 읽은 뒤 정당한 사내 보고를 보내는 흐름까지 함께 막힐 수 있습니다. 관찰 모드에서 실제 트래픽에 대고 며칠 돌려 보는 절차가 필수에 가깝습니다.

지표 쪽에서 문서가 제시하는 신호는 temporal 평가에 걸린 시간을 밀리초로 내보내는 `TemporalLatency` 지표와, 요청별 span 속성입니다. span 속성에는 temporal 평가가 실행되었는지를 나타내는 `aws.agentcore.policy.temporal.evaluation_invoked`와, 평가기가 이벤트 순서를 정할 때 사용한 나노초 단위 타임스탬프인 `aws.agentcore.policy.temporal.event_timestamp_ns`가 포함됩니다. span 데이터는 Gateway 리소스에 트레이스를 켠 뒤 CloudWatch의 `aws/spans` 로그 그룹에서 볼 수 있습니다. 여기서 문서가 덧붙인 주의 사항이 실무적으로 중요합니다. `evaluation_invoked`는 temporal 평가가 실행되었다는 사실만 알려 주고, temporal policy가 일치했거나 판정을 결정했다는 뜻은 아닙니다. 이 값을 정책 적중률로 읽으면 통제가 실제로 작동한다고 잘못 믿게 됩니다.

즉시 중단 수단은 별도로 설계해야 합니다. AgentCore 문서에 kill switch라는 기능은 없습니다. 대신 관련된 동작이 하나 기록되어 있습니다. 엔진의 temporal policy를 추가하거나 변경하면 그 엔진의 활성 temporal 세션이 무효화되고, 무효화된 세션을 재사용하는 다음 요청은 HTTP 409 `ConflictException`으로 실패합니다. 이력이 정책 변경 이전 기준으로 쌓여 있어 새 규칙과 맞지 않는 상태에서 판정하는 것을 막으려는 설계입니다. 이 동작 덕분에 정책 갱신이 진행 중인 세션을 끊는 효과를 내지만, 이것을 긴급 중단 수단으로 쓰기로 한다면 사용자 요청이 409로 떨어지는 구간을 애플리케이션이 어떻게 처리할지 미리 정해 두어야 합니다. 새 세션을 열어 재시도하는 경로가 없으면 장애로 보입니다.

감사 기록에서는 논문이 다소 낙관적인 관측을 하나 제시합니다. 에이전트의 불투명성 때문에 사후 재구성이 어렵다는 점을 인정하면서도, 에이전트가 구조화되고 검증 가능한 실행 추적을 생성할 수 있으므로 잘 설계하면 사람이 수행하는 절차보다 투명성이 높아질 수 있다고 봅니다. 저자들은 모델과 동적으로 발견되는 도구에까지 software bill of materials 원칙을 확장하자고 제안합니다. 이 관측은 검증된 결과 없이 연구 방향으로 제시된 것이고, 실제로 그 수준의 추적을 얻으려면 어떤 이벤트를 어떤 식별자로 묶어 남길지를 처음부터 정해야 합니다. 궤적 판정을 도입하기로 했다면 정책이 참조하는 이벤트와 감사 로그가 남기는 이벤트를 같은 스키마로 맞춰 두는 편이 유리합니다. 정책이 본 이력과 사후에 읽는 기록이 다르면 왜 그 판정이 났는지 재구성할 수 없습니다.

## 정리

논문 한 편이 제기한 문제는 간단합니다. 개별 행동을 심사하는 통제는 그 행동들이 쌓여서 만드는 결과를 심사하지 못합니다. 이 문제에 대한 검증된 해법은 아직 없고, 해당 논문도 해법을 제시하지 않습니다. 열한 개의 연구 방향과 하나의 관점을 제시한 문서입니다.

실무에서 지금 할 수 있는 일은 그보다 좁습니다. 운영 중인 에이전트에서 개별 호출은 모두 허용 범위인데 순서가 문제가 되는 조합을 찾아 목록으로 적어 보는 작업이 출발점입니다. 읽기 검증 없는 쓰기, 민감 데이터 유입 후 외부 전송, 승인 없는 임계값 초과, 위임 체인에서의 권한 확대가 대체로 그 목록의 앞자리를 차지합니다. 그다음에 그 조합들이 현재 통제 구조에서 어느 지점에서 걸리는지 확인하면, 요청 한 건 단위 심사만으로는 걸리지 않는 항목이 남습니다.

그 남은 항목을 세션 이력을 참조하는 정책으로 옮길 수 있는지는 사용하는 플랫폼에 따라 다릅니다. AgentCore의 temporal policy는 그 표현이 가능한 예시 하나이고, 대신 세션 단위 범위, 최대 24시간 구간, 계정과 리전 경계, 호출자가 세션 ID를 공급하는 구조 같은 제약을 함께 받습니다. 세션 경계를 넘는 유량 제어처럼 이 제약 밖에 있는 요구는 다른 수단으로 채워야 합니다. 어느 경로를 택하든 통제를 에이전트 코드 밖에 두는 선택은 그대로 유지하는 편이 낫습니다. 에이전트의 판단이 조작될 수 있다는 전제가 이 논의의 출발점이기 때문입니다.

## References

- Lotfi, A., Karmaker Shanto, S., Karim, I., Bertino, E. (2026). *Securing Agentic AI: From Per-Action Checks to Trajectory Assurance*. arXiv:2608.01558. Accepted to the ACM AI Leadership Summit 2026 (Visionary Track). https://arxiv.org/abs/2608.01558
- Lotfi, A. et al. (2026). *Securing Agentic AI* (HTML full text). https://arxiv.org/html/2608.01558v1
- Srinivasan, B., Nadiminti, A., Dua, P. (2026-03-12). *Secure AI agents with Policy in Amazon Bedrock AgentCore*. AWS Machine Learning Blog. https://aws.amazon.com/blogs/machine-learning/secure-ai-agents-with-policy-in-amazon-bedrock-agentcore/
- AWS. *Policy in Amazon Bedrock AgentCore: Control Agent Interactions*. Amazon Bedrock AgentCore Developer Guide. https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy.html
- AWS. *Temporal policies*. Amazon Bedrock AgentCore Developer Guide. https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-temporal.html
- Dogwood Policy. *The Dogwood policy language*. https://dogwood-policy.github.io/dogwood/index.html
- Cedar Policy. *Cedar policy language*. https://www.cedarpolicy.com/en
