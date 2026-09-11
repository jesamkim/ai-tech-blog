---
title: "에이전트 보안은 한 번의 Tool Call로 끝나지 않는다: Per-Action Check에서 Trajectory Assurance로"
date: 2026-08-23T09:00:00+09:00
lastmod: 2026-09-11T17:15:00+09:00
draft: false
categories: ["논문 리뷰", "AI 에이전트"]
tags: ["Agentic AI", "AI Security", "Trajectory Assurance", "Bedrock AgentCore", "Cedar", "MCP", "A2A"]
author: "Jesam Kim"
cover:
  image: "/ai-tech-blog/images/trajectory-assurance-agent-security/cover.png"
  alt: "에이전트의 여러 실행 궤적이 정책 검증 지점으로 모이는 장면"
  relative: false
description: "개별 tool call이 모두 허용 범위 안에 있어도 그 순서가 쌓이면 규칙을 위반할 수 있습니다. 2026년 8월 공개된 vision paper 한 편을 근거로 현재 요청만 검사하는 통제의 한계를 정리하고, 세션 이력을 보는 정책을 실제로 어떻게 표현하는지 Amazon Bedrock AgentCore의 temporal policy 문서로 확인합니다."
---

> 2026-09-11 정정: A2A와 MCP의 필수 보안 요구사항을 반영하고, 기존 검증 연구와 이력 기반 통제의 적용 범위를 명확히 했습니다.

사내 문서를 검색하고 요약해서 메일로 보내는 에이전트를 하나 상상해 보겠습니다. 이 에이전트에는 세 가지 도구가 붙어 있습니다. 문서 검색, 문서 본문 읽기, 메일 발송입니다. 세 도구는 각각 정당한 권한을 받았고, 호출 한 건씩 떼어 놓고 보면 어느 것도 규칙을 어기지 않습니다. 검색은 읽기 전용이고, 본문 읽기도 사용자가 접근 권한을 가진 문서만 반환하며, 메일 발송은 사용자가 명시적으로 요청한 기능입니다.

그런데 이 세 호출이 다음 순서로 일어나면 상황이 달라집니다. 에이전트가 인사 평가 문서를 검색하고, 그 본문을 읽어 컨텍스트에 담고, 곧바로 외부 도메인 주소로 요약 메일을 발송합니다. 게이트가 발송 권한만 확인하고 자료의 민감도나 앞선 조회 이력을 확인하지 않는다면 이 흐름을 통과시킬 수 있습니다. 도구를 사용할 권한과 그 도구로 특정 정보를 외부에 보낼 권한을 구분해야 하는 예시입니다.

2026년 8월에 arXiv에 올라온 [Securing Agentic AI: From Per-Action Checks to Trajectory Assurance](https://arxiv.org/abs/2608.01558)가 정확히 이 지점을 다룹니다. 논문의 표현으로는 "개별적으로 허용되는 행동의 연쇄가 종합적으로는 시스템 수준의 제약과 안전 불변식을 위반할 수 있다"는 문제입니다. 다만 이 논문의 성격을 먼저 분명히 해 둘 필요가 있습니다. Purdue University의 Alireza Lotfi, Subangkar Karmaker Shanto, Elisa Bertino와 University of Texas at Dallas의 Imtiaz Karim이 쓴 6페이지 분량의 문서이고, ACM AI Leadership Summit 2026의 Visionary Track에 채택되었습니다. 저자들이 본문에서 "vision paper로서 우리의 목표는 완전한 해법을 제안하는 것이 아니라 핵심 문제와 그로부터 생기는 연구 기회를 식별하는 것"이라고 직접 밝힙니다. 실험 결과나 벤치마크로 검증한 방법론은 여기에 없고, 저자들이 열한 개로 정리한 연구 방향이 본문의 골격입니다. 이 글에서 논문을 인용하는 부분은 모두 그 전제 위에서 읽어야 합니다.

이 글은 논문이 제기한 연구 과제와, 그중 일부를 실제 정책으로 표현하는 방법을 구분해서 다룹니다. 특정 시간 순서 제약을 구현하고 평가한 연구는 이미 있습니다. 여기서 살펴보는 제품 기능도 명시한 조건을 강제하는 수단이며, 여러 에이전트와 시스템에 걸친 모든 업무 규칙을 포괄적으로 보장한다는 뜻은 아닙니다.

## 현재 요청만 검사할 때 놓치는 것

게이트웨이나 프록시는 tool call을 가로채고, 호출 주체의 신원과 대상 도구, 입력 파라미터를 정책과 맞춰 허용 여부를 판단합니다. 여기서 <strong>호출마다 검사한다는 것과 이력을 참조하지 않는다는 것은 다릅니다.</strong> Per-action은 검사 시점을, stateless는 판정기가 이전 요청의 상태를 보관하지 않는 특성을 가리킵니다. 호출마다 검사하더라도 신뢰할 수 있는 저장소의 이력이나 누적 상태를 참조할 수 있습니다. 판정기 자체가 stateless여도 호출자가 검증된 상태를 현재 요청의 컨텍스트로 전달할 수 있습니다.

문제는 현재 호출의 주체, 도구, 인자만 보고 관련 상태를 판단에서 빠뜨릴 때 생깁니다. 승인 후 실행해야 한다는 순서 제약이나 여러 호출의 합계에 적용되는 예산 제한은 앞선 사건 또는 그 결과를 나타내는 신뢰할 수 있는 상태가 필요합니다. 반면 외부 전송을 항상 금지하는 규칙처럼 현재 요청만으로도 특정 위험을 막을 수 있는 통제도 있습니다. 어떤 정보가 필요한지는 강제하려는 규칙에 따라 달라집니다.

논문은 이 문제를 <strong>behavioral containment</strong>라는 이름으로 다루면서, 실행환경의 격리와 행동 규칙의 강제를 구분합니다. 샌드박스, 자격 증명의 범위 제한, 송신 경로 통제는 기존 보안 기법으로 대응할 수 있는 영역입니다. 저자들은 5G 스케줄러가 공공안전 서비스의 우선순위를 조금씩 낮춰 복구 요구를 위반하는 상황을 예로 듭니다. 또 다른 예시는 임상 에이전트가 퇴원을 결정하면서 진료 경로에 정해 둔 관찰 시간을 위반하는 상황입니다. 이는 실제 사고나 실험 결과를 보고한 내용이 아니라 가상의 시나리오입니다. 임상 예시의 관찰 시간도 FHIR 표준 자체가 일괄적으로 정한 의무가 아니라, FHIR로 표현한 진료 경로에 특정 요구가 있다고 가정한 것입니다. 저자들은 이런 위반이 적대적 입력 없이도 발생할 수 있다는 점을 강조합니다.

논문은 런타임 가드, 규칙을 도출하는 컴파일러, 확률적 모니터, 시간 순서 제약을 강제하는 연구를 살펴봅니다. 저자들의 비판은 실제 준수해야 하는 규정에서 제약을 빠짐없이 도출하고 여러 에이전트에 걸쳐 강제하는 작업이 충분하지 않다는 것입니다. 그렇다고 기존 방법이 명시된 제약을 강제하지 못하거나, 처음 보는 위반을 모두 놓친다는 뜻은 아닙니다. [Agent-C](https://arxiv.org/abs/2512.23738)는 시간 순서 제약을 형식적으로 표현하고 위반하는 도구 호출의 생성을 막는 방법을 구현해 소매 고객 서비스와 항공권 예약 과제에서 평가했습니다. [Hong 등의 연구](https://arxiv.org/abs/2604.15579)도 symbolic guardrail을 구현하고 평가했습니다. 이 결과는 각 연구가 정의하고 평가한 요구사항에 대한 근거이며, 모든 업무 규칙의 포괄적인 안전 보장으로 확대할 수는 없습니다.

## Trajectory assurance가 판단에 넣어야 하는 것

논문은 개별 행동의 허용 여부에 더해 전체 실행 과정이 시스템의 규칙을 지켰는지도 검증해야 한다고 제안합니다. 이 vision paper 자체가 새로운 알고리즘이나 실험 결과를 제시하는 것은 아닙니다. 아래는 그 문제 제기를 구현 관점에서 살펴보기 위해 정리한 항목입니다. 이 네 가지 분류는 논문의 공식 분류가 아닙니다.

<strong>상태</strong>가 첫 번째입니다. 같은 도구를 같은 파라미터로 호출해도 세션에 어떤 자료가 들어왔는지에 따라 허용 여부가 달라질 수 있습니다. 앞의 예시에서는 발송 권한뿐 아니라 읽은 문서의 민감도를 함께 판단해야 합니다. 그 정보가 현재 인자나 신뢰할 수 있는 조회 결과에 포함되지 않으면 발송 권한 검사만으로는 자료의 외부 반출 가능 여부를 판단할 수 없습니다.

<strong>순서</strong>가 두 번째입니다. 송금 전에 승인받고, 삭제 전에 백업을 확인하고, 외부 전송 전에 검토를 마쳐야 하는 업무가 여기에 해당합니다. 각 도구를 사용할 권한만 확인하면 필요한 선행 단계가 빠진 실행을 놓칠 수 있습니다. 승인 이력이나 유효한 승인 증명처럼 선행 조건의 충족 여부를 확인할 정보가 필요합니다.

<strong>위임되는 권한</strong>이 세 번째입니다. 에이전트가 다른 에이전트에 작업을 맡길 때, 상대가 가진 권한과 최초 사용자가 허용한 범위를 혼동하면 의도하지 않은 작업이 실행될 수 있습니다. A2A는 각 요청의 인증과 인가를 요구하지만, 프로토콜을 연결하는 것만으로 최초 사용자의 신원과 전체 위임 이력이 모든 시스템에 자동으로 보존되는 것은 아닙니다. 직접 호출한 상대의 신원만 전달되는 구현이라면, 재위임된 작업에서 최초 주체와 허용 범위를 판단할 정보가 부족해집니다. 위임 경로에서 어떤 신원과 권한 정보를 유지할지 별도로 설계해야 합니다.

<strong>누적 비용과 데이터 범위</strong>가 네 번째입니다. 개별 조회가 허용 범위 안에 있어도 반복 조회로 수집한 데이터의 합계는 허용량을 넘을 수 있습니다. 호출당 비용이 작아도 반복 실행의 총비용은 예산을 초과할 수 있습니다. 단일 에이전트도 반복 실행으로 비용을 소진할 수 있고, 여러 에이전트가 협업하면 서로의 오류와 재시도를 증폭해 공유 예산이나 rate limit을 소진할 수 있습니다. 이때는 개별 호출의 권한과 함께 누적량을 제한해야 합니다.

## 여러 행동의 조합을 정책으로 제한하기

위의 네 가지를 실제 운영 규칙으로 옮기면 대체로 다음 형태가 됩니다. 아래는 특정 정책 언어의 문법이 아니라 규칙의 구조를 보여 주는 의사 코드입니다.

```text
# 1. 쓰기 전에 같은 대상을 조회했는지 확인
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

1~3과 4a는 선행 작업이나 누적 횟수를 확인하는 예시입니다. 4b의 위임 깊이와 권한 범위는 검증된 위임 정보가 현재 요청에 있다면 그 정보만으로도 평가할 수 있습니다. 따라서 모든 규칙에 전체 실행 이력이 반드시 필요한 것은 아닙니다.

위의 `allow`는 기존 사용자 인가를 통과한 요청에 추가하는 조건입니다. 조회 성공이 쓰기 권한을 부여하지는 않습니다. 송금 승인 예시도 금액과 시간만 표시한 개략적인 구조입니다. 실제로는 거래 식별자, 수취인, 금액, 승인 주체와 만료 시각을 요청에 결부하고, 중복 사용을 막기 위한 승인 소비와 실행의 원자성을 설계해야 합니다. 승인 결과와 자료의 민감도 역시 모델이 임의로 선언한 값이 아닌 신뢰할 수 있는 시스템의 증거를 사용해야 합니다.

이력 관리와 인가는 <strong>모델이 규칙이나 상태를 변경하거나 우회할 수 없는 통제 지점</strong>에 두어야 합니다. 코드가 에이전트 애플리케이션 안에 있다는 사실만으로 프롬프트 인젝션이 고정된 검증 로직을 바꿀 수 있는 것은 아닙니다. 모델로부터 보호된 harness 구성 요소도 규칙을 강제할 수 있습니다. 반대로 외부 게이트웨이를 두더라도 에이전트가 직접 도구를 호출할 자격 증명을 갖고 있으면 우회 경로가 남습니다. 구현 위치와 함께 정책 변경 권한, 이력의 신뢰성, 모든 실행이 검사를 거치도록 하는 구조를 확인해야 합니다.

## AgentCore Policy 문서에 나타난 구현 형태

Amazon Bedrock AgentCore Policy는 여러 도구 호출에 걸친 조건을 표현하고 강제하는 수단 중 하나입니다. 여기서는 공식 문서에 명시된 기능과 제약을 살펴봅니다. 이 제품 기능을 앞서 제기한 모든 연구 과제의 해결책으로 해석하지는 않습니다.

기본 구조는 요청 한 건 단위 심사입니다. [AWS 공식 개발자 문서](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy.html)에 따르면, AgentCore Gateway를 통과하는 에이전트 요청은 정책 엔진에서 평가된 뒤 도구 접근이 허용됩니다. 정책은 [Cedar](https://www.cedarpolicy.com/)를 기반으로 작성하며, `permit`과 `forbid` 규칙을 사용합니다. 기본값은 거부이고, `forbid`가 적용되면 `permit`보다 우선합니다. 자연어로 정책을 작성하는 경로와 Cedar 또는 Dogwood로 직접 작성하는 경로도 제공됩니다. 시간과 관련된 조건으로는 업무 시간대를 제한하는 `context.system.now`(UTC 기준 datetime 값) 조건이 있습니다. 이 조건은 현재 요청 하나에 포함된 컨텍스트만 참조합니다.

이력을 참조하는 부분은 [temporal policy](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-temporal.html)라는 이름으로 개발자 문서에 따로 정리되어 있습니다. 문서의 정의는 "현재 요청만이 아니라 세션 안에서 에이전트가 수행한 행동의 이력에 따라 판정이 달라지는 정책"입니다. 문법은 [Dogwood](https://dogwood-policy.github.io/dogwood/index.html)로 쓰는데, Dogwood는 Cedar 위에 만들어진 오픈소스 정책 언어로 Cedar와 같은 인가 모델을 쓰며, 유효한 Cedar 정책은 모두 유효한 Dogwood 정책이라고 문서가 밝힙니다. 기존 정책을 옮겨 쓸 필요는 없고, 이력을 봐야 하는 규칙에만 temporal 조건을 더하면 됩니다. 문서에 실린 예시는 다음과 같습니다.

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

앞 절의 세 번째 규칙처럼 승인 이력을 현재 요청과 연결하는 구조입니다. 같은 세션에서 한 시간 안에 승인 이벤트가 있었고, 승인의 종목과 수량이 현재 요청과 일치하며 승인 결과가 참이어야 합니다. 이 예시는 과거 승인의 존재를 확인하며 그 승인을 한 번만 사용할 수 있도록 소비하는 전체 거래 절차까지 정의하지는 않습니다. 문서가 나열하는 temporal 연산자는 이력의 일치 이벤트를 찾는 `formerly within`, 기준 이벤트 이후 조건이 유지됐는지 보는 `since within`, 구간 내 집계인 `count`와 `sum`입니다. 반복 호출이나 누적 금액을 제한하는 규칙을 이런 연산자로 표현할 수 있습니다.

이력의 범위는 policy session입니다. 세션 ID는 호출자가 생성해서 `x-amzn-bedrock-agentcore-policy-session-id` 헤더로 매 요청에 실어 보내야 하고, Gateway가 대신 만들어 주지 않습니다. 헤더를 빼면 세션이 성립하지 않으며, 엔진에 temporal policy가 하나라도 있으면 세션 ID 없는 요청은 검증 오류로 실패합니다.

문서가 함께 명시한 제약이 여러 개 있는데, 설계 단계에서 미리 알아야 하는 것들입니다.

| 항목 | 문서에 기재된 내용 |
|---|---|
| 정책 개수 | 정책 엔진당 temporal policy 25개 |
| 연산자 개수 | 정책 하나당 temporal 연산자 3개 |
| 시간 구간 | temporal 조건 하나당 최대 24시간 |
| 계정과 리전 | 세션이 계정 간, 리전 간으로 전파되지 않아 Gateway와 모든 타깃이 같은 계정, 같은 리전에 있어야 함 |
| IAM | Gateway 역할에 `bedrock-agentcore:GetWorkloadAccessToken` 권한 필요 |
| 리전 | 문서의 표에 서울, 도쿄, 오레곤, 버지니아 북부 등 16개 리전이 지원으로, 태국, 밀라노, 말레이시아가 미지원으로 표시됨 |

동작 방식에서 오해하기 쉬운 지점도 문서가 짚어 둡니다. 판정 대상과 같은 행동을 참조하는 조건에서는 현재 요청의 이벤트도 집계에 포함됩니다. 도구가 성공적으로 반환한 결과는 `response` 이벤트로, 정책의 거부나 도구의 오류는 `error` 이벤트로 기록됩니다. 따라서 `response`를 참조하는 조건은 거부된 선행 행동이나 오류 결과와 일치하지 않습니다. 선행 행동의 응답에 의존하는 요청은 그 응답을 받은 뒤 실행해야 합니다. 허용된 호출도 업무상 승인 여부를 `output.approved` 같은 필드로 반환할 수 있으므로, 호출 성공과 업무상 승인을 구분해야 합니다.

보안 관점에서 문서가 직접 언급한 한계가 하나 더 있습니다. `count` 기반 상한은 policy session 내부에서만 유효합니다. 세션 ID를 호출자가 공급하기 때문에 새 policy session을 시작하면 집계가 다시 0에서 출발합니다. 대화마다 새 ID를 발급하는 설계에서는 여러 대화를 합친 총량이 제한되지 않습니다. 애플리케이션이 사용자 단위의 세션을 유지해 대화들을 묶는 설계도 가능하지만, 그 경우에도 세션 만료와 시간 구간의 제약을 받습니다. 세션 ID의 생성과 변경을 누가 통제하는지 확인하고, policy session 경계를 넘는 제한은 별도 상태 저장과 제한 수단으로 보완해야 합니다. 인증된 Gateway는 세션을 호출자 신원과 결부하므로 같은 ID를 보낸 다른 호출자의 이력은 분리됩니다. `authorizerType=NONE`에서는 이런 호출자별 분리가 제공되지 않습니다.

## MCP와 A2A가 검사 범위를 넓히는 이유

도구와 에이전트를 연결하는 표준이 확산되면서 한 작업이 여러 시스템을 통과할 수 있게 됐습니다. 논문은 MCP를 에이전트와 도구의 연결로, A2A를 에이전트 사이의 연결로 구분합니다. A2A가 2025년에 Linux Foundation에 기여됐고 IBM의 Agent Communication Protocol도 A2A에 합류했다는 내용은 각 기관의 자료로 확인할 수 있습니다. Linux Foundation은 2026년 4월 9일 150개 이상의 조직이 A2A를 지원한다고 발표했습니다. 이는 재단의 자체 발표이며, 150개 조직이 모두 운영 환경에 배포했다는 뜻은 아닙니다.

이런 연결에서는 도구 정의와 공급망의 무결성도 확인해야 합니다. 논문은 악성 메타데이터를 삽입하는 tool poisoning과, 신뢰하던 도구 정의가 이후 바뀌는 rug pull을 지적합니다. Microsoft가 2026년 6월 공개한 글에는 2025년 MCP 관련 소프트웨어에 CVE 99건이 보고됐다는 서술이 있습니다. 이는 MCP 프로토콜 자체의 결함 99건이나 tool poisoning 사고 99건을 뜻하지 않으며, 이 글에서 CVE 목록을 독립적으로 재집계하지는 않았습니다.

공급망 사례는 Postmark 팀의 2025년 9월 25일 공지로 확인했습니다. Postmark와 무관한 비공식 `postmark-mcp` 패키지는 15개 버전을 거쳐 신뢰를 얻은 뒤, v1.0.16에서 그 패키지를 통해 발송하는 메일에 외부 BCC 수신자를 몰래 추가했습니다. 사용자의 전체 사서함을 복사했다는 뜻은 아닙니다. 기존에 인용한 Koi URL은 확인 시점에 다른 제품 페이지로 이동하므로, References에는 Postmark의 공식 공지를 사용했습니다. 논문은 이런 공급망 위험이 코드 패키지 외에도 편집기 규칙이나 `SKILL.md`처럼 에이전트가 지침으로 읽는 자료에까지 확장될 수 있다고 봅니다.

A2A에 관한 논문의 비판은 공개 명세의 요구사항과 구분해서 읽어야 합니다. [A2A v1.0.0 명세](https://a2a-protocol.org/v1.0.0/specification/)는 모든 작업에 인가를 적용하고 호출자의 권한 범위로 작업과 결과를 제한하도록 요구합니다. `contextId`는 상호작용을 묶는 식별자이며 그 자체로 접근 권한을 부여하지 않습니다. 접근 권한이 없는 사용자가 식별자만으로 이력에 접근할 수 있다면 구현의 인가 누락을 확인해야 합니다. 인증과 인가를 포함한 보안 요구사항 전체를 선택 사항이라고 설명하면 부정확합니다.

Agent Card에는 선택적인 JWS 서명을 붙일 수 있습니다. 서명 검증으로 카드의 출처와 무결성을 확인할 수 있지만, 카드가 주장하는 능력을 실제로 수행할 수 있는지까지 증명하지는 않습니다. 또한 A2A를 사용한다는 사실만으로 최초 사용자의 신원과 위임 이력이 모든 후속 시스템에 자동으로 보존되는 것은 아닙니다. 이런 범위를 구분해야 프로토콜의 필수 요구사항과 구현자가 추가로 설계할 통제를 혼동하지 않습니다.

관련 OAuth 규격도 제공하는 보장과 구현 조건이 서로 다릅니다. RFC 8693의 Token Exchange와 `act` claim은 위임 신원을 표현하는 구성 요소입니다. 이전 actor들의 기록이 각 단계마다 독립적으로 서명되고, 전체 사슬의 인가가 자동으로 강제된다는 뜻은 아닙니다. RFC 8707은 토큰이 사용될 자원을 지정하는 수단을, RFC 9449의 DPoP는 토큰 사용을 특정 키의 소유 증명에 결부하는 수단을 제공합니다. 특히 [MCP 2025-06-18 인가 프로파일](https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization)은 OAuth 인가를 구현할 때 resource 표시와 대상 audience 검증을 필수로 요구합니다. 인가 기능의 지원 여부가 선택적이라는 점과, 그 기능을 구현할 때 지켜야 하는 요구사항을 구분해야 합니다.

위임이 여러 홉을 지나면 판정에 필요한 이력이 여러 시스템에 흩어질 수 있습니다. AgentCore 문서도 temporal policy의 세션 전파를 같은 계정과 리전으로 제한하며, 직접 운영하는 중간 구성 요소에서는 Workload Access Token을 이어 전달할 로직이 필요하다고 설명합니다. 필요한 이력을 연결하지 못한 구간에서는 그 이력에 의존하는 조건을 검증할 수 없습니다. 이때는 불완전한 정보로 허용하기보다 필요한 증거를 확보하거나 작업 범위를 제한하도록 설계해야 합니다.

## 관측, 중단, 감사 기록

정책의 판정이 실제 업무에 맞는지 관측하고, 필요할 때 작업을 멈추고, 실행 뒤에 과정을 재구성할 수단이 함께 있어야 합니다. 이력을 사용하는 정책은 세션 범위와 선행 사건의 기록 방식에도 영향을 받으므로 정상 업무와 위반 사례를 함께 확인해야 합니다.

AgentCore Policy의 `LOG_ONLY` 모드에서는 정책이 어떤 요청을 거부할지 관찰한 뒤 `ENFORCE`로 전환할 수 있습니다. 예를 들어 앞의 두 번째 규칙은 민감한 문서를 읽은 세션에서 그 문서와 무관한 공개 자료를 외부에 보내는 정상 업무까지 막을 수 있습니다. 실제로 지켜야 하는 업무 규칙과 세션 범위를 비교하고, 대표적인 정상 및 위반 시나리오로 확인해야 합니다. `LOG_ONLY`는 거부 판정을 강제하지 않으므로 관찰 중에도 필요한 기존 보호 장치를 유지해야 합니다.

지표 쪽에서 문서가 제시하는 신호는 temporal 평가에 걸린 시간을 밀리초로 내보내는 `TemporalLatency` 지표와, 요청별 span 속성입니다. span 속성에는 temporal 평가가 실행되었는지를 나타내는 `aws.agentcore.policy.temporal.evaluation_invoked`와, 평가기가 이벤트 순서를 정할 때 사용한 나노초 단위 타임스탬프인 `aws.agentcore.policy.temporal.event_timestamp_ns`가 포함됩니다. span 데이터는 Gateway 리소스에 트레이스를 켠 뒤 CloudWatch의 `aws/spans` 로그 그룹에서 볼 수 있습니다. 여기서 문서가 덧붙인 주의 사항이 실무적으로 중요합니다. `evaluation_invoked`는 temporal 평가가 실행되었다는 사실만 알려 주고, temporal policy가 일치했거나 판정을 결정했다는 뜻은 아닙니다. 이 값을 정책 적중률로 읽으면 통제가 실제로 작동한다고 잘못 믿게 됩니다.

이미 실행 중인 작업을 취소하는 수단은 별도로 확인해야 합니다. [temporal policy 문서](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-temporal.html)에 따르면 정책을 추가하거나 변경하면 엔진의 활성 temporal session이 무효화되고, 그 세션을 재사용하는 다음 요청은 HTTP 409 `ConflictException`으로 실패합니다. 이는 <strong>다음 요청의 차단</strong>이며, 이미 허용돼 실행 중인 도구를 즉시 취소하거나 그 결과를 되돌린다는 뜻은 아닙니다. 정책 갱신 후 정상 업무를 재개하려면 새 세션에서 필요한 선행 조건을 다시 충족해야 합니다. 새 세션의 이력은 비어 있으므로, 과거에 읽은 민감한 자료나 승인 상태를 그대로 둔 채 ID만 바꿔 재시도하면 기존 이력에 근거한 통제를 잃을 수 있습니다.

감사 기록에서는 논문이 다소 낙관적인 관측을 하나 제시합니다. 에이전트의 추론이 불투명하고 행동이 여러 도구와 에이전트에 걸쳐 있어 사후 재구성이 어렵다는 점을 인정하면서도, 사람 운영자와 달리 자율 에이전트는 자신의 결정과 행동을 기록한 구조화되고 검증 가능한 실행 추적을 생성할 수 있으므로 그것이 대체하는 절차보다 오히려 투명하고 감사 가능해질 잠재력이 있다고 봅니다. 논문이 여기서 참조하는 기존 연구는 에이전트 식별자, 실시간 모니터링, 귀속과 포렌식을 지원하는 활동 로그를 요구하지만, 저자들은 그런 수단이 표준화되지도 의무화되지도 않았고 프라이버시와 비용의 트레이드오프를 동반한다고 덧붙입니다. 공급망 쪽에서는 모델과 프롬프트, 배포 시점에 동작이 고정되지 않는 동적으로 발견되는 도구에까지 software bill of materials 원칙을 확장하자고 제안합니다. 이 관측은 검증된 결과 없이 연구 방향으로 제시된 것이고, 실제로 그 수준의 추적을 얻으려면 어떤 이벤트를 어떤 식별자로 묶어 남길지를 처음부터 정해야 합니다. 궤적 판정을 도입하기로 했다면 정책이 참조하는 이벤트와 감사 로그가 남기는 이벤트를 같은 스키마로 맞춰 두는 편이 유리합니다. 정책이 본 이력과 사후에 읽는 기록이 다르면 왜 그 판정이 났는지 재구성할 수 없습니다.

## 정리

이 vision paper는 현재 요청만으로 판단할 수 없는 업무 규칙과, 여러 에이전트에 걸쳐 그 규칙을 검증하는 문제를 제기합니다. 특정 시간 순서 제약을 구현하고 평가한 연구는 이미 있습니다. 남아 있는 과제는 준수해야 하는 규정을 충분히 표현하고, 필요한 증거를 시스템 경계 너머로 연결하며, 그 범위에서 실제로 규칙이 지켜지는지 검증하는 것입니다.

실무에서는 운영 중인 에이전트의 작업 순서를 구체적으로 적는 것부터 시작할 수 있습니다. 선행 조회가 필요한 쓰기, 민감 자료를 읽은 뒤의 외부 전송, 승인이 필요한 금액의 거래, 권한 범위를 넘는 재위임을 살펴봅니다. 그다음 현재 통제가 어떤 증거를 사용해 각 조건을 확인하는지 대조합니다. 도구 호출마다 검사하더라도 필요한 상태와 이력이 빠져 있다면 그 정보를 보완해야 합니다.

AgentCore temporal policy는 세션 이력을 사용하는 규칙을 표현하는 수단 중 하나입니다. 세션 범위, 최대 24시간 구간, 계정과 리전 경계, 세션 ID 관리 방식의 제약을 함께 고려해야 합니다. 이 범위를 넘는 요구는 다른 통제와 상태 관리로 보완해야 합니다. 보호된 harness를 쓰든 외부 게이트웨이를 쓰든, 모델이 정책과 증거를 변경하거나 검사를 우회할 수 없도록 하는 조건은 같습니다.

## References

- Lotfi, A., Karmaker Shanto, S., Karim, I., Bertino, E. (2026). *Securing Agentic AI: From Per-Action Checks to Trajectory Assurance*. arXiv:2608.01558. Accepted to the ACM AI Leadership Summit 2026 (Visionary Track). https://arxiv.org/abs/2608.01558
- Lotfi, A. et al. (2026). *Securing Agentic AI* (HTML full text). https://arxiv.org/html/2608.01558v1
- Srinivasan, B., Nadiminti, A., Dua, P. (2026-03-12). *Secure AI agents with Policy in Amazon Bedrock AgentCore*. AWS Machine Learning Blog. https://aws.amazon.com/blogs/machine-learning/secure-ai-agents-with-policy-in-amazon-bedrock-agentcore/
- AWS. *Policy in Amazon Bedrock AgentCore: Control Agent Interactions*. Amazon Bedrock AgentCore Developer Guide. https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy.html
- AWS. *Temporal policies*. Amazon Bedrock AgentCore Developer Guide. https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-temporal.html
- Dogwood Policy. *The Dogwood policy language*. https://dogwood-policy.github.io/dogwood/index.html
- Cedar Policy. *Cedar policy language*. https://www.cedarpolicy.com/
- The Linux Foundation (2026-04-09). *A2A Protocol Surpasses 150 Organizations, Lands in Major Cloud Platforms, and Sees Enterprise Production Use in First Year*. Press release. https://www.linuxfoundation.org/press/a2a-protocol-surpasses-150-organizations-lands-in-major-cloud-platforms-and-sees-enterprise-production-use-in-first-year (재단이 발표한 지원 조직 수의 출처)
- Microsoft AI Red Team (2026-06-04). *Updating the taxonomy of failure modes in agentic AI systems: what a year of red teaming taught us*. Microsoft Security Blog. https://www.microsoft.com/en-us/security/blog/2026/06/04/updating-taxonomy-failure-modes-agentic-ai-systems-year-red-teaming-taught-us/ (논문이 MCP CVE 건수의 근거로 인용한 출처)
- Postmark Team (2025-09-25). *Information regarding malicious postmark-mcp package*. https://postmarkapp.com/blog/information-regarding-malicious-postmark-mcp-package
- Kamath, A. et al. (2025). *Enforcing Temporal Constraints for LLM Agents*. arXiv:2512.23738. https://arxiv.org/abs/2512.23738
- Hong, Y. et al. (2026). *Don't Make Models Guess Security and Safety: Symbolic Guardrails for Domain-Specific AI Agents*. arXiv:2604.15579v2. https://arxiv.org/html/2604.15579v2
- A2A Project. *A2A Protocol Specification v1.0.0*. https://a2a-protocol.org/v1.0.0/specification/
- A2A Project (2026-03-12). *v1.0.0 release*. https://github.com/a2aproject/A2A/releases/tag/v1.0.0
- Model Context Protocol. *Authorization, specification 2025-06-18*. https://modelcontextprotocol.io/specification/2025-06-18/basic/authorization
- IETF. *OAuth 2.0 Token Exchange*, RFC 8693, section 4.1. https://www.rfc-editor.org/rfc/rfc8693#section-4.1
- IETF. *Resource Indicators for OAuth 2.0*, RFC 8707. https://www.rfc-editor.org/rfc/rfc8707
- IETF. *OAuth 2.0 Demonstrating Proof of Possession (DPoP)*, RFC 9449. https://www.rfc-editor.org/rfc/rfc9449
- IBM Research. *Agent Communication Protocol*. https://research.ibm.com/projects/agent-communication-protocol
- HL7. *FHIR R5 Workflow Module*. https://hl7.org/fhir/R5/workflow.html
- AWS. *Policy sessions and identity propagation*. https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-session-based-temporal.html
- AWS. *Authoring temporal policies*. https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-temporal-authoring.html
- AWS. *Policy enforcement modes*. https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-enforcement-modes.html
