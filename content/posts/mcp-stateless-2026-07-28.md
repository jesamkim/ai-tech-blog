---
title: "MCP가 Stateless로 간다 — 2026-07-28 스펙 변경이 에이전트 인프라에 의미하는 것"
date: 2026-08-01T14:00:00+09:00
draft: false
categories: ["AI 에이전트", "MLOps & Platform"]
tags: ["MCP", "Model Context Protocol", "에이전트 인프라", "AWS AgentCore", "프로토콜"]
author: "Jesam Kim"
description: "Model Context Protocol의 2026-07-28 개정은 initialize handshake와 세션 ID를 프로토콜에서 걷어냈습니다. sticky session과 공유 세션 스토어를 전제로 짜둔 배포 구조가 어떻게 달라지는지, 그리고 상태를 어디로 옮겨야 하는지 정리합니다."
cover:
  image: "/ai-tech-blog/images/mcp-stateless-2026-07-28/cover.png"
  alt: "MCP 2026-07-28 스펙의 stateless 전환과 분산 서버 인프라"
  relative: false
---

MCP 서버를 로컬에서 하나 띄워 쓸 때는 아무 문제가 없습니다. Claude Desktop이 stdio로 프로세스를 하나 붙이고, 그 프로세스가 살아 있는 동안 세션이 유지됩니다. 프로토콜이 처음 설계될 때 상정한 그림이 정확히 이것이었습니다. 단일 앱, 단일 로컬 서버, 하나의 연결.

같은 서버를 사내 100명이 쓰도록 원격 배포하는 순간 이야기가 달라집니다. 로드밸런서 뒤에 인스턴스를 세 개 띄우면 클라이언트가 처음 handshake한 인스턴스에만 세션이 존재합니다. 두 번째 요청이 다른 인스턴스로 가면 그 인스턴스는 세션을 모릅니다. 그래서 sticky routing을 켜거나, 인스턴스들이 공유하는 세션 스토어를 앞에 두거나, 둘 다 하게 됩니다.

[2026-07-28 스펙 개정](https://modelcontextprotocol.io/specification/2026-07-28)이 이 지점을 건드렸습니다. MCP 팀은 이 개정을 [런칭 이후 가장 큰 프로토콜 개정](https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate)이라고 설명하고, breaking change가 포함된다는 점을 함께 명시했습니다. 핵심은 프로토콜에서 세션 개념 자체를 제거한 것입니다.

이 블로그에서 6월에 다룬 Claude Code 플러그인·스킬 생태계나 리뷰어를 스킬로 만드는 이야기는 에이전트 위쪽 레이어였습니다. 이번 글은 그 아래, 도구가 실제로 어떤 규약으로 호출되는지의 레이어를 다룹니다.

## 무엇이 사라졌나

두 개의 SEP가 변경의 축입니다.

첫째, `initialize`와 `notifications/initialized` handshake가 제거됐습니다([SEP-2575](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2575)). 이전 스펙에서 초기화 단계는 [클라이언트와 서버 사이의 첫 상호작용이어야 한다는 필수 요건](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle)이었습니다. 프로토콜 버전을 맞추고, capability를 교환하고, 구현 정보를 공유하는 절차입니다. 이 단계가 없어졌습니다.

둘째, `Mcp-Session-Id` 헤더와 프로토콜 레벨 세션이 Streamable HTTP transport에서 제거됐습니다([SEP-2567](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2567)). 함께 사라진 것들이 더 있습니다. 서버가 클라이언트에게 능동적으로 메시지를 보내기 위한 GET 스트림 엔드포인트, 세션 종료를 위한 HTTP DELETE, `Last-Event-ID`를 통한 스트림 재개가 모두 [이번 개정에서 빠졌습니다](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http).

이 두 변경이 합쳐진 결과가 <strong>요청의 self-contained화</strong>입니다. 초기화 시점에 한 번 교환하던 정보가 이제 매 요청 본문의 `_meta`에 실립니다. 필드 이름에 네임스페이스가 붙었습니다. `io.modelcontextprotocol/protocolVersion`과 `io.modelcontextprotocol/clientCapabilities`가 필수이고, `io.modelcontextprotocol/clientInfo`는 SHOULD입니다. 필수 필드가 빠지면 서버는 `-32602`와 HTTP 400으로 거절합니다.

capability 조회는 필요할 때 하는 방식으로 바뀌었습니다. `server/discover` 메서드가 새로 생겼고, 서버는 이 메서드를 구현해야 하지만 클라이언트가 호출할 의무는 없습니다. 응답에는 `supportedVersions`, `capabilities`, `instructions`가 담기고 여기에도 캐시 힌트인 `ttlMs`와 `cacheScope`가 붙습니다. 도구 목록을 매번 다시 받아올 이유가 없어졌다는 뜻입니다.

![MCP 세션 제거 전후의 배포 구조 비교. Before는 sticky routing으로 특정 인스턴스에 고정되고 공유 세션 스토어가 필요하며, After는 round-robin으로 어느 인스턴스든 요청을 처리합니다](/ai-tech-blog/images/mcp-stateless-2026-07-28/diagram-1-before-after-arch.png)
*세션이 프로토콜에서 빠지면 로드밸런서 설정과 세션 스토어가 필요한 이유가 함께 사라집니다.*

## 요청 하나를 실제로 보면

변화의 크기는 HTTP 왕복 횟수로 보면 분명합니다.

이전 스펙에서 도구를 한 번 호출하려면 세 단계가 필요했습니다. `initialize` 요청을 보내고, 서버가 세션 ID를 발급한 응답을 받고, `notifications/initialized`로 준비 완료를 알린 다음에야 `tools/call`을 보낼 수 있었습니다. 2025-11-25 스펙의 `initialize` 요청은 이런 모양입니다.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-11-25",
    "capabilities": {
      "roots": { "listChanged": true },
      "sampling": {},
      "elicitation": { "form": {}, "url": {} }
    },
    "clientInfo": {
      "name": "ExampleClient",
      "title": "Example Client Display Name",
      "version": "1.0.0"
    }
  }
}
```

서버는 자신의 capability와 `serverInfo`를 담아 응답하고, HTTP 레이어에서 `Mcp-Session-Id`를 발급합니다. 클라이언트는 이후 모든 요청에 그 세션 ID를 실어 보냅니다. 그리고 그 세션 ID는 발급한 인스턴스에서만 유효합니다.

2026-07-28에서는 같은 작업이 요청 하나입니다. 스펙 문서의 `tools/call` 예시를 그대로 옮기면 이렇습니다.

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: get_weather

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "get_weather",
    "arguments": {
      "location": "Seattle, WA"
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "ExampleClient",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

handshake가 없고 세션 ID가 없습니다. 프로토콜 버전과 클라이언트 정보가 본문 안에 들어 있으니 이 요청은 어느 인스턴스에 도착해도 그 자체로 완결됩니다.

![도구 호출까지의 왕복 횟수 비교. Before는 initialize, initialized, tools/call로 세 단계이고 After는 self-contained tools/call 한 번입니다](/ai-tech-blog/images/mcp-stateless-2026-07-28/diagram-2-request-lifecycle.png)
*첫 도구 호출까지 필요한 왕복이 3회에서 1회로 줄어듭니다.*

## 헤더가 라우팅 가능해진 이유

self-contained 요청만으로는 인프라 입장에서 절반만 해결됩니다. 요청 본문에 정보가 다 있다 해도, 로드밸런서가 그걸 읽으려면 JSON을 파싱해야 합니다. 게이트웨이가 매 요청 본문을 열어보는 구조는 좋은 선택이 아닙니다.

[SEP-2243](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2243)이 이 부분을 다뤘습니다. Streamable HTTP는 본문의 일부 필드를 HTTP 헤더로 미러링합니다. `Mcp-Method`는 모든 요청에 붙고, `Mcp-Name`은 `tools/call`, `resources/read`, `prompts/get`에 붙습니다. 스펙은 이 헤더들이 [준수를 위해 필수(REQUIRED)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)라고 명시합니다. 중간 경유 장비가 본문을 파싱하지 않고도 라우팅과 검사를 할 수 있게 하려는 설계입니다.

여기서 흥미로운 건 헤더-본문 불일치 처리입니다. 헤더 값이 본문 값과 다르면 서버는 HTTP 400과 JSON-RPC 에러 `-32020`(`HeaderMismatch`)으로 거절해야 합니다. 스펙이 밝힌 이유가 명확합니다. 네트워크의 서로 다른 구성 요소가 서로 다른 진실 공급원에 의존할 때 생기는 보안 취약점을 막기 위한 것입니다. 로드밸런서는 헤더 값으로 라우팅하는데 MCP 서버는 본문 값으로 실행하는 상황을 차단합니다. 본문이 진실 공급원이고 헤더는 그 사본이라는 원칙이 유지됩니다.

스펙은 중간 경유 장비에도 주의를 남겼습니다. 미러링된 헤더로 정책을 집행하는 장비(테넌트별 라우팅이나 레이트리미팅)는 `MCP-Protocol-Version` 헤더가 헤더-본문 검증을 요구하는 버전인지 확인해야 합니다. 구버전이거나 헤더가 없으면 검증되지 않은 헤더 값을 신뢰하기보다 요청을 거절하는 편이 낫다는 권고입니다.

캐싱도 함께 들어왔습니다. [SEP-2549](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2549)가 `CacheableResult`를 정의해 `tools/list`, `prompts/list`, `resources/list`, `resources/read`, `resources/templates/list` 결과에 `ttlMs`와 `cacheScope`를 필수 필드로 요구합니다. `cacheScope`는 `public` 또는 `private` 값을 갖습니다. 도구 목록이 자주 바뀌지 않는 서버라면 클라이언트가 이 힌트를 보고 재조회를 건너뛸 수 있습니다.

## Stateless 프로토콜은 Stateless 애플리케이션이 아니다

여기가 오해가 생기기 쉬운 지점입니다. 프로토콜에서 세션이 빠졌다는 말이 애플리케이션이 상태를 가질 수 없다는 뜻은 아닙니다.

MCP 팀이 제시한 대안은 전통적인 HTTP API가 오래 써온 방식입니다. 도구가 명시적 핸들을 발급하고, 모델이 그 핸들을 다음 호출의 일반 인자로 되넘깁니다. 스펙 발표 글의 표현으로는 도구에서 [`basket_id`나 `browser_id` 같은 명시적 핸들을 발급](https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate)하고 모델이 그것을 돌려주게 하는 것입니다.

그런데 MCP 팀은 여기에 한 걸음 더 나간 주장을 붙였습니다. 이 패턴이 세션 상태의 대체재로 쓸 만한 수준을 넘어 종종 더 강력하다는 것입니다. 근거는 모델이 핸들을 여러 도구에 걸쳐 조합하고, 그것에 대해 추론하고, 단계 사이에 넘길 수 있다는 점입니다. transport 계층에 숨어 있던 불투명한 세션 상태로는 할 수 없던 일입니다. 상태를 감춰두는 대신 모델에게 보이게 만든다는 관점입니다.

이 논지는 검증 없이 받아들일 성질은 아닙니다. 핸들이 모델의 컨텍스트에 노출된다는 것은 곧 대화 레이어가 새로운 신뢰 경계가 된다는 뜻이기도 합니다. 뒤에서 보안 관점을 다룰 때 같은 사실의 다른 면을 보게 됩니다.

실무 관점에서는 세션 스토어가 사라지는 게 아니라 역할이 바뀐다고 이해하는 편이 정확합니다. Microsoft가 App Service 배포 관점에서 같은 변화를 다루면서 [이전 모델에서 Redis는 프로토콜 세션 상태를 공유하기 위해 있는 경우가 많았고 그 이유가 없어졌다](https://techcommunity.microsoft.com/blog/appsonazureblog/mcp-just-went-stateless-%E2%80%94-what-the-2026-spec-changes-about-scaling-on-app-servic/4530222)고 정리한 부분이 이 구분을 잘 보여줍니다. 애플리케이션 데이터를 담을 저장소는 여전히 필요합니다. 프로토콜 규약을 유지하기 위한 저장소가 필요 없어진 것입니다.

같은 글은 ARR affinity 쿠키에 대해서도 구체적으로 언급합니다. 이전에도 권장 설정은 `clientAffinityEnabled: false`였는데, 이제는 그 설정과 싸우는 프로토콜 세션이 더 이상 없다는 서술입니다. 조정할 affinity 설정이 없어졌다는 표현이 변화의 성격을 압축합니다.

## 서버가 클라이언트에게 물어봐야 할 때

세션과 GET 스트림이 없어지면서 풀어야 할 문제가 하나 생깁니다. 서버가 작업 중간에 사용자 확인이 필요하거나 LLM 완성이 필요할 때, 이전에는 SSE 스트림으로 클라이언트에게 요청을 보냈습니다. 그 통로가 막혔습니다.

이 부분은 브리핑 단계와 최종 스펙 사이에 서술이 달라진 지점이라 정확히 볼 필요가 있습니다. 최종 스펙은 서버가 자체 JSON-RPC 요청을 보내는 패턴을 조건부로 허용한 것이 아니라 없앴습니다. transport 문서는 [서버가 JSON-RPC 요청을 개시하지 않고 클라이언트가 JSON-RPC 응답을 보내지 않는다](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)고 못박습니다. 이 방향 외에 다른 메시지 방향은 존재하지 않는다는 서술입니다.

이 제약을 상류에서 규정한 것이 [SEP-2260](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2260)입니다. 이전 스펙에서는 서버가 클라이언트 요청을 처리하는 동안에만 요청을 개시하는 것이 권장 사항이었는데, 이번 개정은 이를 필수로 끌어올렸습니다. 사용자는 맥락 없이 프롬프트를 받지 않고, 모든 서버발 요청은 클라이언트나 그 에이전트가 시작한 작업으로 추적됩니다.

대신 들어온 것이 Multi Round-Trip Requests입니다([SEP-2322](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2322)). 서버는 스트림을 열지 않고 `InputRequiredResult`를 반환합니다. `resultType`이 `input_required`이고, `inputRequests` 맵에 필요한 입력들이 담기고, `requestState`라는 불투명한 토큰이 함께 옵니다. 클라이언트는 필요한 입력을 모아 원래 요청을 다시 보냅니다. 이때 `inputResponses`와 함께 받은 `requestState`를 그대로 되돌려줍니다. 재전송은 새 JSON-RPC `id`를 씁니다.

흐름을 두 단계로 보면 이렇습니다. 서버가 사용자 입력을 요구하는 응답을 먼저 돌려줍니다. `inputRequests`는 서버가 부여한 문자열 키로 요청을 묶은 맵이고, 값은 `ElicitRequest`, `CreateMessageRequest`, `ListRootsRequest` 중 하나여야 합니다.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "github_login": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Please provide your GitHub username",
          "requestedSchema": {
            "type": "object",
            "properties": { "name": { "type": "string" } },
            "required": ["name"]
          }
        }
      }
    },
    "requestState": "AEAD-protected blob"
  }
}
```

클라이언트는 사용자에게 물어 답을 받고, 원래 요청을 새 `id`로 다시 보냅니다. `inputResponses`의 키는 `inputRequests`의 키와 대응합니다.

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "create_issue",
    "arguments": { "title": "Bug report" },
    "inputResponses": {
      "github_login": {
        "action": "accept",
        "content": { "name": "octocat" }
      }
    },
    "requestState": "AEAD-protected blob",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

`InputRequiredResult`를 돌려줄 수 있는 요청은 `tools/call`, `resources/read`, `prompts/get` 세 가지로 제한됩니다. 클라이언트가 capability로 선언하지 않은 종류의 요청은 서버가 `inputRequests`에 담을 수 없습니다.

핵심은 이 재전송이 어느 인스턴스로 가도 된다는 점입니다. 이어받을 상태가 `requestState` 안에 들어 있으니까요. 그런데 이 편의에는 조건이 붙습니다. 스펙은 서버가 `requestState`를 <strong>공격자가 통제하는 입력으로 취급해야 한다(MUST)</strong>고 규정합니다. 이 값이 authorization이나 리소스 접근, 비즈니스 로직에 영향을 준다면 서버는 HMAC이나 AEAD로 무결성을 보호하고 검증에 실패한 상태는 거절해야 합니다. 무결성 보호를 생략할 수 있는 경우는 변조가 요청 실패보다 나쁜 결과를 만들 수 없을 때로 한정됩니다.

replay 방어 요구사항도 구체적입니다. 스펙은 무결성 보호된 `requestState` 안에 인증된 principal, 짧은 만료 시간, 원 요청 식별자(메서드 이름과 주요 파라미터의 다이제스트)를 담고 수신 시 각각 검증하도록 권고합니다. 다른 principal이 제시한 상태를 거절하고, 만료된 것을 거절하고, 일치하지 않는 요청에 실린 것을 거절하라는 뜻입니다. 스펙은 이 조치들이 replay 창을 좁히고 사용자 간·요청 간 재사용을 막지만 단일 사용을 보장하지는 않는다는 점도 함께 경고합니다. 일회성 처리가 필요한 경우 서버가 그 불변식을 직접 강제해야 합니다.

이 부분이 마이그레이션에서 가장 놓치기 쉬운 지점입니다. 이전에는 세션이 서버 안에 있었으니 상태의 무결성을 고민할 필요가 없었습니다. 이제 그 상태가 클라이언트를 왕복하므로 암호학적 보호가 애플리케이션 책임이 됩니다.

`resultType`은 모든 result에 필수 필드가 됐고 정상 완료는 `complete` 값을 갖습니다. 클라이언트 코드에서 결과를 받으면 이 필드로 분기해야 한다는 뜻입니다. 기존에 결과를 바로 파싱하던 코드는 `input_required`를 만나면 예상하지 못한 구조를 보게 됩니다.

장기 실행 알림이 필요한 경우는 `subscriptions/listen` 요청으로 처리합니다. 이 요청의 응답 자체가 열린 채 유지되는 SSE 스트림이고, 클라이언트가 옵트인한 종류의 변경 알림만 흘러갑니다. 요청 범위 알림인 `notifications/progress`나 `notifications/message`는 이 스트림으로 오지 않고 해당 요청의 응답 스트림에만 흐릅니다.

## 코어에서 확장으로

이번 개정은 코어 스펙의 표면적을 줄이는 방향도 함께 잡았습니다.

Tasks가 확장으로 옮겨졌습니다([SEP-2663](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2663)). 이전에 실험적 코어 기능으로 출시됐던 것인데, MCP 팀 설명으로는 프로덕션 사용에서 재설계가 필요한 부분이 충분히 드러나 확장이 적절한 위치라고 판단했습니다. 이름은 `io.modelcontextprotocol/tasks`이고, 서버가 durable `taskId`를 반환하면 클라이언트가 `tasks/get`으로 폴링하고 `tasks/update`로 입력을 넣습니다. `tasks/cancel`도 있고, `tasks/list`는 제거됐습니다.

MCP Apps도 확장으로 분리됐습니다. 서버가 선언한 인터랙티브 HTML 인터페이스를 호스트가 sandboxed iframe으로 렌더하는 기능입니다.

여기서 브리핑 단계 정보 하나를 정정해 둘 필요가 있습니다. Skills over MCP는 확장이 아닙니다. MCP 커뮤니티의 working group으로 운영되고 있어서, 확장 스펙으로 분리된 Tasks나 MCP Apps와는 성격이 다릅니다.

인증 쪽도 정리가 필요합니다. Authorization 자체는 여전히 OPTIONAL입니다. 다만 authorization server를 구현한다면 OAuth 2.1을 구현해야 합니다. OIDC가 requirement가 된 것은 아닙니다. authorization server는 RFC 8414 방식 또는 OIDC Discovery 중 최소 하나를 제공하면 되고, 클라이언트는 양쪽을 모두 지원해야 합니다. 엔터프라이즈 환경을 위한 Enterprise-Managed Authorization은 `ext-auth` 확장으로 추가됐습니다.

## Deprecation과 그 정책

기능 세 개가 deprecated 상태로 들어갔습니다([SEP-2577](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2577)). Roots는 도구 파라미터나 resource URI, 서버 설정으로 대체하도록 권고되고, Sampling은 서버가 LLM provider API를 직접 호출하는 방식으로, Logging은 stdio 환경에서 `stderr` 또는 OpenTelemetry로 대체하도록 권고됩니다. HTTP+SSE transport도 deprecated 분류로 정리됐습니다.

deprecated는 제거가 아닙니다. 이 구분이 실무에서 중요합니다. 세 기능은 이번 릴리스와 이후 12개월 내에 발표되는 어떤 스펙 버전에서도 동작을 유지하고, 가장 이른 제거 시점은 2027-07-28 이후 첫 개정입니다. 마이그레이션할 시간이 있다는 뜻입니다.

이 기간을 보장하는 것이 새로 생긴 정식 deprecation 정책입니다([SEP-2596](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2596)). 최소 12개월의 deprecation window와 공개 deprecated 레지스트리를 규정합니다. 다만 90일 신속 제거 예외 조항이 있어서 모든 경우에 12개월이 보장되는 것은 아닙니다.

정책 자체가 없었던 이전 상황과 비교하면 이 변화가 프로토콜 채택 판단에 미치는 영향이 작지 않습니다. 이번처럼 breaking change가 들어오는 개정이 예고 없이 반복될 수 있는 구조에서는 프로덕션 의존을 결정하기 어렵습니다. MCP 팀도 이번 릴리스에 breaking change가 있다는 점을 밝히면서 그것이 앞으로 기본이 되기를 의도하지 않는다고 함께 적었습니다.

## 매니지드 서비스 쪽 대응

AWS는 스펙 공개일에 맞춰 AgentCore Gateway의 2026-07-28 스펙 지원 내용을 [블로그로 공개했습니다](https://aws.amazon.com/blogs/machine-learning/how-agentcore-gateway-supports-the-mcp-2026-07-28-spec). 게시글에서 확인되는 범위를 팩트만 옮기면 다음과 같습니다.

Gateway는 요청마다 `MCP-Protocol-Version` 헤더를 읽고, 지원 목록에 없는 버전이면 HTTP 400과 코드 `-32022`로 거절합니다. 헤더-본문 바인딩도 집행해서 `Mcp-Method`나 `Mcp-Name`이 본문과 모순되면 HTTP 400과 `-32020`을 반환합니다. 도구 결과는 새 구조화 결과 봉투를 실어 보내고, 캐시 가능한 결과에는 TTL과 스코프 힌트가 붙습니다. MRTR 방식의 elicitation과 sampling도 요청 단위 메커니즘으로 전달하며, 클라이언트가 해당 capability를 선언하지 않은 경우 `-32021`을 반환합니다.

하위 호환 처리도 문서에 정리돼 있습니다. 지원 버전은 2025-03-26, 2025-06-18, 2025-11-25, 2026-07-28이고, 헤더가 없으면 2025-03-26으로 간주합니다. 구버전 클라이언트와 신버전 타깃 사이에서는 버전 간 변환이 이뤄지지만, 이 변환은 현재 서버에서 클라이언트로 향하는 elicitation과 sampling 호출을 지원하지 않는다는 제약이 명시돼 있습니다. 스펙 채택은 타깃별 작업 없이 `UpdateGateway` 호출 하나로 처리되는 설정 변경으로 설명되며, `UpdateGateway`가 추가가 아니라 교체 방식이라는 주의가 함께 적혀 있습니다. 게시글에는 GA나 프리뷰 라벨, 리전 정보가 언급되지 않습니다.

세션 제거가 배포에 주는 의미에 대해서는 같은 글이 세션을 제거함으로써 원격 MCP 서버가 표준 HTTPS 엔드포인트나 다름없어지고 엔터프라이즈 워크로드에서 깔끔하게 스케일한다고 서술합니다. 이전 문제로는 운영자가 로드밸런서에 sticky session을 두거나 fleet 뒤에 공유 세션 스토어를 두거나 둘 다 해야 했던 상황을 지적합니다. Microsoft의 App Service 글도 SEP-2575와 SEP-2567을 근거로 어떤 MCP 요청이든 어느 인스턴스에 도착해도 된다는 같은 결론에 도달합니다. 두 회사가 서로 다른 플랫폼 관점에서 같은 병목을 지목했다는 점이, 이 변경이 특정 벤더의 편의를 위한 것이 아니라 원격 배포의 공통 문제였음을 보여줍니다.

## 보안 경계가 이동한다

스케일링 이점만 보고 넘어가면 이번 개정의 절반을 놓칩니다.

보안 회사 Backslash Security가 새 스펙이 새로운 공격면을 연다는 분석을 냈습니다. 인용에 앞서 시점을 정확히 해둘 필요가 있습니다. 이 글은 [2026년 5월 28일 작성](https://www.backslash.security/blog/new-mcp-spec-opens-new-attack-surfaces)으로, 7월 28일 최종 스펙 확정 두 달 전 release candidate 단계를 대상으로 한 사전 분석입니다. 확정 스펙에 대한 사후 감사가 아닙니다.

분석의 첫 지적이 세션 하이재킹이 핸들 하이재킹으로 대체된다는 것입니다. 악성 도구가 공격자가 통제하는 `task_id` 값을 반환하면 모델이 그것을 따른다는 시나리오입니다. 다른 사용자 대화의 핸들을 재사용하거나 `requestState` blob을 변조해 금액이나 수신자를 바꾸는 경우도 함께 제시됩니다. 이 글이 제안하는 대응은 서버가 핸들 단독이 아니라 핸들과 인증 컨텍스트의 쌍을 검증해야 한다는 것입니다. 서버가 되돌아온 값을 검증 없이 신뢰하면 침투가 성립한다는 지적입니다.

공격 비용 구조가 어떻게 달라지는지가 이 분석의 요점입니다. 세션 하이재킹은 서버의 세션 스토어에 접근해야 성립했습니다. 핸들 하이재킹은 대화에 접근하면 됩니다. 상태를 모델에게 보이게 만든다는 설계 결정이 가진 다른 얼굴입니다. 앞서 MCP 팀이 강점으로 제시한 바로 그 성질입니다.

`requestState` 변조 시나리오에 관해서는 최종 스펙이 이 분석보다 뒤에 나왔다는 점을 함께 봐야 합니다. 앞서 본 대로 확정 스펙의 MRTR 문서는 `requestState`를 공격자 통제 입력으로 다루고 무결성을 보호하라는 요구사항을 명시적으로 담았습니다. 문서 마지막의 보안 고려사항 항목도 악성이거나 침해된 클라이언트가 서버 동작을 바꾸거나 authorization 검사를 우회하거나 서버 로직을 훼손하려 `requestState`를 수정할 수 있다는 위험을 직접 서술합니다. 이 지적이 RC 단계에서 나와 최종 스펙에 반영된 것인지, 아니면 병렬로 정리된 것인지는 공개된 자료로 판단하기 어렵습니다. 확인할 수 있는 것은 확정 스펙이 이 공격 벡터를 인지하고 서버 측 요구사항을 규정했다는 사실입니다. 다만 규정이 있다는 것과 구현이 그것을 따른다는 것은 별개입니다. SDK와 자체 구현 서버가 이 요구사항을 실제로 집행하는지는 각자 확인할 몫으로 남습니다.

같은 글이 Roots deprecation을 두고 프로토콜이 유일한 구조적 집행 지점을 잃었다고 표현하는데, 이 부분은 인용에 주의가 필요합니다. Roots는 제거된 것이 아니라 deprecated이고, 이번 릴리스와 이후 12개월 내 버전에서 동작을 유지합니다. 글 자체도 도입부에서는 deprecates라는 표현을 씁니다. MCP Apps에 대해서는 IDE가 서버가 보낸 HTML을 렌더한다는 점에서 XSS, 클릭재킹, 공급망 오염, 그리고 네이티브 인증 프롬프트를 위장하는 UI 모방을 벡터로 꼽습니다. 이 분석의 논조가 스펙 자체에 대한 비판은 아니라는 점도 덧붙일 만합니다. 글은 스펙이 좋은 엔지니어링이고 스케일 논거가 실재한다고 평가합니다.

공식 스펙의 입장도 확인해 둘 지점입니다. 보안 및 신뢰·안전 섹션의 구현 가이드라인은 MCP 자체가 프로토콜 레벨에서 보안 원칙을 집행할 수 없다고 명시하면서, 구현자가 견고한 consent와 authorization flow를 애플리케이션에 구축해야 한다고 권고합니다. 같은 문서의 기본 프로토콜 항목에 stateless self-contained 요청과 요청 단위 capability 협상이 나란히 적혀 있습니다. 이 두 서술을 붙여 읽으면 검증 책임이 어디로 갔는지가 분명해집니다. 요청마다 자기 완결적이라는 것은 요청마다 검증해야 한다는 뜻입니다.

## 마이그레이션할 때 확인할 것

지금 원격 MCP 서버를 운영하고 있다면 검토 순서를 이렇게 잡을 수 있습니다.

먼저 클라이언트와 서버 중 어느 쪽을 먼저 올릴지 정해야 합니다. 스펙은 하위 호환 감지 방식을 규정해 뒀습니다. 두 시대를 모두 지원하는 클라이언트는 최신 방식 요청을 먼저 시도하고, HTTP 400을 받으면 본문을 확인합니다. 본문에 최신 스펙의 JSON-RPC 에러가 있으면 서버는 최신 버전을 말하는 것이므로 fallback하지 말고 요청을 고쳐 재시도해야 합니다. 본문이 비어 있거나 알려진 형식이 아니면 `initialize` 방식으로 내려갑니다. 이 구분을 구현하지 않으면 최신 서버의 버전 오류를 구버전 서버로 오인하게 됩니다.

이전 방식 트래픽에 대한 서버 동작도 규정돼 있습니다. MCP 엔드포인트로 오는 GET이나 DELETE에는 405를 응답하고, `Mcp-Session-Id` 헤더는 무시하며 세션 ID를 발급하거나 되돌려 보내지 않고, `Last-Event-ID`도 무시합니다.

다음으로 애플리케이션 상태를 어떻게 다시 표현할지 설계해야 합니다. 세션에 얹어 두던 것을 명시적 핸들로 바꾸는 작업입니다. 이 단계에서 함께 결정해야 하는 것이 핸들 검증 방식입니다. 앞서 본 핸들 하이재킹 시나리오가 여기서 갈립니다. 핸들을 발급할 때 어떤 인증 컨텍스트에 묶어 둘지, 되돌아온 핸들을 어떻게 그 컨텍스트와 대조할지를 함께 설계해야 합니다. 프로토콜이 세션으로 암묵적으로 제공하던 신뢰 경계를 애플리케이션이 명시적으로 세우는 일입니다.

서버에서 클라이언트로 향하는 상호작용이 있다면 MRTR 재작성이 필요합니다. elicitation이나 sampling을 쓰고 있었다면 SSE 스트림에 요청을 실어 보내던 코드가 `InputRequiredResult` 반환으로 바뀝니다. 매니지드 게이트웨이를 경유하는 경우 이 경로의 버전 간 변환 지원 여부를 확인해 두는 편이 안전합니다.

마지막으로 deprecated 기능 사용 현황을 훑어야 합니다. Roots, Sampling, Logging을 쓰고 있다면 12개월 window 안에서 옮길 계획을 세울 수 있습니다. 급하지 않지만 90일 신속 제거 예외가 있다는 점은 알고 있어야 합니다.

## 정리

<strong>이번 개정의 성격은 프로토콜이 배포 현실에 맞춰진 것입니다.</strong> MCP는 로컬 단일 연결을 전제로 출발했고, 원격 스케일 배포는 그 전제 밖의 일이었습니다. sticky session과 공유 세션 스토어는 그 간극을 배포 레이어에서 메우던 방식이었습니다. 이번 변경은 그 간극을 프로토콜 쪽에서 없앴습니다.

대가가 없는 변경은 아닙니다. breaking change이고, 상태 관리 책임이 프로토콜에서 애플리케이션으로 넘어왔고, 그와 함께 검증 책임도 넘어왔습니다. 세션이라는 형태로 프로토콜이 제공하던 암묵적 신뢰 경계를 이제 각 서버가 명시적으로 세워야 합니다. 스케일링이 쉬워진 것과 보안 설계가 쉬워진 것은 다른 이야기입니다.

원격 MCP 서버를 운영 중이라면 마이그레이션 계획에서 두 축을 나눠 보는 편이 낫습니다. 하나는 transport와 라이프사이클 변경을 따라가는 작업이고, 다른 하나는 세션이 사라진 자리에 무엇을 놓을지 설계하는 작업입니다. 첫 번째는 스펙 문서를 따라가면 되지만 두 번째는 각 서버가 다루는 데이터와 신뢰 모델에 따라 답이 달라집니다.

## References

- [Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) &mdash; Model Context Protocol
- [Streamable HTTP transport (2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http) &mdash; Model Context Protocol
- [Transports overview (2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports) &mdash; Model Context Protocol
- [Lifecycle (2025-11-25)](https://modelcontextprotocol.io/specification/2025-11-25/basic/lifecycle) &mdash; Model Context Protocol
- [The 2026-07-28 release candidate](https://blog.modelcontextprotocol.io/posts/2026-07-28-release-candidate) &mdash; MCP Blog
- [MCP 2026-07-28: what's changing and how to migrate](https://aaif.io/blog/mcp-2026-07-28-whats-changing-and-how-to-migrate) &mdash; Akash Jaiswal, AAIF
- [How AgentCore Gateway supports the MCP 2026-07-28 spec](https://aws.amazon.com/blogs/machine-learning/how-agentcore-gateway-supports-the-mcp-2026-07-28-spec) &mdash; AWS Machine Learning Blog
- [New MCP spec opens new attack surfaces](https://www.backslash.security/blog/new-mcp-spec-opens-new-attack-surfaces) &mdash; Maya Pik, Backslash Security (2026-05-28, release candidate 기준)
- [MCP just went stateless — what the 2026 spec changes about scaling on App Service](https://techcommunity.microsoft.com/blog/appsonazureblog/mcp-just-went-stateless-%E2%80%94-what-the-2026-spec-changes-about-scaling-on-app-servic/4530222) &mdash; Microsoft Tech Community
- [MCP goes stateless in the 2026-07-28 specification](https://appwrite.io/blog/post/mcp-goes-stateless-in-the-2026-07-28-specification) &mdash; Appwrite
- [SEP-2575: Make MCP Stateless](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2575) &mdash; modelcontextprotocol GitHub
- [SEP-2567: Sessionless MCP via Explicit State Handles](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2567) &mdash; modelcontextprotocol GitHub
- [SEP-2322: Multi Round-Trip Requests](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2322) &mdash; modelcontextprotocol GitHub
- [SEP-2243: HTTP Standardization](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2243) &mdash; modelcontextprotocol GitHub
- [SEP-2549: TTL for List Results](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2549) &mdash; modelcontextprotocol GitHub
- [SEP-2596: Specification Feature Lifecycle and Deprecation Policy](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2596) &mdash; modelcontextprotocol GitHub
- [SEP-2577: Deprecate Roots, Sampling, and Logging](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2577) &mdash; modelcontextprotocol GitHub
- [SEP-2663: Tasks Extension](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/2663) &mdash; modelcontextprotocol GitHub
