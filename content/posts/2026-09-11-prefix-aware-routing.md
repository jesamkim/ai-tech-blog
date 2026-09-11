---
title: "캐시를 켰는데도 LLM이 느린 이유: Prefix-aware Routing의 원리"
date: 2026-09-11T21:44:07+09:00
draft: false
slug: "prefix-aware-routing"
author: "Jesam Kim"
description: "LLM의 prefix cache를 여러 서버에서 효율적으로 재사용하려면 요청 라우팅도 함께 설계해야 합니다. SageMaker의 prefix-aware routing을 사례로 캐시 적중률과 부하 분산, 첫 토큰 지연의 관계를 살펴봅니다."
categories: ["MLOps & Platform"]
tags: ["LLM Serving", "KV Cache", "Prefix Caching", "Amazon SageMaker", "vLLM", "Inference", "Load Balancing"]
cover:
  image: "/ai-tech-blog/images/prefix-aware-routing/cover.png"
  alt: "같은 모양의 메시지 캡슐을 하나의 보관소로 모으고 일부는 다른 보관소로 보내는 우주 정거장"
  relative: false
---

LLM 서버에서 prefix caching을 켰습니다. 같은 시스템 프롬프트를 반복해서 보내면 이미 계산한 부분을 재사용할 수 있습니다. 그런데 서버를 여러 대로 늘린 뒤에는 캐시를 켜기 전과 응답 속도가 크게 다르지 않습니다.

이런 상황을 가정해 보겠습니다. 첫 요청은 서버 A로 들어갔고, 같은 프롬프트로 시작하는 다음 요청은 서버 B로 들어갔습니다. A에 계산 결과가 남아 있어도 B가 그 결과를 사용할 수 없다면, B는 같은 부분을 다시 계산해야 합니다. 캐시 기능과 요청을 보내는 방식이 서로 맞지 않는 상황입니다.

2026년 9월 10일 공개된 Amazon SageMaker Inference의 prefix-aware routing은 이 문제를 다룹니다. 여러 요청에서 반복되는 앞부분을 라우팅 기준으로 사용해 캐시를 재사용할 가능성을 높입니다. 이 글에서는 공식 발표와 API 문서, vLLM 문서를 바탕으로 동작 원리와 적용 조건을 살펴봅니다. 아래 설정과 사례는 설명용이며 직접 배포하거나 성능을 측정한 결과는 아닙니다.

## Prefix cache가 줄이는 시간

LLM은 입력 프롬프트를 처리하는 prefill 단계에서 토큰의 Key와 Value 텐서를 계산합니다. Prefix caching은 이전 요청과 동일한 앞부분의 KV cache를 재사용해 해당 계산을 줄입니다. 이미 생성한 답변을 그대로 돌려주는 응답 캐시와는 저장하는 대상이 다릅니다.

설명용으로 다음과 같은 입력을 생각할 수 있습니다.

```text
요청 1: [동일한 상담 지침] [동일한 반품 규정] [개봉한 제품도 반품할 수 있나요?]
요청 2: [동일한 상담 지침] [동일한 반품 규정] [반품 배송비는 누가 부담하나요?]
```

상담 지침과 반품 규정의 계산 결과를 재사용하더라도, 각 질문의 처리와 새 답변 생성은 필요합니다. 따라서 prefix caching의 직접적인 효과는 입력 처리 시간을 줄이는 데 있습니다. vLLM 문서도 prefill과 decode를 구분하며, 새 토큰을 생성하는 decode 단계의 시간을 직접 줄여주는 기능은 아니라고 설명합니다.

이 구분은 성능 지표를 읽을 때도 필요합니다. <strong>첫 토큰이 도착하는 시간인 TTFT가 줄었다고 해서 답변 전체가 끝나는 시간도 같은 비율로 줄어드는 것은 아닙니다.</strong> 긴 답변을 생성하는 작업에서는 decode가 전체 시간에서 차지하는 비중이 클 수 있습니다.

기존 [vLLM 아키텍처 글](/ai-tech-blog/posts/2026-03-17-vllm-architecture-pagedattention-continuous-batching/)에서는 KV cache의 메모리 관리와 batching을 다뤘습니다. 이번에는 그 계산 결과를 어느 서버에서 재사용할지에 집중합니다.

## 여러 서버에 흩어진 캐시

여기서는 서버 A와 B가 각각 로컬 KV cache를 가지고 있고, 서버 간 캐시 전송이나 공유 저장소를 별도로 구성하지 않았다고 가정합니다. A에 도착한 요청의 계산 결과는 A가 다시 처리하는 요청에서 재사용할 수 있습니다. 같은 내용의 요청이 B로 가면 B에도 해당 cache가 있어야 합니다.

무작위 라우팅에서도 시간이 지나면 여러 서버에 같은 prefix가 저장될 수 있습니다. 따라서 서버를 늘린다고 캐시가 반드시 무효가 되거나 성능이 떨어진다고 단정할 수는 없습니다. 다만 서로 다른 prefix가 많고 저장 공간이 제한돼 있다면, 여러 서버가 같은 계산을 중복 수행하고 같은 데이터를 중복 보관하는 일이 늘어날 수 있습니다.

vLLM의 캐시 설계 문서는 토큰 블록과 그 앞의 prefix를 이용해 블록을 식별하고, 공간을 다시 할당할 때 기존 블록을 퇴출하는 과정을 설명합니다. <strong>같은 서버에 도착하는 것과 실제 cache hit가 발생하는 것은 별개의 조건</strong>입니다. 필요한 블록이 이미 퇴출됐다면 다시 계산해야 합니다.

또한 같은 의미의 문장만으로 재사용이 성립하지는 않습니다. 실제 캐시가 비교하는 토큰과 모델 실행 조건이 맞아야 합니다. 이 글의 prefix는 의미가 비슷한 질문을 묶는 semantic routing과 구분해서 읽어야 합니다.

## 재사용할 서버를 고르고 과부하를 피하기

SageMaker의 `PREFIX_AWARE` 전략은 같은 prompt prefix를 공유하는 요청을 같은 인스턴스로 보내는 방식을 제공합니다. 선택된 인스턴스에서 처리 중인 요청이 설정한 한도에 도달하면, 여유가 있는 다른 인스턴스로 요청을 보냅니다.

[![공통 prefix P는 KV 캐시를 보유한 서버 A로 우선 전달하고, A의 처리 중 요청이 한도에 도달하면 서버 C로 우회하는 개념도](/ai-tech-blog/images/prefix-aware-routing/routing.png)](/ai-tech-blog/images/prefix-aware-routing/routing.svg)
*요청의 prefix에 따른 서버 선택과 과부하 시 우회를 단순화한 예시입니다. 실제 캐시 적중은 각 서버에 남아 있는 KV 블록에 따라 달라집니다. 그림을 누르면 크게 볼 수 있습니다. 출처: [SageMaker 라우팅 API 문서](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariantRoutingConfig.html)를 바탕으로 재구성했습니다.*

이 우회 동작이 필요한 이유는 대기 시간 때문입니다. 이미 계산한 prefix를 가진 서버라도 요청이 많이 밀려 있으면 첫 토큰을 늦게 반환할 수 있습니다. 반대로 덜 바쁜 서버에서는 prefix를 다시 계산하더라도 먼저 응답할 수 있습니다. 캐시 적중률을 높이는 목표와 요청 대기를 줄이는 목표를 함께 고려해야 합니다.

SageMaker의 라우팅 전략을 선택할 때 비교할 기준은 다음과 같습니다.

| 전략 | 문서에 정의된 서버 선택 기준 | 워크로드에서 확인할 점 |
|---|---|---|
| `RANDOM` | 인스턴스를 무작위로 선택 | 반복 prefix가 적거나 캐시 재사용 효과가 작은지 |
| `LEAST_OUTSTANDING_REQUESTS` | 처리할 여유가 있는 인스턴스로 전달 | 요청별 처리 시간이 달라 대기가 쌓이는지 |
| `PREFIX_AWARE` | 공통 prefix를 기준으로 선택하고, 한도에 도달하면 우회 | 같은 prefix가 반복되고 실제 캐시 재사용이 가능한지 |

오른쪽 열은 API의 동작 정의를 바탕으로 정리한 적용 판단 기준입니다. 어느 전략이 유리한지는 실제 요청 분포와 처리 시간으로 비교해야 합니다.

## 라우팅의 prefix와 모델의 토큰은 다른 단위

설정에서 특히 주의할 부분은 `PrefixLength`입니다. API에 따라 이 값의 단위가 다릅니다.

| 호출 API | `PrefixLength`가 읽는 범위 |
|---|---|
| `InvokeEndpoint`, `InvokeEndpointWithResponseStream` | 요청 본문 처음부터의 바이트 수 |
| OpenAI 호환 API | `messages`의 텍스트 내용에서 읽는 문자 수 |

<strong>`PrefixLength: 4096`을 입력 토큰 4,096개라는 뜻으로 해석하면 안 됩니다.</strong> 특히 native API에서는 프롬프트 앞의 JSON 필드와 공백도 요청 본문의 일부입니다. 같은 프롬프트를 담더라도 JSON 키 순서나 직렬화 형식이 달라지면, 라우터가 읽는 앞부분이 달라질 수 있습니다.

문자 수와 바이트 수도 같지 않습니다. 예를 들어 `가`는 문자 하나이지만 UTF-8로는 3바이트입니다. 같은 `PrefixLength` 값을 설정해도 호출 API에 따라 실제로 읽는 텍스트 분량이 달라질 수 있습니다.

반면 모델 내부의 prefix cache는 토큰을 기준으로 계산 결과를 재사용합니다. 라우터가 요청 앞부분을 보고 서버를 선택한 뒤, 해당 서버의 추론 엔진이 자신의 캐시에서 재사용할 블록을 찾습니다. 라우팅에 사용하는 범위가 모델의 실제 재사용 가능한 prefix 전체를 그대로 나타내는 것은 아닙니다.

다음 JSON은 production variant의 `RoutingConfig`에 넣는 <strong>설정 일부</strong>입니다. 새 엔드포인트를 생성하는 완전한 배포 예시는 아닙니다.

```json
{
  "RoutingStrategy": "PREFIX_AWARE",
  "PrefixAwareRoutingConfig": {
    "PrefixLength": 4096,
    "ConcurrencyThreshold": 10
  }
}
```

문서상 `PrefixLength`의 범위는 1,024에서 65,536까지이고, `ConcurrencyThreshold`는 1에서 1,024까지입니다. `PREFIX_AWARE`를 선택하면 두 값을 지정해야 합니다. 예시의 4,096과 10은 설명을 위한 값이며 모든 서비스에 맞는 권장값은 아닙니다.

공유 구간보다 훨씬 짧은 범위를 읽으면 서로 다른 작업까지 같은 기준으로 묶일 수 있습니다. 반대로 매 요청마다 달라지는 내용까지 길게 읽으면, 재사용할 부분이 많은 요청도 다른 서버로 분산될 수 있습니다. 실제 요청 본문을 확인하면서 범위를 정하고, 추론 엔진에서도 prefix caching이 활성화돼 있는지 확인해야 합니다.

인스턴스가 한 대뿐이면 어떤 전략을 선택해도 요청의 도착 지점은 같습니다. 여러 인스턴스 사이의 라우팅 효과를 비교하려면 최소 두 대가 필요합니다.

## 발표된 성능 수치의 조건

공식 발표의 비교 대상은 prefix caching을 켠 vLLM에 적용한 무작위 라우팅입니다. Llama 3.1 70B Instruct를 `ml.p5.48xlarge` 7대에서 서빙했으며, 라우팅 전략을 바꿔 비교했습니다.

| 워크로드 | P50 TTFT 감소 | P90 TTFT 감소 | 처리량 증가 |
|---|---:|---:|---:|
| 8,000토큰의 공통 prefix를 사용하는 요청, 1시간 | 71~77% | 33~37% | 15~16% |
| 가변 길이 ShareGPT 유형 대화, 30분 | 13~16% | 24~37% | 1.7~2.0% |

*출처: [AWS 공식 벤치마크](https://aws.amazon.com/blogs/machine-learning/reduce-llm-latency-with-prefix-aware-routing-on-amazon-sagemaker-inference/). 여러 테스트 구성에서 보고한 범위이며, 이 글에서 재현한 측정값은 아닙니다.*

긴 공통 prefix를 반복하는 조건에서 P50 TTFT 개선 폭이 컸습니다. 하지만 같은 조건의 P90 개선 폭과 처리량 증가는 그보다 작았습니다. 발표 도입부에 요약된 최대 수치를 전체 응답 시간이나 모든 요청의 개선율로 바꾸어 읽으면 실제 기대 성능을 과대평가하게 됩니다.

비용도 별도로 확인해야 합니다. 지연 감소율이 곧바로 인스턴스 비용 감소율이 되지는 않습니다. 비용을 평가하려면 같은 서비스 목표를 만족하면서 필요한 인스턴스 수나 처리 가능한 요청량이 얼마나 달라지는지 측정해야 합니다.

## 우리 요청으로 검증할 항목

적용 검증에서는 모델과 컨테이너 버전, 인스턴스 구성, 요청 집합, 부하 조건을 고정하고 라우팅 전략을 바꿔 비교하는 방식이 적절합니다. 아래는 직접 측정할 때 사용할 수 있는 제안입니다.

- <strong>요청 집합:</strong> 긴 공통 prefix를 반복하는 요청, 매번 다른 문서를 넣는 요청, 특정 prefix에 요청이 집중되는 경우를 구분합니다.
- <strong>초기 상태:</strong> 캐시가 비어 있는 시작 구간과, 요청을 반복해 캐시가 채워진 구간을 나누어 기록합니다.
- <strong>지연:</strong> TTFT의 P50과 P90, 전체 응답 시간을 함께 봅니다. 캐시 적중률만으로 성능을 판단하지 않습니다.
- <strong>부하:</strong> 인스턴스 단위 지표를 수집할 수 있는지 먼저 점검합니다. 필요하면 계측을 추가하고, 처리 중 요청 수와 처리량, 오류를 함께 확인합니다.
- <strong>변경 상황:</strong> 인스턴스 증감과 prefix 변경 이후의 응답 지연을 따로 확인합니다.

결과를 읽는 순서도 중요합니다. Cache hit가 늘었는데 P90 TTFT가 줄지 않았다면, 다음으로 살펴볼 것은 요청의 집중과 대기 시간입니다. TTFT는 좋아졌지만 전체 응답 시간이 거의 같다면 decode가 차지하는 시간을 확인해야 합니다. 반복되는 입력이 거의 없다면 라우팅을 바꾸어도 재사용할 계산량 자체가 작을 수 있습니다.

Prefix-aware routing을 적용할 때는 실제 요청의 앞부분, 서버에 남아 있는 캐시, 각 서버의 부하를 함께 봐야 합니다. 이 세 조건을 측정하면 캐시를 켜고도 기대한 속도가 나오지 않는 이유를 더 구체적으로 찾을 수 있습니다.

## References

- AWS, [Reduce LLM latency with prefix-aware routing on Amazon SageMaker Inference](https://aws.amazon.com/blogs/machine-learning/reduce-llm-latency-with-prefix-aware-routing-on-amazon-sagemaker-inference/), 2026-09-10.
- Amazon SageMaker API Reference, [ProductionVariantRoutingConfig](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_ProductionVariantRoutingConfig.html), 2026-09-11 확인.
- Amazon SageMaker API Reference, [PrefixAwareRoutingConfig](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_PrefixAwareRoutingConfig.html), 2026-09-11 확인.
- vLLM 문서, [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), 2026-09-11 확인.
- vLLM 설계 문서, [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/), 2026-09-11 확인.
