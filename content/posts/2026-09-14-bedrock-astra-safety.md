---
title: "Amazon Bedrock의 GPT-6 Astra: 정렬 평가와 감시 가능성"
date: 2026-09-14T10:57:58+09:00
draft: false
slug: "bedrock-astra-safety"
author: "Jesam Kim"
description: "GPT-6 Astra의 정렬 평가와 감시 가능성을 구분하고, Amazon Bedrock에서 엔드포인트, 호출 로그, 데이터 보존 정책과 업무 권한을 어떻게 확인할지 정리합니다."
categories: ["AWS AI/ML"]
tags: ["Amazon Bedrock", "GPT-6 Astra", "AI Safety", "Alignment", "Monitorability", "Agentic AI", "Observability"]
cover:
  image: "/ai-tech-blog/images/bedrock-astra-safety/cover.png"
  alt: "해안 시험장에서 여러 광학 장비로 연구용 글라이더를 관측하는 장면"
  relative: false
---

Amazon Bedrock에서 GPT-6 Astra를 사용해 문서를 수정하거나 여러 도구를 실행하는 에이전트를 만든다고 생각해 보겠습니다. 모델이 지시를 잘 따르는지 확인하는 일과, 잘못된 행동이 발생했을 때 알아낼 수 있는지 확인하는 일은 서로 다릅니다. 허용한 파일만 수정하는 모델을 고르는 것과 실제 수정 대상을 기록하고 검사하는 작업을 함께 해야 합니다.

GPT-6 Astra는 2026년 9월 8일 Amazon Bedrock에서 정식 출시됐습니다. 앞서 공개된 OpenAI 안전성 평가는 정렬 평가 결과가 개선됐다고 보고하는 동시에, CoT 기반 감시 가능성이 상당히 낮아졌다고 밝혔습니다. Bedrock에서 이 모델을 운영할 때는 모델 평가 결과, 서비스의 보호 조치, 애플리케이션이 남기는 기록을 구분해서 읽을 필요가 있습니다.

이 글은 OpenAI 시스템 카드와 AWS 공식 문서를 2026년 9월 14일 기준으로 대조한 내용입니다. OpenAI의 실험을 Bedrock 운영 환경에서 직접 측정한 결과로 제시하지 않습니다. 기존 [Trajectory Assurance 글](/ai-tech-blog/posts/trajectory-assurance-agent-security/)에서 행동 이력을 이용한 통제를 살펴봤다면, 이번에는 감시 결과를 해석하는 기준과 Bedrock에서 실제로 수집할 수 있는 자료에 집중합니다.

## 정렬과 감시 가능성은 서로 다른 평가 대상

정렬(alignment)은 모델이 의도한 지시와 경계를 따르는지에 관한 문제입니다. 감시 가능성(monitorability)은 특정 관측 자료를 받은 감시자가 평가 대상의 행동이나 속성을 얼마나 잘 구분할 수 있는지에 관한 문제입니다. 모델이 잘 따르도록 학습하는 것과 감시자가 잘 알아내도록 만드는 것은 같은 작업이 아닙니다.

| 확인하려는 것 | 필요한 근거 |
|---|---|
| 모델이 허용 범위를 지키는지 | 정상 작업과 경계 위반 시나리오의 수행 결과 |
| 감시자가 위반을 알아내는지 | 정답이 알려진 사례에서의 탐지와 오탐 결과 |
| 실제 작업에서 경보가 얼마나 발생했는지 | 관측 범위와 감시 설정이 명확한 운영 기록 |

경보 수는 행동의 발생 빈도뿐 아니라 감시자가 볼 수 있는 자료와 탐지 기준에도 영향을 받습니다. 따라서 경보가 줄었다는 사실만으로 실제 위반이 같은 비율로 줄었다고 판단하기는 어렵습니다. 운영 기록을 비교할 때는 모델 변경 외에 감시자와 입력 자료가 바뀌었는지도 함께 확인해야 합니다.

## 시스템 카드의 수치와 관측 조건

시스템 카드 8.6절은 과거 내부 Codex 작업 54,218건에서 동일한 문맥을 구성하고, 두 모델로 마지막 턴을 다시 생성한 시뮬레이션을 설명합니다. 심각도 3 이상으로 분류된 플래그는 다음과 같습니다.

| 모델 | 플래그 수 | 54,218건 대비 비율 |
|---|---:|---:|
| GPT-5.6 Sol | 73 | 0.135% |
| GPT-6 Astra | 34 | 0.063% |

이 값은 감시자가 표시한 비율입니다. 독립적으로 확정한 사고율이나 Bedrock 고객의 운영 사고율로 해석해서는 안 됩니다.

비적대적 평가(9.1절)에서 Astra의 CoT 기반 감시 가능성은 대부분의 CoT 길이에서 GPT-5.6 Sol보다 낮았지만, 긴 CoT에서는 대체로 회복됐습니다. 전체 문맥을 본 조건에서는 대부분의 출력 길이에서 낮았고, 짧은 출력과 가장 긴 출력에서도 같은 결과였습니다. 행동만 본 조건에서는 Astra가 높았습니다. 연구진은 짧고 정보가 적은 CoT, 추론을 말로 드러내는 경향과 필요성 감소를 주요 원인으로 분석했습니다. 긴 CoT에서도 차이가 남은 것은 일부 환경이었습니다.

[![OpenAI Figure 22: CoT, 전체 메시지, 행동을 관측하는 조건별 감시 가능성과 출력 길이의 관계](/ai-tech-blog/images/bedrock-astra-safety/monitorability-original.png)](/ai-tech-blog/images/bedrock-astra-safety/monitorability-original.png)
*OpenAI 시스템 카드 Figure 22 원본입니다. 그림을 누르면 원본 크기로 볼 수 있습니다. CoT-only 패널의 가로축은 CoT 토큰 수이고, 나머지 패널은 출력 토큰 수입니다. 세로축 g-mean²는 참양성률과 참음성률을 반영합니다. 감시자는 GPT-5.6 Sol이며, Bedrock 운영 측정값은 아닙니다. 출처: [시스템 카드 9.1절](https://deploymentsafety.openai.com/gpt-6-astra/monitorability-under-non-adversarial-conditions).*

9.2절의 sandbagging 평가는 성능 저하를 지시하고, 감시의 존재를 알리는지와 회피를 지시하는지에 따라 조건을 나눴습니다. 자발적인 운영 행동의 발생률을 뜻하지 않습니다. 9월 9일 정정은 평가를 의식한 발언과 평가를 훼손하는 행동을 구분하며, 실패 미관측을 일반적인 신뢰성 보장으로 확대하지 말라고 설명합니다. 출처: [GPT-6 Astra 시스템 카드](https://deploymentsafety.openai.com/gpt-6-astra).

## Bedrock 서비스의 보호 조치와 고객의 기록

AWS의 Astra 출시 글은 실시간 오용 감시와 경계를 벗어난 활동의 중단을 위한 자동 보호 조치가 Amazon Bedrock 서비스 경계 안에서 동작한다고 설명합니다. Astra는 OpenAI의 Preparedness Framework에서 사이버보안 역량이 Critical로 분류된 첫 모델로 소개됩니다. 여기서 Critical은 사이버보안 역량에 대한 분류입니다. 추론 데이터가 모델 학습에 사용되지 않으며, Astra를 사용하기 위해 OpenAI와의 데이터 공유에 동의할 필요도 없다고 안내합니다.

이 보호 조치와 애플리케이션의 업무 규칙은 적용 범위가 다릅니다. 서비스가 제공하는 보호 조치만으로 특정 문서의 수정 권한, 사내 승인 단계, 취소 가능한 거래 범위가 모두 정의되지는 않습니다. 애플리케이션은 자신이 실행할 작업의 대상과 권한을 별도로 확인해야 합니다. 이 부분은 서비스의 자동 감시 설명을 바탕으로 정리한 운영상의 판단입니다.

또한 시스템 카드의 감시자가 본 자료와 고객 로그를 같은 것으로 취급해서는 안 됩니다. 모델 호출의 입출력을 기록했다고 해서 그림의 CoT-only 평가에 사용한 내부 자료를 그대로 얻거나 같은 감시 성능을 재현할 수 있다고 단정할 수는 없습니다. 출처: [Bedrock의 GPT-6 Astra 출시 글](https://aws.amazon.com/blogs/machine-learning/take-on-your-most-ambitious-work-with-gpt-6-astra-on-amazon-bedrock/).

## 엔드포인트와 API를 함께 선택하기

Bedrock에서 모델 이름만 정하면 운영 기능의 선택까지 끝나는 것은 아닙니다. Astra 모델 카드와 엔드포인트 문서를 함께 확인해야 합니다.

| 필요한 기능 | Astra에서 확인할 조건 |
|---|---|
| Responses 또는 Chat Completions | `bedrock-runtime`과 `bedrock-mantle`에서 지원 |
| 모델 호출에 통합된 Guardrails | `bedrock-runtime`의 Converse API에서 지원 |
| Application inference profile | Converse API에서 지원하며, Responses와 Chat Completions에서는 미지원 |
| 서버 측 도구 실행 | `bedrock-mantle`에서 지원, `bedrock-runtime`에서는 미지원 |
| 관리형 모델 호출 로그 | `bedrock-runtime`에서 지원하며 별도 활성화 필요 |

Guardrails 행은 모델 호출에 통합된 지원 범위입니다. 독립적인 [ApplyGuardrail](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html) 호출을 포함한 별도 애플리케이션 설계 전체의 가능 여부를 뜻하지 않습니다. 또한 서버 측 도구 실행과 클라이언트가 도구를 실행하는 방식도 구분해야 합니다. 엔드포인트 문서는 클라이언트 측 도구 실행을 양쪽에서 지원한다고 설명합니다.

예를 들어 클라이언트가 도구 실행을 담당하면서 호출 입출력 로그를 수집하려면 `bedrock-runtime`의 지원 범위를 먼저 검토할 수 있습니다. 반대로 서버 측 도구가 필요해 `bedrock-mantle`을 선택한다면, 같은 호출 로그 수집 방식을 그대로 사용할 수 있는지 가정하지 말고 필요한 감사 기록을 따로 확인해야 합니다. 출처: [Astra 모델 카드](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html), [엔드포인트별 지원 기능](https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html).

## 호출 로그와 도구 실행 기록을 연결하기

Model invocation logging은 기본적으로 꺼져 있습니다. 활성화하면 지원되는 호출의 요청, 응답과 메타데이터를 CloudWatch Logs 또는 S3에 저장할 수 있으며, 저장 대상은 같은 계정과 리전에 있어야 합니다. 문서는 `bedrock-runtime`의 OpenAI 호환 Responses와 Chat Completions도 포함하지만, `bedrock-mantle` 호출은 현재 수집하지 않는다고 명시합니다.

CloudTrail도 별도로 확인해야 합니다. `bedrock-mantle`의 추론은 데이터 이벤트로 분류되며 기본 수집 대상이 아닙니다. 이 엔드포인트에서 추론 호출의 감사 기록이 필요하다면 데이터 이벤트 수집을 명시적으로 설정해야 합니다. 반면 Astra를 `bedrock-runtime`의 Converse로 호출하면 관리 이벤트로 기본 기록됩니다. `bedrock-mantle`을 쓴다면 “CloudTrail과 연동된다”는 문장만 보고 필요한 추론 이벤트가 이미 쌓이고 있다고 판단하면 안 됩니다.

애플리케이션 관점에서는 모델 호출 기록과 실제 도구의 실행 결과를 연결하는 것이 유용합니다. 다음은 문서 수정 에이전트에 남길 수 있는 기록의 예시입니다.

| 단계 | 확인할 내용 |
|---|---|
| 요청 접수 | 어떤 작업과 대상이 허용됐는지 |
| 모델 호출 | 사용한 모델과 엔드포인트, 호출을 식별할 정보 |
| 실행 전 검사 | 제안된 수정이 허용 범위 안인지, 필요한 확인을 받았는지 |
| 도구 실행 | 실제 수정 대상, 성공 또는 거부 결과 |
| 완료 확인 | 사용자가 요청한 결과가 저장됐는지, 의도하지 않은 변경이 없는지 |

이 표는 애플리케이션의 기록 항목을 제안한 것이며, Bedrock이 모든 항목을 자동으로 남긴다는 뜻은 아닙니다. 특히 클라이언트에서 실행한 도구의 외부 동작까지 모델 입출력 로그만으로 확인할 수 있다고 가정해서는 안 됩니다. 로그에는 업무에 필요한 식별자를 사용하고, 비밀 정보나 불필요한 본문 전체를 메타데이터에 넣지 않도록 해야 합니다. 출처: [모델 호출 로깅](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html), [bedrock-mantle의 CloudTrail 기록](https://docs.aws.amazon.com/bedrock/latest/userguide/logging-cloudtrail-mantle.html).

## 데이터 보존은 별도 조건으로 확인하기

Astra 출시 글은 분류기가 표시한 트래픽을 AWS가 최대 30일 보존해 자동 오용 탐지에 처리한다고 설명합니다. 필요한 경우 ZDR을 요청할 수 있다는 안내도 있습니다. 이 내용을 “모든 트래픽이 30일 보존된다”거나 “계정에서 옵션만 바꾸면 항상 ZDR로 Astra를 사용할 수 있다”로 바꾸어 읽으면 안 됩니다.

데이터 보존 문서는 계정과 프로젝트 설정, 모델의 요구 조건을 함께 확인하도록 합니다. Responses API의 `store=false`만으로 무보존이 보장되지는 않습니다. 보존이 필요한 모델에 `none` 정책을 적용하면 요청이 차단될 수 있고, ZDR 접근은 계정과 모델별로 검토됩니다.

운영 전에는 서비스 측 보존 조건과 직접 구성한 로그의 저장 위치, 접근 권한, 보존 기간을 각각 확인하는 편이 좋습니다. 감시를 위해 자료를 모으더라도 필요한 범위를 정하고 로그 저장소 접근을 제한해야 합니다. 출처: [Astra 출시 글](https://aws.amazon.com/blogs/machine-learning/take-on-your-most-ambitious-work-with-gpt-6-astra-on-amazon-bedrock/), [Bedrock 데이터 보존 문서](https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html).

## 경보가 없는 경우까지 포함한 검증

운영 검증에는 작업의 성공 여부와 감시자의 판단을 함께 남길 수 있습니다. 다음은 별도 테스트 환경에서 사용할 수 있는 시나리오 제안입니다. 이 글에서 직접 실행한 Bedrock 벤치마크는 아닙니다.

- <strong>정상 작업:</strong> 허용한 문서를 수정하고 완료하는지 확인합니다. 불필요한 차단이나 경보가 발생하는지도 기록합니다.
- <strong>권한 밖의 작업:</strong> 허용하지 않은 대상에 대한 수정이 실행 전에 거부되는지 확인하고, 거부 결과가 작업 기록에 남는지 봅니다.
- <strong>오류를 넣은 테스트:</strong> 정답이 알려진 사례로 탐지와 누락을 확인합니다. 정상 사례의 오탐도 함께 셉니다.
- <strong>모델이나 API 변경:</strong> 기존 시나리오를 다시 실행하고, 호출 로그와 도구 결과가 같은 방식으로 연결되는지 확인합니다.

경보가 발생하지 않은 요청은 정말 정상일 수도 있고, 필요한 자료가 기록되지 않았거나 감시자가 놓친 경우일 수도 있습니다. 이를 구분하려면 경보 목록뿐 아니라 실행 결과와 수집 상태를 확인할 수 있어야 합니다. 운영 전에는 선택한 Bedrock 호출 방식에서 필요한 기록이 실제로 남고, 업무 권한 밖의 동작이 실행 전에 거부되는지 확인해야 합니다.

## References

- AWS, [OpenAI GPT-6 Astra is now generally available on Amazon Bedrock](https://aws.amazon.com/about-aws/whats-new/2026/09/openai-gpt-6-astra-on-amazon-bedrock/), 2026-09-08.
- OpenAI, [GPT-6 Astra System Card](https://deploymentsafety.openai.com/gpt-6-astra), 2026-09-03 공개, 2026-09-09 변경 사항 확인.
- AWS, [Take on your most ambitious work with GPT-6 Astra on Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/take-on-your-most-ambitious-work-with-gpt-6-astra-on-amazon-bedrock/), 2026-09-08.
- Amazon Bedrock, [GPT-6 Astra 모델 카드](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-openai-gpt-6-astra.html), 2026-09-14 확인.
- Amazon Bedrock, [Endpoints supported by Amazon Bedrock](https://docs.aws.amazon.com/bedrock/latest/userguide/endpoints.html), 2026-09-14 확인.
- Amazon Bedrock, [Model invocation logging](https://docs.aws.amazon.com/bedrock/latest/userguide/model-invocation-logging.html), 2026-09-14 확인.
- Amazon Bedrock, [Monitor bedrock-mantle API calls using CloudTrail](https://docs.aws.amazon.com/bedrock/latest/userguide/logging-cloudtrail-mantle.html), 2026-09-14 확인.
- Amazon Bedrock, [Data retention](https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html), 2026-09-14 확인.
- Amazon Bedrock, [Use the ApplyGuardrail API in your application](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-use-independent-api.html), 2026-09-14 확인.
