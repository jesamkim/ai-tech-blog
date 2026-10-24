---
title: "Deep Insight로 살펴보는 에이전트 하네스 아키텍처 모범사례"
slug: "deep-insight-harness-architecture"
date: 2026-10-24T13:39:16+09:00
draft: false
author: "Jesam Kim"
description: "Deep Insight의 공개 설계를 바탕으로 계획과 승인, 컨텍스트 관리, 실행 격리, 검증과 복구를 연결하는 에이전트 하네스 아키텍처를 살펴봅니다."
categories: ["AWS AI/ML"]
tags: ["Deep Insight", "Harness Engineering", "Context Engineering", "Amazon Bedrock AgentCore", "Strands Agents"]
cover:
  image: "/ai-tech-blog/images/deep-insight-harness-architecture/cover.png"
  alt: "지능 코어와 격리된 실험 공간을 연결하는 구조물을 점검하는 기술자들"
  relative: false
---

에이전트가 보고서를 완성했다고 응답했습니다. 그런데 필수 분석 하나가 빠져 있고, 차트의 수치를 다시 계산할 근거가 남아 있지 않습니다. 재실행하려니 어떤 단계부터 시작해야 할지도 알기 어렵습니다. 모델의 답변 품질만 평가했다면 발견하기 어려운 문제입니다.

이런 상황을 가정하면 에이전트 아키텍처에서 확인할 대상이 분명해집니다. 작업의 시작과 완료를 누가 결정하는지, 중간 결과를 어디에 남기는지, 실패했을 때 무엇을 다시 실행하는지까지 살펴봐야 합니다. 이 글에서는 모델의 추론을 실제 작업으로 연결하고 통제하는 소프트웨어와 실행 환경을 <strong>하네스</strong>라고 부르겠습니다.

Deep Insight는 AWS Korea Solutions Architect(SA) 팀이 개발하고 오픈소스로 공개한 멀티 에이전트 기반 데이터 분석 솔루션입니다. CSV 파일과 자연어로 작성한 분석 요청을 입력하면, 여러 에이전트가 분석 계획 수립, 코드 실행, 수치 검증, 차트 생성과 보고서 작성을 나누어 수행합니다. AWS 공식 블로그에는 이 시스템의 실행 흐름, 컨텍스트 관리, 프로덕션 실행 환경, 대화형 분석을 다룬 글들이 공개되어 있습니다. 여기서는 그 내용을 하네스의 책임과 경계라는 관점으로 재구성합니다. 아래에 제시한 적용 방안은 공개 사례를 바탕으로 저자가 해석한 내용이며, AWS의 공식 표준 아키텍처를 뜻하지 않습니다.

<small>참조: [Deep Insight 공개 저장소](https://github.com/aws-samples/sample-deep-insight), [Deep Insight 아키텍처](https://aws.amazon.com/ko/blogs/tech/practical-design-lessons-from-the-deep-insight-arch/), [하네스 엔지니어링](https://aws.amazon.com/ko/blogs/tech/harness-engineering-from-deep-insight/)</small>

## 하네스의 책임을 먼저 나눕니다

하네스를 프롬프트와 도구 호출의 묶음으로만 보면 운영 중 생기는 문제를 놓치기 쉽습니다. 다음 표는 이 글에서 사용할 설계 관점입니다. Deep Insight의 공식 계층도를 그대로 옮긴 표는 아닙니다.

![하네스는 계획과 승인, 모델과 도구의 작업, 검증과 복구, 상태 보존을 연결합니다.](/ai-tech-blog/images/deep-insight-harness-architecture/harness-overview.png)

*계획에서 검증까지의 작업 흐름과 상태 보존을 함께 나타낸 개념도입니다. 본문의 설계 관점을 재구성했으며 공식 배포도를 뜻하지 않습니다.*

<small>참조: [하네스 엔지니어링](https://aws.amazon.com/ko/blogs/tech/harness-engineering-from-deep-insight/)</small>

| 설계 대상 | 하네스가 관리할 내용 | 빠졌을 때 나타나는 문제 |
|---|---|---|
| 계획과 승인 | 작업 범위, 승인한 계획, 다음 단계의 실행 조건 | 잘못 이해한 요청을 끝까지 실행 |
| 상태와 산출물 | 진행 상태, 파일 위치, 입력과 결과의 연결 | 대화가 길어지거나 세션이 끝나면 근거 소실 |
| 실행 환경 | 도구 권한, 자원 한도, 세션 생명주기 | 생성 코드의 오류가 다른 작업에 영향 |
| 검증과 복구 | 검증 증거, 실패 분류, 재실행 범위 | 잘못된 결과를 완료로 처리하거나 전체 재실행 |
| 사용자 경험 | 장시간 작업과 빠른 질의의 처리 방식 | 짧은 질문도 긴 분석을 기다림 |

이 책임을 반드시 서로 다른 서비스로 구현할 필요는 없습니다. 작은 시스템에서는 하나의 프로세스 안에서도 구분할 수 있습니다. 다만 상태를 바꾸는 주체와 접근 권한은 명시적으로 정해야 합니다.

## 계획은 실행과 검증이 공유하는 기준입니다

Deep Insight에서는 Coordinator가 요청을 분류하고, Planner가 계획을 작성하며, Plan Reviewer가 사용자 검토를 받습니다. Supervisor는 Coder, Validator, Reporter, Tracker를 도구처럼 호출합니다. 계획과 검토의 흐름에는 Graph 패턴을, 전문 에이전트 호출에는 Agents-as-tools 패턴을 사용합니다.

<small>참조: [Deep Insight의 멀티 에이전트 구조](https://aws.amazon.com/ko/blogs/tech/practical-design-lessons-from-the-deep-insight-arch/)</small>

이 구조에서 참고할 점은 에이전트의 수보다 <strong>실행 전에 작업 기준을 남긴다는 점</strong>입니다. 예를 들어 매출 분석 요청이라면 대상 기간, 취소 거래 처리, 매출의 정의를 계획에 기록할 수 있습니다. 실행자는 이 기준으로 계산하고, 검증자는 같은 기준으로 결과를 확인합니다. 사용자가 중간에 기간을 바꾸면 어느 결과부터 다시 만들어야 하는지도 판단하기 쉬워집니다.

승인 정책에는 별도 판단이 필요합니다. 공식 글의 계획 검토 구현은 300초 동안 응답이 없으면 자동 승인합니다. 따라서 검토 화면이 있다는 사실만으로 사람의 명시적 승인을 보장한다고 볼 수는 없습니다.

<small>참조: [계획 검토와 시간 초과 처리](https://aws.amazon.com/ko/blogs/tech/practical-design-lessons-from-the-deep-insight-arch/)</small>

읽기 전용 탐색과 외부 발송은 실패했을 때의 영향이 다릅니다. 외부 시스템을 변경하는 작업이라면 응답이 없을 때 대기하거나 중단하도록 정하고, 승인한 사용자와 계획 버전을 기록하는 편이 적절합니다. 승인 후 계획이 바뀌었을 때 재승인이 필요한 범위도 정해야 합니다.

## 컨텍스트를 줄이려면 산출물의 형식을 정해야 합니다

Deep Insight는 에이전트마다 컨텍스트를 분리하고, 공유 상태에는 계획과 압축된 결과를 전달합니다. 긴 분석 결과와 검증 정보는 파일로 남깁니다. 또한 각 스크립트가 필요한 데이터를 직접 불러오도록 해 이전 실행의 메모리 변수에 대한 의존을 줄입니다.

<small>참조: [Deep Insight의 Context Engineering](https://aws.amazon.com/ko/blogs/tech/context-engineering-from-deep-insight/)</small>

![다음 에이전트에는 요약과 파일 경로를 전달하고 상세 근거는 필요할 때 조회합니다.](/ai-tech-blog/images/deep-insight-harness-architecture/context-artifacts.png)

*요약과 파일을 통한 인계를 구분했습니다. 아래쪽의 입력 버전과 검증 상태 기록은 본문에서 제안하는 보완 사항입니다.*

<small>참조: [Context Engineering](https://aws.amazon.com/ko/blogs/tech/context-engineering-from-deep-insight/)</small>

이 방식은 토큰 절약과 함께 산출물 관리 문제를 만듭니다. 대화에 파일 경로만 남겼는데 파일이 덮어써졌거나 사라졌다면 다음 에이전트가 작업을 이어갈 수 없습니다. 따라서 파일 외부화를 적용할 때는 경로와 함께 입력 버전, 생성 단계, 검증 상태를 기록할 것을 제안합니다.

예를 들어 다음과 같은 인계 정보를 정의할 수 있습니다. 이는 Deep Insight의 실제 스키마를 복사한 것이 아닌 설명용 예시입니다.

| 인계 항목 | 설명용 값 | 목적 |
|---|---|---|
| 작업과 단계 | `job-123 / sales-summary` | 결과가 속한 작업 식별 |
| 입력 버전 | 원본 파일의 해시 또는 객체 버전 | 재계산할 입력 고정 |
| 산출물 위치 | 작업별 저장 경로 | 다음 단계에서 결과 조회 |
| 검증 상태 | `pending`, `passed`, `failed` | 생성과 검증 통과 구분 |
| 완료한 요구사항 | 승인한 계획의 항목 ID | 작업 누락 확인 |

모델에는 다음 판단에 필요한 요약을 주고, 시스템에는 재현에 필요한 근거를 남기는 방식입니다. 파일 저장 성공을 확인하기 전에 완료 상태를 기록하지 않는 것까지 포함해야 인계가 안정적으로 동작합니다.

## 추론과 코드 실행의 자원과 권한을 분리합니다

Deep Insight의 관리형 구성에서는 AgentCore Runtime이 에이전트 추론을 수행하고, 생성한 코드는 Fargate의 별도 실행 환경에서 처리합니다. 세션 관리 모듈은 Task 생성, 준비 상태 확인, 요청 연결, 종료를 관리합니다. 같은 세션의 작업은 컨테이너 파일을 공유하고, 공식 글에 설명된 구성에서는 세션 완료 시 산출물을 S3에 일괄 보관합니다.

<small>참조: [추론과 실행의 분리, S3 활용 구조](https://aws.amazon.com/ko/blogs/tech/harness-engineering-from-deep-insight/)</small>

![추론 환경은 실행을 요청하고, 세션별 실행 환경은 결과를 반환하며 완료 시 산출물을 보관합니다.](/ai-tech-blog/images/deep-insight-harness-architecture/reasoning-execution.png)

*추론, 코드 실행, 산출물 보관의 역할을 단순화했습니다. 실제 AWS 서비스 배치나 네트워크 경계를 나타내는 그림은 아닙니다.*

<small>참조: [추론과 실행의 분리](https://aws.amazon.com/ko/blogs/tech/harness-engineering-from-deep-insight/)</small>

이 설계를 적용할 때는 컨테이너 분리와 함께 권한도 검토해야 합니다. 실행 컨테이너가 다른 작업의 데이터에 접근할 수 있다면 프로세스가 분리되어 있어도 데이터 접근 범위는 넓게 남습니다. 세션 ID가 붙은 S3 경로는 파일을 구분하는 수단이며, 인증과 권한 검사를 대신하지 않습니다.

적용 환경에서는 입력을 읽는 권한, 결과를 쓰는 권한, 외부 네트워크 접근, 실행 시간과 메모리 한도를 각각 정해야 합니다. 공개 접근이 필요한 진입점에는 인증과 HTTPS를 적용하고, S3는 비공개로 유지합니다. 사용자가 제출한 작업 ID가 실제로 그 사용자의 작업인지 서버에서 확인하는 절차도 필요합니다.

복구 요구사항은 저장 시점에도 영향을 줍니다. 세션 마지막에만 보관하면 종료 직전 장애로 중간 결과를 잃을 수 있습니다. 재실행 비용이 큰 단계가 있다면 단계별 체크포인트를 추가하는 편이 적절합니다. 이는 원문 구현에 이미 있는 기능이라는 주장이 아닌 확장 제안입니다.

## 생성 완료와 검증 통과를 다른 상태로 관리합니다

Deep Insight의 Validator는 Coder가 남긴 계산 메타데이터를 바탕으로 소스 데이터에서 다시 계산하고 결과를 비교합니다. Reporter는 분석 결과와 검증 관련 파일을 참조해 보고서를 작성합니다.

<small>참조: [Coder, Validator, Reporter의 역할](https://aws.amazon.com/ko/blogs/tech/practical-design-lessons-from-the-deep-insight-arch/)</small>

재계산은 수치 오류를 찾는 데 도움이 됩니다. 다만 같은 잘못된 필터를 적용하면 두 계산이 일치할 수도 있습니다. 따라서 검증 항목은 계산의 일치 여부와 요청의 충족 여부를 구분해야 합니다. 앞의 매출 분석 예시라면 합계를 다시 구하는 검사와 취소 거래를 올바르게 제외했는지 확인하는 검사가 모두 필요합니다.

완료 조건을 어디서 집행하는지도 중요합니다. 공식 Context Engineering 글은 Supervisor의 섹션 완료 규칙을 프롬프트로 지시한다고 설명합니다. 체크리스트가 있다는 사실과 프로그램이 다음 단계 진입을 차단한다는 사실은 다릅니다.

<small>참조: [Supervisor의 섹션 완료 규칙](https://aws.amazon.com/ko/blogs/tech/context-engineering-from-deep-insight/)</small>

누락을 허용할 수 없는 요구사항에는 코드로 확인하는 조건을 함께 둘 것을 제안합니다. 필수 산출물 존재 여부와 검증 상태를 확인한 뒤에만 배포나 발송 단계를 허용하는 방식입니다. 의미 판단은 모델의 도움을 받더라도, 승인 기록과 필수 검증 결과가 없으면 실행을 막는 조건은 명확하게 구현할 수 있습니다.

![생성한 산출물은 검증을 거쳐야 완료로 처리하며 불일치하면 수정하거나 사용자에게 확인합니다.](/ai-tech-blog/images/deep-insight-harness-architecture/validation-recovery.png)

*본문에서 제안하는 완료 조건과 복구 흐름입니다. Deep Insight의 현재 구현을 그대로 옮긴 그림이 아닙니다.*

<small>참조: [검증과 완료 규칙을 다룬 Context Engineering](https://aws.amazon.com/ko/blogs/tech/context-engineering-from-deep-insight/)</small>

재시도 역시 실패 원인에 따라 달라야 합니다. 일시적인 실행 오류에는 제한된 재시도를 적용할 수 있지만, 데이터 정의가 불명확한 경우에는 사용자 확인이 필요합니다. 외부 쓰기 작업을 다시 실행할 때는 같은 요청이 중복 반영되지 않도록 멱등성도 고려해야 합니다.

## 긴 분석과 빠른 탐색에 맞춰 하네스를 구성합니다

Deep Insight의 대화형 분석 구성은 보고서를 만드는 작업과 별도 ECS 서비스로 분리됩니다. 챗에서는 세션별 DuckDB를 메모리에 유지하고, 모델이 만든 SQL을 실행하며 사용자가 SQL을 확인하고 수정할 수 있게 합니다. 데이터 적재 후에는 DuckDB의 외부 접근을 제한하고 설정 변경을 잠그는 방식을 설명합니다.

<small>참조: [Deep Insight Chatbot의 Inner Loop 설계](https://aws.amazon.com/ko/blogs/tech/inner-loop-engineering-deep-insight-chat-bot/)</small>

보고서 생성에서는 여러 단계의 완결성과 산출물 보존이 중요합니다. 반면 분석가가 조건을 바꿔 가며 질문하는 상황에서는 응답 지연과 세션 상태가 더 직접적으로 사용자 경험에 영향을 줍니다. 같은 데이터를 쓰더라도 작업의 성격에 맞춰 실행 수명과 복구 방식을 달리할 수 있습니다.

Text2SQL에서도 검증은 필요합니다. 데이터베이스가 계산을 수행해도 모델이 잘못된 집계 기준을 선택할 수 있습니다. SQL을 공개하면 사용자가 계산 과정을 검토할 수 있지만, SQL 공개 자체가 정답을 보장하지는 않습니다. 또한 DuckDB 연결 설정만으로 차트 코드 실행이나 애플리케이션의 다른 도구까지 안전해졌다고 해석해서는 안 됩니다.

메모리에 상태를 유지한다면 프로세스 재시작 시의 데이터 재적재, 후속 요청의 라우팅, 유휴 세션 정리도 함께 정해야 합니다. 빠른 응답을 위해 유지한 상태가 운영 중에는 관리해야 할 자원이 됩니다.

## 모범사례를 적용하는 순서

새 시스템에 처음부터 여덟 에이전트와 모든 인프라를 도입할 필요는 없습니다. 먼저 하나의 업무에서 실패할 수 있는 지점을 찾고, 해당 실패를 확인하고 제한할 장치를 추가하는 편이 낫습니다.

1. 성공 조건을 적습니다. 생성할 파일뿐 아니라 반드시 충족할 계산 기준과 승인 조건을 포함합니다.
2. 작업 상태를 남깁니다. 실행 중, 생성 완료, 검증 통과, 실패를 구분하고 산출물과 연결합니다.
3. 실행 권한을 제한합니다. 모델이 호출할 수 있는 도구와 실행 환경의 접근 범위를 함께 검토합니다.
4. 실패를 재현합니다. 승인 시간 초과, 검증 불일치, 컨테이너 종료, 중복 요청을 넣어 실제 동작을 확인합니다.
5. 측정 후 분리합니다. 지연, 비용, 권한 또는 컨텍스트 문제가 확인된 부분부터 에이전트와 실행 환경을 나눕니다.

Deep Insight에서 참고할 만한 설계는 계획, 파일, 실행 환경, 검증 결과가 서로 연결된다는 점입니다. 이 연결을 갖추면 결과가 잘못되었을 때 어떤 입력과 단계에서 문제가 생겼는지 조사할 수 있습니다. 하네스 아키텍처의 완성도는 정상 응답을 만드는 능력과 함께, 실패를 발견하고 통제하며 작업을 다시 이어갈 수 있는지로 평가해야 합니다.

## References

- [프로덕션 Multi-Agent 시스템이 해결해야 할 5가지 문제](https://aws.amazon.com/ko/blogs/tech/practical-design-lessons-from-the-deep-insight-arch/) (2026-04-01)
- [Context Window 한계를 넘어서: Deep Insight의 Context Engineering](https://aws.amazon.com/ko/blogs/tech/context-engineering-from-deep-insight/) (2026-04-08)
- [하네스 엔지니어링으로 본 Deep Insight](https://aws.amazon.com/ko/blogs/tech/harness-engineering-from-deep-insight/) (2026-04-22)
- [Inner Loop 엔지니어링으로 본 Deep Insight Chatbot](https://aws.amazon.com/ko/blogs/tech/inner-loop-engineering-deep-insight-chat-bot/) (2026-06-15)
- [5 production challenges in multi-agent systems and how Deep Insight solved them](https://aws.amazon.com/blogs/machine-learning/5-production-challenges-in-multi-agent-systems-and-how-deep-insight-solved-them/) (2026-10-21)
- [Deep Insight 공개 저장소](https://github.com/aws-samples/sample-deep-insight)
