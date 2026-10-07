---
title: "Strands Decider 2B와 Jev 비교: 공개 구조와 A10G 실험"
slug: "strands-decider-vs-jev"
date: 2026-10-07T15:52:32+09:00
draft: false
author: "Jesam Kim"
description: "AWS Strands Decider 2B의 공개 구조와 운영 조건을 Jev와 비교합니다. A10G에서 v19와 v21을 실행해 영어와 한국어 분류, 질문 반전, 입력 길이에 따른 지연을 확인합니다."
categories: ["AWS AI/ML"]
tags: ["Strands Agents", "Strands Decider", "Jev", "System One", "AI Agent", "Calibration"]
cover:
  image: "/ai-tech-blog/images/strands-decider-vs-jev/cover.png"
  alt: "작은 철도 분기 장치의 내부 회로를 살펴보는 기술자와 멀리 연결된 관제소"
  relative: false
---

에이전트가 도구를 호출하기 전에는 작은 판단이 반복됩니다. 문의를 어느 담당자에게 보낼지, 도구 인자를 사용자가 실제로 말했는지, 추가 질문이 필요한지 결정해야 합니다. 이런 단계에는 긴 답변보다 프로그램이 바로 읽을 선택지와 확률이 유용합니다.

AWS Strands 팀은 2026년 10월 1일, 이런 용도를 위한 <strong>Strands Decider 2B</strong>를 공개했습니다. 로컬 CPU나 GPU에서 실행할 수 있고, 모델 구성과 학습 방법을 살펴보며 수정할 수 있는 판단 모델입니다. TypeSafe의 Jev와 같은 종류의 문제를 다루지만, 개발자가 직접 운영하는 범위는 상당히 다릅니다.

<small>참조: [Strands Decider 발표](https://strandsagents.com/blog/introducing-strands-decider/)</small>

[앞선 Jev 글](/ai-tech-blog/posts/jev-system-one-deep-dive/)에서는 입력 자료인 state와 Choice, Noul, Score를 설명했습니다. 이번에는 Decider의 내부 구조, Jev API를 대체할 때 확인할 차이, 직접 실행한 결과에 집중합니다. 문서와 모델 정보는 2026년 10월 7일 기준입니다.

## Decider가 공개한 것은 판단 모델의 내부입니다

Decider는 사전 학습한 `Qwen3.5-2B-Base`를 바탕으로 만듭니다. 일반적인 언어 모델은 입력을 처리한 뒤 LM head에서 다음 토큰을 고릅니다. Decider는 이 출력부를 제거하고 <strong>주어진 선택지에 점수를 매기는 pointer head</strong>를 붙입니다. 모델 본체는 rank-16 LoRA로 조정합니다.

입력에는 상황, 질문, 선택지 설명이 함께 들어갑니다. 모델이 이 입력을 처리하면 각 위치에 문맥을 반영한 벡터가 생깁니다. Pointer head는 답변 위치의 벡터와 각 선택지 끝의 벡터를 비교해 점수를 구하고, softmax로 선택지별 분포를 만듭니다. 답변 문장을 한 토큰씩 생성하는 반복 과정은 없습니다.

<small>참조: [공개 아키텍처 설명](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/docs/architecture.md)</small>

[![일반 언어 모델의 LM head를 제거하고 선택지와 답변 위치의 벡터를 비교하는 pointer head를 붙인 구조입니다.](/ai-tech-blog/images/strands-decider-vs-jev/architecture-original.svg)](/ai-tech-blog/images/strands-decider-vs-jev/architecture-original.svg)

*공식 저장소의 원본 구조도입니다. 왼쪽은 일반적인 언어 모델, 오른쪽은 선택지에 점수를 매기는 Decider입니다. 그림에서 사용하는 Hobson은 체크포인트 이름에도 쓰입니다. 그림을 누르면 크게 볼 수 있습니다.*

<small>참조: [Strands Decider 원본 그림, Apache-2.0](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/research/figures/architecture.svg)</small>

이 방식에서는 분류할 이름과 설명을 요청할 때 전달할 수 있습니다. 미리 학습한 고정 범주의 번호만 반환하는 분류기와 차이가 있습니다. 다만 새 범주를 입력할 수 있다는 것과 그 범주를 정확하게 구분한다는 것은 별개의 조건입니다. 공개 평가 문서도 낯선 yes/no 질문과 새로운 평가 기준에서 성능이 떨어질 수 있다고 설명합니다.

<small>참조: [구조와 동적 선택지](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/docs/architecture.md), [평가와 알려진 한계](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/evaluation/README.md)</small>

## Jev와 비교하면 운영 책임이 달라집니다

| 비교 항목 | Strands Decider | Jev |
|---|---|---|
| 사용 방식 | 공개 모델을 내려받아 로컬 또는 자체 인프라에서 실행 | TypeSafe의 관리형 API 호출 |
| 모델 공개 범위 | 베이스 모델, LoRA, 출력부, 학습 레시피와 데이터 출처 확인 가능 | 확인한 공식 자료에서 가중치와 상세 구조의 공개 자료를 찾지 못함 |
| 수정 방법 | 모델과 서빙 코드를 수정하고 재학습 가능 | 요청의 state, 질문과 선택지 설명을 조정 |
| 이 글의 비교 버전 | 발표 당시 v19와 이후 공개된 v21 | 공식 문서의 `jev-1.13.0` |
| 텍스트 길이 | 이번 실험은 4,096토큰 윈도 사용 | 요청 전체 64k, state와 가장 긴 질문의 합 32k |
| 비용을 계산할 기준 | 장비 이용 시간, 실제 처리량, 유휴 시간, 운영 비용 | 입력 100만 토큰당 0.042달러, 출력 무료 |
| 운영자가 맡을 부분 | 배포, 인증, 용량, 장애 대응, 모델 업데이트 | API 연동, 한도와 오류 처리, 버전 및 품질 관리 |

<small>참조: [Decider 모델 카드](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21), [서빙 문서](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/docs/inference.md), [Jev 모델과 가격](https://docs.typesafe.ai/models), [Jev 학습 개념 설명](https://docs.typesafe.ai/introduction/machine-learning-primer)</small>

Decider는 판단 단계의 입력을 자체 환경 안에서 처리하도록 구성할 수 있습니다. 그렇다고 전체 에이전트의 데이터가 자동으로 내부에 머무르는 것은 아닙니다. 답변을 만드는 LLM이나 외부 도구를 별도로 호출한다면 그 단계의 데이터 흐름도 확인해야 합니다. 반대로 관리형 API를 사용하면 모델을 설치하고 GPU 용량을 맞추는 작업을 서비스 제공자에게 맡길 수 있습니다.

공개 모델을 무료로 내려받는다는 사실만으로 요청당 비용이 Jev보다 낮다고 결론낼 수는 없습니다. 같은 장비에서도 사용률과 요청 크기에 따라 비용이 달라집니다. 이 글의 EC2 실험은 이미 실행 중인 GPU를 사용하므로 새 인스턴스를 생성하지 않았지만, 장비 이용 비용까지 없어지는 것은 아닙니다.

Jev도 SDK와 워크플로 평가 코드를 공개합니다. TypeSafe의 `WorkflowEvals`를 통해 API 기반 평가를 재현하는 것과, Decider의 가중치와 학습 코드를 이용해 모델 자체를 재현하는 것은 공개 범위가 다릅니다.

<small>참조: [TypeSafe WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals/tree/0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb), [Decider 학습 레시피](https://github.com/strands-labs/strands-decider/tree/3e94e9d84c620ed5a95f1a3310c3decb971e261c/training)</small>

## 응답 형태가 비슷해도 교체 전에 확인할 내용이 있습니다

두 모델은 `state`와 `questions`를 받고 Choice, Noul, Score를 반환하는 형태가 비슷합니다. Decider의 로컬 서버도 `/v1/systemone`을 제공합니다. 하지만 공개 서빙 문서는 <strong>Jev API 자체와의 호환성을 검증하지 않았다고 명시</strong>합니다. JevBench의 어댑터로 평가가 실행됐다는 사실을 SDK 전체의 교체 호환성으로 확대하면 안 됩니다.

긴 입력을 처리하는 방식도 살펴봐야 합니다. Decider는 기본적으로 입력이 윈도를 넘으면 질문의 선택지를 보존하면서 state를 줄입니다. `--strict-window`를 사용하면 대신 오류로 거절합니다. 원문 일부가 잘려도 응답이 나온다는 점은 정책 문서나 긴 대화를 분류할 때 특히 중요합니다.

<small>참조: [HTTP 서버와 입력 윈도 동작](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/docs/inference.md#serve)</small>

`confidence`라는 필드명도 뜻을 확인해야 합니다. Choice에서는 두 구현 모두 가장 높은 선택지 확률을 균등분포 기준으로 정규화합니다. 예를 들어 선택지가 3개이고 가장 높은 확률이 0.6이면 confidence는 0.4입니다. 따라서 confidence 0.9를 곧바로 “이 답이 맞을 확률 90%”로 읽을 수 없습니다.

Score의 계산법에는 차이가 있습니다. Jev 문서는 가장 가능성이 높은 수준으로부터의 평균 절대거리를 사용합니다. Decider는 표준편차를 이용하며 학습 시 적용한 ordinal smoothing도 보정합니다. <strong>기존 Score 임계값을 같은 숫자로 복사하면 처리 결과가 달라질 수 있습니다.</strong> 모델을 바꿀 때에는 업무 데이터로 임계값을 다시 평가해야 합니다.

<small>참조: [Jev confidence 계산식](https://docs.typesafe.ai/confidence), [Decider의 Choice와 Score 계산법](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/docs/architecture.md#one-mechanism-three-primitives)</small>

## 공개 성능 수치는 버전과 조건을 함께 봐야 합니다

발표 글의 모델은 v19이며, 10월 7일 확인한 공식 저장소에는 v21도 안내돼 있습니다. 아래는 각 모델 카드에 기록된 <strong>공개 배포 파일의 JevBench 평가</strong>입니다.

| 체크포인트 | 공개 과제 정답 수 | 정확도 | Brier score | 윈도 |
|---|---:|---:|---:|---:|
| v19 | 167 / 231 | 72.3% | 0.348 | 4,096 |
| v21 | 176 / 231 | 76.2% | 0.323 | 4,096 |

<small>참조: [v19 모델 카드의 Results](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19), [v21 모델 카드의 Results와 선택 방법](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)</small>

Brier score는 예측한 확률과 실제 정답의 차이를 평가하며, 이 표에서는 낮을수록 좋습니다. 다만 표의 두 줄은 각각 한 체크포인트의 결과입니다. v21 모델 카드는 별도 호스트에서 두 학습 레시피를 각각 6개 시드로 반복했을 때 평균 정답 수가 v19 172.8개, v21 설정 172.3개였다고 기록합니다. 카드 자체도 두 배포본의 차이를 그대로 정확도 향상으로 해석하지 말라고 설명합니다. v19의 레시피 평가 문서와 배포 파일의 모델 카드에도 서로 다른 측정값이 있으므로 출처를 섞지 않았습니다.

<small>참조: [v21 체크포인트 선택과 반복 실험](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21#how-this-checkpoint-was-chosen)</small>

이 표에 Jev 행을 억지로 추가하지 않은 이유도 같습니다. Decider가 사용하는 JevBench 공개 과제와 TypeSafe의 공식 `WorkflowEvals`는 다른 평가입니다. 후자는 네 업무 흐름에서 참조 모델의 결과와 얼마나 일치하는지 평가합니다. 이름이 비슷하거나 모두 정확도라고 표시돼도 숫자를 같은 척도로 취급할 수 없습니다.

<small>참조: [Decider의 JevBench 평가 범위](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/evaluation/jevbench.md), [TypeSafe 공식 평가](https://evals.typesafe.ai/)</small>

발표 글의 RTX 3090 중앙값 약 115ms도 측정 조건이 있는 수치입니다. 같은 글의 지연 그래프는 v18 결과라고 적혀 있습니다. 이를 아래 A10G 실험이나 Jev의 API 지연과 합쳐 모델 간 속도 순위를 만들지는 않았습니다.

<small>참조: [발표 글의 지연 설명과 Figure 3](https://strandsagents.com/blog/introducing-strands-decider/)</small>

## A10G에서 직접 확인한 결과

기존 EC2의 NVIDIA A10G 24GB 한 대에서 v19와 v21을 실행했습니다. BF16, 4,096토큰 윈도, 요청당 질문 하나를 사용했습니다. 소스 커밋과 Hugging Face의 모델 revision을 고정했고, Python 3.13.12, PyTorch 2.10.0, Transformers 5.17.0 환경에서 FLA와 causal-conv1d의 최적화 구현이 선택됐는지 확인했습니다. Jev API는 이번에 실행하지 않았습니다.

<strong>입력과 정답, 판정 기준은 첫 추론 전에 고정했습니다.</strong> 분류 실험에는 결제, 기술 지원, 영업 문의를 각각 4개씩 만들고 영어와 한국어로 같은 뜻을 적었습니다. 총 24개 입력이며 질문과 선택지 설명은 영어로 유지했습니다. 따라서 이 실험은 한국어 state를 읽는지 확인하는 범위입니다.

별도로 언어마다 환불을 요청하는 문장 3개와 요청하지 않는 문장 3개를 만들었습니다. 각 문장에 “환불을 요청하는가?”와 “환불을 요청하지 않는가?”를 따로 물었습니다. 두 질문을 모두 맞힌 경우에만 한 쌍을 맞혔다고 셌습니다. Noul의 yes 확률이 0.5 이상이면 yes로 판정했으며, 결과를 본 뒤 기준을 바꾸지 않았습니다.

| 확인한 작업 | v19 | v21 |
|---|---:|---:|
| 영어 문의의 담당 분류 | 12 / 12 | 12 / 12 |
| 한국어 문의의 담당 분류 | 12 / 12 | 12 / 12 |
| 영어의 일반 질문과 부정 질문을 모두 맞힌 쌍 | 6 / 6 | 6 / 6 |
| 한국어의 일반 질문과 부정 질문을 모두 맞힌 쌍 | 6 / 6 | 5 / 6 |

<small>참조: [직접 실행한 입력, 원시 응답, 집계와 재현 스크립트](/ai-tech-blog/downloads/strands-decider-vs-jev/experiment.zip)</small>

두 모델 모두 단순한 담당 분류는 맞혔습니다. 이 12개 문의는 제가 만든 짧고 명확한 문장이므로, 한국어 업무 전반의 정확도나 실제 고객 문의의 자동 처리율을 추정할 수는 없습니다.

v21에서 정답과 달랐던 입력은 “환불을 요청하는 것이 아닙니다. 비밀번호 재설정을 도와주세요.”였습니다. “고객이 환불을 요청하고 있나요?”에는 yes 확률 0.0739를 반환했습니다. 반대로 “고객이 환불을 요청하지 않고 있나요?”에는 0.4583을 반환해, 미리 정한 0.5 기준에서 no로 분류됐습니다. 같은 부정 질문에 v19는 0.6145를 반환했습니다.

이 한 사례로 v19가 전반적으로 더 낫다고 결론낼 수는 없습니다. 다만 최신 버전으로 바꿀 때 기존 질문의 판정이 유지되는지는 확인해야 합니다. 서로 반대인 두 질문의 확률이 정확히 보완 관계를 이룬다고 가정하는 것도 피해야 합니다. 이번 실험은 확률 보정이나 운영 임계값을 평가할 만큼 크지 않습니다.

<small>참조: [실험 원시 응답의 negation-05-ko 항목](/ai-tech-blog/downloads/strands-decider-vs-jev/experiment.zip)</small>

### 짧은 입력은 약 62-67ms, 긴 입력은 약 265ms였습니다

지연은 내용 이해 실험과 분리했습니다. 같은 영어 문장을 반복해 세 길이의 입력을 만들고, 각 길이에서 3회 워밍업 후 20회씩 순차 실행했습니다. 아래 토큰 수에는 엔진이 렌더링한 state와 질문이 모두 포함됩니다. 모델 엔진 호출 전후에 CUDA 동기화를 적용했으며 HTTP, 네트워크, 동시 요청의 대기 시간은 포함하지 않았습니다.

| 실제 입력 토큰 수 | v19 중앙값 / p95 | v21 중앙값 / p95 |
|---|---:|---:|
| 132 | 67.3 / 67.5ms | 62.1 / 62.3ms |
| 510 | 70.6 / 70.7ms | 69.7 / 69.7ms |
| 2,049 | 266.2 / 266.3ms | 264.5 / 264.6ms |

<small>참조: [실험의 raw.jsonl 및 summary.json](/ai-tech-blog/downloads/strands-decider-vs-jev/experiment.zip)</small>

[![A10G에서 두 모델의 워밍업 후 중앙값은 132토큰에서 약 62-67ms, 510토큰에서 약 70ms, 2,049토큰에서 약 265ms였습니다.](/ai-tech-blog/images/strands-decider-vs-jev/a10g-latency.png)](/ai-tech-blog/images/strands-decider-vs-jev/a10g-latency.png)

*직접 측정한 중앙값입니다. 각 길이에서 같은 합성 입력을 20회 반복한 결과이며, 긴 문서의 이해 능력이나 운영 서비스의 지연을 평가한 그래프는 아닙니다.*

<small>참조: [직접 실행한 원시 기록과 측정 방법](/ai-tech-blog/downloads/strands-decider-vs-jev/experiment.zip)</small>

이번 실행에서는 두 버전의 지연이 비슷했고 입력이 길어지면 처리 시간이 늘었습니다. 실행 순서는 v19 다음 v21이었으며, 순서를 바꿔 여러 번 반복하는 통제 실험은 하지 않았습니다. 20회의 p95는 이 실행에서 관찰한 값으로, 운영 환경의 지연 상한을 뜻하지 않습니다.

<strong>첫 요청은 워밍업 후와 크게 달랐습니다.</strong> v19의 모델 로드는 약 3.65초, 로드 후 첫 요청은 초기 커널 컴파일 등을 포함해 약 33.07초였습니다. 이후 별도 프로세스에서 실행한 v21은 첫 요청에 약 0.95초가 걸렸지만, v19가 만든 디스크의 컴파일 캐시를 재사용할 수 있는 상태였습니다. 두 값을 버전별 콜드 스타트 성능으로 비교할 수는 없습니다. 짧은 요청의 지연을 활용하려면 모델을 상주시킬지, 시작 시 워밍업할지까지 운영 조건에 넣어야 합니다.

### 재현 자료에 포함한 것

[실험 자료 ZIP](/ai-tech-blog/downloads/strands-decider-vs-jev/experiment.zip)에는 고정한 입력과 정답, 실행 전 해시, 모델 revision, 실행 스크립트와 라이브러리 목록, 원시 응답과 지연 기록을 담았습니다. 모델 가중치는 포함하지 않았습니다. 압축을 푼 뒤 README의 환경 준비 절차를 따르면 고정된 모델을 내려받아 다시 실행할 수 있습니다.

```bash
.venv/bin/python download_models.py
.venv/bin/python run_probe.py --model v19
.venv/bin/python run_probe.py --model v21
```

파일에는 측정 이후 추가한 재현성 검사도 구분해 기록했습니다. 첫 v19 실행본을 그대로 보존하고, 실제 불러온 소스와 커밋이 계획과 같았는지 사후에 별도 확인했습니다. 이 보강으로 입력, 정답, 측정값을 바꾸지는 않았습니다.



## 에이전트에 붙일 때 남는 운영 작업

공식 Strands 예제는 날씨 도구를 호출하기 직전에 Decider로 두 가지를 판단합니다. 사용자가 도구 인자를 실제로 제공했는지, 더 확인하기 전에 호출하려는 것은 아닌지 묻습니다. `before_tool_call`에서 실행하는 코드가 결과를 읽고, 호출을 진행하거나 LLM에 추가 질문을 하도록 돌려보냅니다.

이 예제의 질문과 임계값은 설명을 위해 수동으로 정한 값입니다. 실제 업무에 적용하려면 누락된 인자를 잡는 비율과 정상 호출을 잘못 막는 비율을 함께 평가해야 합니다. 모델이 허용하더라도 실행할 수 있는 도구와 데이터의 범위는 애플리케이션 권한 검사로 제한해야 합니다.

<small>참조: [공식 Strands 도구 호출 개입 예제](https://github.com/strands-labs/strands-decider/tree/3e94e9d84c620ed5a95f1a3310c3decb971e261c/examples/strands)</small>

Decider의 기본 HTTP 서버는 `127.0.0.1`에 바인딩하며 인증이 없습니다. 문서는 동시 요청 동작도 검증하지 않았다고 밝힙니다. 따라서 이 서버를 그대로 외부에 열기보다 로컬 실험에서 시작하고, 운영 배포 시에는 인증, 요청 제한, 오류 처리와 동시성 검증을 별도로 준비해야 합니다. AWS에 배포한다면 비공개 네트워크와 최소 권한을 적용하고, 외부 접근이 필요한 경우에도 HTTPS와 인증을 갖추는 방식이 적절합니다.

<small>참조: [Decider 서빙 문서의 운영 제한](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/docs/inference.md#serve)</small>

Jev는 모델 서빙을 맡기는 선택이고, Decider는 모델을 직접 살펴보고 바꾸며 실행 환경도 통제하는 선택입니다. 로컬 처리나 재학습이 필요한 좁은 판단 업무라면 Decider를 시험할 이유가 있습니다. GPU 운영을 추가하지 않고 판단 기능부터 붙이려면 Jev API를 검토할 수 있습니다. 어느 쪽이든 먼저 정할 것은 자동으로 실행할 행동과 잘못 판단했을 때의 처리 방식입니다. 그 조건을 만족하는지 같은 업무 데이터로 확인해야 모델 선택이 실제 운영 결정으로 이어집니다.

## References

- [Strands Agents, Introducing Strands Decider 2B, 2026-10-01](https://strandsagents.com/blog/introducing-strands-decider/)
- [Strands Decider 공개 저장소, 본문이 참고한 커밋](https://github.com/strands-labs/strands-decider/tree/3e94e9d84c620ed5a95f1a3310c3decb971e261c)
- [Strands Decider 아키텍처](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/docs/architecture.md)
- [Strands Decider 추론과 서빙](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/docs/inference.md)
- [Strands Decider 평가와 한계](https://github.com/strands-labs/strands-decider/blob/3e94e9d84c620ed5a95f1a3310c3decb971e261c/evaluation/README.md)
- [Strands Decider v19 모델 카드](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19)
- [Strands Decider v21 모델 카드](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21)
- [Strands 도구 호출 개입 예제](https://github.com/strands-labs/strands-decider/tree/3e94e9d84c620ed5a95f1a3310c3decb971e261c/examples/strands)
- [TypeSafe, Models](https://docs.typesafe.ai/models)
- [TypeSafe, Confidence](https://docs.typesafe.ai/confidence)
- [TypeSafe, Machine learning primer](https://docs.typesafe.ai/introduction/machine-learning-primer)
- [TypeSafe, WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals/tree/0ac3b8ad845429f0d8e064ecfb2430a47c5a25cb)
- [TypeSafe, Workflow evaluations](https://evals.typesafe.ai/)
