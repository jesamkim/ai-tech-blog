---
title: "IoT × Generative AI: 시계열 Foundation Model과 AWS IoT+Bedrock 통합 설비 예방정비 아키텍처"
date: 2026-02-16T21:16:28+09:00
draft: false
author: "Jesam Kim"
description: "TimesFM, Chronos 등 시계열 Foundation Model의 zero-shot 성능을 활용하고, AWS IoT SiteWise + Bedrock 파이프라인으로 설비 예방정비를 구현하는 통합 아키텍처를 분석합니다."
categories:
  - "Physical AI"
tags:
  - "Predictive Maintenance"
  - "Time Series Foundation Model"
  - "AWS IoT"
  - "Amazon Bedrock"
  - "TimesFM"
  - "Chronos"
  - "IoT"
  - "RUL"
  - "이상탐지"
ShowToc: true
TocOpen: true
---

## 1. 설비 예방정비(Predictive Maintenance)의 핵심 과제

제조·플랜트·테마파크 할 것 없이, 설비가 멈추는 순간 비용은 기하급수적으로 늘어납니다. 예방정비(Predictive Maintenance, PM)는 이 다운타임을 줄이기 위한 핵심 전략이며, 기술적으로는 크게 두 가지 축으로 나뉩니다.

### PM의 두 축: 이상탐지와 잔여수명 예측

| 과제 | 핵심 질문 | 비즈니스 임팩트 |
|---|---|---|
| 시계열 이상탐지 (Anomaly Detection) | "지금 설비가 정상인가?" | 돌발 고장 방지, 즉각 대응 |
| 잔여수명 예측 (Remaining Useful Life, RUL) | "이 부품이 언제 교체 시점에 도달하는가?" | 정비 일정 최적화, 부품 재고 관리 |

이상탐지는 실시간성이 생명이고, RUL 예측은 장기 트렌드를 읽어야 하므로 모델 설계 철학 자체가 다릅니다. 개인적으로 현장에서 느끼는 건, 이상탐지는 비교적 빠르게 도입할 수 있지만 RUL은 충분한 고장 이력 데이터가 확보되지 않으면 정확도를 담보하기 어렵다는 점입니다.

### 전통적 접근의 한계

통계 기반 임계값(threshold) 룰이나 도메인 전문가가 수작업으로 정의한 규칙은 여전히 많은 현장에서 쓰이고 있습니다. 하지만 이 방식에는 구조적 한계가 있습니다.

- **설비별 개별 모델 학습 비용**: 펌프, 모터, 컴프레서 등 설비 유형마다 별도 모델을 구축해야 하므로 확장성이 크게 떨어집니다.
- **라벨 데이터 부족**: 고장은 드문 이벤트(rare event)이기 때문에 지도학습에 필요한 양질의 라벨을 확보하기가 매우 어렵습니다.
- 신규 설비나 신규 라인이 투입되면 충분한 운전 이력이 쌓이기 전까지 모델이 사실상 무력합니다. 이른바 Cold-start 문제입니다.

```python
# 전통적 임계값 기반 이상탐지 — 단순하지만 한계가 명확합니다
def rule_based_alert(vibration: float, temp: float) -> bool:
    VIBRATION_THRESHOLD = 7.1  # mm/s, 도메인 전문가 정의
    TEMP_THRESHOLD = 85.0      # °C
    return vibration > VIBRATION_THRESHOLD or temp > TEMP_THRESHOLD
```

위 코드처럼 단순 임계값 방식은 구현이 쉽습니다. 그 대신 설비 노화에 따른 정상 범위의 점진적 변화(concept drift)를 전혀 반영하지 못한다는 문제가 있습니다.

### Generative AI 시대의 패러다임 전환

이런 한계를 근본적으로 바꾸고 있는 것이 **사전학습 시계열 Foundation Model**과 LLM 기반 진단의 결합입니다. Google의 TimesFM이나 Amazon의 Chronos처럼 대규모 시계열 코퍼스로 사전학습된 모델은 zero-shot 또는 few-shot만으로도 새로운 설비의 이상 패턴을 포착할 수 있습니다. Cold-start 문제를 상당 부분 완화해 주는 셈입니다. 여기에 LLM이 탐지 결과를 해석하고 정비 가이드를 자연어로 생성하면, 현장 엔지니어가 별도 분석 도구 없이도 바로 의사결정을 내릴 수 있습니다.

![PM 파이프라인 비교](/ai-tech-blog/images/posts/2026-02-17/iot-generative-ai-시계열-foundation-model과-aws-iotbedrock-통합-설비/diagram-1.png)

실제로 써보면, 가장 체감이 큰 변화는 "모델 하나로 여러 설비를 커버할 수 있다"는 확장성입니다. 다음 섹션에서는 이 패러다임 전환의 핵심인 TimesFM과 Chronos 두 논문을 구체적으로 살펴보겠습니다.

## 2. 시계열 Foundation Model 논문 리뷰: TimesFM & Chronos

앞서 살펴본 PM의 핵심 과제인 시계열 이상탐지와 RUL 예측을 해결하려면 결국 강력한 시계열 예측 모델이 필요합니다. 최근 NLP 분야의 Foundation Model 패러다임이 시계열 영역으로 확장되면서, 도메인 특화 학습 없이도 zero-shot으로 예측이 가능한 모델들이 등장했습니다. 그중 주목할 만한 두 연구를 살펴보겠습니다.

### Google TimesFM (arXiv 2310.10688)

TimesFM은 디코더 전용(decoder-only) 트랜스포머 아키텍처를 기반으로, 대규모 시계열 코퍼스에서 사전학습된 Foundation Model입니다. 핵심 아이디어는 LLM이 텍스트의 다음 토큰을 예측하듯, 시계열의 다음 패치(patch)를 예측하는 방식으로 학습한다는 점입니다. 논문에 따르면 대규모 실제 시계열 데이터와 합성 데이터를 혼합하여 사전학습했고, Monash Forecasting Archive 등 여러 벤치마크에서 zero-shot 상태로도 supervised 모델에 준하는 성능을 기록했습니다.

```python
# TimesFM 추론 예시 (공식 API 기준)
import timesfm

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        per_core_batch_size=32,
        horizon_len=96,  # 예측 구간
    ),
    checkpoint=timesfm.TimesFmCheckpoint(
        huggingface_repo_id="google/timesfm-1.0-200m"
    ),
)

# 설비 진동 센서 시계열 입력
vibration_data = [sensor_readings]  # shape: (batch, context_len)
frequency_input = [0]  # 0: high-freq 데이터

point_forecast, experimental_quantiles = tfm.forecast(
    vibration_data,
    freq=frequency_input,
)
```

### Amazon Chronos (arXiv 2403.07815)

Chronos는 완전히 다른 접근을 취합니다. 시계열 값을 양자화(quantization)하여 토큰 시퀀스로 변환한 뒤, T5 기반 언어 모델 아키텍처에서 cross-entropy 손실로 학습합니다. 이 방식의 가장 큰 강점은 **확률적 예측(probabilistic forecasting)**이 자연스럽게 가능하다는 점입니다. 모델이 각 시점에서 토큰 분포를 출력하기 때문에 예측 불확실성(uncertainty)을 직접 정량화할 수 있습니다. PM 관점에서 이 특성은 실용적 가치가 큽니다. "고장이 날 수도 있다"가 아니라 "고장 확률이 이 범위에 있다"는 식으로 의사결정을 지원할 수 있기 때문입니다.

![TimesFM(패치 기반 디코더)과 Chronos(토큰화 기반 인코더-디코더)의 아키텍처 비교 및 입력→추론→출력 흐름](/ai-tech-blog/images/posts/2026-02-17/iot-generative-ai-시계열-foundation-model과-aws-iotbedrock-통합-설비/diagram-2.png)

### PM 도메인 적용 시 비교 고려사항

| 관점 | TimesFM | Chronos |
|------|---------|---------|
| 아키텍처 | 디코더 전용, 패치 단위 예측 | T5 인코더-디코더, 토큰 단위 예측 |
| 출력 형태 | 포인트 예측 중심 (+ 실험적 분위수) | 네이티브 확률 분포 |
| Zero-shot 강점 | 긴 컨텍스트, 다양한 주기성 처리 | 불확실성 정량화, 짧은 시계열 |
| 파인튜닝 | 논문 기준 미공개 (zero-shot 특화) | T5 기반이라 LoRA 등 적용 용이 |

개인적으로, PM 도메인에서는 두 모델의 선택이 출력의 활용 방식에 따라 달라진다고 봅니다. 진동·온도·압력 센서 데이터로 단순 임계값 초과 여부를 판단하는 이상탐지에는 TimesFM의 빠른 포인트 예측이 효율적입니다. 반면 RUL 예측처럼 "언제 고장날 것인가"에 대한 신뢰구간이 필요한 경우에는 Chronos의 확률적 예측이 더 적합합니다.

실제로 써보면, 두 모델 모두 zero-shot 상태에서 범용 시계열에는 인상적인 성능을 보입니다. 다만 산업 설비 센서처럼 도메인 특이적인 패턴, 예를 들어 베어링 열화의 점진적 주파수 변화 같은 경우에는 fine-tuning 없이 한계가 분명합니다. 다음 섹션에서는 이러한 모델들을 AWS IoT 파이프라인과 통합하여 실시간 PM 시스템을 구축하는 방법을 다루겠습니다.

## 3. AWS IoT + Bedrock 통합 아키텍처 설계

앞서 살펴본 TimesFM과 Chronos가 모델 관점의 혁신이라면, 이를 실제 산업 현장에 배포하려면 센서 데이터 수집부터 진단 리포트 전달까지 이어지는 엔드투엔드(End-to-End) 파이프라인이 필요합니다. AWS의 IoT 서비스군과 Bedrock을 결합하면 이 파이프라인을 비교적 빠르게 구성할 수 있습니다.

### 3-1. 데이터 수집 계층

설비 현장의 진동·온도·전류 센서 데이터는 AWS IoT Greengrass가 설치된 엣지 게이트웨이에서 1차 수집됩니다. Greengrass의 로컬 Lambda 컴포넌트를 활용하면 엣지 단에서 간단한 필터링(노이즈 제거, 다운샘플링)과 경량 추론까지 처리할 수 있습니다.

이후 데이터는 AWS IoT SiteWise로 전송되어 산업 자산 모델(Asset Model)에 매핑되고, 시계열 형태로 저장됩니다. 장기 보관 및 배치 학습용 데이터는 S3로, 실시간 쿼리가 필요한 경우 Amazon Timestream으로 이중 적재하는 구성을 권장합니다.

![IoT Greengrass(엣지) → IoT SiteWise(자산 모델링) → S3/Timestream 이중 적재 파이프라인](/ai-tech-blog/images/posts/2026-02-17/iot-generative-ai-시계열-foundation-model과-aws-iotbedrock-통합-설비/diagram-3.png)

### 3-2. 이상탐지·예측 계층

수집된 시계열 데이터에 대한 추론은 두 가지 경로로 설계할 수 있습니다.

**실시간 추론** — SageMaker 실시간 엔드포인트에 Chronos 또는 TimesFM을 서빙하여, IoT Rule Action이 트리거될 때마다 즉시 예측값을 반환합니다. 개인적으로 실시간 경로는 잔여수명(RUL) 예측보다는 급격한 이상 패턴 탐지에 적합하다고 봅니다.

**배치 추론** — SageMaker Processing Job이나 Step Functions 워크플로우를 통해 주기적(예: 1시간/1일)으로 S3의 누적 데이터를 대상으로 RUL 예측을 수행합니다. 모델 크기가 큰 TimesFM-200M 급은 배치 방식이 비용 효율적입니다.

아래는 SageMaker 엔드포인트에서 Chronos를 호출하는 간단한 예시입니다.

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime")

payload = {
    "timeseries": sensor_readings,  # List[float], 최근 512 포인트
    "prediction_length": 64,
    "num_samples": 20  # 확률적 예측을 위한 샘플 수
}

response = runtime.invoke_endpoint(
    EndpointName="chronos-bolt-endpoint",
    ContentType="application/json",
    Body=json.dumps(payload)
)

forecast = json.loads(response["Body"].read())
# forecast["median"], forecast["quantile_90"] 등으로 이상 임계값 판단
```

### 3-3. LLM 진단·알림 계층

이상탐지 계층에서 임계값을 초과하는 이벤트가 발생하면, Amazon Bedrock Agent가 이를 컨텍스트로 수신합니다. Agent는 설비 매뉴얼과 과거 정비 이력이 저장된 Knowledge Base(OpenSearch Serverless 기반 RAG)를 참조하여 자연어 진단 리포트를 생성합니다.

실제로 써보면, 단순히 "진동값 초과"라는 알림보다 "베어링 외륜 결함 패턴과 유사하며, 지난 정비 이후 가동 시간을 고려할 때 윤활유 교체를 우선 점검하시기 바랍니다"와 같은 맥락 있는 진단이 현장 엔지니어의 판단 속도를 확실히 높여 줍니다.

생성된 리포트는 Amazon SNS를 통해 Slack·Teams 채널로 즉시 전달됩니다. 여기에 CMMS(Computerized Maintenance Management System)의 API를 호출하여 작업 오더(Work Order)를 자동 생성하는 워크플로우까지 연결할 수도 있습니다.

![이상탐지 이벤트 → Bedrock Agent(RAG + Knowledge Base) → 진단 리포트 생성 → SNS/Slack 알림 + CMMS 작업 오더 생성 흐름](/ai-tech-blog/images/posts/2026-02-17/iot-generative-ai-시계열-foundation-model과-aws-iotbedrock-통합-설비/diagram-4.png)

이 세 계층을 조합하면, 센서 신호가 발생한 시점부터 현장 엔지니어가 조치 가능한 진단 리포트를 받기까지 걸리는 시간을 크게 줄일 수 있습니다. 다음 섹션에서는 이 아키텍처를 실제 현장에 적용한 시나리오를 구체적으로 살펴보겠습니다.

## 4. 실제 적용 시나리오

앞서 설계한 AWS IoT + Bedrock 통합 아키텍처가 실제 산업 현장에서 어떻게 동작하는지, 두 가지 시나리오를 통해 구체적으로 살펴보겠습니다.

### 4-1. 테마파크 어트랙션 설비 PM

대형 테마파크의 어트랙션은 고속 회전체, 유압 시스템, 레일 구동부 등 수백 개의 센서가 부착된 복합 설비입니다. 안전이 곧 서비스 품질이기 때문에, 예방정비(Predictive Maintenance)에 대한 요구 수준이 높을 수밖에 없습니다.

핵심 모니터링 대상은 진동(vibration) 센서, 모터 전류(current) 센서, 유압 온도/압력 센서입니다. IoT SiteWise에서 이들 시계열 데이터를 수집한 뒤, Chronos 모델로 zero-shot 이상탐지를 수행하는 흐름을 예시로 보겠습니다.

```python
# 어트랙션 유압 펌프 진동 데이터 이상탐지 예시
import boto3
import json

bedrock_agent = boto3.client("bedrock-agent-runtime")

# SiteWise에서 수집된 최근 1시간 진동 데이터 (sampling: 1초 간격)
vibration_context = {
    "asset": "T-Express_Hydraulic_Pump_03",
    "sensor_type": "vibration_rms",
    "unit": "mm/s",
    "recent_values": [4.2, 4.3, 4.1, 7.8, 8.1, 8.5, 9.2],  # 급격한 상승 패턴
    "baseline_mean": 4.5,
    "anomaly_detected": True,
    "predicted_rul_hours": 72
}

response = bedrock_agent.invoke_agent(
    agentId="attraction-pm-agent",
    sessionId="session-001",
    inputText=json.dumps({
        "task": "diagnose_and_recommend",
        "data": vibration_context
    })
)
# Bedrock Agent 응답: 베어링 마모 가능성 진단 + 72시간 내 교체 권고 + 운영팀 알림 트리거
```

개인적으로 이 시나리오에서 가장 가치 있다고 느끼는 부분은, LLM이 단순히 "이상 발생"이라는 알림을 넘어 **"베어링 마모 초기 징후로 판단되며, 운행 중단 없이 야간 정비 시간에 교체를 권고합니다"**와 같은 맥락적 진단을 제공한다는 점입니다. 운영자가 의사결정에 바로 활용할 수 있는 수준의 출력이 나옵니다.

### 4-2. 건설 현장 중장비 모니터링

테마파크와는 환경이 사뭇 다릅니다. 건설 현장의 굴삭기, 크레인, 덤프트럭 등은 가혹한 조건에서 운용되고, 고장이 나면 공정 지연 비용이 상당합니다. 이런 환경에서는 AWS IoT Greengrass를 엣지에 배포하여 현장에서 1차 필터링을 수행하고, 이상 징후가 포착된 데이터만 클라우드로 전송하는 구조가 효과적입니다.

실제로 써보면, 건설 현장은 네트워크 환경이 불안정한 경우가 많아 **엣지 추론(Edge Inference)**이 거의 필수입니다. Greengrass에 경량화된 TimesFM 모델을 배포하여 엔진 오일 온도, 유압 라인 압력 등의 시계열을 로컬에서 분석하고, 임계치 초과 시에만 Bedrock 에이전트로 상세 진단을 요청하는 2단계 구조를 쓰면 비용과 지연 시간을 모두 줄일 수 있습니다.

![건설 현장 엣지-클라우드 2단계 PM 파이프라인 — Greengrass 엣지 추론 → 이상 감지 시 클라우드 Bedrock Agent 진단 → 현장 관리자 모바일 알림](/ai-tech-blog/images/posts/2026-02-17/iot-generative-ai-시계열-foundation-model과-aws-iotbedrock-통합-설비/diagram-5.png)

두 시나리오 모두 Swann의 IoT+Bedrock 통합 사례에서 확인된 패턴과 맥이 통합니다. Swann은 IoT 디바이스 데이터를 Bedrock 기반 에이전트로 처리하여 사용자에게 자연어 인사이트를 제공하는 구조를 채택했는데, 같은 접근 방식이 산업 설비 영역에도 충분히 적용 가능합니다.

## References

1. Das, A., Kong, W., Leber, A., Mathews, J., & Rajmohan, S. (2024). "A Decoder-Only Foundation Model for Time-Series Forecasting." *arXiv preprint*. https://arxiv.org/abs/2310.10688 — Google TimesFM 논문. 대규모 시계열 코퍼스에서 사전학습한 디코더 전용 Foundation Model의 zero-shot 예측 성능을 다룸.

2. Ansari, A. F., Stella, L., Turkmen, C., Zhang, X., Mercado, P., Shen, H., Shchur, O., Rangapuram, S. S., Arango, S. P., Kapoor, S., Zschiegner, J., Maddix, D. C., Wang, H., Mahoney, M. W., Torber, K., Wilson, A. G., Bohlke-Schneider, M., & Gasthaus, J. (2024). "Chronos: Learning the Language of Time Series." *arXiv preprint*. https://arxiv.org/abs/2403.07815 — Amazon Chronos 논문. 시계열을 토큰 시퀀스로 변환하여 언어 모델 아키텍처로 학습하는 접근법 제시.

3. Amazon Chronos 공식 GitHub 리포지토리. https://github.com/amazon-science/chronos-forecasting — Chronos 모델 코드, 사전학습 가중치 및 벤치마크 재현 스크립트 제공.

4. AWS IoT SiteWise 공식 문서. https://docs.aws.amazon.com/iot-sitewise/latest/userguide/what-is-sitewise.html — 산업 설비 센서 데이터 수집·모니터링·분석을 위한 AWS IoT SiteWise 서비스 개요 및 아키텍처 가이드.

5. AWS IoT Greengrass 공식 문서. https://docs.aws.amazon.com/greengrass/v2/developerguide/what-is-iot-greengrass.html — 엣지 디바이스에서의 로컬 컴퓨팅·ML 추론·데이터 동기화를 위한 Greengrass V2 개발자 가이드.

6. Amazon Bedrock Agents 공식 문서. https://docs.aws.amazon.com/bedrock/latest/userguide/agents.html — Bedrock 에이전트를 활용한 자율적 태스크 오케스트레이션(API 호출, 지식 기반 조회 등) 구성 방법.

7. Swann Communications의 Amazon Bedrock 활용 사례. AWS Machine Learning Blog. https://aws.amazon.com/blogs/machine-learning/swann-provides-generative-ai-to-millions-of-iot-devices-using-amazon-bedrock/ — Swann이 IoT 보안 카메라 플랫폼에 Bedrock LLM을 통합하여 이벤트 요약 및 자연어 알림을 구현한 아키텍처 사례.

8. Lai, S., Zha, D., Wang, G., Xu, J., Zhao, L., Kumar, V., Ding, C., Liao, W., & Hu, X. (2024). "Are Language Models Actually Useful for Time Series Forecasting?" *arXiv preprint*. https://arxiv.org/abs/2406.16964 — LLM/Foundation Model의 시계열 예측 유효성을 비판적으로 분석한 연구. 시계열 FM 도입 시 기대 효과와 한계를 균형 있게 이해하는 