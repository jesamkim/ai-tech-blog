---
title: "파인튜닝의 딜레마: Catastrophic Forgetting에서 Nova Forge까지"
date: 2026-03-11T18:00:00+09:00
description: "대규모 언어 모델을 파인튜닝할 때 직면하는 치명적 망각 문제와 이를 해결하기 위한 PEFT 기법들의 진화 과정을 살펴봅니다. LoRA부터 Amazon Nova Forge까지, 실전 커스터마이징 전략을 제시합니다."
categories: ["AI/ML 기술 심층분석"]
tags: ["Fine-tuning", "PEFT", "LoRA", "Catastrophic Forgetting", "Amazon Nova", "LLM"]
author: "Jesam Kim"
---

## 1. 왜 Fine-tuning인가: RAG vs Fine-tuning 판단 기준

대규모 언어 모델을 특정 도메인이나 태스크에 맞추려 할 때, 두 가지 주요 접근법이 있습니다. <strong>Retrieval-Augmented Generation(RAG)</strong>과 <strong>Fine-tuning</strong>입니다.

### RAG가 적합한 경우

- <strong>최신 정보나 사실 지식</strong>이 필요한 경우 (예: 제품 카탈로그, 법률 문서)
- <strong>지식이 자주 변경</strong>되는 경우
- <strong>출처 추적이 중요</strong>한 경우 (환각 방지)
- 프롬프트만으로 해결 가능한 경우

### Fine-tuning이 적합한 경우

- <strong>일관된 스타일이나 포맷</strong>을 학습해야 하는 경우 (예: 브랜드 톤, 응답 구조)
- <strong>복잡한 추론 패턴</strong>을 학습해야 하는 경우
- <strong>새로운 행동 양식</strong>을 학습해야 하는 경우 (예: 코드 생성 스타일)
- <strong>레이턴시가 중요</strong>한 경우 (RAG의 검색 오버헤드 제거)

간단히 말하면, <strong>무엇을 아는가(knowledge)</strong>의 문제라면 RAG를, <strong>어떻게 행동하는가(behavior)</strong>의 문제라면 Fine-tuning을 선택하는 것이 일반적입니다.

## 2. Fine-tuning의 함정: Catastrophic Forgetting

하지만 Fine-tuning에는 치명적인 함정이 있습니다. 바로 <strong>Catastrophic Forgetting(치명적 망각)</strong>입니다.

### 문제의 본질

전통적인 Full Fine-tuning은 모델의 <strong>모든 파라미터를 업데이트</strong>합니다. 특정 도메인 데이터로 훈련하면 해당 태스크 성능은 향상되지만, 원래 학습했던 다른 능력들이 급격히 저하되는 현상이 발생합니다.

예를 들어:
- 의료 도메인으로 파인튜닝 → 일반 상식 질문에 대한 답변 능력 저하
- 한국어 데이터로 파인튜닝 → 영어 능력 저하
- 코드 생성으로 파인튜닝 → 자연어 이해 능력 저하

이는 <strong>신경망의 가중치가 새로운 정보로 덮어써지기</strong> 때문입니다. 수십억 개의 파라미터가 모두 변경되면서, 사전 학습 단계에서 얻은 범용 지식이 손실됩니다.

### Catastrophic Forgetting의 측정

이 현상은 벤치마크 점수 하락으로 확인할 수 있습니다:

- <strong>MMLU(Massive Multitask Language Understanding)</strong>: 범용 지식
- <strong>HellaSwag</strong>: 상식 추론
- <strong>GSM8K</strong>: 수학 능력
- <strong>HumanEval</strong>: 코드 생성

도메인 데이터로 Full Fine-tuning 후 이 벤치마크들의 점수가 떨어지면, Catastrophic Forgetting이 발생한 것입니다.

## 3. PEFT의 등장: 원본 가중치를 지키는 전략

<strong>Parameter-Efficient Fine-Tuning(PEFT)</strong>는 이 문제를 해결하기 위해 등장했습니다. 핵심 아이디어는 간단합니다:

> <strong>원본 모델의 가중치는 동결(freeze)하고, 작은 수의 추가 파라미터만 훈련한다.</strong>

이 접근법은 두 가지 이점을 제공합니다:

1. <strong>Catastrophic Forgetting 방지</strong>: 원본 가중치가 보존되므로 기존 능력 유지
2. <strong>메모리/연산 효율성</strong>: 훈련할 파라미터 수가 1% 미만으로 감소

### PEFT 진화 타임라인

![PEFT Evolution Timeline](/ai-tech-blog/images/peft-evolution-timeline.png)

PEFT 기법들은 지난 몇 년간 빠르게 진화했습니다:

- <strong>2019: Adapter Layers</strong> — 각 Transformer 레이어에 작은 병목 네트워크 삽입
- <strong>2021: LoRA</strong> — Low-rank 행렬 분해로 어댑터 단순화
- <strong>2023: QLoRA</strong> — 양자화 + LoRA로 메모리 사용량 극적 감소
- <strong>2024: DoRA</strong> — 가중치를 magnitude와 direction으로 분해
- <strong>2024: LoRA+</strong> — 학습률 스케일링으로 성능 개선
- <strong>2024: GaLore</strong> — 그래디언트 저차원 투영으로 메모리 효율 극대화

## 4. PEFT 기법 비교: 구조와 성능

### LoRA: 저차원 어댑터의 시작

![LoRA Architecture](/ai-tech-blog/images/lora-architecture.png)

LoRA는 사전 학습된 가중치 행렬 W<sub>0</sub>에 저차원 분해 행렬을 추가하는 방식입니다:

<strong>h = W<sub>0</sub>x + BAx</strong>

여기서:
- W<sub>0</sub> ∈ ℝ<sup>d×k</sup>는 동결된 원본 가중치
- B ∈ ℝ<sup>d×r</sup>, A ∈ ℝ<sup>r×k</sup>는 훈련 가능한 저차원 행렬
- r ≪ min(d,k)는 rank (보통 4~64)

<strong>핵심 하이퍼파라미터</strong>:
- <strong>rank(r)</strong>: 어댑터 용량. 높을수록 표현력 증가하지만 overfitting 위험
- <strong>alpha(α)</strong>: 스케일링 팩터. ΔW는 α/r로 스케일링됨
- <strong>target_modules</strong>: 어느 레이어에 적용할지 (q, k, v, o, ffn 등)

### QLoRA: 양자화로 메모리 장벽 돌파

QLoRA는 LoRA에 <strong>4-bit 양자화</strong>를 결합합니다:

1. <strong>Base model을 4-bit NormalFloat(NF4)</strong>로 양자화
2. <strong>LoRA 어댑터만 FP16/BF16</strong>으로 훈련
3. <strong>Paged Optimizers</strong>로 CPU 메모리 활용

결과: 65B 파라미터 모델을 단일 48GB GPU에서 파인튜닝 가능. 원래는 8개 A100(80GB)이 필요했던 작업입니다.

### DoRA: 가중치 분해로 표현력 향상

DoRA는 가중치를 <strong>magnitude</strong>와 <strong>direction</strong>으로 분해합니다:

<strong>W' = W<sub>0</sub> + BA = ||W'|| · (V + BA) / ||V + BA||</strong>

여기서:
- ||W'||는 magnitude (스칼라)
- 분자는 direction (단위 벡터)

이 분해는 LoRA보다 <strong>학습 역학(learning dynamics)</strong>이 Full Fine-tuning에 가까워 성능이 향상됩니다.

### LoRA+: 학습률 최적화

LoRA+는 간단하지만 효과적인 개선입니다:

<strong>A 행렬과 B 행렬에 서로 다른 학습률 적용</strong>

- A 행렬: 낮은 학습률
- B 행렬: 높은 학습률 (보통 16~64배)

Amazon Nova Forge에서는 `lora_plus_lr_ratio: 64.0`로 설정 가능합니다.

### GaLore: 그래디언트 차원 축소

GaLore는 완전히 다른 접근법을 취합니다. 가중치가 아닌 <strong>그래디언트를 저차원으로 투영</strong>합니다:

<strong>G<sub>low-rank</sub> = P<sup>T</sup> · ∇L · Q</strong>

이를 통해 Full Fine-tuning과 유사한 성능을 유지하면서도 메모리 사용량을 65% 이상 감소시킵니다.

### 성능 비교

| 기법 | 훈련 파라미터 | 메모리 사용량 | Full FT 대비 성능 | Forgetting 방어 |
|------|--------------|--------------|------------------|----------------|
| Full Fine-tuning | 100% | 100% | 100% | ⚠️ 매우 취약 |
| Adapter | ~2% | ~70% | 95-98% | ✅ 양호 |
| LoRA (r=16) | ~0.5% | ~50% | 97-99% | ✅ 우수 |
| QLoRA (4-bit) | ~0.5% | ~25% | 96-98% | ✅ 우수 |
| DoRA | ~0.6% | ~55% | 98-99.5% | ✅ 우수 |
| LoRA+ | ~0.5% | ~50% | 98-99.5% | ✅ 우수 |
| GaLore | 100% (메모리 효율) | ~35% | 99-100% | ⚠️ 보통 |

<strong>Forgetting 방어 메커니즘</strong>:
- <strong>LoRA 계열</strong>: 원본 가중치 동결 → 근본적으로 방지
- <strong>GaLore</strong>: 모든 파라미터 업데이트 → 일부 forgetting 발생 가능 (단, 적은 편)

## 5. Amazon Nova Forge: 최신 커스터마이징 파이프라인

Amazon Nova Forge는 AWS가 제공하는 <strong>Nova 모델 전용 파인튜닝 플랫폼</strong>입니다. 다음 커스터마이징 방법을 지원합니다:

### 지원 방법론

1. <strong>Supervised Fine-Tuning(SFT)</strong>
   - Input-output 페어로 학습
   - LoRA, LoRA+, Full-rank 지원
   - 1K~10K 샘플 권장

2. <strong>Direct Preference Optimization(DPO)</strong>
   - 선호도 데이터로 학습 (chosen vs rejected)
   - 인간 피드백 없이 alignment 가능

3. <strong>Reinforcement Fine-Tuning(RFT)</strong>
   - 보상 함수 기반 강화학습
   - 복잡한 추론 태스크에 적합

4. <strong>Continued Pre-training</strong>
   - 대량의 비구조화 데이터로 지식 주입
   - 100B+ 토큰 규모

### Nova Forge의 Catastrophic Forgetting 방어 전략

Nova Forge는 <strong>Data Mixing</strong> 기능으로 이 문제를 해결합니다:

```yaml
data_mixing:
  dataset_catalog: sft_1p5_text_chat
  sources:
    customer_data:
      percent: 50
    nova_data:
      reasoning-instruction-following: 45
      planning: 10
      code: 10
      instruction-following: 13
      math: 2
      chat: 0.5
```

<strong>작동 원리</strong>:
1. <strong>고객 데이터 50%</strong> + <strong>Nova 원본 데이터 50%</strong> 혼합
2. Nova 데이터가 범용 능력 유지 (reasoning, code, math 등)
3. LoRA와 결합하면 이중 보호: 가중치 동결 + 데이터 다양성

## 6. Nova Forge 실전 설정 가이드

### 기본 LoRA 설정

```yaml
run:
  name: my-domain-lora-sft
  model_type: amazon.nova-2-lite-v1:0:256k
  data_s3_path: s3://my-bucket/train.jsonl
  replicas: 4
  output_s3_path: s3://my-bucket/outputs/

training_config:
  max_steps: 100
  save_steps: 10
  max_length: 32768
  global_batch_size: 32
  reasoning_enabled: true

  lr_scheduler:
    warmup_steps: 15
    min_lr: 1e-6

  optim_config:
    lr: 1e-5
    weight_decay: 0.0

  peft:
    peft_scheme: "lora"
    lora_tuning:
      alpha: 64
      lora_plus_lr_ratio: 64.0
```

### Rank/Alpha 튜닝 전략

| 시나리오 | Rank | Alpha | 이유 |
|---------|------|-------|------|
| 작은 도메인 변화 | 8 | 32 | Overfitting 방지 |
| 중간 복잡도 태스크 | 16 | 64 | 균형 잡힌 선택 |
| 복잡한 추론 패턴 | 32 | 128 | 높은 표현력 필요 |
| 매우 특수한 도메인 | 64 | 192 | 최대 용량 |

<strong>경험 법칙</strong>:
- Alpha는 보통 Rank의 2~4배
- 데이터가 적을수록 낮은 Rank (overfitting 방지)
- 데이터가 많고 복잡할수록 높은 Rank

### Forgetting 방지 체크리스트

✅ <strong>LoRA/LoRA+ 사용</strong> (Full-rank보다 우선)
✅ <strong>Data Mixing 활성화</strong> (50% Nova 데이터 혼합)
✅ <strong>Reasoning 데이터 포함</strong> (범용 벤치마크 유지)
✅ <strong>중간 체크포인트 검증</strong> (MMLU, HellaSwag 등)
✅ <strong>Learning rate 보수적 설정</strong> (1e-5 이하)
✅ <strong>적절한 Warmup</strong> (max_steps의 15%)

### 데이터 준비 팁

<strong>JSONL 형식 예시</strong>:

```json
{"messages": [
  {"role": "user", "content": "법률 문서 요약해줘: [문서 내용]"},
  {"role": "assistant", "content": "주요 내용은 다음과 같습니다.."}
]}
```

<strong>품질 > 수량</strong>:
- 1,000개 고품질 샘플 > 10,000개 저품질 샘플
- 다양한 패턴 포함 (edge case 커버)
- 일관된 포맷 유지

## 7. 의사결정 가이드: 언제 어떤 기법을 쓸 것인가

![PEFT Decision Flowchart](/ai-tech-blog/images/peft-decision-flowchart.png)

### 시나리오별 추천

#### 시나리오 1: 메모리가 충분한 경우 (A100 80GB 이상)

```
데이터 크기 < 5K → LoRA (r=16, α=64)
데이터 크기 5K~50K → LoRA+ (r=32, α=128)
데이터 크기 > 50K → DoRA 또는 Full-rank SFT
```

#### 시나리오 2: 메모리가 제한적인 경우 (RTX 4090, A10G 등)

```
모든 경우 → QLoRA (4-bit, r=16)
극단적 제약 → GaLore
```

#### 시나리오 3: Catastrophic Forgetting이 중요한 경우

```
1순위: LoRA + Data Mixing
2순위: DoRA + Data Mixing
3순위: Adapter Layers
```

#### 시나리오 4: 최고 성능이 필요한 경우

```
충분한 데이터(>10K) → Full-rank SFT + Data Mixing
제한된 데이터 → DoRA 또는 LoRA+
```

### AWS 환경에서의 추천

| AWS 서비스 | 추천 기법 | 이유 |
|------------|---------|------|
| <strong>Amazon Nova Forge</strong> | LoRA+ with Data Mixing | 기본 지원, 최적화됨 |
| <strong>SageMaker Training</strong> | QLoRA (PEFT library) | 비용 효율성 |
| <strong>SageMaker JumpStart</strong> | LoRA | 빠른 시작 |
| <strong>Bedrock Continued Pre-training</strong> | Full-parameter | 대량 도메인 지식 주입 |

## 결론

파인튜닝은 강력하지만 Catastrophic Forgetting이라는 치명적 함정을 내포하고 있습니다. PEFT 기법들은 원본 가중치를 보존하면서도 도메인 적응을 가능하게 하는 우아한 해결책입니다.

<strong>핵심 교훈</strong>:

1. <strong>RAG vs Fine-tuning</strong>: 지식은 RAG, 행동은 Fine-tuning
2. <strong>LoRA 계열이 기본 선택</strong>: 효율성과 Forgetting 방어의 균형
3. <strong>Data Mixing은 필수</strong>: 특히 Full Fine-tuning 시
4. <strong>메모리 제약 시 QLoRA</strong>: 성능 손실 최소
5. <strong>Amazon Nova Forge 활용</strong>: LoRA+와 Data Mixing으로 프로덕션급 파인튜닝 가능

파인튜닝을 시작하기 전에, 벤치마크 baseline을 측정하고, 중간 체크포인트마다 범용 성능을 모니터링하세요. Forgetting은 예방이 치료보다 쉽습니다.

---

## References

- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685
- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized LLMs. arXiv:2305.14314. https://arxiv.org/abs/2305.14314
- Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., & Chen, M.-H. (2024). DoRA: Weight-Decomposed Low-Rank Adaptation. arXiv:2402.09353. https://arxiv.org/abs/2402.09353
- Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., & Tian, Y. (2024). GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. arXiv:2403.03507. https://arxiv.org/abs/2403.03507
- AWS Documentation. Amazon Nova Forge Supervised Fine-Tuning. https://docs.aws.amazon.com/nova/latest/nova2-userguide/nova-forge-sft.html
- AWS Documentation. Customizing Nova models with Amazon SageMaker AI. https://docs.aws.amazon.com/sagemaker/latest/dg/nova-model.html
- AWS Blog. Announcing Amazon Nova customization in Amazon SageMaker AI. https://aws.amazon.com/blogs/aws/announcing-amazon-nova-customization-in-amazon-sagemaker-ai/
- AWS Machine Learning Blog. Reinforcement fine-tuning for Amazon Nova: Teaching AI through feedback. https://aws.amazon.com/blogs/machine-learning/reinforcement-fine-tuning-for-amazon-nova-teaching-ai-through-feedback/
