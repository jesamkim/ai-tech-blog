---
title: "분산 학습 vs 연합 학습: 같은 뿌리, 다른 철학"
date: 2026-03-04T10:00:00+09:00
description: "분산 학습과 연합 학습은 모두 여러 노드에서 모델을 학습하지만, 목적과 제약 조건이 완전히 다릅니다. 핵심 논문들의 Problem/Contribution/Method를 비교하며 두 접근법의 차이를 살펴봅니다."
categories: ["AI/ML 기술 심층분석"]
tags: ["Distributed Learning", "Federated Learning", "FedAvg", "Data Parallel", "Privacy", "Non-IID", "Communication Efficiency"]
author: "Jesam Kim"
cover:
  image: "/ai-tech-blog/images/cover-distributed-vs-federated-learning.png"
---

## 1. 왜 분산해야 하나?

2012년 구글의 DistBelief 논문 이후, 딥러닝 모델과 데이터셋은 빠르게 커졌습니다. GPT-3는 1750억 개의 파라미터를 가지고 있고, 최신 멀티모달 모델은 수조 개의 토큰으로 학습됩니다. 단일 GPU 메모리로는 감당할 수 없습니다.

그런데 "여러 노드에서 학습한다"는 전제는 같지만, 목적은 다릅니다.

<strong>분산 학습(Distributed Training)</strong>은 속도가 목적입니다. 중앙 데이터센터에서 데이터를 여러 GPU로 분할하여 학습 시간을 단축합니다. Facebook은 2017년 논문에서 ImageNet을 1시간 만에 학습했다고 보고했습니다.

<strong>연합 학습(Federated Learning)</strong>은 프라이버시가 목적입니다. 데이터를 중앙으로 모을 수 없을 때, 모델을 각 클라이언트로 보내서 로컬 학습 후 결과만 집계합니다. 구글 키보드는 사용자의 타이핑 데이터를 서버로 보내지 않고도 예측 모델을 개선합니다.

둘 다 SGD(Stochastic Gradient Descent)를 기반으로 하지만, 전제 조건과 해결해야 할 문제가 다릅니다.

---

## 2. 분산 학습 (Distributed Training) — 속도가 목적

분산 학습은 단일 GPU로는 감당할 수 없는 모델이나 데이터를 여러 GPU에 나누어 처리하는 방법입니다. 크게 세 가지 패러다임이 있습니다.

### Data Parallel

모델 전체를 각 GPU에 복사하고, 데이터를 분할합니다. 각 GPU는 미니배치를 병렬로 처리한 후 그래디언트를 동기화합니다.

![Data Parallel Architecture](/ai-tech-blog/images/distributed-federated-learning/data-parallel.svg)
*Data Parallel 학습 구조 - 각 GPU가 동일한 모델로 다른 데이터를 처리*

<strong>Problem</strong>: 단일 GPU로는 큰 데이터셋을 빠르게 학습할 수 없습니다. ImageNet(120만 장)을 단일 GPU로 학습하면 수일~수주가 걸립니다.

<strong>Contribution</strong>: Goyal et al. (2017)은 8192개 미니배치를 256개 GPU로 병렬 처리하여 ImageNet을 1시간 만에 학습했습니다. 핵심은 Linear Scaling Rule(학습률을 배치 크기에 비례하여 증가)과 Warmup(초기 몇 에폭 동안 학습률을 점진적으로 증가)입니다.

<strong>Method</strong>: All-Reduce 통신 패턴을 사용합니다. 각 GPU가 그래디언트를 계산하면, Ring All-Reduce 알고리즘으로 평균을 구한 뒤 모든 GPU에 브로드캐스트합니다. PyTorch의 DistributedDataParallel(DDP)이 이 방식을 구현합니다.

### Model Parallel

모델이 너무 커서 단일 GPU 메모리에 들어가지 않을 때, 모델을 레이어 단위로 분할합니다. GPT-3 같은 초대형 모델은 단일 GPU에 로드할 수 없습니다.

<strong>Problem</strong>: Transformer 모델이 수백억 파라미터로 커지면서 단일 GPU 메모리(A100 80GB)를 초과합니다.

<strong>Contribution</strong>: Megatron-LM(NVIDIA 2019)은 Transformer를 텐서 병렬화(Tensor Parallelism)로 분할했습니다. Self-Attention과 MLP 레이어를 여러 GPU에 분산하여 83억 파라미터 모델을 학습했습니다.

<strong>Method</strong>: 행렬 곱셈을 열/행 단위로 분할하여 여러 GPU에서 병렬 계산합니다. 각 GPU는 모델의 일부분만 저장하고, Forward/Backward 시 필요한 활성화 값만 통신합니다.

### Pipeline Parallel

모델을 레이어 그룹으로 나누어 각 GPU에 할당하고, 미니배치를 여러 마이크로배치로 분할하여 파이프라인 방식으로 처리합니다.

<strong>Problem</strong>: 단순 Model Parallel은 GPU 유휴 시간(bubble)이 많습니다. GPU 1이 계산하는 동안 GPU 2는 대기하고, GPU 2가 계산할 때 GPU 1은 놉니다.

<strong>Contribution</strong>: Huang et al. (2019) GPipe는 미니배치를 M개의 마이크로배치로 분할하여 파이프라인 처리함으로써 bubble을 최소화했습니다. AmoebaNet 모델을 단일 GPU 대비 25배 빠르게 학습했습니다.

<strong>Method</strong>: Forward 단계에서 마이크로배치 1이 GPU 1 → 2 → 3을 거치는 동안, 마이크로배치 2가 GPU 1에서 시작됩니다. Backward도 역순으로 파이프라인 처리하여 GPU 활용률을 높입니다.

cover:
  image: "/ai-tech-blog/images/cover-distributed-vs-federated-learning.png"
---

## 3. 연합 학습 (Federated Learning) — 프라이버시가 목적

연합 학습은 데이터를 중앙으로 모으지 않고, 모델을 각 클라이언트로 보내서 로컬 학습 후 모델 업데이트만 집계하는 방법입니다.

### FedAvg: 연합 학습의 기초

McMahan et al. (2017)이 제안한 Federated Averaging(FedAvg)는 연합 학습의 표준 알고리즘입니다.

![FedAvg Architecture](/ai-tech-blog/images/distributed-federated-learning/fedavg-architecture.svg)
*FedAvg 프로토콜 - 로컬 학습 후 모델 업데이트만 서버에 전송*

<strong>Problem</strong>: 모바일 기기, 병원, 금융기관 등은 데이터를 외부로 보낼 수 없습니다. GDPR 같은 규제나 보안 정책 때문입니다. 기존 분산 학습은 데이터를 중앙 서버로 모아야 하므로 불가능합니다.

<strong>Contribution</strong>: FedAvg는 각 클라이언트가 로컬 데이터로 E 에폭 학습한 후 모델 업데이트만 서버로 보냅니다. 서버는 이를 가중 평균하여 글로벌 모델을 업데이트합니다. MNIST 데이터셋에서 중앙 학습 대비 통신 횟수를 10~100배 줄였습니다.

<strong>Method</strong>:
1. 서버가 글로벌 모델 w<sub>t</sub>를 K개 클라이언트에 브로드캐스트
2. 각 클라이언트가 로컬 데이터 D<sub>k</sub>로 E 에폭 SGD 학습 → w<sub>t+1</sub><sup>(k)</sup>
3. 서버가 가중 평균: w<sub>t+1</sub> = Σ (n<sub>k</sub> / n) × w<sub>t+1</sub><sup>(k)</sup>
4. 다음 라운드 반복

여기서 n<sub>k</sub>는 클라이언트 k의 데이터 개수, n은 전체 데이터 개수입니다.

### 통신 효율성

Konečný et al. (2016)은 연합 학습의 통신 비용을 줄이는 두 가지 방법을 제시했습니다.

<strong>Problem</strong>: 모바일 네트워크는 불안정하고 느립니다. 매 스텝마다 그래디언트를 보내는 기존 분산 학습은 비현실적입니다.

<strong>Contribution</strong>:
- Structured updates: 저랭크 행렬로 업데이트를 근사
- Sketched updates: 랜덤 마스크로 중요한 파라미터만 전송

LSTM 언어 모델에서 통신량을 100배 줄이면서도 정확도 손실은 1% 미만이었습니다.

<strong>Method</strong>: 업데이트 Δw를 저랭크 행렬 UV<sup>T</sup>로 근사하거나, Count Sketch로 압축합니다. 클라이언트는 압축된 업데이트만 전송하고, 서버는 압축 해제 후 집계합니다.

---

## 4. 핵심 차이: 같은 SGD, 다른 전제

분산 학습과 연합 학습은 모두 SGD를 여러 노드에서 실행합니다. 하지만 전제 조건이 다릅니다.

![Distributed vs Federated Comparison](/ai-tech-blog/images/distributed-federated-learning/distributed-vs-federated-comparison.svg)
*분산 학습과 연합 학습의 체계적 비교*

### IID vs Non-IID

분산 학습은 데이터가 IID(Independent and Identically Distributed)라고 가정합니다. 중앙 서버가 데이터를 균등하게 분할하므로 각 GPU는 동일한 분포를 봅니다.

연합 학습은 Non-IID가 기본입니다. 각 클라이언트의 데이터는 고유한 특성을 가집니다. 병원 A는 심장병 환자가 많고, 병원 B는 당뇨 환자가 많을 수 있습니다. 이 불균형이 수렴을 어렵게 만듭니다.

Li et al. (2020) FedProx는 이 문제를 해결하기 위해 Proximal Term을 추가했습니다.

<strong>Problem</strong>: Non-IID 데이터에서 FedAvg는 수렴 속도가 느리거나 발산할 수 있습니다. 각 클라이언트가 다른 방향으로 학습하면 글로벌 모델이 진동합니다.

<strong>Contribution</strong>: FedProx는 로컬 업데이트를 글로벌 모델에 가깝게 유지하는 정규화 항을 추가했습니다. FEMNIST(Non-IID 손글씨 데이터셋)에서 수렴 속도가 FedAvg 대비 2배 빨랐습니다.

<strong>Method</strong>: 로컬 손실 함수에 (μ/2) ||w - w<sub>t</sub>||<sup>2</sup> 항을 추가합니다. μ는 얼마나 글로벌 모델에 가까이 있을지 조절하는 하이퍼파라미터입니다.

### 통신 비용

분산 학습은 데이터센터 내부의 고속 네트워크(InfiniBand, NVLink)를 사용합니다. 매 스텝마다 그래디언트를 동기화해도 병목이 되지 않습니다.

연합 학습은 인터넷을 통해 통신합니다. 모바일 기기는 4G/5G 네트워크를 사용하고, 병원은 방화벽 뒤에 있습니다. 통신 횟수를 최소화해야 합니다. 그래서 로컬에서 여러 에폭 학습 후 1회만 전송합니다.

### 신뢰 모델

분산 학습은 완전 신뢰 환경입니다. 모든 GPU는 같은 조직 소속이고, 악의적인 노드는 없습니다.

연합 학습은 Semi-honest 모델을 가정합니다. 클라이언트는 프로토콜을 따르지만, 데이터를 노출하지 않습니다. 악의적인 클라이언트는 Byzantine-robust 알고리즘으로 방어합니다.

### 데이터 소유권

분산 학습은 중앙 집중형입니다. 데이터는 조직이 소유하고, GPU는 단순히 계산 자원입니다.

연합 학습은 분산 소유입니다. 각 클라이언트가 데이터를 소유하고 통제합니다. 서버는 글로벌 모델만 관리하고, 원본 데이터를 볼 수 없습니다.

cover:
  image: "/ai-tech-blog/images/cover-distributed-vs-federated-learning.png"
---

## 5. 최신 트렌드: LLM 시대의 분산+연합

최근 Foundation Model은 분산 학습과 연합 학습에 새로운 문제를 던졌습니다.

### Foundation Model 분산 학습

GPT-3(1750억 파라미터), PaLM(5400억), GPT-4(추정 1조 이상)는 기존 분산 학습 기법의 한계를 시험했습니다.

<strong>DeepSpeed ZeRO</strong>(Microsoft 2020)는 메모리 최적화 기법입니다. Optimizer State, Gradient, Parameter를 여러 GPU에 분할하여 메모리 사용량을 N배 줄입니다. 1조 파라미터 모델을 800개 GPU로 학습할 수 있습니다.

<strong>FSDP</strong>(Meta 2021)는 PyTorch의 Fully Sharded Data Parallel입니다. ZeRO와 유사하게 파라미터를 샤딩하지만, PyTorch 네이티브로 구현되어 사용이 쉽습니다.

<strong>Megatron-LM</strong>(NVIDIA 2021)은 3D 병렬화를 제안했습니다. Data Parallel + Tensor Parallel + Pipeline Parallel을 조합하여 수천 개 GPU로 확장합니다.

### Federated Learning + LLM

Foundation Model을 연합 학습으로 파인튜닝하는 시도가 늘고 있습니다.

<strong>FederatedScope-LLM</strong>(Alibaba 2023)은 LLaMA를 연합 학습으로 파인튜닝하는 프레임워크입니다. 각 클라이언트가 도메인별 데이터(의료, 법률)로 LoRA 어댑터를 학습하고, 서버가 이를 집계합니다.

<strong>FedPAQ</strong>(2024)는 양자화와 어댑터를 결합했습니다. 모델을 4bit로 양자화하여 통신량을 줄이고, QLoRA로 파라미터 효율적 파인튜닝을 수행합니다.

문제는 여전히 Non-IID입니다. 병원 A의 의료 데이터와 은행 B의 금융 데이터는 분포가 다릅니다. Personalized Federated Learning(각 클라이언트에 맞춤형 모델)이 대안으로 제시되고 있습니다.

---

## 6. 실무 관점 정리

언제 분산 학습을 쓰고, 언제 연합 학습을 써야 할까요?

### 분산 학습을 선택하는 경우

- 데이터를 중앙으로 모을 수 있습니다 (법적/기술적 제약 없음)
- 빠른 학습 속도가 중요합니다 (연구, 프로덕션 모델 학습)
- 고속 네트워크 인프라가 있습니다 (AWS/GCP 같은 클라우드)
- 데이터가 IID이거나 사전에 균등 분할 가능합니다

<strong>도구</strong>: PyTorch DDP, DeepSpeed, FSDP, Megatron-LM, Ray Train

### 연합 학습을 선택하는 경우

- 데이터를 외부로 보낼 수 없습니다 (GDPR, 의료법, 금융 규제)
- 데이터가 여러 조직에 분산되어 있습니다 (병원, 모바일 기기)
- 통신 비용이 높거나 네트워크가 불안정합니다
- 데이터 소유권을 각 참여자가 유지해야 합니다

<strong>도구</strong>: TensorFlow Federated, PySyft, Flower, FederatedScope, FATE

### 하이브리드 접근

실제로는 두 가지를 조합합니다. 각 병원 내부에서는 분산 학습(Data Parallel)로 로컬 GPU를 활용하고, 병원 간에는 연합 학습(FedAvg)로 글로벌 모델을 학습합니다.

구글 키보드는 디바이스 클러스터 내부에서 분산 학습으로 미니 모델을 학습하고, 클러스터 간에는 연합 학습으로 집계합니다. 이렇게 하면 통신 횟수를 더욱 줄일 수 있습니다.

cover:
  image: "/ai-tech-blog/images/cover-distributed-vs-federated-learning.png"
---

## References

- Dean, J., et al. (2012). "Large Scale Distributed Deep Networks." *NIPS 2012*. https://papers.nips.cc/paper/2012/hash/6aca97005c68f1206823815f66102863-Abstract.html
- Goyal, P., et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour." *arXiv:1706.02677*. https://arxiv.org/abs/1706.02677
- Huang, Y., et al. (2019). "GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism." *NeurIPS 2019*. https://arxiv.org/abs/1811.06965
- McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." *AISTATS 2017*. https://arxiv.org/abs/1602.05629
- Konečný, J., et al. (2016). "Federated Learning: Strategies for Improving Communication Efficiency." *arXiv:1610.05492*. https://arxiv.org/abs/1610.05492
- Li, T., et al. (2020). "Federated Optimization in Heterogeneous Networks." *MLSys 2020*. https://arxiv.org/abs/1812.06127
- Kairouz, P., et al. (2021). "Advances and Open Problems in Federated Learning." *Foundations and Trends in Machine Learning*. https://arxiv.org/abs/1912.04977
- Shoeybi, M., et al. (2019). "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism." *arXiv:1909.08053*. https://arxiv.org/abs/1909.08053
- Rajbhandari, S., et al. (2020). "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models." *SC20*. https://arxiv.org/abs/1910.02054
- Zhao, S., et al. (2018). "Federated Learning with Non-IID Data." *arXiv:1806.00582*. https://arxiv.org/abs/1806.00582
