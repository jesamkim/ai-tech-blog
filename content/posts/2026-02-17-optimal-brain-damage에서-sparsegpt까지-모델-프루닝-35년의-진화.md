---
title: "Optimal Brain Damage에서 SparseGPT까지 — 모델 프루닝 35년의 진화"
date: 2026-02-17T14:55:35+09:00
draft: false
author: "Jesam Kim"
description: "1989년 OBD부터 2023년 SparseGPT·Wanda까지, 뉴럴 네트워크 프루닝 기법 35년의 핵심 아이디어와 진화 과정을 총정리합니다."
categories:
  - "AI/ML 기술 심층분석"
tags:
  - "프루닝"
  - "모델 압축"
  - "SparseGPT"
  - "Wanda"
  - "Lottery Ticket Hypothesis"
  - "LLM 경량화"
  - "Optimal Brain Damage"
  - "Sparsity"
ShowToc: true
TocOpen: true
---

## 왜 프루닝인가 — 모델 압축의 필요성과 프루닝의 위치

GPT-3의 175B 파라미터가 화제가 된 지 불과 몇 년, 이제 LLaMA 65B조차 "작은 모델"로 불리는 시대입니다. 그런데 파라미터 수가 기하급수적으로 늘어나는 동안, GPU 메모리와 추론 비용이 같은 속도로 저렴해지진 않았습니다. 개인적으로 175B 모델을 A100 여러 장에 올려본 적이 있는데, 단일 추론 요청의 레이턴시와 비용을 체감하고 나면 "이 가중치가 전부 정말 필요한가?"라는 질문이 자연스럽게 떠오릅니다.

이 질문에 답하기 위해 연구 커뮤니티는 크게 네 가지 압축 전략을 발전시켜 왔습니다.

<svg viewBox="0 0 800 400" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0a0a2e"/>
      <stop offset="100%" stop-color="#1a1a3e"/>
    </linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/>
    </marker>
  </defs>

  <!-- Background -->
  <rect width="800" height="400" rx="12" fill="url(#bg)"/>

  <!-- Title -->
  <text x="400" y="38" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif"
        font-size="18" fill="#fff" font-weight="bold">OBD vs OBS — 핵심 차이 비교</text>

  <!-- OBD Box -->
  <rect x="40" y="70" width="330" height="260" rx="8" fill="#00d4aa" fill-opacity="0.15" stroke="#00d4aa" stroke-width="1"/>
  <text x="205" y="100" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif"
        font-size="15" fill="#00d4aa" font-weight="bold">OBD (LeCun et al., 1989)</text>

  <text x="205" y="135" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif"
        font-size="11" fill="#fff">Hessian Diagonal Approximation</text>
  <text x="205" y="160" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif"
        font-size="10" fill="#aaa">δL ≈ ½ · h_ii · w_i²</text>

  <!-- OBD flow boxes -->
  <rect x="70" y="185" width="100" height="36" rx="8" fill="#00d4aa" fill-opacity="0.25" stroke="#00d4aa" stroke-width="1"/>
  <text x="120" y="207" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif"
        font-size="9" fill="#fff">Saliency 계산</text>

  <line x1="170" y1="203" x2="200" y2="203" stroke="#555" stroke-width="1.5" marker-end="url(#arrow)"/>

  <rect x="200" y="185" width="100" height="36" rx="8" fill="#00d4aa" fill-opacity="0.25" stroke="#00d4aa" stroke-width="1"/>
  <text x="250" y="207" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif"
        font-size="9" fill="#fff">가중치 제거</text>

  <line x1="250" y1="221" x2="250" y2="250" stroke="#555" stroke-width="1.5" marker-end="url(#arrow)"/>

  <rect x="200" y="255" width="100" height="36" rx="8" fill="#00d4aa" fill-opacity="0.25" stroke="#00d4aa" stroke-width="1"/>
  <text x="250" y="277" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif"
        font-size="9" fill="#fff">Retrain 필요</text>

  <text x="205" y="315" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif"
        font-size="10" fill="#ff9900">⚠ Off-diagonal 무시</text>

  <!-- OBS Box -->
  <rect x="430" y="70" width="330" height="260" rx="8" fill="#00a8ff" fill-opacity="0.15" stroke="#

## 패러다임 전환: Lottery Ticket Hypothesis (2018)

앞서 살펴본 OBD와 OBS가 "학습된 네트워크에서 무엇을 잘라낼 것인가"에 집중했다면, 2018년 Frankle & Carlin이 제시한 Lottery Ticket Hypothesis(LTH)는 질문 자체를 뒤집었습니다. "애초에 dense 네트워크 안에 처음부터 잘 학습될 수 있는 sparse subnetwork가 숨어 있는 것 아닌가?"라는 주장이었습니다.

### 핵심 주장: Winning Ticket의 존재

LTH의 골자는 명쾌합니다. 랜덤 초기화된 dense 네트워크 $f(x;\theta_0)$ 안에는, 독립적으로 학습시켰을 때 원래 네트워크와 동일하거나 더 나은 성능을 동일 학습 스텝 내에 달성할 수 있는 sparse subnetwork $f(x; m \odot \theta_0)$가 존재한다는 것입니다. 여기서 $m \in \{0,1\}^{|\theta|}$는 이진 마스크(binary mask)이고, 이 subnetwork를 **winning ticket**이라 부릅니다.

### Iterative Magnitude Pruning(IMP)과 Rewinding

Winning ticket을 찾는 알고리즘이 Iterative Magnitude Pruning(IMP)입니다. 한 번에 프루닝하는 것이 아니라, 학습 → 프루닝 → 초기 가중치로 되감기(rewinding) → 재학습 사이클을 반복하는 방식입니다.

```python
import torch
import copy

def iterative_magnitude_pruning(model, train_fn, prune_rate=0.2, rounds=10):
    # 1) 초기 가중치 저장 (theta_0)
    initial_state = copy.deepcopy(model.state_dict())
    mask = {name: torch.ones_like(p) for name, p in model.named_parameters()}

    for r in range(rounds):
        # 2) 마스크 적용 후 학습
        apply_mask(model, mask)
        train_fn(model)

        # 3) 크기 기준 하위 prune_rate% 제거
        for name, param in model.named_parameters():
            alive = mask[name].bool()
            threshold = param.abs()[alive].quantile(prune_rate)
            mask[name] = mask[name] * (param.abs() > threshold).float()

        # 4) 초기 가중치로 되감기 (rewinding)
        model.load_state_dict(initial_state)
        print(f"Round {r+1}: 잔존 파라미터 비율 = "
              f"{sum(m.sum() for m in mask.values()) / sum(m.numel() for m in mask.values()):.1%}")

    return model, mask
```

개인적으로 LTH를 처음 접했을 때 가장 인상적이었던 부분은 rewinding이었습니다. 프루닝된 구조만으로는 부족하고, 반드시 **초기화 시점의 가중치 $\theta_0$**로 되돌려야 winning ticket이 성립합니다. 후속 연구(Frankle et al., 2020)에서는 학습 초반 $k$ 스텝 시점으로 되감는 late rewinding이 대규모 모델에서 더 안정적이라는 점도 밝혀졌습니다.

![Lottery Ticket Hypothesis](/ai-tech-blog/images/posts/2026-02-17/optimal-brain-damage에서-sparsegpt까지-모델-프루닝-35년의-진화/diagram-2.png)

import torch

def wanda_prune(W: torch.Tensor, X: torch.Tensor, sparsity: float = 0.5):
    """
    W: 가중치 행렬 (out_features, in_features)
    X: 캘리브레이션 입력 활성화 (n_samples, in_features)
    """
    # 입력 활성화의 열(column)별 L2 노름
    act_norm = X.norm(p=2, dim=0)  # (in_features,)

    # 중요도 점수 = |가중치| * 활성화 노름
    importance = W.abs() * act_norm.unsqueeze(0)  # (out, in)

    # 행(row)별로 하위 sparsity% 제거
    k = int(W.shape[1] * sparsity)
    threshold = importance.kthvalue(k, dim=1).values.unsqueeze(1)
    mask = importance > threshold

    return W * mask
```

```svg
<svg viewBox="0 0 800 420" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#0a0a2e"/>
      <stop offset="100%" stop-color="#1a1a3e"/>
    </linearGradient>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="6" markerHeight="6" orient="auto-start-reverse">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#555"/>
    </marker>
  </defs>

  <rect width="800" height="420" rx="12" fill="url(#bg)"/>

  <!-- Title -->
  <text x="400" y="35" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif"
        font-size="18" fill="#fff" font-weight="bold">SparseGPT vs Wanda — 프루닝 파이프라인 비교</text>

  <!-- ===== SparseGPT Row (y ~ 80-190) ===== -->
  <text x="400" y="72" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif"
        font-size="14" fill="#00d4aa" font-weight="bold">SparseGPT</text>

  <!-- Box 1: Calibration -->
  <rect x="30" y="85" width="140" height="55" rx="8" fill="#00a8ff" fill-opacity="0.15" stroke="#00a8ff" stroke-width="1"/>
  <text x="100" y="108" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif" font-size="10" fill="#00a8ff">캘리브레이션 데이터</text>
  <text x="100" y="125" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif" font-size="9" fill="#fff">(128 샘플)</text>

  <!-- Arrow 1→2 -->
  <line x1="170" y1="112" x2="210" y2="112" stroke="#555" stroke-width="1.5" marker-end="url(#arrow)"/>

  <!-- Box 2: Hessian -->
  <rect x="215" y="85" width="140" height="55" rx="8" fill="#aa88ff" fill-opacity="0.15" stroke="#aa88ff" stroke-width="1"/>
  <text x="285" y="108" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif" font-size="10" fill="#aa88ff">Hessian 근사</text>
  <text x="285" y="125" text-anchor="middle" font-family="NanumBarunGothic, NanumSquare, sans-serif" font-size="9" fill="#fff">H = 2X^T X</text>

  <!-- Arrow 2→3 -->
  <line x1="355" y1="112" x2="395" y2="112" stroke="#555" stroke-width="1.5" marker-end="url

![프루닝 진화 타임라인](/ai-tech-blog/images/posts/2026-02-17/optimal-brain-damage에서-sparsegpt까지-모델-프루닝-35년의-진화/diagram-3.png)

## References

1. LeCun, Y., Denker, J. S., & Solla, S. A. (1989). "Optimal Brain Damage." *Advances in Neural Information Processing Systems (NeurIPS) 2*. [https://proceedings.neurips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf](https://proceedings.neurips.cc/paper/1989/file/6c9882bbac1c7093bd25041881277658-Paper.pdf)

2. Hassibi, B. & Stork, D. G. (1993). "Second Order Derivatives for Network Pruning: Optimal Brain Surgeon." *IEEE International Conference on Neural Networks*. [https://proceedings.neurips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf](https://proceedings.neurips.cc/paper/1992/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf)

3. Frankle, J. & Carlin, M. (2019). "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." *ICLR 2019*. [https://arxiv.org/abs/1803.03635](https://arxiv.org/abs/1803.03635)

4. Frantar, E. & Alistarh, D. (2023). "SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot." *ICML 2023*. [https://arxiv.org/abs/2301.00774](https://arxiv.org/abs/2301.00774)

5. Sun, M., Liu, Z., Bair, A., & Kolter, J. Z. (2024). "A Simple and Effective Pruning Approach for Large Language Models (Wanda)." *ICLR 2024*. [https://arxiv.org/abs/2306.11695](https://arxiv.org/abs/2306.11695)

6. Han, S., Pool, J., Tran, J., & Dally, W. J. (2015). "Learning both Weights and Connections for Efficient Neural Networks." *NeurIPS 2015*. 크기 기반(magnitude-based) 프루닝의 현대적 부활을 이끈 핵심 연구로, 구조적·비구조적 프루닝 파이프라인의 사실상 표준 베이스라인이 되었다. [https://arxiv.org/abs/1506.02626](https://arxiv.org/abs/1506.02626)

7. Hoefler, T., Alistarh, D., Ben-Nun, T., Dryden, N., & Peste, A. (2021). "Sparsity in Deep Learning: Pruning and Growth for Efficient Inference and Training in Neural Networks." *Journal of Machine Learning Research, 22*(241), 1–124. 프루닝을 포함한 희소성 기법 전반을 체계적으로 정리한 서베이 논문으로, 35년간의 연구 흐름을 조망하는 데 유용하다. [https://arxiv.org/abs/2102.00554](https://arxiv.org/abs/2102.00554)

8. Frantar, E., Ashkboos, S., Hoefler, T., & Alistarh, D. (2022). "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." *ICLR 2023*. SparseGPT와 동일한 Hessian 기반 근사 프레임워크를 양자화에 적용한 연구로, 프루닝과 양자화의 기술적 연결 고리를 이해하는 데 중요하다. [https://arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)