---
title: "Inkling 해부: 975B MoE 오픈웨이트가 노리는 '기업 커스터마이징'"
date: 2026-07-25T10:00:00+09:00
draft: false
categories: ["AI/ML 기술 심층분석"]
tags: ["Inkling", "Thinking Machines Lab", "MoE", "Open Weights", "Long Context", "Fine-tuning", "Architecture"]
author: "Jesam Kim"
description: "Thinking Machines Lab의 첫 오픈웨이트 모델 Inkling(975B MoE)의 아키텍처 선택을 분해합니다. sliding-window 55/66, learned relative-position bias, kernel-4 convolution이 왜 '벤치마크 1등'보다 '파인튜닝 베이스'를 겨냥하는지 살펴봅니다."
cover:
  image: "/ai-tech-blog/images/inkling-975b-moe-open-weights-anatomy/cover.png"
  alt: "Inkling 975B MoE 오픈웨이트 아키텍처 해부"
  relative: false
---

오픈웨이트 모델 출시 발표문에서 개발사가 "이건 현존 최강 모델이 아닙니다"라고 먼저 말하는 건 흔한 순서가 아닙니다. Thinking Machines Lab이 2026년 7월 15일 공개한 [Inkling](https://thinkingmachines.ai/news/introducing-inkling/)의 발표문은 그렇게 시작합니다. 오픈이든 클로즈드든 최고 성능 모델은 아니고, 대신 <strong>커스터마이징하기 좋은 오픈웨이트 베이스</strong>를 목표로 했다는 자기규정입니다.

이 포지셔닝이 진심인지 마케팅 수사인지는 벤치마크 표만 봐도 알기 어렵습니다. 그래서 아키텍처를 봅니다. 975B 파라미터 중 token당 41B만 활성화하는 Mixture-of-Experts, 66개 decoder layer 중 55개를 512-token 창으로 제한한 attention, RoPE를 걷어내고 학습된 상대위치 bias를 넣은 선택. 리더보드 점수를 몇 %p 끌어올리는 종류의 튜닝과는 결이 다른 결정들이고, 긴 컨텍스트 처리 비용을 구조로 낮추는 쪽을 향합니다.

이 글은 발표문과 모델카드, 그리고 Sebastian Raschka의 [아키텍처 분석 노트](https://sebastianraschka.com/blog/2026/inkling-architecture-benchmark-notes.html)를 근거로 Inkling의 설계를 엔지니어 관점에서 분해합니다.

## 왜 지금 Inkling인가

Thinking Machines Lab이 자사 모델을 공개 가중치로 내놓은 건 이번이 처음입니다. 그동안 이 회사의 이름이 걸려 있던 건 파인튜닝 플랫폼 Tinker 쪽이었습니다. Inkling은 [Apache 2.0으로 배포](https://huggingface.co/thinkingmachines/Inkling)되므로 상업적 이용과 파생 모델 배포에 별도 협상이 필요하지 않습니다. 물론 라이선스 고지와 변경 표시 같은 준수 의무는 남습니다.

여기서 "베이스 모델"이라는 단어를 조심해서 읽어야 합니다. Inkling은 정렬 학습까지 거쳐 그대로 쓸 수 있는 모델이고, 발표문의 벤치마크도 그 상태에서 측정한 값입니다. 그런데도 개발사가 스스로를 커스터마이징용 베이스라고 규정하고 Tinker에서 즉시 fine-tune 가능하다는 점을 앞세운 건, 최종 사용 형태를 추가 학습 이후로 보고 있다는 신호로 읽힙니다.

기업 입장에서 이 구분이 걸리는 지점은 조직 고유의 판단 기준을 모델에 반영하는 방법이 하나가 아니라는 데 있습니다. 프롬프트와 RAG로 푸는 경로가 있고, 가중치를 학습시키는 경로가 있습니다. 어느 쪽이 나은지는 작업과 데이터에 따라 갈리지만, 후자를 선택지로 두려면 조건이 붙습니다. 가중치가 공개돼 있어야 하고, 학습 파이프라인이 있어야 하고, 라이선스가 허용해야 합니다. Inkling은 이 세 조건을 함께 맞춘 구성입니다.

## 아키텍처 개요

기본 스펙부터 정리합니다.

| 항목 | 값 |
|---|---|
| 아키텍처 | Mixture-of-Experts Transformer |
| 전체 파라미터 | 975B |
| token당 활성 파라미터 | 41B (4.2%) |
| decoder layer | 66개 (55개 local, 11개 global) |
| context 길이 | 최대 1M tokens |
| 사전학습 데이터 | 45조 tokens (text, image, audio, video) |
| 입력 모달리티 | text, image, audio 네이티브 |
| thinking effort | 0.2 &ndash; 0.99 조절 가능 |
| 라이선스 | Apache 2.0 |

MoE 설계는 대체로 DeepSeek-V3 계열을 따릅니다. 대규모 오픈웨이트 MoE에서 자주 보이는 구성입니다. Nemotron 3 Ultra처럼 Mamba를 섞은 하이브리드가 아니라 일반적인 Transformer decoder이고, attention도 MLA가 아닌 conventional GQA입니다.

Raschka는 이 GQA 선택을 두고 raw decoding 속도가 Inkling의 주된 무기는 아닐 수 있다고 관찰합니다. 단정은 아니지만 설계를 읽는 단서가 됩니다. MLA가 KV 캐시를 압축하는 방향으로 긴 컨텍스트 비용을 다루는 기법이라면, Inkling은 attention 구성 자체를 손댔습니다. 그쪽이 다음 두 절의 주제입니다.

![Inkling MoE decoder layer 구조. token embedding 직후 추가 RMSNorm, 512-token sliding-window attention, K/V projection 뒤 kernel-4 convolution, 256 routed + 2 shared expert 구성을 표시한 다이어그램](/ai-tech-blog/images/inkling-975b-moe-open-weights-anatomy/moe-decoder-layer.png)

*Inkling의 decoder layer 구조. 스펙 출처: [Thinking Machines Lab 발표문](https://thinkingmachines.ai/news/introducing-inkling/), [모델카드](https://thinkingmachines.ai/model-card/inkling/), [Raschka 아키텍처 노트](https://sebastianraschka.com/blog/2026/inkling-architecture-benchmark-notes.html)*

## MoE 라우팅 뜯어보기

각 MoE layer는 256개 routed expert와 2개 shared expert로 구성되고, token 하나는 routed 중 6개만 통과합니다. shared expert는 라우팅과 무관하게 모든 token이 지납니다. 라우팅 대상과 무조건 통과 경로를 나눠 둔 이 배치는 DeepSeek-V3 계열에서 넘어온 형태입니다.

라우터는 sigmoid 기반이고, load balancing은 auxiliary loss 대신 <strong>bias 항</strong>으로 처리합니다. MoE 학습에서 expert 사용량을 고르게 만드는 전통적 방법은 별도 손실 항을 더하는 것인데, 이 항은 본래 목표인 언어 모델링과 같은 손실 안에서 경쟁합니다. aux-loss-free 방식은 라우터 점수에 학습되는 bias를 붙여 균형을 맞추기 때문에 그 항이 없습니다. 선택된 routed expert와 shared expert의 점수는 함께 normalize됩니다.

추가 학습 관점에서 이 구성이 어떻게 작동할지는 실제로 돌려봐야 알 수 있는 영역입니다. auxiliary loss가 없다는 건 downstream 학습에서 손실을 재구성할 때 다룰 항이 하나 적다는 뜻이지만, 도메인 데이터로 계속 학습할 때 expert 사용 분포가 어떻게 움직이는지는 공개된 정보로는 판단할 수 없습니다.

sparsity 수준은 비교해서 볼 만합니다.

![Inkling 975B/41B, Kimi K2.5 1T/32B, GLM-5.2 744B/40B의 total 대비 active 파라미터 비교. 하단은 활성 비율 Inkling 4.2% vs Kimi K2.5 3.2%](/ai-tech-blog/images/inkling-975b-moe-open-weights-anatomy/sparsity-comparison.png)

*활성 파라미터 비율 비교. 수치 출처: [Thinking Machines Lab 발표문](https://thinkingmachines.ai/news/introducing-inkling/), [Raschka 아키텍처 노트](https://sebastianraschka.com/blog/2026/inkling-architecture-benchmark-notes.html)*

Inkling의 4.2%는 Kimi K2.5의 3.2%보다 덜 sparse합니다. GLM-5.2는 744B 중 40B를 쓰기 때문에 active footprint가 Inkling과 거의 같은데, total은 231B 작습니다.

이 두 숫자는 서빙 비용의 서로 다른 축을 건드립니다. active 파라미터는 token당 연산량 쪽에, total 파라미터는 가중치를 올려둘 메모리 쪽에 걸립니다. Inkling을 자체 인프라에 올린다면 GLM-5.2와 비슷한 active 규모를 다루면서 975B를 담을 메모리를 준비해야 합니다. 실제 지연과 처리량은 라우팅 오버헤드, attention 구성, 메모리 대역폭까지 함께 결정하므로 active 수치만으로 환산되지는 않습니다.

## long-context를 위한 세 가지 선택

Raschka가 "surprises"라고 표현한 설계가 세 가지 있습니다. 흔한 레시피에서 벗어난 지점들입니다.

<strong>sliding-window attention의 비율.</strong> 66개 decoder layer 중 55개가 512-token 창의 local attention이고, global attention은 11개뿐입니다. 전체의 83%가 국소 문맥만 봅니다. 이 비율은 KV 캐시와 직접 연결됩니다. global attention layer의 캐시는 시퀀스 길이에 비례해 커지지만, 512-token 창 layer는 상한이 고정됩니다. 1M context를 지원하는 모델에서 layer 대부분의 캐시가 상수로 묶이는 구성입니다. 실제 절감량은 구현과 배치 조건에 따라 달라지므로 별도 측정이 필요합니다.

<strong>RoPE 제거.</strong> 대신 학습된, 입력에 의존하는 relative-position bias를 씁니다. TML은 이 방식이 더 좋고 긴 시퀀스 외삽에도 유리하다고 주장합니다. 세부가 흥미롭습니다. global attention layer에서도 learned bias는 직전 1,024 tokens 범위에만 적용되고, 그보다 먼 거리는 위치 정보 없이 내용 기반으로만 처리됩니다. 위치 인코딩을 아예 두지 않는 NoPE 계열 연구가 관찰한 방향과 통하는 구성입니다.

<strong>짧은 convolution.</strong> 각 decoder layer에서 K/V projection 뒤, 그리고 attention과 MLP branch의 출력에 kernel-4 convolution이 붙습니다. 커널 4는 인접 4개 token만 섞는 좁은 연산이고, Raschka는 이를 값싼 국소 token mixing이자 단거리 inductive bias로 설명합니다. attention 대부분을 512 창으로 좁힌 구조와 나란히 놓고 보면 근거리 정보를 저렴한 연산에 맡기는 배분으로 읽힙니다.

세 선택이 향하는 곳은 같아 보입니다. 긴 컨텍스트를 지원하면서 그 비용을 attention 구성 쪽에서 눌러 두는 방향입니다. 긴 문서나 코드베이스, 대화 로그를 학습 입력으로 쓰는 쪽에는 유리한 형태이지만, 실제 학습 비용은 하드웨어와 배치 크기에 따라 달라집니다.

## 멀티모달: encoder-free

이미지와 오디오 처리에서 Inkling은 별도 encoder를 두지 않습니다. 전용 vision encoder로 임베딩을 뽑아 언어 모델에 이어붙이는 방식이 아니라, 모달리티를 토큰화해 본체에 직접 넣는 구성입니다.

오디오는 dMel spectrogram 표현으로 입력됩니다. [dMel 논문](https://arxiv.org/abs/2407.15835)이 제안한 방식으로, mel filterbank 에너지를 이산화해 별도 음성 토크나이저 학습 없이 언어 모델 입력으로 쓸 수 있게 만든 표현입니다. 실무 스펙으로는 WAV 16kHz 입력을 받고 20분 이내가 최적입니다.

이미지는 40x40 픽셀 패치로 자르고 4-layer hMLP를 거쳐 임베딩이 됩니다. hMLP patch embedding은 [Touvron 등의 연구](https://arxiv.org/abs/2203.09795)에서 온 구성입니다.

두 선택 모두 모달리티별 encoder를 두지 않는 쪽입니다. 컴포넌트 수가 적으면 추가 학습에서 다룰 부분도 줄어들지만, 멀티모달 파인튜닝이 실제로 어떻게 동작하는지는 공개 자료에 없습니다.

## 벤치마크 정직하게 읽기

발표문 벤치마크 표는 effort 0.99, temperature 1.0 기준입니다. 결과를 정리하면 성격이 뚜렷하게 갈립니다.

추론과 에이전틱 코딩에서는 상위 오픈웨이트 모델보다 낮은 점수가 나왔습니다.

| 벤치마크 | Inkling | GLM-5.2 | Kimi K2.6 |
|---|---|---|---|
| HLE (text only) | 29.7% | 40.1% | 35.9% |
| HLE (with tools) | 46.0% | 54.7% | - |
| SWEBench Verified | 77.6% | 80.0% | 80.2% |
| Terminal Bench 2.1 | 63.8% | 82.7% | 71.3% |

SWEBench Verified는 bash-only harness 조건에서 77.6%로 2&ndash;3%p 차이지만, Terminal Bench 2.1에서는 격차가 훨씬 큽니다.

반대 방향의 지표들도 있습니다.

| 벤치마크 | Inkling | 비교 |
|---|---|---|
| IFBench (instruction following) | 79.8% | GLM-5.2 73.3% |
| SimpleQA Verified | 43.9% | GLM-5.2 38.1% |
| ForecastBench (calibration, no search) | 61.1 | Gemini 3.1 Pro 동률, GPT-5.5 59.1 |
| FORTRESS Adversarial | 78.0% | 비교 오픈웨이트 중 최고 |
| StrongREJECT | 98.6% | - |

지시 따르기, 짧은 사실 질의(SimpleQA Verified), 안전성 쪽에서는 상위권입니다. ForecastBench는 사실 정확도가 아니라 예측 확률의 calibration을 재는 평가인데, 여기서 Inkling은 검색 없이 61.1로 Gemini 3.1 Pro와 동률이고 GPT-5.5(59.1)보다 높습니다. 수학과 과학에서는 AIME 2026 97.1%, GPQA Diamond 87.2%이고, 멀티모달은 MMMU Pro 73.5%, VoiceBench 91.4%, MMAU 77.2%, Audio MC 56.6%입니다.

이 프로필은 정렬 학습 설계와 방향이 맞습니다. calibration은 결과가 확정된 실세계 질문에 proper scoring rule을 걸고 RL로 학습시켰고, instruction following에는 rubric grader와 claims grader를 썼습니다. claims grader는 에이전트가 웹 검색으로 주장의 사실성을 검증하는 방식입니다. 여기에 abstention-aware reward가 붙어 확신이 있을 때만 답하고 아니면 모른다고 말하도록 학습됐습니다. 검열 저항성 쪽에서는 Cognition의 Propaganda/Censorship Eval에서 강한 결과를 보였습니다. 다만 특정 벤치마크 점수를 특정 학습 기법의 결과로 인과적으로 연결하는 건 발표문이 보여준 범위를 넘습니다.

Raschka의 평가가 이 프로필을 잘 요약합니다. 모든 벤치마크 1등이 아니어서 실망할 수 있지만, 넓고 섞인 프로필이 오히려 정직하다는 것입니다. 추가 학습으로 목표하는 건 대개 도메인 성능이고, 지시 따르기나 확신도 관리 같은 일반 역량은 베이스에서 출발합니다. 다만 도메인 학습이 일반 역량을 깎을 수 있다는 건 잘 알려진 문제이므로, 어느 프로필이 출발점으로 나은지는 목적과 학습 설계에 따라 갈립니다.

## 커스터마이징이 실제로 뜻하는 것

발표문의 데모 구성은 커스터마이징 주장을 뒷받침하는 쪽으로 짜여 있습니다.

가장 눈에 띄는 건 Inkling이 스스로 자신의 fine-tuning job을 작성하고 실행하고 평가한 데모입니다. Tinker 위에서 벌어지는 일이므로 학습 API가 모델이 다룰 수 있는 표면으로 노출되어 있다는 뜻이 됩니다. 학습 루프를 코드로 짜는 대신 모델에 맡기는 방향을 보여주는 데모입니다.

thinking effort를 0.2에서 0.99까지 조절할 수 있다는 점도 배포 관점에서 의미가 있습니다. 같은 가중치로 지연과 품질 사이를 옮겨 다닐 수 있으면, 워크로드마다 별도 모델을 준비하는 대신 파라미터로 대응해 볼 여지가 생깁니다. 어느 effort 값이 어떤 작업에 맞는지는 자체 평가로 잡아야 하는 부분입니다.

나머지 데모는 에이전틱 활용 쪽입니다. 브라우저 사용 에이전트를 내장한 웹앱을 한 번에 만들어냈고, GPT Codex를 리뷰어로 두고 40회 반복하며 멀티플레이어 스네이크 게임을 개선했습니다. Design Arena의 Agentic Web Dev 부문에서는 1257점으로 오픈웨이트 상위권이고 Claude Opus 4.6과 동률입니다. Terminal Bench 점수가 낮은 모델이 웹 개발 에이전트 평가에서는 상위권이라는 대비는, 에이전틱 성능을 하나의 숫자로 요약하기 어렵다는 점을 보여줍니다.

경량판도 있습니다. Inkling-Small은 12B active 규모로 동일한 학습 레시피를 적용한 프리뷰 버전입니다. 같은 레시피의 작은 모델이 있으면 실험을 작은 쪽에서 먼저 돌려보는 선택지가 생깁니다.

## 엔지니어 관점 정리

공개된 스펙과 벤치마크만으로 후보를 좁히는 기준을 정리하면 이렇습니다. 아래는 검증된 제품 특성이 아니라 그 자료를 읽은 제 판단입니다.

<strong>후보로 볼 만한 경우.</strong> 도메인 데이터로 가중치를 학습시킬 계획이 있고, 파생 모델을 직접 배포해야 하는 상황입니다. 긴 문서나 로그가 입력의 중심이면 sliding-window 중심 설계와 1M context가 맞는 방향입니다. 정해진 형식과 제약을 지켜야 하는 작업이라면 IFBench 점수가, 확률로 답해야 하는 판단 업무라면 ForecastBench의 calibration 결과가 참고가 됩니다. 텍스트와 이미지와 오디오를 한 모델로 받아야 하는 경우도 후보에 들어갑니다.

<strong>다른 선택이 나은 경우.</strong> 파인튜닝 계획 없이 성능만 필요하다면 벤치마크가 앞서는 모델을 API로 쓰는 쪽이 단순합니다. 에이전틱 코딩과 터미널 자동화가 핵심이면 Terminal Bench 격차를 무시하기 어렵습니다. 대량 서빙에서는 41B active를 위해 975B를 메모리에 얹는 구조를 계산에 넣어야 하고, MLA가 아닌 GQA라는 점도 처리량 추정에 영향을 줍니다.

Inkling의 아키텍처 선택들에서 읽히는 건 벤치마크 상위 몇 %p를 겨냥한 최적화가 아닙니다. sliding-window 비율, RoPE 제거, kernel-4 convolution, aux-loss-free 라우팅은 모두 긴 컨텍스트 처리 비용과 학습 구조 쪽을 건드리는 결정들입니다.

그래서 발표문의 "최강이 아니다"라는 문장은 겸양 표현이라기보다 설계 목표의 서술로 읽힙니다. 최강 모델이 필요한 자리와 커스터마이징 베이스가 필요한 자리는 다르고, Inkling은 후자를 향해 있습니다. 다만 이 글은 공개 스펙과 발표 벤치마크를 근거로 한 분석입니다. 파인튜닝이 실제로 얼마나 잘 되는지, 학습이 얼마나 안정적인지는 자체 데이터로 돌려봐야 알 수 있고, 가중치가 공개돼 있다는 사실이 그 확인을 가능하게 만든다는 점이 이 출시의 실용적인 부분입니다.

## References

- Thinking Machines Lab. (2026). *Introducing Inkling.* [thinkingmachines.ai/news/introducing-inkling](https://thinkingmachines.ai/news/introducing-inkling/)
- Thinking Machines Lab. (2026). *Inkling Model Card.* [thinkingmachines.ai/model-card/inkling](https://thinkingmachines.ai/model-card/inkling/)
- Raschka, S. (2026). *Inkling Architecture and Benchmark Notes.* [sebastianraschka.com/blog/2026/inkling-architecture-benchmark-notes.html](https://sebastianraschka.com/blog/2026/inkling-architecture-benchmark-notes.html)
- Thinking Machines Lab. *Inkling config.json (Hugging Face).* [huggingface.co/thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling/blob/main/config.json)
- Thinking Machines Lab. *Inkling 모델 저장소 (라이선스 표기).* [huggingface.co/thinkingmachines/Inkling](https://huggingface.co/thinkingmachines/Inkling)
- Apache Software Foundation. *Apache License, Version 2.0.* [apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0)
- Bai, R. H., et al. (2024). *dMel: Speech Tokenization made Simple.* [arxiv.org/abs/2407.15835](https://arxiv.org/abs/2407.15835)
- Touvron, H., et al. (2022). *Three things everyone should know about Vision Transformers.* [arxiv.org/abs/2203.09795](https://arxiv.org/abs/2203.09795)
