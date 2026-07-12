---
title: "Hybrid Thinking은 왜 갈라섰나 — Reasoning에서 Agentic Thinking으로"
date: 2026-07-12T17:00:00+09:00
draft: false
categories: ["논문 리뷰"]
tags: ["Reasoning", "Agentic AI", "Reinforcement Learning", "Qwen", "LLM", "Agentic RL"]
author: "Jesam Kim"
description: "Qwen 前 테크리드 Junyang Lin이 정리한 hybrid thinking의 한계, 그리고 reasoning thinking에서 agentic thinking으로의 이동을 학술 인프라(agentic RL) 흐름과 교차해 읽습니다."
cover:
  image: "/ai-tech-blog/images/end-of-hybrid-thinking-agentic-shift/cover.png"
  alt: "Hybrid Thinking은 왜 갈라섰나 — Reasoning에서 Agentic Thinking으로"
  relative: false
---

2026년 상반기 AI reasoning 담론에는 한 가지 방향 전환이 있었습니다. o1과 DeepSeek-R1이 연 "긴 사고 사슬(long chain-of-thought)"의 시대에서, 사고 자체를 행동의 도구로 재정의하는 흐름으로의 이동입니다. 이 전환을 비교적 명료하게 정리한 사람이 Alibaba Qwen 프로젝트의 前 테크리드 Junyang Lin입니다. 그는 [2026년 3월 3일 Qwen 테크리드에서 사임](https://techcrunch.com/2026/03/03/alibabas-qwen-tech-lead-steps-down-after-major-ai-push/)했고, 현재는 독립연구자로서 "training models에서 training agents로"라는 명제를 이야기하고 있습니다.

이 글은 지난 7월 초 "reasoning trace를 읽는 법"에서 다룬 논의의 후속입니다. 이번에는 Lin의 정리를 축으로, 담론과 학술 인프라(agentic RL)가 어디서 만나는지를 살펴봅니다.

## Qwen의 하이브리드 실험과 그 정리

Lin의 강연 [Qwen: Towards a Generalist Model / Agent](https://www.youtube.com/watch?v=b0xlsQ_6wUQ)는 Qwen 패밀리의 궤적을 훑는 투어입니다. QwQ-32B, Qwen2.5-Max, Qwen3, Qwen2.5-VL, Qwen2.5-Omni를 거치며 DeepSeek-R1, Grok 3 Beta, Gemini 2.5 Pro, OpenAI o-series와의 벤치마크 비교를 지나갑니다. 그리고 마지막 슬라이드에 한 줄이 남습니다. "Training models -> training agents."

Qwen3는 <strong>하이브리드 thinking</strong>을 내세운 모델이었습니다. 단계적으로 추론하는 thinking mode와 즉답하는 non-thinking mode를 한 모델에 담고, 추론량의 상한을 설정하는 dynamic thinking budget을 두었습니다. 지원 언어를 29개에서 119개로 넓혔고, 0.6B부터 235B까지의 규모를 Apache 2.0으로, GGUF/GPTQ/AWQ/MLX 양자화와 함께 공개했습니다. 두 모드를 한 가중치에 통합하려는 시도였습니다.

## 두 모드는 왜 상반되나

Lin의 진단은 두 모드의 최적화 방향이 서로 반대라는 데서 출발합니다. instruct mode는 직접성, 간결함, 낮은 지연을 보상받습니다. thinking mode는 어려운 문제에 토큰을 더 쓰는 쪽을 보상받습니다. 무리하게 병합하면 양쪽이 다 나빠집니다. thinking은 비대해지고, instruct는 덜 명료해집니다.

Qwen3는 4단계 post-training으로 이를 붙이려 했습니다. long-CoT cold start, reasoning RL, 그리고 thinking mode fusion을 포함한 단계들입니다. 하지만 2025년 후반의 [2507 라인은 Instruct와 Thinking을 별도 변형으로 분리 출시](https://www.theregister.com/software/2025/07/31/alibaba-admits-qwen3s-hybrid-thinking-mode-was-dumb/1284173)했습니다. Lin은 이를 모델의 문제라기보다 데이터의 문제로 규정합니다.

여기서 Lin은 Anthropic의 반대 경로를 "유용한 교정(useful corrective)"으로 언급합니다. Claude 3.7 Sonnet은 사용자가 thinking budget을 설정하는 하이브리드로 출시됐고, Claude 4는 reasoning을 tool use와 interleave하는 방향, 즉 코딩과 장기 작업을 겨냥한 설계로 갔습니다. Lin의 요점은 이렇게 정리됩니다. 더 긴 reasoning trace가 모델을 더 똑똑하게 만들지는 않으며, thinking은 벤치마크가 아니라 타깃 워크로드에 맞춰 형성되어야 한다는 것입니다.

## Reasoning thinking과 Agentic thinking

Lin은 사고의 두 시대를 구분합니다.

1시대는 reasoning thinking입니다. o1과 DeepSeek-R1이 정의했습니다. 여기서 RL은 결정론적이고 검증 가능한 보상을 필요로 했고, 그래서 math, code, logic이 중심이 되었습니다. 대규모 rollout과 검증이 시스템 차원의 문제로 떠올랐습니다.

2시대는 agentic thinking, 곧 "행동하기 위한 사고(thinking in order to act)"입니다. 에이전트가 계획을 세우고, 언제 행동할지 결정하고, 도구를 쓰고, 환경의 피드백을 읽고, 계획을 고칩니다. 긴 내적 독백이 아니라 세계와의 닫힌 루프(closed-loop) 상호작용으로 사고가 정의됩니다.

reasoning은 회피할 수 있었지만 agentic thinking은 반드시 다뤄야 하는 문제들이 있습니다. 언제 사고를 멈추고 행동할지, 어떤 도구를 어떤 순서로 부를지, 환경의 noisy하고 부분적인 관측을 어떻게 통합할지, 실패 후 계획을 어떻게 고칠지, 여러 턴과 여러 도구 호출에 걸친 일관성을 어떻게 유지할지가 그것입니다.

![reasoning thinking에서 agentic thinking으로의 전환: 평가 기준, 보상 신호, 학습 대상, 인프라 병목, 실패 모드 5개 축 대비](/ai-tech-blog/images/end-of-hybrid-thinking-agentic-shift/reasoning-to-agentic.png)

*reasoning thinking과 agentic thinking의 다섯 축 대비. 출처: Junyang Lin의 정리를 재구성.*

두 사고를 다섯 축으로 대비하면 다음과 같습니다.

| 차원 | Reasoning thinking | Agentic thinking |
|------|--------------------|------------------|
| 평가 기준 | 답 전 내부 숙고의 질 | 행동하며 진전이 유지되는가 |
| 보상 신호 | 검증 가능한 답(math/code/logic) | 상호작용 환경에서의 과업 성공 |
| 학습의 핵심 대상 | 모델 | 모델 + 환경(하네스) |
| 인프라 병목 | rollout, 검증, 안정적 policy 업데이트 | tool server, sandbox, train-serve 분리 |
| 주 실패 모드 | 장황하고 저가치인 reasoning trace | 도구 접근·환경 누수를 통한 reward hacking |

이 표에서 눈여겨볼 지점은 학습 대상의 이동입니다. reasoning thinking에서는 모델만 학습하면 됐지만, agentic thinking에서는 모델과 함께 환경, 즉 하네스가 학습 시스템의 일부가 됩니다. 실패 모드도 옮겨갑니다. 장황한 trace 문제에서, 도구 접근과 환경 누수를 통한 reward hacking 문제로 무게중심이 바뀝니다.

## 담론과 학술 인프라의 교차점

Lin이 말한 "인프라 병목의 이동", 곧 train-serve 분리와 tool server/sandbox의 중요성은 담론에만 머무르지 않습니다. 같은 시기 학술 쪽에서도 agentic RL을 시스템 레벨에서 뒷받침하는 작업들이 나오고 있습니다.

[AGENTRL](https://openreview.net/forum?id=zq3vAmuUk9)은 비동기 rollout–training 파이프라인으로 여러 종류의 task와 환경에서 LLM 에이전트의 RL 학습을 다루는 시스템입니다. [ProRL Agent](https://huggingface.co/papers/2603.18815)는 "Rollout-as-a-Service"를 표방하며 멀티턴 LLM 에이전트 RL 학습을 위한 확장 가능한 rollout 서비스를 제안합니다. 이 밖에도 칭화대의 분산 RL 인프라 프레임워크 RLinf가 embodied·agentic AI를 겨냥하고 있고, "Training Recipes for Agentic RL in LLMs"처럼 이 분야의 학습 레시피를 정리하는 서베이 작업도 진행되고 있습니다.

이들이 공통으로 건드리는 것이 바로 rollout과 tool server, sandbox, 그리고 train-serve 분리입니다. Lin이 담론 차원에서 지목한 병목의 이동을, 학술 인프라 쪽에서 실제 시스템 설계로 확인하는 셈입니다.

## 정리

Lin의 정리를 한 문장으로 줄이면, 사고를 벤치마크 점수를 위한 것이 아니라 워크로드를 위한 것으로 다시 보자는 제안입니다. 하이브리드 thinking이 한 모델 안에서 두 목적을 병합하려다 분리로 돌아간 경험은, 사고의 방향이 보상 설계와 인프라 구조에 종속된다는 점을 드러냅니다.

reasoning thinking에서 agentic thinking으로의 이동이 곧바로 전자의 종말을 뜻하지는 않습니다. math와 code의 검증 가능한 보상은 여전히 유효한 학습 신호입니다. 다만 평가와 보상, 학습 대상과 인프라의 무게중심이 "행동하며 진전을 유지하는가"로 옮겨가고 있다는 것이, 前 Qwen 리드가 담론과 학술의 교차점에서 그린 그림입니다.

## References

- MarkTechPost, "Qwen's former lead on what hybrid thinking got wrong and why he now backs agents" — https://www.marktechpost.com/2026/07/04/qwens-former-lead-on-what-hybrid-thinking-got-wrong-and-why-he-now-backs-agents/
- Junyang Lin, "Qwen: Towards a Generalist Model / Agent" (강연 영상) — https://www.youtube.com/watch?v=b0xlsQ_6wUQ
- TechCrunch, "Alibaba's Qwen tech lead steps down after major AI push" — https://techcrunch.com/2026/03/03/alibabas-qwen-tech-lead-steps-down-after-major-ai-push/
- 36Kr, Junyang Lin 관련 보도 — https://eu.36kr.com/en/p/3740825962135558
- The Register, "Alibaba admits Qwen3's hybrid thinking mode was dumb" — https://www.theregister.com/software/2025/07/31/alibaba-admits-qwen3s-hybrid-thinking-mode-was-dumb/1284173
- AGENTRL (OpenReview) — https://openreview.net/forum?id=zq3vAmuUk9
- ProRL Agent (Hugging Face Papers) — https://huggingface.co/papers/2603.18815
