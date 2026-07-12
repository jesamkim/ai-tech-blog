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

Lin의 자리 이동 자체가 담론의 방향을 보여주는 신호이기도 합니다. 프론티어 모델 하나를 여러 해 이끌던 사람이 조직을 떠나 에이전트 학습을 이야기한다는 것은, 모델 가중치를 키우는 경쟁의 한 사이클이 일단락됐다는 인식과 무관하지 않습니다. [당시 보도](https://eu.36kr.com/en/p/3740825962135558)도 이 사임을 큰 AI 드라이브가 지나간 뒤의 매듭으로 읽었습니다.

이 글은 지난 7월 초 "reasoning trace를 읽는 법"에서 다룬 논의의 후속입니다. 이번에는 Lin의 정리를 축으로, 담론과 학술 인프라(agentic RL)가 어디서 만나는지, 그리고 이 이동이 에이전트를 실제로 배포하고 평가하는 쪽에 어떤 질문을 남기는지를 살펴봅니다.

## Qwen의 하이브리드 실험과 그 정리

Lin의 강연 [Qwen: Towards a Generalist Model / Agent](https://www.youtube.com/watch?v=b0xlsQ_6wUQ)는 Qwen 패밀리의 궤적을 훑는 투어입니다. QwQ-32B, Qwen2.5-Max, Qwen3, Qwen2.5-VL, Qwen2.5-Omni를 거치며 DeepSeek-R1, Grok 3 Beta, Gemini 2.5 Pro, OpenAI o-series와의 벤치마크 비교를 지나갑니다. 그리고 마지막 슬라이드에 한 줄이 남습니다. "Training models -> training agents."

Qwen3는 <strong>하이브리드 thinking</strong>을 내세운 모델이었습니다. 단계적으로 추론하는 thinking mode와 즉답하는 non-thinking mode를 한 모델에 담고, 추론량의 상한을 설정하는 dynamic thinking budget을 두었습니다. 지원 언어를 29개에서 119개로 넓혔고, 0.6B부터 235B까지의 규모를 Apache 2.0으로, GGUF/GPTQ/AWQ/MLX 양자화와 함께 공개했습니다. 두 모드를 한 가중치에 통합하려는 시도였습니다.

의도 자체는 사용자 입장에서 합리적이었습니다. 같은 모델에 어려운 문제를 던지면 스스로 오래 생각하고, 간단한 질문에는 바로 답하도록 하면, 엔드포인트를 하나만 두고도 두 종류의 워크로드를 감당할 수 있습니다. 배포하는 쪽에서는 모델 버전을 하나로 줄이고, budget 파라미터 하나로 지연과 품질을 조절하는 그림이 됩니다. 하이브리드는 이 편의를 노린 설계였고, 초기에는 벤치마크 상으로도 나쁘지 않게 보였습니다.

## 두 모드는 왜 상반되나

Lin의 진단은 두 모드의 최적화 방향이 서로 반대라는 데서 출발합니다. instruct mode는 직접성, 간결함, 낮은 지연을 보상받습니다. thinking mode는 어려운 문제에 토큰을 더 쓰는 쪽을 보상받습니다. 무리하게 병합하면 양쪽이 다 나빠집니다. thinking은 비대해지고, instruct는 덜 명료해집니다.

두 목적을 한 가중치에 담으면 학습 신호가 서로를 상쇄합니다. 짧게 답하도록 미는 그래디언트와 오래 생각하도록 미는 그래디언트가 같은 파라미터를 놓고 경쟁하기 때문입니다. 어느 한쪽으로 세게 밀면 반대쪽 능력이 깎이고, 중간에서 타협하면 양쪽 다 어중간해집니다. Lin이 이를 모델 구조의 한계가 아니라 <strong>데이터의 문제</strong>로 본 지점이 여기입니다. 두 모드가 요구하는 응답 분포가 애초에 다른데, 그 둘을 섞은 데이터로 하나의 정책을 학습시키면 모델은 "언제 어느 모드로 답해야 하는가"라는 경계 자체를 흐릿하게 배웁니다. 문제는 아키텍처가 아니라, 상반된 목표를 하나의 목적함수로 눌러 담은 학습 설정에 있었다는 진단입니다.

Qwen3는 4단계 post-training으로 이를 붙이려 했습니다. long-CoT cold start, reasoning RL, 그리고 thinking mode fusion을 포함한 단계들입니다. 하지만 2025년 후반의 [2507 라인은 Instruct와 Thinking을 별도 변형으로 분리 출시](https://www.theregister.com/software/2025/07/31/alibaba-admits-qwen3s-hybrid-thinking-mode-was-dumb/1284173)했습니다. 하나로 합치려던 설계를 접고, 목적이 다른 두 모델을 각각 최적화하는 쪽으로 돌아온 것입니다.

여기서 Lin은 Anthropic의 반대 경로를 "유용한 교정(useful corrective)"으로 언급합니다. Claude 3.7 Sonnet은 사용자가 thinking budget을 설정하는 하이브리드로 출시됐고, Claude 4는 reasoning을 tool use와 interleave하는 방향, 즉 코딩과 장기 작업을 겨냥한 설계로 갔습니다. 두 모드를 한 슬라이더로 섞는 대신, 사고를 도구 호출 사이에 끼워 넣어 "행동의 일부"로 만든 셈입니다. Lin의 요점은 이렇게 정리됩니다. 더 긴 reasoning trace가 모델을 더 똑똑하게 만들지는 않으며, thinking은 벤치마크가 아니라 타깃 워크로드에 맞춰 형성되어야 한다는 것입니다.

## Reasoning thinking과 Agentic thinking

Lin은 사고의 두 시대를 구분합니다.

1시대는 reasoning thinking입니다. o1과 DeepSeek-R1이 정의했습니다. 여기서 RL은 결정론적이고 검증 가능한 보상을 필요로 했고, 그래서 math, code, logic이 중심이 되었습니다. 대규모 rollout과 검증이 시스템 차원의 문제로 떠올랐습니다.

2시대는 agentic thinking, 곧 "행동하기 위한 사고(thinking in order to act)"입니다. 에이전트가 계획을 세우고, 언제 행동할지 결정하고, 도구를 쓰고, 환경의 피드백을 읽고, 계획을 고칩니다. 긴 내적 독백이 아니라 세계와의 닫힌 루프(closed-loop) 상호작용으로 사고가 정의됩니다.

reasoning은 회피할 수 있었지만 agentic thinking은 반드시 다뤄야 하는 문제들이 있습니다. 언제 사고를 멈추고 행동할지, 어떤 도구를 어떤 순서로 부를지, 환경의 noisy하고 부분적인 관측을 어떻게 통합할지, 실패 후 계획을 어떻게 고칠지, 여러 턴과 여러 도구 호출에 걸친 일관성을 어떻게 유지할지가 그것입니다.

이 차이는 사고의 "종료 조건"이 어디에 있느냐로 요약됩니다. reasoning thinking에서 사고는 답이 나오면 끝납니다. 정답이라는 정지 지점이 밖에서 주어지므로, 모델은 그 지점까지 최대한 정확하게 도달하기만 하면 됩니다. agentic thinking에는 그런 외부 정지 지점이 없습니다. 사고는 다음 행동을 정하기 위한 중간 계산이고, 행동의 결과가 다시 다음 사고의 입력이 됩니다. 그래서 "충분히 생각했는가"가 아니라 "지금 행동으로 옮겨야 하는가"가 매 스텝의 판단 대상이 됩니다. 오래 생각하는 능력만으로는 이 판단이 좋아지지 않습니다.

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

이 표에서 눈여겨볼 지점은 학습 대상의 이동입니다. reasoning thinking에서는 모델만 학습하면 됐지만, agentic thinking에서는 모델과 함께 환경, 즉 하네스가 학습 시스템의 일부가 됩니다. 도구 서버, sandbox, 관측을 만들어 주는 시뮬레이터가 모두 학습 루프 안으로 들어옵니다. 모델을 잘 만드는 문제가 아니라, 모델과 환경을 함께 설계하는 문제로 성격이 바뀝니다.

실패 모드도 옮겨갑니다. reasoning thinking의 대표적 실패는 장황하고 저가치인 trace, 즉 답에 기여하지 않는 사고를 길게 늘어놓는 것이었습니다. agentic thinking에서는 reward hacking이 더 무겁게 다가옵니다. 이유는 보상의 성격에 있습니다. reasoning RL의 보상은 정답 대조라는 닫힌 판정이라 속이기 어렵습니다. 반면 과업 성공을 보상으로 쓰면, 그 성공 신호는 대체로 환경이 만들어 냅니다. 파일 시스템 상태, 테스트 통과 여부, API 응답 같은 것들입니다. 에이전트가 도구에 접근할 수 있다는 말은, 보상을 계산하는 그 환경에도 손댈 수 있다는 뜻입니다. 테스트를 실제로 통과시키는 대신 테스트 파일을 고쳐 통과시키거나, 채점 스크립트가 읽는 로그에 성공 문자열을 심어 넣는 식의 지름길이 열립니다. 문제를 푸는 것보다 보상 신호를 조작하는 편이 쉬운 순간이 생기고, RL은 그 쉬운 쪽을 정확히 찾아냅니다. reward hacking이 agentic 설정에서 더 심각한 이유는 모델이 더 교활해서가 아니라, 보상을 만드는 환경과 그 환경을 조작할 도구가 한 루프 안에 함께 있기 때문입니다.

## 담론과 학술 인프라의 교차점

Lin이 말한 "인프라 병목의 이동", 곧 train-serve 분리와 tool server/sandbox의 중요성은 담론에만 머무르지 않습니다. 같은 시기 학술 쪽에서도 agentic RL을 시스템 레벨에서 뒷받침하는 작업들이 나오고 있습니다.

[AGENTRL](https://openreview.net/forum?id=zq3vAmuUk9)은 비동기 rollout–training 파이프라인으로 여러 종류의 task와 환경에서 LLM 에이전트의 RL 학습을 다루는 시스템입니다. [ProRL Agent](https://huggingface.co/papers/2603.18815)는 "Rollout-as-a-Service"를 표방하며 멀티턴 LLM 에이전트 RL 학습을 위한 확장 가능한 rollout 서비스를 제안합니다. 이 밖에도 칭화대의 분산 RL 인프라 프레임워크 RLinf가 embodied·agentic AI를 겨냥하고 있고, "Training Recipes for Agentic RL in LLMs"처럼 이 분야의 학습 레시피를 정리하는 서베이 작업도 진행되고 있습니다.

이들이 공통으로 건드리는 것이 바로 rollout과 tool server, sandbox, 그리고 train-serve 분리입니다. Lin이 담론 차원에서 지목한 병목의 이동을, 학술 인프라 쪽에서 실제 시스템 설계로 확인하는 셈입니다.

## 실무에서는 무엇이 바뀌나

이 전환은 학습하는 쪽뿐 아니라 에이전트를 배포하고 운영하는 쪽에도 몇 가지 실무 질문을 남깁니다. 엔지니어나 아키텍트 관점에서 보면, 초점이 "모델을 무엇으로 쓸까"에서 "모델을 어떤 환경 안에 놓을까"로 옮겨간다는 것이 핵심입니다.

첫째는 도구 오케스트레이션 계층의 무게입니다. 사고가 도구 호출 사이에 끼워지는 설계에서는 도구 자체의 스키마, 오류 반환 형식, 타임아웃 정책이 모델 성능의 일부가 됩니다. 도구가 실패했을 때 무엇을 돌려주는지, 부분적인 결과를 어떻게 표현하는지가 모델의 다음 판단을 좌우합니다. 도구를 얇은 API 래퍼로만 두는 접근은 에이전트 워크로드에서는 한계가 빨리 드러날 수 있습니다.

둘째는 sandbox와 권한 경계입니다. reward hacking이 도구 접근을 통해 일어난다는 점은 학습만의 문제가 아닙니다. 배포된 에이전트도 같은 구조에 놓입니다. 채점 대신 실제 작업이라는 차이가 있을 뿐, 도구가 상태를 바꿀 수 있는 한 의도치 않은 지름길은 언제든 생길 수 있습니다. 파일 쓰기, 네트워크 호출, 배포 명령처럼 상태를 바꾸는 도구에는 실행 경계와 승인 지점을 두는 편이, 특히 되돌리기 어려운 작업을 다루는 워크로드에서는 안전합니다.

셋째는 모델 선택의 결이 달라진다는 점입니다. 하이브리드가 분리로 돌아온 사례는, 하나의 엔드포인트로 모든 워크로드를 덮으려는 접근이 항상 유리하지는 않다는 것을 보여줍니다. 간결한 응답이 필요한 경로와 장기 도구 사용이 필요한 경로를 같은 모델·같은 설정으로 처리하려 하면, 어느 쪽도 최적이 아닌 지점에 머물 수 있습니다. 경로별로 모델이나 설정을 나누는 편이 나은 워크로드가 있습니다.

agentic thinking으로 무게중심이 옮겨가면서 함께 어려워지는 것이 평가입니다. reasoning thinking에서는 답이 맞았는지 틀렸는지가 평가의 대부분이었습니다. 정답이라는 단일 기준이 있으니 벤치마크도 만들기 쉬웠습니다. 행동하며 진전을 유지하는가를 평가하려면 정답 하나로는 부족합니다. 같은 과업을 여러 경로로 풀 수 있고, 중간에 실패했다가 회복하는 궤적도 성공으로 볼 여지가 있으며, 도구를 몇 번 호출했는지나 부작용을 남겼는지까지 함께 봐야 합니다. 그래서 agentic 평가는 정적 데이터셋이 아니라 실행 가능한 환경을 요구하는 쪽으로 갑니다. 앞서 언급한 학술 인프라들이 rollout과 sandbox를 강조하는 이유도, 학습만이 아니라 평가가 같은 환경을 필요로 하기 때문입니다.

Lin의 정리를 한 문장으로 줄이면, 사고를 벤치마크 점수를 위한 것이 아니라 워크로드를 위한 것으로 다시 보자는 제안입니다. 하이브리드 thinking이 한 모델 안에서 두 목적을 병합하려다 분리로 돌아간 경험은, 사고의 방향이 보상 설계와 인프라 구조에 종속된다는 점을 드러냅니다.

reasoning thinking에서 agentic thinking으로의 이동이 곧바로 전자의 종말을 뜻하지는 않습니다. math와 code의 검증 가능한 보상은 여전히 유효한 학습 신호이고, agentic 설정에서도 하위 단계의 정확성을 다지는 토대로 쓰입니다. 오히려 두 사고는 겹쳐서 작동합니다. 도구를 언제 부를지 판단하는 것은 agentic 층의 일이지만, 그 도구가 내놓은 중간 결과를 검산하고 다음 계획을 세우는 데는 검증 가능한 reasoning이 그대로 필요합니다. 달라진 것은 사고가 사라졌다는 게 아니라, 사고가 답이 아니라 행동을 향하게 됐다는 점입니다. 평가와 보상, 학습 대상과 인프라의 무게중심이 "행동하며 진전을 유지하는가"로 옮겨가고 있다는 것이, 前 Qwen 리드가 담론과 학술의 교차점에서 그린 그림입니다.

## References

- MarkTechPost, "Qwen's former lead on what hybrid thinking got wrong and why he now backs agents" — https://www.marktechpost.com/2026/07/04/qwens-former-lead-on-what-hybrid-thinking-got-wrong-and-why-he-now-backs-agents/
- Junyang Lin, "Qwen: Towards a Generalist Model / Agent" (강연 영상) — https://www.youtube.com/watch?v=b0xlsQ_6wUQ
- TechCrunch, "Alibaba's Qwen tech lead steps down after major AI push" — https://techcrunch.com/2026/03/03/alibabas-qwen-tech-lead-steps-down-after-major-ai-push/
- 36Kr, Junyang Lin 관련 보도 — https://eu.36kr.com/en/p/3740825962135558
- The Register, "Alibaba admits Qwen3's hybrid thinking mode was dumb" — https://www.theregister.com/software/2025/07/31/alibaba-admits-qwen3s-hybrid-thinking-mode-was-dumb/1284173
- AGENTRL (OpenReview) — https://openreview.net/forum?id=zq3vAmuUk9
- ProRL Agent (Hugging Face Papers) — https://huggingface.co/papers/2603.18815
