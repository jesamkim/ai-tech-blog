---
title: "9B 모델이 진화한 스킬로 27B 모델을 앞섰다: WikiSkill이 에이전트 경험을 지식으로 컴파일하는 방법"
date: 2026-08-30T09:00:00+09:00
draft: false
categories: ["논문 리뷰", "AI 에이전트"]
tags: ["WikiSkill", "Agent Skills", "Skill Evolution", "Persistent Knowledge", "LLM Agent", "Agentic AI", "Google Research"]
author: "Jesam Kim"
slug: "wikiskill-persistent-knowledge-skill-evolution"
description: "Google Research가 2026년 8월 27일 arXiv에 올린 WikiSkill은 실행 기록, 누적 지식, 실행 스킬을 서로 다른 저장소로 분리해서 스킬을 진화시킵니다. 다섯 벤치마크와 다섯 모델에서 확인된 수치를 정리하고, 이 구조를 엔터프라이즈 에이전트 운영으로 옮길 때 필요한 승격과 검증 경계를 함께 다룹니다."
cover:
  image: "/ai-tech-blog/images/wikiskill-persistent-knowledge-skill-evolution/wikiskill-cover.png"
  alt: "WikiSkill: 에이전트 경험을 지속 지식으로 컴파일하는 스킬 진화. raw, wiki, skills 세 저장소를 분리한 개념도"
---

스프레드시트를 다루는 에이전트에게 병합된 셀이 섞인 표를 정리하라고 시키면, 처음에는 대개 실패합니다. 라이브러리가 병합 셀을 어떻게 반환하는지 모른 채로 코드를 쓰기 때문입니다. 사람이 옆에서 원인을 알려주면 그 세션에서는 해결됩니다. 그런데 다음 주에 비슷한 표를 주면 같은 자리에서 다시 실패합니다.

이 상황에서 잃어버린 것이 무엇인지 따져 보면 조금 이상합니다. 실행 기록은 남아 있습니다. 어떤 코드를 썼고 무엇이 어떻게 실패했는지가 trace에 전부 들어 있습니다. 문제는 그 기록을 읽고 정리해 두는 자리가 없다는 것입니다. 최근에는 에이전트가 스스로 실행 기록을 분석해서 스킬 파일을 고치는 방법들이 나왔지만, 거기서도 분석의 결과물은 스킬 파일 안으로 흡수되거나 최적화 이력 속에 흩어져 남습니다. 다음 반복이 참조할 정리된 지식은 따로 존재하지 않습니다.

[2026년 8월 27일 arXiv에 v1으로 올라온 WikiSkill](https://arxiv.org/abs/2608.27454)이 이 지점을 다룹니다. Liyan Tang, Cyrus Rashtchian, Chun-Sung Ferng, Andrew Tomkins, Da-Cheng Juan, Tu Vu가 저자이고, 소속은 모두 Google Research로 표기되어 있으며 교신저자는 Liyan Tang과 Tu Vu 두 사람입니다. Tu Vu에게는 Virginia Tech 소속이 함께 적혀 있습니다. 아직 peer review를 거치지 않은 preprint이므로 이 글에서 인용하는 수치는 모두 그 전제 위에서 읽어야 합니다.

이 블로그에서 에이전트 스킬을 다룬 글이 이미 몇 편 있으므로 무엇이 다른지 먼저 밝혀 두겠습니다. [Agent Plugins 1.0](/ai-tech-blog/posts/agent-plugins-1-0/)은 이미 만들어진 스킬을 어떤 상자에 담아 배포하느냐를 정하는 패키징 표준이었고, [memory poisoning](/ai-tech-blog/posts/agent-memory-poisoning/)은 지속 메모리에 악성 항목이 들어가는 공격 경로였습니다. WikiSkill이 다루는 것은 그 사이에 있는 문제입니다. 스킬의 <strong>내용 자체를 무엇을 근거로 어떻게 고쳐 나가느냐</strong>입니다.

글은 두 부분으로 나뉩니다. 앞부분은 논문이 제시한 구조와 실험에서 확인된 수치이고, 뒷부분은 이 구조를 엔터프라이즈 환경에 옮길 때 어디에 경계를 두어야 하는지에 대한 저자의 분석입니다. 근거의 성격이 다르므로 절 제목에 표시해 두었습니다.

## 기존 스킬 진화 방법이 남기지 못하는 것

자동 스킬 진화 방법들은 대체로 같은 루프를 돕니다. 학습 과제에서 에이전트를 실행하고, 성공과 실패 trajectory를 분석하고, 스킬 수정안을 만들고, validation 점수로 그 수정안을 받아들일지 결정합니다. 논문이 비교 대상으로 삼은 세 방법도 이 골격을 공유합니다.

세 방법이 갈리는 지점은 "무엇을 어떻게 보존하느냐"입니다. 논문의 정리로는 [EvoSkill](https://arxiv.org/abs/2603.02766)이 과거 제안과 그 평가 결과를 누적 이력으로 유지하고, [Trace2Skill](https://arxiv.org/abs/2603.25158)은 실행 trajectory에서 교훈을 뽑아 스킬 업데이트로 통합하며, [SkillOpt](https://arxiv.org/abs/2605.23904)은 거부된 편집에 대한 피드백과 epoch 단위 meta guidance를 사용합니다.

논문은 이 세 방법이 공통으로 하지 않는 일을 지적합니다. 학습한 내용을 <strong>별도로 존재하면서 계속 갱신되는 지식 표현</strong>으로 유지하지 않는다는 것입니다. 지식이 스킬 파일과 최적화 산출물 곳곳에 흩어져 있으면, 다음 반복은 잘 정리된 지식이 아니라 흩어진 기록 위에서 판단하게 됩니다. 논문은 이 문제 설정을 Karpathy가 LLM Wiki라는 이름으로 정리한 관점에서 가져왔다고 밝히고, 그 출처를 GitHub Gist로 표기합니다. 경험을 지속적이고 누적되는 지식으로 컴파일하자는 주장입니다.

## WikiSkill이 나눈 세 개의 저장소

WikiSkill은 에이전트 작업 공간을 세 계층으로 나눕니다. 각 계층이 무엇을 담고 누가 접근하는지가 이 프레임워크의 핵심입니다.

<strong>raw/</strong>는 각 반복에서 학습 과제를 실행해 얻은 원본 trace를 담습니다. 추론 과정, tool 호출, tool 출력, 최종 답변이 단계별로 그대로 들어갑니다. 이 계층은 immutable입니다. 원본 이력을 보존하기 위해 고치지 않습니다.

<strong>wiki/</strong>는 그 원본 trace를 구조화된 지식으로 정리해서 누적합니다. `patterns/` 디렉토리에는 개별 실패 유형이나 성공 전략을 실행 가능한 대응 방법과 함께 기록한 markdown 파일들이 들어가고, `index.md`가 그 목록을 관리합니다. 여기에 두 종류의 이력 파일이 더 있습니다. `logs.md`는 Wiki Maintainer가 반복마다 발견한 내용을 요약해서 덧붙이는 진화 로그이고, `skill-impact.md`는 validation gating이 끝난 뒤 외부 루프 harness가 프로그램으로 기록을 덧붙이는 스킬 영향 추적 파일입니다. 이 계층은 반복 사이에 초기화되지 않고 계속 누적됩니다.

<strong>skills/</strong>는 Inference Agent가 실제로 읽는 활성 스킬을 담습니다. 각 스킬 디렉토리에는 두 파일이 있습니다. `SKILL.md`가 스킬 본문이고, `PURPOSE.md`는 그 스킬이 생성되거나 수정된 계기가 된 wiki pattern으로 되짚어 가는 매핑입니다.

이 구조에서 눈여겨볼 부분은 `PURPOSE.md`와 `skill-impact.md`입니다. `skill-impact.md`에는 제안 메타데이터, 대상 스킬 이름, 수정 내용의 unified diff, validation 점수, 그리고 수락 여부가 기록됩니다. 논문은 이것을 과거 개입에 대한 객관적인 감사 기록으로 설명하고, Skill Proposer가 이후 반복에서 이 기록을 참조해 실패한 수정을 다시 제안하지 않게 하는 용도로 씁니다.

![WikiSkill 진화 루프를 그린 구조도입니다. raw/, wiki/, skills/ 세 저장소를 왼쪽에서 오른쪽으로 배치하고, Wiki Maintainer가 raw에서 wiki로, Skill Proposer가 wiki에서 skills로 정보를 옮깁니다. 하단에서 Inference Agent가 새 trace를 만들어 다음 반복으로 이어지는 루프를 닫으며, 학습 rollout 중에는 wiki 접근이 차단되고 gating이 거부하면 skills만 롤백됩니다.](/ai-tech-blog/images/wikiskill-persistent-knowledge-skill-evolution/wikiskill-architecture.png)

<em>WikiSkill의 세 저장소와 네 구성 요소, 그리고 스킬만 되돌리고 wiki는 되돌리지 않는 비대칭 구조를 정리했습니다. arXiv:2608.27454v1의 Figure 2 구조를 바탕으로 직접 작성했습니다.</em>

## 진화 루프에서 되돌리는 것과 되돌리지 않는 것

한 반복은 네 단계로 진행됩니다.

<strong>Inference Agent</strong>가 현재 활성 스킬로 학습 과제를 실행해서 trace를 만듭니다. 이때 활성 스킬의 전체 내용은 system prompt에 직접 주입됩니다. 논문은 이 선택의 이유를 명시합니다. 스킬 검색이나 트리거 실패가 실험 결과를 흐리는 변수로 끼어들지 않게 하려는 것입니다. 그리고 이 단계에서 Inference Agent는 wiki 계층에 접근할 수 없습니다. 학습 rollout 중에 wiki 접근을 허용하면 스킬 개발에 나쁜 영향을 준다는 ablation 결과가 근거입니다.

<strong>Wiki Maintainer</strong>가 trace를 읽고 wiki를 갱신합니다. 컨텍스트 한계 때문에 전체 trace를 넣지는 않고, 반복마다 최대 8개를 층화 표집합니다. 실패 trace는 최대 5개로 근본 원인을 분석하고, 성공 trace는 최대 3개로 효과적인 전략을 뽑으면서 이미 잘 동작하는 행동이 퇴화하지 않게 확인합니다. 개별 실행 로그는 프롬프트에 넣기 전에 15,000자로 자릅니다. pattern 페이지 수정은 덧붙이기, 교체, 삽입 같은 증분 패치 방식으로 적용하고, 반복당 생성하거나 수정할 pattern 개수에는 상한이 없습니다.

<strong>Skill Proposer</strong>가 스킬 수정안을 만듭니다. 이 구성 요소는 ReAct 방식의 다중 턴 에이전트로 동작합니다. 미리 뽑아 둔 trace 묶음을 받는 것이 아니라, 처음에는 wiki index와 `skill-impact.md`, 그리고 전체 학습 과제 결과 요약만 받습니다. 그 다음부터는 `read_file` 도구로 필요한 pattern 페이지와 원본 trace를 직접 골라 읽으면서 원인을 진단합니다. 한 반복에서 내놓는 제안은 스킬 하나만 대상으로 하는 atomic proposal입니다. 새 스킬을 만들거나, 기존 스킬 하나에 증분 패치를 적용합니다.

<strong>Gating과 rollback</strong>이 그 제안을 판정합니다. 후보 스킬 세트를 validation split에서 평가해서, 점수가 지금까지의 최고 점수를 넘을 때만 수락합니다. 진화를 시작하기 전 빈 스킬 세트로 측정한 validation 점수가 그 기준의 출발점이고, 도중에 validation 점수가 1.0에 도달하면 루프를 조기에 종료합니다. 거부되면 스킬 세트를 직전의 성공 상태로 되돌립니다.

여기서 갈리는 지점이 이 논문의 설계 핵심입니다. <strong>스킬은 되돌리지만 wiki는 되돌리지 않습니다.</strong> 수락이든 거부든 관계없이 누적된 pattern과 로그는 남습니다. 그리고 거부된 제안의 diff와 거부 사실 자체가 `skill-impact.md`에 기록되어 다음 반복의 입력이 됩니다. 실패한 시도도 지식으로 남는다는 것이 이 구조가 노리는 부분입니다.

최적화 비용 쪽도 짚어 둘 만합니다. 논문은 학습 배치 크기를 전체 학습 세트 크기와 같게 두었고, 그 결과 반복당 optimizer LLM 호출은 Wiki Maintainer 1회와 Skill Proposer의 ReAct 턴 수를 합한 값이 됩니다. 실험에서 ReAct 턴 수는 대략 10에서 20 사이였습니다. 학습 과제 개수와 무관하다는 뜻입니다. 비교하면 Trace2Skill은 모든 학습 trajectory마다 별도 LLM 호출이 필요해서 학습 세트 크기에 비례하고, EvoSkill과 SkillOpt는 최고 성능이 나오는 minibatch 설정에서 역시 학습 세트 크기에 비례합니다. 다만 논문은 이 상수 호출 구조가 일부 데이터셋에서는 오히려 더 높은 inference 비용을 유발할 수 있다고 직접 적어 두었습니다.

## 실험 설정과 다섯 모델의 결과

논문은 다섯 개 벤치마크를 씁니다. 과제 분할과 도구 구성은 선행 연구와 엄격하게 맞췄다고 밝히고 있습니다.

| 벤치마크 | 상호작용 | Train | Val | Test | 환경 도구 |
|---|---|---|---|---|---|
| LiveMath | 단일 단계 | 35 | 18 | 124 | 없음(직접 추론) |
| SealQA | 다단계 | 16 | 10 | 85 | `web_search`, `read_file` |
| SpreadSheet | 다단계 | 80 | 40 | 280 | `bash` |
| OfficeQA | 다단계 | 50 | 24 | 172 | `glob`, `grep`, `read` |
| ALFWorld | 다단계 | 39 | 18 | 134 | 허용 행동 집합 |

LiveMath는 최근 수학 경시 문제로 구성된 객관식 벤치마크이고, SealQA는 검색 도구로 답을 찾는 사실 질의응답입니다. 실험에는 2026년 7월 버전을 사용했고 검색은 Google Search API를 씁니다. SpreadSheet는 라이브러리 제약 아래에서 표 변환 코드를 작성하는 과제이고, OfficeQA는 과거 Treasury bulletin 문서를 대상으로 하는 긴 컨텍스트 질의응답입니다. ALFWorld는 텍스트 시뮬레이터에서 가정 내 다단계 과제를 수행하는 상호작용 환경입니다.

모델은 다섯 개입니다. 폐쇄 모델로 Gemini-3.5-Flash를 쓰고, open-weight 모델로 Qwen-3.5-4B-Instruct, Qwen-3.5-9B-Instruct, Qwen-3.6-27B, Gemma-4-31B-It을 vLLM으로 서빙했습니다.

다음은 논문 Table 1의 test 성능입니다. 모든 방법은 빈 스킬 세트에서 출발하고, 진화한 스킬은 추론 시점에 Inference Agent 프롬프트에 주입됩니다. 각 숫자는 전체 진화 과정을 3회 독립 실행한 결과의 평균입니다.

| 모델 | 방법 | LiveMath | SealQA | SpreadSheet | OfficeQA | ALFWorld | 평균 |
|---|---|---|---|---|---|---|---|
| Qwen-3.5-4B | 스킬 없음 | 29.1 | 32.5 | 14.6 | 30.2 | 24.4 | 26.2 |
| | Trace2Skill | 31.5 | 37.6 | 17.5 | 31.0 | 42.8 | 32.1 |
| | EvoSkill | 41.7 | 37.3 | 18.6 | 29.5 | 41.5 | 33.7 |
| | SkillOpt | 48.7 | 33.3 | 14.0 | 34.5 | 45.3 | 35.2 |
| | WikiSkill | 49.7 | 39.4 | 21.1 | 28.5 | 53.7 | 38.5 |
| Qwen-3.5-9B | 스킬 없음 | 28.2 | 26.3 | 24.3 | 35.9 | 34.7 | 29.9 |
| | Trace2Skill | 33.1 | 36.9 | 26.5 | 38.4 | 48.8 | 36.7 |
| | EvoSkill | 58.1 | 34.5 | 35.4 | 34.9 | 48.5 | 42.3 |
| | SkillOpt | 48.7 | 29.4 | 29.0 | 38.0 | 55.7 | 40.2 |
| | WikiSkill | 56.3 | 43.1 | 33.6 | 40.5 | 63.4 | 47.4 |
| Qwen-3.6-27B | 스킬 없음 | 33.9 | 27.5 | 40.8 | 42.1 | 52.8 | 39.4 |
| | Trace2Skill | 36.3 | 37.3 | 53.3 | 54.3 | 55.5 | 47.3 |
| | EvoSkill | 57.3 | 32.9 | 59.5 | 52.5 | 64.2 | 53.3 |
| | SkillOpt | 51.9 | 34.5 | 53.2 | 54.8 | 59.2 | 50.7 |
| | WikiSkill | 61.9 | 41.6 | 81.7 | 53.7 | 77.6 | 63.3 |
| Gemma-4-31B | 스킬 없음 | 33.9 | 30.6 | 48.3 | 43.3 | 50.4 | 41.3 |
| | Trace2Skill | 32.3 | 37.7 | 58.5 | 43.2 | 57.2 | 45.8 |
| | EvoSkill | 29.8 | 38.4 | 56.4 | 39.9 | 52.6 | 43.4 |
| | SkillOpt | 40.1 | 36.1 | 63.1 | 44.4 | 61.9 | 49.1 |
| | WikiSkill | 56.7 | 41.2 | 68.0 | 44.2 | 64.4 | 54.9 |
| Gemini-3.5-Flash | 스킬 없음 | 33.0 | 29.4 | 50.5 | 48.6 | 85.9 | 49.5 |
| | Trace2Skill | 41.9 | 44.3 | 56.0 | 50.0 | 85.9 | 55.6 |
| | EvoSkill | 44.6 | 43.6 | 55.4 | 51.2 | 85.9 | 56.1 |
| | SkillOpt | 49.7 | 28.2 | 66.1 | 49.8 | 85.9 | 55.9 |
| | WikiSkill | 72.6 | 44.7 | 76.6 | 60.7 | 85.9 | 68.1 |

WikiSkill은 다섯 모델 모두에서 가장 높은 평균 성능을 냈습니다. 모델별로 가장 강한 경쟁 방법과 비교하면 평균이 Qwen-3.5-4B에서 3.3점, Qwen-3.5-9B에서 5.1점, Qwen-3.6-27B에서 10.0점, Gemma-4-31B에서 5.8점, Gemini-3.5-Flash에서 12.0점 올라갔습니다.

기존 방법들은 개선의 방향이 일정하지 않습니다. EvoSkill은 Qwen-3.5-9B의 LiveMath를 28.2%에서 58.1%로 크게 올렸지만 같은 벤치마크에서 Gemma-4-31B를 33.9%에서 29.8%로 떨어뜨렸고, SkillOpt은 Gemini-3.5-Flash의 SealQA를 29.4%에서 28.2%로 떨어뜨렸습니다.

표에서 Gemini-3.5-Flash의 ALFWorld 점수가 모든 방법에서 85.9%로 같은 것은 오기가 아닙니다. 이 모델이 진화를 시작하기 전에 이미 validation split에서 100%를 기록해서 루프가 조기 종료되었고, 그래서 스킬이 만들어지지 않았습니다.

통계 검정은 벤치마크별로 1,000회 반복하는 paired bootstrap으로 수행하고 p < 0.05를 기준으로 삼았습니다. 여러 방법이 최상위와 통계적으로 구별되지 않으면 단독 최고 성능을 선언하지 않고 동순위로 묶는 방식입니다.

## 스킬 진화와 모델 크기가 함께 작동하는 방식

Qwen 계열 안에서 WikiSkill이 스킬 없음 대비 올린 평균 점수는 모델이 커질수록 커집니다. 4B에서 12.3점, 9B에서 17.5점, 27B에서 23.9점입니다. SpreadSheet에서 이 경향이 특히 뚜렷했고, 같은 순서로 6.5점, 9.3점, 40.9점이 올라갔습니다.

반대 방향의 결과도 같이 나옵니다. WikiSkill을 적용한 Qwen-3.5-9B의 평균은 47.4%인데, 스킬 없는 Qwen-3.6-27B는 39.4%입니다. 진화한 스킬이 상당한 모델 크기 차이를 메울 수 있다는 뜻입니다. WikiSkill을 적용한 Qwen-3.5-4B는 38.5%였습니다. 논문은 이 두 결과를 함께 놓고, 모델 능력과 진화한 절차 지식이 서로를 대체하는 관계가 아니라 각각 다른 방향에서 성능을 만든다고 해석합니다. 강한 모델은 더 효과적인 스킬을 만들고 실행할 수 있어서 진화의 이득을 더 크게 얻고, 좋은 스킬은 작은 모델이 훨씬 큰 모델을 앞서게 할 수 있습니다.

벤치마크에 따른 차이도 큽니다. LiveMath는 다섯 모델 전부에서 20.6점에서 39.6점 사이의 개선을 보였고, ALFWorld는 스킬이 진화한 네 모델에서 14.0점에서 29.3점 사이였습니다. OfficeQA는 사정이 다릅니다. Qwen-3.6-27B에서 11.6점, Gemini-3.5-Flash에서 12.1점이 올라갔지만, Qwen-3.5-4B는 30.2%에서 28.5%로 오히려 조금 내려갔습니다. 논문의 분석으로는 작은 모델이 긴 컨텍스트에서 여러 단계로 이어지는 검색 절차를 끝까지 따라가지 못하고 기본 문서 읽기 방식으로 되돌아갔기 때문입니다.

## 모델 사이의 스킬 전이와 negative transfer

논문은 한 모델이 진화시킨 스킬을 다른 모델에 붙였을 때 무슨 일이 일어나는지도 측정했습니다. Table 2에서 확인되는 대표적인 조합을 옮기면 다음과 같습니다.

| 추론 모델 | 벤치마크 | 스킬 없음 | 자기 스킬 | 다른 모델 스킬 |
|---|---|---|---|---|
| Qwen-3.5-9B | SpreadSheet | 24.3 | 33.6 | 50.5 (Qwen-3.6-27B) |
| Qwen-3.5-9B | ALFWorld | 34.7 | 63.4 | 70.2 (Qwen-3.6-27B) |
| Qwen-3.5-9B | LiveMath | 28.2 | 56.3 | 61.0 (Qwen-3.5-4B) |
| Gemma-4-31B | LiveMath | 33.9 | 56.7 | 73.1 (Qwen-3.5-4B), 73.7 (Qwen-3.6-27B) |
| Gemma-4-31B | ALFWorld | 50.4 | 64.4 | 66.9 (Qwen-3.5-4B) |
| Qwen-3.6-27B | OfficeQA | 42.1 | 53.7 | 52.9 (Qwen-3.5-4B) |
| Gemini-3.5-Flash | SpreadSheet | 50.5 | 76.6 | 63.4 (Qwen-3.6-27B), 18.1 (Qwen-3.5-4B) |

전이된 스킬이 자기 스킬을 앞서는 경우가 자주 나옵니다. 작은 모델에서 큰 모델로 가는 방향도 작동합니다. Qwen-3.5-4B가 만든 LiveMath 스킬은 Gemma-4-31B를 33.9%에서 73.1%로 올렸습니다. 논문은 여기서 강한 source 모델이 반드시 더 좋은 스킬을 만드는 것은 아니라고 결론짓습니다.

마지막 줄이 반대 사례입니다. Qwen-3.5-4B가 만든 SpreadSheet 스킬은 Gemini-3.5-Flash의 성능을 50.5%에서 18.1%로 떨어뜨렸습니다. 논문의 오류 분석은 두 가지를 원인으로 지목합니다. 첫째, 작은 모델이 만든 스킬에는 한 줄짜리 Python 명령이나 문자열 변환 규칙처럼 저수준 우회 방법이 들어 있는데, 이것이 작은 모델의 실행 실패는 막아 주지만 강한 모델이 처음부터 끝까지 이어지는 완결된 스크립트를 쓰지 못하게 제약합니다. 둘째, 잘게 쪼개진 진단 절차가 중복 tool 호출을 만들어서 Gemini-3.5-Flash가 과제를 끝내기 전에 상호작용 예산을 소진시킵니다.

같은 source 스킬이 추론 모델에 따라 다른 값을 만들기도 합니다. Qwen-3.6-27B가 만든 SpreadSheet 스킬은 Qwen-3.5-4B, Qwen-3.5-9B, Qwen-3.6-27B의 스킬 없음 기준선을 각각 18.4점, 26.2점, 40.9점 올렸습니다. 반대 방향의 예도 있습니다. Qwen-3.5-4B가 만든 OfficeQA 스킬은 자기 성능은 30.2%에서 28.5%로 떨어뜨렸지만 Qwen-3.6-27B를 42.1%에서 52.9%로 올렸습니다.

논문은 이 결과들을 근거로 자기 진화가 뭉쳐서 다루던 두 능력을 분리합니다. 경험에서 유용한 절차 지식을 <strong>발견하는</strong> 능력과, 그 지식을 추론 시점에 <strong>실행하는</strong> 능력입니다.

## persistent knowledge를 제거했을 때의 결과

논문의 ablation은 Gemini-3.5-Flash로 진행하고 wiki 접근 권한을 두 구성 요소에서 각각 켜고 끕니다. Skill Proposer의 wiki 접근을 끄면 Wiki Maintainer도 함께 제거하므로, 반복 사이의 지식 누적 자체가 사라집니다. 평균은 ALFWorld를 제외한 네 벤치마크의 값입니다.

| Inference Agent wiki 접근 | Skill Proposer wiki 접근 | LiveMath | SealQA | SpreadSheet | OfficeQA | 평균 |
|---|---|---|---|---|---|---|
| 스킬 없음 | 스킬 없음 | 33.0 | 29.4 | 50.5 | 48.6 | 40.4 |
| 허용 | 차단 | 43.8 | 42.0 | 44.4 | 51.0 | 45.3 |
| 차단 | 차단 | 51.3 | 38.4 | 49.9 | 55.2 | 48.7 |
| 허용 | 허용 | 64.8 | 42.8 | 80.2 | 55.6 | 60.9 |
| 차단 | 허용 (기본 설정) | 72.6 | 44.7 | 76.6 | 60.7 | 63.7 |

Inference Agent의 wiki 접근이 차단된 상태에서 Skill Proposer에게 지속 wiki를 주면 평균이 48.7%에서 63.7%로 15.0점 올라갑니다. LiveMath는 51.3%에서 72.6%, SpreadSheet는 49.9%에서 76.6%로 움직입니다. 논문의 설명으로는 반복에 걸쳐 누적된 지식이 없으면 Skill Proposer가 복잡한 실패 유형을 풀어내지 못합니다.

반대 방향의 결과가 더 흥미롭습니다. Skill Proposer가 wiki를 쓰는 상태에서 Inference Agent에게도 wiki를 열어 주면 평균이 63.7%에서 60.9%로 내려가고, LiveMath는 72.6%에서 64.8%로 떨어집니다. 논문이 내놓은 가설은 이렇습니다. 학습 rollout 중에 Inference Agent가 스킬과 wiki를 둘 다 볼 수 있으면 과제 해결에 필요한 지식 일부를 스킬이 아니라 wiki에서 직접 얻게 되고, 그러면 그 rollout에서 나온 trace가 스킬 개발에 덜 유용해집니다.

진화가 어느 시점에 일어나는지도 측정되어 있습니다. 부록 Table 5는 수락된 스킬 업데이트를 초기(반복 0에서 1), 중기(2에서 4), 후기(5에서 7)로 나눕니다. 모델별로 초기 구간이 39%에서 52%를 차지하고, 나머지 상당 부분이 중기와 후기에 걸쳐 있습니다. SealQA에서는 중기 33%, 후기 28%로 후반까지 개선이 이어졌습니다.

만들어진 산출물의 형태도 모델과 벤치마크에 따라 갈립니다. Qwen 계열은 118.9줄에서 128.6줄 사이의 긴 절차 스킬을 만들고, Gemma-4-31B와 Gemini-3.5-Flash는 각각 45.1줄과 81.2줄로 더 간결한 스킬을 만듭니다. wiki pattern은 모델별로 평균 6.3개에서 8.9개가 생성되고 7.0회에서 18.4회 수정됩니다. 벤치마크로 보면 SpreadSheet가 가장 긴 스킬(142.5줄)과 가장 많은 pattern(9.8개)을 만들고, LiveMath가 가장 짧은 스킬(84.6줄)과 가장 적은 pattern(4.4개)을 만듭니다.

논문의 사례 연구가 이 구조가 실제로 어떻게 작동하는지 보여 줍니다. Qwen-3.6-27B의 ALFWorld 진화 과정입니다. 반복 0에서 Wiki Maintainer가 반복적인 순환 행동을 `take-examine-move-loop.md`로 기록하고, Skill Proposer는 `goal-directed-action`을 제안했지만 validation 점수가 오르지 않아 거부되었습니다. 여기서 `skill-impact.md`가 그 제안의 diff와 거부 결과를 보존합니다. 반복 1에서 Skill Proposer는 그 기록을 참고해 `break-repetition-loop`를 만들었고, 여기에는 "물건을 원래 위치로 되돌려 놓지 않는다"는 구체적인 행동 규칙이 들어갔습니다. 이 제안은 수락되었습니다. 이후 새로운 순환 변종이 rollout에서 나타나면서 Wiki Maintainer가 `multi-operation-loop.md`에 증거를 쌓았고, 반복 4에서 Skill Proposer가 "물건마다 각 동작을 한 번만 수행한다"는 규칙을 추가해 스킬을 다시 다듬었습니다.

![wiki 접근 조건별 성능을 비교한 막대 그래프입니다. Gemini-3.5-Flash로 측정한 네 벤치마크(LiveMath, SealQA, SpreadSheet, OfficeQA)와 평균에 대해 다섯 설정을 그룹 막대로 비교합니다. Skill Proposer에게 wiki를 준 기본 설정이 대부분의 벤치마크에서 가장 높고, 평균은 지속 wiki가 없을 때보다 15.0점 높습니다.](/ai-tech-blog/images/wikiskill-persistent-knowledge-skill-evolution/wikiskill-ablation.png)

<em>wiki 접근을 구성 요소별로 켜고 끈 ablation 결과를 정리했습니다. 수치는 arXiv:2608.27454v1 Table 3(Gemini-3.5-Flash)에서 가져와 직접 작성했습니다.</em>

## 엔터프라이즈 에이전트 운영으로 옮길 때의 경계 (저자 분석)

여기서부터는 논문의 실험 결과가 아닙니다. 위 구조를 사내 에이전트 운영에 적용한다고 가정했을 때 어디에 경계를 두어야 하는지에 대한 저자의 분석입니다. 논문은 이 문제를 다루지 않았으므로 아래 내용을 논문의 주장으로 읽으면 안 됩니다.

<strong>승격 기준이 성능 게이트에서 끝나면 안 됩니다.</strong> WikiSkill의 gating은 validation 점수가 올랐는지만 봅니다. 벤치마크에는 정답과 validation split이 있으니 이 기준이 성립하지만, 사내 업무에는 그런 정답 집합이 대개 없습니다. 그리고 점수가 오르는 스킬이 반드시 허용 가능한 스킬은 아닙니다. 예를 들어 승인 절차를 건너뛰는 우회 방법을 학습한 스킬은 과제 성공률을 올리면서 동시에 통제를 무너뜨립니다. 스킬을 활성화하기 전에 성능 게이트와 권한 게이트를 따로 두고, 권한 게이트는 사람이 판정하도록 설계할 수 있습니다.

<strong>provenance는 논문 구조를 그대로 쓸 수 있습니다.</strong> `PURPOSE.md`가 스킬을 유발한 wiki pattern으로 되짚고, `skill-impact.md`가 제안 diff와 수락 여부를 남깁니다. human review가 성립하려면 검토자가 "이 규칙이 왜 여기 있는지"를 추적할 수 있어야 하는데, 이 두 파일이 그 경로를 제공합니다. 반대로 이 경로가 없는 자동 스킬 진화 시스템은 검토 자체가 불가능합니다. 검토자가 볼 수 있는 것이 결과 문서뿐이기 때문입니다.

<strong>wiki를 되돌리지 않는 설계에는 대가가 따릅니다.</strong> 잘못된 pattern이 한 번 들어가면 그것도 남습니다. 논문은 한계 절에서 wiki를 자동으로 정리하는 메커니즘이 아직 없고, 진화를 길게 돌리면 그런 정리가 필요해질 수 있다고 적었습니다. 학습 효율 측면에서는 실패 기록도 자산이지만, 운영 측면에서는 지식 항목에도 만료 기한과 철회 경로가 필요합니다. wiki pattern마다 근거가 된 trace의 식별자와 기록 시점을 함께 남기고, 근거가 무효화되면 그 pattern을 폐기 상태로 표시하는 방식을 설계할 수 있습니다.

<strong>Inference Agent의 wiki 차단은 격리 경계로도 쓸 수 있습니다.</strong> 논문에서 이 차단의 근거는 스킬 품질이었습니다. 그런데 결과적으로 이 설계는 사용자 입력과 외부 데이터를 직접 다루는 실행 경로가 지식 저장소에 손대지 못하게 만듭니다. [memory poisoning](/ai-tech-blog/posts/agent-memory-poisoning/)에서 정리한 것처럼 오염은 쓰기 경로를 타고 들어오므로, 실행 경로와 지식 쓰기 경로를 분리해 두면 그 경로 하나가 줄어듭니다. 운영 시스템에서는 이 분리를 품질 최적화가 아니라 신뢰 경계로 명시하고, wiki 쓰기 권한을 Wiki Maintainer 역할에만 부여하는 방식으로 설계할 수 있습니다.

<strong>negative transfer는 배포 문제로 그대로 넘어옵니다.</strong> 논문의 Table 2에서 한 모델이 만든 스킬이 다른 모델의 성능을 50.5%에서 18.1%까지 끌어내린 사례가 나왔습니다. 스킬 저장소를 여러 에이전트가 공유하는 구성에서는 이 결과가 그대로 위험이 됩니다. 스킬을 조직 전체에 배포하기 전에 대상 모델별로 재검증하고, 어떤 모델에서 진화했는지를 스킬 메타데이터에 남겨 두는 것이 최소 조건입니다.

<strong>rollback 단위를 스킬 세트보다 작게 잡아야 합니다.</strong> 논문은 validation 점수가 떨어지면 스킬 세트를 직전 상태로 되돌립니다. 운영에서는 어떤 스킬의 어떤 버전이 언제부터 어떤 에이전트에 붙어 있었는지를 고정할 수 있어야, 사후에 문제가 드러난 스킬만 골라 되돌릴 수 있습니다. 스킬 파일을 형상 관리 대상으로 두고 활성화 이력을 별도로 기록하는 구성이 필요합니다.

<strong>진화 비용을 optimizer 호출로만 계산하면 안 됩니다.</strong> 논문의 optimizer 호출 분석은 반복당 상수라는 좋은 특성을 보여 주지만, 실제 비용의 큰 부분은 그 옆에 있습니다. 반복마다 학습 rollout과 validation rollout이 함께 돌기 때문입니다. 사내에서 이 루프를 돌린다면 스킬 하나를 다듬는 데 드는 총 실행 비용을 먼저 측정하고, 그 비용이 정당화되는 반복 업무에만 적용 범위를 한정하는 편이 안전합니다.

## 이 논문을 읽을 때 감안할 한계

논문이 직접 밝힌 한계가 네 가지입니다. 첫째, 스킬 품질만 떼어 보기 위해 스킬을 프롬프트에 직접 주입했으므로 스킬 검색과 트리거는 평가되지 않았습니다. 스킬 개수가 늘어나면 이 부분이 중요해집니다. 둘째, gating이 validation 점수 상승만 수락하므로, 당장 성능을 유지하면서 이후 반복의 개선을 가능하게 하는 중립적인 제안은 제외됩니다. 셋째, wiki를 자동으로 정리하는 기능이 없습니다. 넷째, 벤치마크에 긴 컨텍스트 문서 추론과 다단계 도구 상호작용은 포함되지만, 수백 개의 환경 행동이나 여러 시간에 걸치는 매우 긴 horizon 과제는 포함되지 않습니다.

실험 규모도 감안해야 합니다. validation split이 작습니다. SealQA는 10개, LiveMath와 ALFWorld는 18개, OfficeQA는 24개, SpreadSheet는 40개입니다. 논문도 이 크기가 gating 판정에 평가 노이즈를 넣을 수 있다고 인정하고, 전체 파이프라인을 3회 독립 실행한 평균과 paired bootstrap 검정으로 대응했습니다. 그래도 이 규모의 validation에서 나온 승격 결정이 실제 업무 분포에서 같은 판정을 낸다고 보기는 어렵습니다.

한 가지 더 있습니다. 논문은 AI Disclosure 절에서 대규모 언어 모델과 코딩 에이전트를 사용해 글을 다듬고 일부 표와 그림을 생성했다고 밝혔습니다.

이 글에서 다루지 못한 항목도 적어 둡니다. 전체 반복 횟수의 설정값은 본문에서 확인하지 못했고, 부록 Table 5가 반복 0에서 7까지의 구간을 보고한다는 사실까지만 확인했습니다. 실제 토큰 소비량이나 금액 기준 비용, 그리고 코드와 데이터 공개 여부도 확인하지 못했습니다.

## 정리

WikiSkill이 바꾼 것은 스킬을 만드는 알고리즘이 아니라 저장 구조입니다. 실행 기록, 정리된 지식, 실행 스킬을 서로 다른 파일 집합으로 분리하고, 스킬은 되돌리되 지식은 되돌리지 않는 비대칭을 두었습니다. ablation에서 Skill Proposer의 wiki 접근 하나가 평균 15.0점을 만든 것이 이 분리의 효과를 보여 줍니다.

운영 관점에서 이 논문이 남기는 것은 성능 수치보다 감사 기록 구조입니다. 어떤 지식에서 어떤 스킬이 나왔고 그 제안이 어떤 판정을 받았는지가 파일로 남아 있으면, 자동으로 진화하는 스킬에도 사람의 검토를 붙일 수 있습니다. 경험이 곧 권한이 되지 않게 하려면 성능 게이트 옆에 권한 게이트를 따로 세워야 하는데, 그 게이트가 판정할 근거를 이 구조가 이미 만들어 둡니다.

## References

- Liyan Tang, Cyrus Rashtchian, Chun-Sung Ferng, Andrew Tomkins, Da-Cheng Juan, Tu Vu. "WikiSkill: Compiling Agent Experience into Persistent Knowledge for Skill Evolution." arXiv:2608.27454v1 [cs.AI], 2026년 8월 27일 제출. https://arxiv.org/abs/2608.27454 (HTML 전문: https://arxiv.org/html/2608.27454v1)
- Salaheddin Alzubi, Nicholas Provenzano, Joshua Bingham, Wenhu Chen, Tu Vu. "EvoSkill: Automated Skill Discovery for Multi-Agent Systems." arXiv:2603.02766. https://arxiv.org/abs/2603.02766
- Jian Ni 외. "Trace2Skill: Distill Trajectory-Local Lessons into Transferable Agent Skills." arXiv:2603.25158. https://arxiv.org/abs/2603.25158
- Y. Yang 외. "SkillOpt: Executive Strategy for Self-Evolving Agent Skills." arXiv:2605.23904. https://arxiv.org/abs/2605.23904
- Andrej Karpathy. "LLM Wiki." GitHub Gist. https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f (WikiSkill 논문의 참고문헌에 기재된 항목이며, 이 글에서는 논문의 인용을 통해서만 참조했습니다.)
