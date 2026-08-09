---
title: "중국발 오픈웨이트 러시: Kimi K3 이후 7주간의 지형 변화"
date: 2026-08-08T10:00:00+09:00
draft: false
categories: ["GenAI"]
tags: ["Open Weights", "Kimi K3", "DeepSeek", "GLM", "Qwen", "MoE", "Cost Optimization", "AI Industry"]
author: "Jesam Kim"
description: "GLM-5.2에서 Qwen3.8-Max까지 7주 동안 중국 대형 모델이 연속으로 나왔습니다. 규모와 가격이 어디까지 왔는지, 경쟁 축이 미국 대 중국에서 오픈 대 클로즈드로 옮겨가고 있다는 진단이 얼마나 타당한지 살펴봅니다."
cover:
  image: "/ai-tech-blog/images/china-open-weight-rush/cover.png"
  alt: "중국발 오픈웨이트 모델 릴리스 러시와 오픈 대 클로즈드 경쟁 구도"
  relative: false
---

2주 전 이 블로그에서 Thinking Machines Lab의 [Inkling](/ai-tech-blog/posts/2026-07-25-inkling-975b-moe-open-weights-anatomy/)을 다뤘습니다. 975B MoE 오픈웨이트 한 건을 아키텍처 관점에서 뜯어보는 글이었습니다. 그 글을 쓰던 시점에는 개별 출시로 보였던 사건이, 지금 돌아보면 훨씬 빠른 흐름의 한 지점이었습니다.

6월 중순부터 8월 초까지 7주 동안 대형 모델이 연달아 나왔습니다. Z.ai의 GLM-5.2, Moonshot AI의 Kimi K3, DeepSeek의 V4-Flash-0731, Alibaba의 Qwen3.8-Max. 네 건 모두 중국 기업입니다. 앞의 세 건은 가중치를 지금 내려받을 수 있고, Qwen3.8-Max는 공개를 예고한 상태입니다. Kimi K3는 2.8조 파라미터로 세계 최대 오픈웨이트 모델 자리를 가져갔습니다.

이 글은 그 7주를 타임라인과 수치로 정리하고, 여기서 나온 "경쟁 축이 미국 대 중국에서 오픈 대 클로즈드로 옮겨갔다"는 진단이 어디까지 타당한지 따져봅니다. 응원할 진영을 고르는 글은 아닙니다. 자체 호스팅이나 비용 민감 워크로드를 설계하는 입장에서 지금 무엇이 달라졌고 무엇이 그대로인지가 관심사입니다.

## 7주 동안 무슨 일이 있었나

![2026년 6월 16일부터 8월 3일까지 중국 대형 모델 릴리스 타임라인. GLM-5.2, Kimi K3, DeepSeek V4-Flash-0731, Qwen3.8-Max의 공개 날짜와 주요 스펙을 표시한 다이어그램](/ai-tech-blog/images/china-open-weight-rush/diagram-1-release-timeline.png)

*2026년 6월 &ndash; 8월 릴리스 타임라인. Qwen3.8-Max는 가중치 공개를 예고한 상태입니다. 출처: [Z.ai](https://z.ai/blog/glm-5.2), [Kimi](https://www.kimi.com/blog/kimi-k3), [DeepSeek API 문서](https://api-docs.deepseek.com/updates), [Qwen](https://qwen.ai/blog?id=qwen3.8)*

시작은 GLM-5.2였습니다. Z.ai가 [6월 16일 공식 블로그](https://z.ai/blog/glm-5.2)로 상세를 공개했습니다. 753B 파라미터에 활성 40B, MIT 라이선스, SWE-bench Pro 62.1점, 1M 토큰 컨텍스트입니다. Artificial Analysis는 같은 날 이 모델이 Intelligence Index 51점으로 [오픈웨이트 선두에 올랐다](https://artificialanalysis.ai/articles/glm-5-2-is-the-new-leading-open-weights-model-on-the-artificial-analysis-intelligence-index)고 정리했습니다. 당시 2위권은 MiniMax-M3와 DeepSeek V4-Pro가 44점, Kimi K2.6이 43점이었습니다.

한 달 뒤 Moonshot AI가 [Kimi K3를 발표](https://www.kimi.com/blog/kimi-k3)했습니다. 2.8조 파라미터 MoE에 1M 컨텍스트, 발표문의 표현은 "세계 최초 open 3T-class 모델"입니다. 특이한 건 순서였습니다. 7월 16일에는 API로만 열고, 가중치는 [7월 27일에 Hugging Face에 올렸습니다](https://huggingface.co/moonshotai/Kimi-K3). 성능이 검증되는 11일 동안 가중치는 없는 상태였다는 뜻입니다. Reuters는 이를 [미국 경쟁사와의 격차를 좁힌 세계 최대 오픈웨이트 시스템](https://www.reuters.com/world/china/chinas-moonshot-unveils-worlds-largest-open-ai-model-closing-us-rivals-2026-07-17)으로 보도했습니다.

7월 31일에는 DeepSeek이 V4-Flash 공개 베타를 냈습니다. 빌드명 V4-Flash-0731입니다. [모델 카드](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)를 보면 아키텍처와 크기는 그대로 두고 post-training만 다시 한 버전인데, 에이전트 벤치마크 9종에서 자사 플래그십 V4-Pro 프리뷰를 앞섭니다. 활성 파라미터가 훨씬 적은 모델이 더 큰 모델을 앞선 결과입니다.

후처리 학습의 효과는 같은 Flash 계열끼리 비교할 때 드러납니다. 프리뷰에서 0731로 오면서 Terminal Bench 2.1이 61.8에서 82.7로, DeepSWE는 7.3에서 54.4로 올랐습니다. 가중치 크기를 그대로 두고 후처리만 다시 해서 이만큼 움직인 사례입니다.

8월 3일 Alibaba가 [Qwen3.8-Max를 공개](https://qwen.ai/blog?id=qwen3.8)했습니다. 2.4조 파라미터에 활성 파라미터는 95B, Qwen 팀이 낸 것 중 가장 큽니다. 발표문이 앞세운 건 벤치마크 점수보다 시뮬레이션 이커머스 환경에서의 장기 태스크 결과였습니다. 시작 자본을 41만 6252위안까지 늘려 4.16배를 만들었고, 2위 GLM-5.2보다 38% 높았습니다.

Qwen3.8-Max는 앞의 세 건과 상태가 다릅니다. Max 등급 Qwen 중 처음으로 가중치를 공개하겠다고 선언했지만, 8월 8일 현재 Hugging Face와 ModelScope에 저장소가 올라오지 않았고 라이선스도 발표되지 않았습니다. 발표문의 표현은 다음 주 공개 예정입니다. 지금 쓸 수 있는 건 API뿐이므로, 이 글에서 Qwen3.8-Max는 오픈웨이트로 확정된 사례가 아니라 예고된 사례로 다룹니다.

## 규모 경쟁: 753B에서 2.8조까지 7주

파라미터 규모의 기준선이 빠르게 올라갔습니다. 6월의 GLM-5.2가 753B였고, 7월의 Kimi K3가 2.8조입니다. 8월의 Qwen3.8-Max는 2.4조인데 아직 가중치가 나오지 않았으므로, 지금 내려받을 수 있는 것 중에서는 K3가 최대입니다. Qwen 쪽 가중치가 예고대로 공개되면 조 단위 오픈웨이트가 3주 안에 두 번 등장한 셈이 됩니다.

이걸 가능하게 한 건 MoE입니다. Inkling 글에서 다뤘던 구조가 여기서도 전제로 작동합니다. Inkling은 975B 중 토큰당 41B만 활성화하는 구성이었고, Qwen3.8-Max는 2.4조 중 95B를 활성화합니다. 전체 파라미터가 늘어나도 토큰 하나를 처리하는 연산량은 활성 파라미터에 묶이므로, 모델 크기를 키우는 비용과 서빙 비용이 분리됩니다. 조 단위 숫자가 헤드라인에 계속 오르는 배경입니다.

다만 이 분리가 무료는 아닙니다. 가중치를 자체 호스팅한다면 활성 파라미터가 아니라 전체 파라미터를 메모리에 올려야 합니다. 2.8조 파라미터 모델을 내려받을 수 있다는 사실과 그것을 돌릴 수 있다는 사실은 다른 문제입니다.

오픈웨이트가 곧 저렴함을 뜻하지도 않습니다. Artificial Analysis의 태스크당 비용 측정에서 K3는 테스트 1건당 86센트로, 3센트인 DeepSeek V4-Flash보다 훨씬 비싸고 1.86달러인 GPT-5.6 Sol과 같은 자릿수입니다. 규모가 커지면 그만큼 서빙 비용이 붙습니다. 라이선스가 열려 있다는 속성과 싸게 돌아간다는 속성은 별개의 축입니다.

성능 쪽에서 K3의 위치는 이렇습니다. CNBC 보도에 따르면 K3는 [Claude Fable 5와 GPT-5.6 Sol에는 전체 성능에서 뒤지지만](https://www.cnbc.com/2026/07/17/moonshot-ai-kimi-k3-model-openai-anthropic-china.html) 코딩과 일반 에이전트 벤치마크에서는 Claude Opus 4.8과 GPT-5.5를 앞섭니다. Reuters는 Arena.ai가 웹 인터페이스 구축 능력 평가에서 K3를 1위에 뒀고 Vals AI 종합에서는 [Fable 5 다음, GPT-5.6 Sol 앞인 2위](https://www.reuters.com/world/china/chinas-moonshot-unveils-worlds-largest-open-ai-model-closing-us-rivals-2026-07-17)라고 전했습니다. Nathan Lambert의 [정리](https://www.interconnects.ai/p/kimi-k3-the-open-weights-escalation)도 Vals AI 2위, Frontend Code Arena 1위로 같은 방향입니다.

## 비용: 출력 토큰 89배 차이

![위쪽은 DeepSeek V4-Flash와 Claude Opus 4.8의 100만 토큰당 정가 비교, 아래쪽은 V4-Flash, Kimi K3, GPT-5.6 Sol, Claude Fable 5의 테스트 1건당 평균 비용 비교 막대 차트. 둘 다 로그 스케일](/ai-tech-blog/images/china-open-weight-rush/diagram-2-cost-comparison.png)

*위쪽 정가는 [DeepSeek](https://api-docs.deepseek.com/quick_start/pricing)과 [Anthropic](https://platform.claude.com/docs/en/docs/about-claude/pricing) 공식 가격 페이지 기준(2026년 8월 8일 확인)입니다. 아래쪽 태스크당 비용은 Artificial Analysis 측정치를 인용한 [Reuters 보도](https://www.reuters.com/business/retail-consumer/deepseeks-new-ai-model-is-by-far-cheapest-well-known-models-run-research-firm-2026-08-03) 기준으로, 오픈웨이트인 Kimi K3가 클로즈드 모델과 같은 자릿수에 있습니다*

이 7주에서 엔지니어 관점으로 가장 눈에 걸리는 숫자는 파라미터가 아니라 가격입니다. DeepSeek 공식 가격 페이지의 V4-Flash는 100만 토큰당 [입력 $0.14, 출력 $0.28](https://api-docs.deepseek.com/quick_start/pricing)입니다. 같은 시점 Anthropic 가격 페이지의 [Claude Opus 4.8은 입력 $5.00, 출력 $25.00](https://platform.claude.com/docs/en/docs/about-claude/pricing)입니다. 출력 토큰 기준 약 89배입니다.

정가 비교는 컨텍스트 캐싱이나 배치 할인, 실제 토큰 소비량을 반영하지 않으므로 그 자체로는 반쪽입니다. 토큰당 싸도 정답에 이르기까지 단계를 많이 밟으면 청구서는 커집니다. 그래서 Artificial Analysis는 정가보다 태스크당 비용을 앞세웁니다.

그 측정에서 V4-Flash는 [테스트 1건당 평균 3센트](https://www.reuters.com/business/retail-consumer/deepseeks-new-ai-model-is-by-far-cheapest-well-known-models-run-research-firm-2026-08-03)입니다. Kimi K3가 86센트, GPT-5.6 Sol이 1.86달러, Claude Fable 5가 3.15달러입니다. 두 자리 배수 차이가 정가만의 착시는 아니라는 뜻입니다. 보도 시점의 같은 측정에서 V4-Flash의 Intelligence Index는 100점 만점에 50점으로 Gemini 3.6 Flash와 동점이고 GLM-5.2보다 1점 낮았습니다. 당시 Kimi K3는 57점이었고, V4-Flash와 상위권 모델 사이는 10점 이상 벌어져 있었습니다.

여기에 붙여야 할 조건이 둘 있습니다. 하나는 이 가격이 시점 값이라는 점입니다. DeepSeek은 같은 가격 페이지에서 전반적인 가격 인상을 예고했고 상당한 폭을 언급했습니다. 지금의 89배는 고정된 조건이 아닙니다. 다른 하나는 지수 점수가 버전과 설정에 따라 움직인다는 점입니다. 보도 시점 50점이던 V4-Flash의 값은 이후 갱신된 지수에서 다르게 나옵니다. K3도 같은 기간에 지수가 v4.1.1로 재채점되면서 값과 순위가 옮겨갔습니다. 이런 지수를 소수점까지 따지는 건 의미가 옅습니다.

가격이 이렇게 벌어진 이유를 성능 격차만으로 설명하기는 어렵습니다. Stanford HAI의 [2026 AI Index 기술 성능 장](https://hai.stanford.edu/ai-index/2026-ai-index-report/technical-performance)은 2026년 3월 Arena 리더보드 기준으로 최상위 클로즈드 모델이 최상위 오픈 모델을 3.3% 앞선다고 정리합니다. 2024년 8월의 0.5%에서 오히려 다시 벌어진 값이지만, 여전히 한 자릿수입니다. 성능은 몇 퍼센트 차이인데 가격은 두 자리 배수 차이라는 구도입니다. 물론 각 진영 최상위 모델 사이의 간격과 특정 저가 모델의 성능은 다른 이야기이므로, 두 수치를 곧바로 이어 붙이면 과한 해석이 됩니다.

## 벤치마크 지형을 어디까지 믿을 수 있나

오픈웨이트가 클로즈드에 얼마나 근접했는지 재려면 어떤 지표를 보느냐가 결론을 크게 흔듭니다. 이 7주가 그걸 잘 보여줍니다.

GLM-5.2는 6월 출시 당시 Artificial Analysis Intelligence Index 51점으로 오픈웨이트 선두였습니다. 이 문장에는 시점이 붙어야 합니다. K3가 나온 뒤 오픈웨이트 선두는 바뀌었고, 같은 지수의 오픈웨이트 랭킹은 지금 K3가 앞에 있습니다. "오픈웨이트 1위"는 몇 주 단위로 갈리는 타이틀입니다.

같은 모델의 점수가 인용 시점마다 다르다는 문제도 있습니다. Artificial Analysis는 출시 시점 [발표 글](https://artificialanalysis.ai/articles/kimi-k3-achieves-3-in-the-artificial-analysis-intelligence-index-comparable-to-opus-4-8-and-gpt-5-5)에서 K3를 Intelligence Index 57점, 종합 3위로 소개하며 Claude Opus 4.8·GPT-5.5와 비슷한 수준이라고 평했습니다. 이후 지수가 v4.1.1로 재채점되면서 [모델 페이지](https://artificialanalysis.ai/models/kimi-k3)의 값과 순위는 출시 시점과 달라졌습니다(2026년 8월 8일 확인). 채점 모델과 평가 구성이 바뀌면 값이 재계산되고, 같은 모델도 추론 설정에 따라 다른 점수를 받기 때문입니다. 특정 순위를 근거로 도입 결정을 내리려면 그 순위가 언제, 어떤 지수 버전과 설정에서 나온 값인지 확인해야 합니다.

벤치마크 종류에 따라 순서가 뒤집히기도 합니다. 여기서 주의할 점은 벤더 표의 측정 조건입니다. Moonshot의 발표 표는 모델마다 다른 하네스를 쓴다고 각주에 밝혀둡니다. K3는 Kimi Code, 경쟁 모델은 각자의 에이전트 환경입니다. 서로 다른 실행 환경의 숫자를 한 표에 나란히 놓은 것이므로, 격차의 절대값을 그대로 받기는 어렵습니다.

DeepSeek 모델 카드는 같은 조건에서 측정한 비교를 함께 싣고 있어 참고가 됩니다. 이 표에서 V4-Flash-0731은 Terminal Bench 2.1에서 82.7로 GLM-5.2(81.0)를 앞서지만 Opus 4.8(85.0)에는 못 미치고, DeepSWE에서는 54.4로 GLM-5.2(46.2)를 앞서고 Opus 4.8(58.0)에 근접합니다. 오픈웨이트 상위권이 클로즈드 프론티어 바로 아래 구간에 들어와 있다는 진술의 근거가 이런 형태입니다.

반대 방향의 예도 있습니다. GLM-5.2는 SWE-bench Pro에서 62.1점으로 GPT-5.5(58.6)를 앞섭니다. 어느 쪽이 "더 좋은 모델"인지는 코딩 에이전트를 돌릴 것인지, 대량 요청을 낮은 지연으로 처리할 것인지에 따라 갈립니다.

정리하면, 오픈웨이트 상위 모델이 클로즈드 프론티어 바로 아래 구간에 들어와 있다는 진술 정도는 여러 집계에서 공통으로 확인됩니다. 그 이상으로 몇 점 차이라거나 몇 위라는 식의 정밀한 주장은 출처와 시점을 함께 명시하지 않으면 며칠 만에 틀린 문장이 됩니다.

## 축의 이동이라는 진단

이 흐름을 두고 Fortune에는 8월 4일 [경쟁 축이 미국 대 중국에서 오픈 대 클로즈드로 옮겨갔는지 묻는 논평](https://fortune.com/2026/08/04/has-the-ai-race-shifted-from-u-s-vs-china-to-open-vs-closed)이 실렸습니다. Grace Shao와 Alvin Wang Graylin은 주류 서사가 미국 AI와 중국 AI를 맞붙이지만 실제 경쟁은 오픈과 클로즈드 사이에서 벌어진다고 주장합니다.

이 프레임에는 근거가 있습니다. 위에서 본 대로 오픈웨이트 상위 모델과 클로즈드 프론티어의 성능 간격은 좁고, 가격 간격은 넓습니다. 다운로드 가능한 가중치로 프론티어 근처 성능을 얻을 수 있다면 조달 의사결정의 축이 국적에서 배포 방식으로 옮겨간다는 논리입니다.

동시에 이 프레임을 그대로 받기 어려운 지점이 둘 있습니다.

하나는 오픈웨이트가 오픈소스와 같지 않다는 문제입니다. Stanford HAI는 8월 4일 [가중치 공개만으로는 부족하다는 기사](https://hai.stanford.edu/news/open-weight-models-arent-enough-we-need-truly-open-source-ai-models-for-science-and-society)를 냈습니다. Shana Lynch가 쓴 이 글에서 HAI Denning Director인 James Landay는 "그건 오픈 모델이 아니라 오픈 배포"라고 말합니다. HAI가 기준으로 삼는 건 Linux Foundation의 Model Openness Framework 최상위 등급, 즉 학습 코드와 데이터와 도구까지 공개된 상태입니다. 이 글에 등장하는 네 모델은 모두 그 기준에 못 미칩니다. 가중치가 MIT로 풀려 있어도 학습 데이터가 공개되지 않으면 결과를 재현하거나 데이터에 들어간 것을 감사할 수 없습니다. 파인튜닝은 되지만 검증은 안 되는 상태입니다.

다른 하나는 국적 축이 사라지지 않았다는 점입니다. 같은 AI Index는 2025년 민간 AI 투자가 미국 2859억 달러, 중국 124억 달러로 23배 차이라고 집계합니다. 미국은 최상위 모델 총량과 민간 투자에서 앞서고 중국은 논문·특허·인용에서 앞선다는 구도도 그대로입니다. 성능 격차 2.7%가 이 비대칭 위에서 나온 값이라는 점을 빼면 그림이 단순해집니다.

규제 논쟁도 이 두 축을 오갑니다. Rest of World는 8월 3일 [실리콘밸리가 중국 오픈웨이트 모델을 두고 갈라져 있다고 정리](https://restofworld.org/2026/silicon-valley-debate-chinese-open-weight-ai-models)했습니다. 한쪽에는 군사적 우위와 오용 가능성을 이유로 우려하는 입장이 있고, 다른 쪽에는 규제가 미국 경쟁력을 깎는다는 입장이 있습니다. 앞의 Fortune 논평은 후자를 더 밀어서, 오픈웨이트 금지는 중국보다 미국에 더 해로우며 미국 기업은 국외에서 훨씬 저렴하게 얻는 지능에 프리미엄을 지불해야 할 것이라고 씁니다. 어느 쪽이 맞는지는 이 글에서 판정할 문제가 아니지만, 논쟁 자체가 오픈웨이트를 국가 전략 변수로 취급하기 시작했다는 신호입니다.

## 엔지니어 관점 정리

이 7주가 실무에서 바꿔놓은 건 선택지의 폭입니다. 아래는 검증된 제품 특성이 아니라 위 자료를 읽은 제 판단입니다.

<strong>오픈웨이트를 후보로 볼 만한 경우.</strong> 데이터가 조직 경계를 벗어날 수 없어 자체 호스팅이 전제인 워크로드가 첫 번째입니다. 이 경우 선택지는 원래 좁았는데, 프론티어 바로 아래 구간에 다운로드 가능한 모델이 여러 개 생겼습니다. 두 번째는 토큰 소비량이 큰 대량 처리입니다. 분류, 추출, 1차 요약처럼 최상위 추론이 필요하지 않은 작업에서 두 자리 배수 가격 차이는 설계를 바꿀 만한 크기입니다. 세 번째는 가중치에 직접 학습을 걸어야 하는 경우입니다.

<strong>클로즈드 API가 여전히 단순한 경우.</strong> 최상위 성능이 필요한 구간에서는 격차가 좁아졌다고 해도 순서는 아직 유지됩니다. 2.8조 파라미터 모델을 자체 서빙하는 데 드는 GPU 메모리와 운영 인력을 계산에 넣으면, 절감액이 운영 비용에 잠식되는 규모 구간이 존재합니다. 태스크당 86센트였던 K3와 3센트였던 V4-Flash의 차이가 오픈웨이트 안에서도 이 구분이 필요하다는 걸 보여줍니다. 오픈웨이트를 API로 쓴다면 자체 호스팅의 데이터 통제 이점은 얻지 못하면서 가격만 취하는 구성이 됩니다. 그 자체가 나쁜 선택은 아니지만, 무엇을 얻고 있는지는 구분해야 합니다.

<strong>어느 쪽을 고르든 남는 조건.</strong> 라이선스는 모델마다 다르고 실제 조항을 읽어야 합니다. GLM-5.2와 DeepSeek V4-Flash는 MIT지만, Kimi K3는 [자체 라이선스](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE)입니다. 모델을 서비스형으로 제공하는 사업의 매출이 12개월 누적 2000만 달러를 넘으면 Moonshot AI와 별도 계약을 맺어야 하고, 월 활성 사용자 1억 명이나 월 매출 2000만 달러를 넘는 제품에는 "Kimi K3" 표기를 UI에 노출해야 합니다. 내부 사용에는 두 조항이 적용되지 않습니다. Qwen3.8-Max는 라이선스가 아직 발표되지 않았습니다. 가격도 고정이 아닙니다. DeepSeek은 인상을 예고한 상태이고, 오늘의 비용 우위로 몇 년치 계산을 세우는 건 위험합니다. 재현성은 여전히 닫혀 있습니다. 가중치를 받아도 학습 데이터를 알 수 없으므로, 규제 산업에서 요구하는 수준의 데이터 출처 감사는 오픈웨이트로도 해결되지 않습니다. Stanford HAI의 문제 제기가 겨냥하는 지점입니다.

7주 동안 네 개의 모델이 나왔고, 최대 기록과 최저가 기록이 각각 갈렸습니다. 이 속도가 계속될지는 알 수 없습니다. 확실한 건 "성능이 필요하면 클로즈드 API, 저렴함이 필요하면 품질을 포기"라는 이분법이 예전보다 덜 들어맞게 됐다는 것입니다. 그렇다고 오픈웨이트가 클로즈드를 대체하는 단계는 아닙니다. 두 선택지의 겹치는 구간이 넓어졌고, 그래서 워크로드별로 따져야 할 일이 늘었습니다.

## References

- Z.ai. (2026). *GLM-5.2.* [z.ai/blog/glm-5.2](https://z.ai/blog/glm-5.2)
- Artificial Analysis. (2026). *GLM-5.2 is the new leading open weights model on the Artificial Analysis Intelligence Index.* [artificialanalysis.ai](https://artificialanalysis.ai/articles/glm-5-2-is-the-new-leading-open-weights-model-on-the-artificial-analysis-intelligence-index)
- Moonshot AI. (2026). *Kimi K3.* [kimi.com/blog/kimi-k3](https://www.kimi.com/blog/kimi-k3)
- Moonshot AI. *Kimi-K3 모델 저장소.* [huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3)
- Artificial Analysis. (2026). *Kimi K3 achieves #3 in the Artificial Analysis Intelligence Index, comparable to Opus 4.8 and GPT-5.5.* [artificialanalysis.ai](https://artificialanalysis.ai/articles/kimi-k3-achieves-3-in-the-artificial-analysis-intelligence-index-comparable-to-opus-4-8-and-gpt-5-5)
- Artificial Analysis. *Kimi K3 모델 페이지.* [artificialanalysis.ai/models/kimi-k3](https://artificialanalysis.ai/models/kimi-k3)
- Z.ai. *GLM-5.2 모델 저장소.* [huggingface.co/zai-org/GLM-5.2](https://huggingface.co/zai-org/GLM-5.2)
- Reuters. (2026). *China's Moonshot unveils world's largest open AI model, closing in on US rivals.* [reuters.com](https://www.reuters.com/world/china/chinas-moonshot-unveils-worlds-largest-open-ai-model-closing-us-rivals-2026-07-17)
- CNBC. (2026). *Moonshot AI releases Kimi K3.* [cnbc.com](https://www.cnbc.com/2026/07/17/moonshot-ai-kimi-k3-model-openai-anthropic-china.html)
- Lambert, N. (2026). *Kimi K3: the open weights escalation.* [interconnects.ai](https://www.interconnects.ai/p/kimi-k3-the-open-weights-escalation)
- DeepSeek. *API 가격 페이지.* [api-docs.deepseek.com/quick_start/pricing](https://api-docs.deepseek.com/quick_start/pricing)
- DeepSeek. *API 변경 이력.* [api-docs.deepseek.com/updates](https://api-docs.deepseek.com/updates)
- DeepSeek. *DeepSeek-V4-Flash-0731 모델 카드.* [huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
- Baptista, E. (2026). *DeepSeek's new AI model is by far cheapest of well-known models to run, research firm says.* Reuters. [reuters.com](https://www.reuters.com/business/retail-consumer/deepseeks-new-ai-model-is-by-far-cheapest-well-known-models-run-research-firm-2026-08-03)
- Anthropic. *Claude 모델 가격.* [platform.claude.com/docs/en/docs/about-claude/pricing](https://platform.claude.com/docs/en/docs/about-claude/pricing)
- Qwen Team. (2026). *Qwen3.8-Max.* [qwen.ai/blog?id=qwen3.8](https://qwen.ai/blog?id=qwen3.8)
- Lynch, S. (2026). *Open-Weight Models Aren't Enough. We Need Truly Open Source AI Models for Science and Society.* Stanford HAI. [hai.stanford.edu](https://hai.stanford.edu/news/open-weight-models-arent-enough-we-need-truly-open-source-ai-models-for-science-and-society)
- Stanford HAI. (2026). *The 2026 AI Index Report.* [hai.stanford.edu/ai-index/2026-ai-index-report](https://hai.stanford.edu/ai-index/2026-ai-index-report)
- Stanford HAI. (2026). *The 2026 AI Index Report: Technical Performance.* [hai.stanford.edu/ai-index/2026-ai-index-report/technical-performance](https://hai.stanford.edu/ai-index/2026-ai-index-report/technical-performance)
- Shao, G., & Graylin, A. W. (2026). *Has the AI race shifted from U.S. vs China to open vs closed?* Fortune. [fortune.com](https://fortune.com/2026/08/04/has-the-ai-race-shifted-from-u-s-vs-china-to-open-vs-closed)
- Zhou, V. (2026). *Why Silicon Valley is divided over China's powerful, cheap AI models.* Rest of World. [restofworld.org](https://restofworld.org/2026/silicon-valley-debate-chinese-open-weight-ai-models)
- Moonshot AI. *Kimi K3 라이선스.* [huggingface.co/moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE)
