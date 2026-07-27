---
title: "Opus 5와 Fable 5 — 코딩 벤치마크와 실무 태스크의 간극"
date: 2026-07-27T12:00:00+09:00
draft: false
categories: ["AWS AI/ML"]
tags: ["Claude Opus 5", "Claude Fable 5", "Amazon Bedrock", "벤치마크", "모델 선택"]
author: "Jesam Kim"
description: "Claude Opus 5는 agentic 코딩 벤치마크에서 Fable 5를 앞섰지만, 시각적 생성이나 구조적 산출 같은 태스크에서는 다른 이야기가 나옵니다. 벤치마크 커버리지와 실무 워크로드가 어긋나는 지점을 정리합니다."
cover:
  image: "/ai-tech-blog/images/opus5-vs-fable5-benchmark-vs-real-tasks/cover.png"
  alt: "Opus 5와 Fable 5의 벤치마크 점수와 실무 태스크 비교"
  relative: false
---

새 모델이 나오면 가장 먼저 보게 되는 건 벤치마크 표입니다. 그런데 표에서 이긴 모델을 실제 작업에 붙여 보면 기대와 다른 결과가 나오는 경우가 있습니다. 2026년 7월에 출시된 Claude Opus 5가 그런 사례를 만들었습니다.

[Amazon Bedrock과 Claude Platform on AWS에서 이용 가능한](https://aws.amazon.com/blogs/machine-learning/introducing-claude-opus-5-on-aws-anthropics-most-capable-opus-model/) Opus 5는 API model id `claude-opus-5`, 컨텍스트 1M tokens, 최대 출력 128K tokens, knowledge cutoff는 2026년 5월입니다. adaptive thinking이 기본으로 켜져 있고, thinking을 끄면 effort가 high로 캡됩니다.

가격은 input $5 / MTok, output $25 / MTok입니다. Opus 4.8과 동일한 단가이고, Fable 5의 $10 / $50 대비 절반입니다. 벤치마크 점수와 별개로, 이 가격 구조가 모델 선택 판단에 먼저 개입하는 조건입니다.

## 벤치마크에서는 Opus 5가 앞섭니다

Anthropic이 공개한 agentic terminal 코딩 벤치마크 Frontier-Bench v0.1에서 Opus 5는 43.3점을 기록했습니다. Fable 5는 33.7점, Opus 4.8은 18.7점입니다. 10점 가까운 차이는 벤치마크 표에서 흔히 보는 소수점 단위 경쟁과 다른 폭입니다.

반면 SWE-bench Pro에서는 순서가 뒤집힙니다. Fable 5가 80.0%, Opus 5가 79.2%로 [Fable 5가 근소하게 앞섭니다](https://codingfleet.com/blog/claude-opus-5-vs-claude-fable-5/). 0.8%p 차이는 측정 노이즈 범위로 봐도 무리가 없는 수준입니다.

![Frontier-Bench v0.1과 SWE-bench Pro에서의 Claude Opus 5, Fable 5, Opus 4.8 점수 비교 막대 차트](/ai-tech-blog/images/opus5-vs-fable5-benchmark-vs-real-tasks/bench-compare.png)
*Frontier-Bench v0.1(왼쪽)과 SWE-bench Pro(오른쪽). 벤치마크에 따라 순위가 달라집니다. 출처: Anthropic(Frontier-Bench v0.1, 수치는 [Vellum](https://www.vellum.ai/blog/claude-opus-5-benchmarks-explained) 정리 기준), [CodingFleet](https://codingfleet.com/blog/claude-opus-5-vs-claude-fable-5/)(SWE-bench Pro)*

비용 대비 성능 쪽으로 보면 CursorBench 3.2 결과가 참고할 만합니다. max effort 설정에서 Opus 5는 [Fable 5 최고점의 0.5% 이내](https://www.vellum.ai/blog/claude-opus-5-benchmarks-explained)에 들어오면서 절반 비용으로 그 성능을 냈습니다.

코딩 외 축에서는 격차가 더 벌어집니다. novel problem-solving을 측정하는 ARC-AGI 3에서 Opus 5는 30.2점, GPT-5.6 Sol은 7.8점, Opus 4.8은 1.5점입니다. 처음 보는 문제 구조를 다루는 능력에서 세대 차이가 드러나는 수치입니다.

## 초기 사용 보고는 다른 방향을 가리킵니다

여기까지가 표의 이야기입니다. 초기 사용 보고는 다른 축을 짚습니다.

출시 직후 벤치마크를 정리한 한 [분석](https://tensorboyofficial.substack.com/p/opus-5-vs-fable-5)은 <strong>태스크 길이라는 축</strong>을 짚습니다. 대부분의 벤치마크에서 Opus 5가 앞서지만, 가장 긴 long-horizon 자율 작업에는 Anthropic 자신이 여전히 Fable 5를 권장한다는 점, 그리고 FrontierCode Main 같은 일부 코딩 벤치에서는 Fable 5가 53.5% 대 53.4%로 근소하게 앞선다는 점을 함께 지적합니다. 코드가 돌아가느냐를 넘어 한 번에 얼마나 완성된 산출물이 나오느냐가 별개의 축이라는 이야기입니다.

프론트엔드 작업에서는 태스크 규모가 변수로 등장합니다. Fable 5 출시 시점의 [Hacker News 논의](https://news.ycombinator.com/item?id=48495500)에서 한 사용자는 toy-scale 와이어프레임에서 Fable 5가 당시 Opus보다 눈에 띄게 나았다고 적었습니다. 동시에 레이아웃과 심미성을 모델이 스스로 결정해야 하는 중대형 멀티페이지 웹앱에서는 두 모델의 결과가 사람 평가자 기준으로 구분되지 않았다고 덧붙였습니다. 이 관찰은 Opus 5 출시 이전 시점이므로 비교 대상이 Opus 4.8이라는 점은 감안해야 합니다.

한쪽에서 Anthropic은 Opus 5가 interactive artifacts, 3D, 애니메이션, 데이터 시각화 같은 시각적 산출에서 이전 Opus 대비 크게 향상됐다고 [설명합니다](https://www.anthropic.com/news/claude-opus-5). Lovable과 Gamma 같은 파트너 피드백을 근거로 제시합니다. 그런데 시각적 태스크를 두고 커뮤니티에서 나온 이야기는 Fable 5 쪽이었습니다. 두 진술은 직접 충돌하지는 않습니다. 공식 발표는 Opus 계열 안에서의 세대 비교이고, 커뮤니티 관찰은 Fable 5와의 횡비교이기 때문입니다. 다만 이 둘을 나란히 놓았을 때 벤치마크 표만 보고 모델을 고르기 어렵다는 점은 분명해집니다.

이 관찰들은 개인 경험 보고입니다. 통제된 측정이 아니고, 사용한 프롬프트나 태스크 조건도 공개되지 않았습니다. 비교 시점의 상대 모델이 다른 경우도 있습니다. 일반화할 근거로는 약합니다. 그런데 서로 다른 채널에서 비슷한 방향의 관찰이 나온다면 그 자체가 확인해 볼 가설이 됩니다.

## 왜 간극이 생기나

Frontier-Bench v0.1이 무엇을 측정하는지 보면 설명이 됩니다. 이 벤치마크는 terminal 환경의 agentic 코딩을 다룹니다. 여러 파일을 수정하고, 디버깅하고, 스펙에서 기능을 구현하는 작업입니다. 자동 채점이 가능해야 벤치마크가 성립하므로, 테스트 통과나 명시적 정답 판정이 가능한 태스크로 구성됩니다.

실무 태스크 중 상당 부분은 그 조건을 만족하지 않습니다. SVG 다이어그램을 그리는 작업을 생각해 보면, 좌표를 배치하고 요소 간 겹침을 피하고 시각적 균형을 맞추는 판단이 필요합니다. 문법이 유효한 SVG인지는 기계로 검증할 수 있지만, 보기 좋은지는 그렇지 않습니다. 프론트엔드 심미성도 같은 성질입니다. 구조적 문서 산출, 슬라이드 구성, 레이아웃 설계 모두 사람 판단이 개입하는 영역입니다.

![벤치마크가 커버하는 terminal 코딩 영역과 실무 태스크 영역이 부분적으로만 겹치는 개념도](/ai-tech-blog/images/opus5-vs-fable5-benchmark-vs-real-tasks/coverage-gap.png)
*벤치마크 커버리지와 실무 태스크 분포. 겹치는 구간에서는 벤치마크 점수가 유효한 신호이지만, 겹치지 않는 구간은 별도 확인이 필요합니다.*

정리하면 벤치마크 태스크 분포와 실무 태스크 분포가 같지 않습니다. 특정 벤치마크에서 SOTA를 기록한 모델이 그 벤치마크가 커버하지 않는 워크로드에서도 우위라는 보장은 없습니다. SWE-bench Pro와 Frontier-Bench가 서로 다른 순위를 내놓는 것 자체가 같은 코딩 도메인 안에서도 측정 대상이 갈린다는 증거입니다.

## 모델을 고르는 순서

AWS에서 고객 워크로드를 다루면서 반복해서 확인하는 지점이 있습니다. 벤더가 공개하는 벤치마크는 방향성 신호이고, 최종 판단 근거는 아닙니다. 실무에 적용할 때는 순서를 이렇게 잡는 편이 안전합니다.

먼저 본인 워크로드에서 대표 태스크를 몇 개 뽑습니다. 실제로 반복 수행하는 작업이어야 하고, 평가 기준이 명시돼 있어야 합니다. 다음으로 후보 모델들을 같은 조건에서 돌려 비교합니다. Bedrock에서는 모델 id만 바꿔 동일 코드로 비교할 수 있으므로 이 실험 비용이 크지 않습니다.

그다음이 태스크별 모델 배정입니다. 전체 파이프라인을 단일 모델로 통일할 이유는 없습니다. 예를 들어 멀티파일 리팩터링과 디버깅 루프에는 Frontier-Bench에서 강한 모델을, 시각적 산출이 결과물인 단계에서는 그쪽에서 검증된 모델을 붙이는 구성이 가능합니다. 비용 축도 같이 봐야 합니다. Opus 5의 절반 단가는 반복 호출이 많은 단계에서 누적 차이를 만듭니다.

<strong>벤치마크 순위와 실무 체감은 서로 다른 축입니다.</strong> 두 축이 어긋난다고 해서 어느 쪽이 틀린 것은 아니고, 측정 대상이 다를 뿐입니다. 모델 선택 판단은 자신의 태스크에서 직접 측정한 값 위에 세우는 편이 낫습니다.

## References

- [Introducing Claude Opus 5 on AWS: Anthropic's most capable Opus model](https://aws.amazon.com/blogs/machine-learning/introducing-claude-opus-5-on-aws-anthropics-most-capable-opus-model/) &mdash; AWS Machine Learning Blog
- [Introducing Claude Opus 5](https://www.anthropic.com/news/claude-opus-5) &mdash; Anthropic
- [Anthropic Claude Opus 5 model card](https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-opus-5.html) &mdash; Amazon Bedrock User Guide
- [Claude Opus 5 benchmarks explained](https://www.vellum.ai/blog/claude-opus-5-benchmarks-explained) &mdash; Vellum
- [Claude Opus 5 vs Claude Fable 5](https://codingfleet.com/blog/claude-opus-5-vs-claude-fable-5/) &mdash; CodingFleet
- [Opus 5 vs Fable 5](https://tensorboyofficial.substack.com/p/opus-5-vs-fable-5) &mdash; Tensor Protocol
- [Claude Fable 5: mid-tier results on coding tasks](https://news.ycombinator.com/item?id=48495500) &mdash; Hacker News 논의
