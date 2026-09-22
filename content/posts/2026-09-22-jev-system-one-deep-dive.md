---
title: "Jev: 판단 전용 AI의 원리와 활용"
slug: "jev-system-one-deep-dive"
date: 2026-09-22T09:38:04+09:00
draft: false
author: "Jesam Kim"
description: "TypeSafe AI의 Jev를 쉬운 애니메이션과 함께 살펴봅니다. Choice, Score, Noul부터 병렬 평가, RLCD, 확률 보정, 공식 벤치마크의 조건과 실제 적용 방법까지 설명합니다."
categories: ["AI/ML 기술 심층분석"]
tags: ["Jev", "TypeSafe AI", "System One", "RLCD", "Calibration", "AI Agent"]
cover:
  image: "/ai-tech-blog/images/jev-system-one-deep-dive/cover.png"
  alt: "들어온 물건을 여러 센서가 함께 살펴보고 정해진 경로로 분류하는 복고풍 로봇"
  relative: false
---

고객이 “어제 주문한 물건을 취소하고 싶습니다”라고 썼습니다. 프로그램에는 몇 가지 판단이 필요합니다. 주문 취소 요청인지, 이미 배송한 주문인지, 바로 처리할 수 있는지 확인해야 합니다. 고객에게 보낼 답장을 쓰기 전에도 이런 결정이 여러 번 일어납니다.

TypeSafe AI가 2026년 9월 15일 early access로 공개한 <strong>Jev는 프로그램이 사용할 판단값을 반환하는 AI 모델</strong>입니다. 개발자가 상황과 질문을 보내면, 정해진 선택지와 점수, 확률로 결과를 돌려줍니다. 회사는 이 모델 계열을 ‘System One Model’이라고 부릅니다. 이 글은 2026년 9월 22일 확인한 공식 문서와 업체 평가를 기준으로 작성했습니다. [출시 발표](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

## 고객 문의 하나로 보는 Jev

Jev를 정해진 항목이 있는 접수표를 채우는 담당자에 비유할 수 있습니다. 다만 실제로 고객에게 전화하거나 취소 버튼을 누르는 주체는 애플리케이션입니다. Jev는 상황을 읽고 질문에 대한 판단을 반환합니다.

다음 그림에서는 중복 결제 문의를 예로 들어, 필요한 판단을 얻은 프로그램이 처리 방향을 정하는 과정을 살펴봅니다. 화면에 들어오면 짧은 영상이 음소거 상태로 반복 재생됩니다. 재생을 멈춰 읽거나 아래 링크에서 직접 단계를 넘겨 볼 수 있습니다. <strong>애니메이션의 입력과 숫자는 설명용 예시이며 실제 Jev 호출 결과가 아닙니다.</strong>

<div class="jev-media">
<video id="jev-motion" autoplay loop muted playsinline controls preload="none" poster="/ai-tech-blog/images/jev-system-one-deep-dive/jev-eli5-poster.png" data-desktop-src="/ai-tech-blog/images/jev-system-one-deep-dive/jev-eli5.mp4" data-mobile-src="/ai-tech-blog/images/jev-system-one-deep-dive/jev-eli5-mobile.mp4" data-mobile-poster="/ai-tech-blog/images/jev-system-one-deep-dive/jev-eli5-mobile-poster.png" width="1092" height="996" aria-label="고객 문의에서 판단과 코드의 분기로 이어지는 설명용 반복 애니메이션" style="display:block;width:100%;height:auto;border:1px solid #dbe2ea;border-radius:16px;background:#fff;"></video>
<noscript>
<style>#jev-motion{display:none!important}</style>
<img src="/ai-tech-blog/images/jev-system-one-deep-dive/jev-eli5-poster.png" alt="고객 문의를 함께 평가한 결과를 코드가 읽고 사람 검토와 자동 처리를 구분합니다.">
</noscript>
<script>
(() => {
  const video = document.getElementById('jev-motion');
  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const mobile = matchMedia('(max-width: 600px)').matches;
  let visible = false, userPaused = false, suppressPause = false, loaded = false;
  video.muted = true;
  video.defaultMuted = true;
  video.autoplay = false;
  if (mobile) {
    video.poster = video.dataset.mobilePoster;
    video.width = 720;
    video.height = 1000;
  }
  const load = () => {
    if (loaded) return;
    loaded = true;
    video.src = mobile ? video.dataset.mobileSrc : video.dataset.desktopSrc;
    video.preload = 'metadata';
    video.load();
  };
  const pauseAutomatically = () => {
    if (!video.paused) { suppressPause = true; video.pause(); }
  };
  const update = () => {
    if (!visible || document.hidden) { pauseAutomatically(); return; }
    load();
    if (reduced.matches || userPaused) return;
    video.play().catch(() => { /* Native controls and poster remain available. */ });
  };
  video.addEventListener('pause', () => {
    if (suppressPause) suppressPause = false;
    else userPaused = true;
  });
  video.addEventListener('play', () => { userPaused = false; });
  document.addEventListener('visibilitychange', update);
  reduced.addEventListener('change', () => {
    if (reduced.matches) pauseAutomatically();
    else update();
  });
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(entries => {
      visible = entries[0].isIntersecting;
      update();
    }, { threshold: 0.1 }).observe(video);
  } else { visible = true; update(); }
})();
</script>
</div>

[직접 단계를 넘겨 보기](/ai-tech-blog/images/jev-system-one-deep-dive/jev-eli5.html) / [GIF 버전](/ai-tech-blog/images/jev-system-one-deep-dive/jev-eli5.gif)

<details>
<summary>정지 그림으로 보기</summary>

[![같은 문의에서 여러 판단을 얻은 뒤 코드가 자동 처리와 사람 검토를 구분합니다.](/ai-tech-blog/images/jev-system-one-deep-dive/jev-eli5-poster.png)](/ai-tech-blog/images/jev-system-one-deep-dive/jev-eli5-poster.png)

</details>

*고객 문의에 대한 판단을 코드의 처리 규칙과 연결한 예시입니다. 출처: [TypeSafe의 소프트웨어 설계 가이드](https://docs.typesafe.ai/concepts/how-to-build-with-system-one).*

이 흐름에서 `state`는 모델이 읽을 자료입니다. 고객의 말, 주문 상태, 필요한 정책을 담습니다. `questions`에는 자료를 어떻게 판단할지 적습니다. 프로그램은 반환된 값을 읽고 다음 작업을 정합니다. 현재 Jev가 받는 입력은 텍스트입니다. 이미지나 녹음은 별도 처리로 텍스트 또는 구조화된 필드로 바꿔야 합니다. [State](https://docs.typesafe.ai/concepts/state)

## 판단을 표현하는 세 가지 타입

Jev의 질문은 Choice, Noul, Score로 나뉩니다. 이름보다 먼저 답의 형태를 보면 구분하기 쉽습니다.

| 질문 | 사용할 타입 | 반환값의 의미 |
|---|---|---|
| 취소, 배송 문의, 기타 중 어떤 요청입니까? | Choice | 선택지 하나, 선택지별 확률, confidence |
| 고객이 취소 의사를 명시했습니까? | Noul | ‘그렇다’의 확률 |
| 업무가 얼마나 중단됐습니까? | Score | 설명한 수준들의 위치를 확률로 가중한 점수, 수준별 확률, confidence |

Choice는 개발자가 제시한 선택지 안에서 고릅니다. 문서상 `choice`는 가장 높은 확률의 선택지입니다. 입력이 어느 항목에도 맞지 않을 수 있다면 ‘기타’나 ‘정보 부족’을 선택지에 넣어야 합니다. 허용한 답의 목록을 잘 만드는 일도 설계에 포함됩니다. [Choice](https://docs.typesafe.ai/primitives/choice)

Noul은 yes의 확률입니다. `0.9`는 yes 쪽으로 판단했다는 뜻이고, `0.1`은 no 쪽으로 판단했다는 뜻입니다. `0.5` 부근은 판단이 불확실한 상태입니다. <strong>Noul에는 별도 confidence 필드가 없습니다.</strong> “업무가 중단됐는가?”의 `0.5`를 “업무가 절반 중단됐다”로 읽으면 의미가 달라집니다. [Noul](https://docs.typesafe.ai/primitives/noul)

Score에서는 수준을 먼저 말로 정의합니다. 예를 들어 “업무에 영향 없음”, “불편하지만 대체 방법이 있음”, “대체 방법 없이 업무가 중단됨”의 순서입니다. 배열의 위치가 0, 1, 2가 되고, 점수는 각 위치에 확률을 곱해 더한 값입니다. ‘조금’, ‘중간’, ‘많이’만 쓰기보다 각 수준을 판단할 구체적인 상황을 적는 편이 좋습니다. 이 점수는 설명한 수준에서의 위치이므로, 1.5를 ‘업무가 75% 중단됐다’로 환산해서는 안 됩니다. [Score](https://docs.typesafe.ai/primitives/score)

## 같은 점수에도 서로 다른 판단이 들어갑니다

수준이 0, 1, 2인 경우를 생각해 보겠습니다. 확률이 `[0, 1, 0]`이면 점수는 1입니다. 확률이 `[0.5, 0, 0.5]`여도 점수는 1입니다. 전자는 중간 수준 하나에 확률이 모여 있고, 후자는 양 끝으로 나뉘어 있습니다. 평균만 저장하면 이 차이를 잃습니다.

Choice와 Score가 반환하는 `confidence`는 확률분포 모양에서 계산한 0부터 1 사이의 통계량입니다. 공식 문서에는 세 선택지용 데모 근사식이 있지만, 이를 실제 API가 쓰는 정확한 계산식으로 명시하지는 않습니다. 따라서 `confidence`를 최고 확률과 같은 값으로 가정하거나, `0.9`를 “이 답이 정답일 확률 90%”로 바꾸어 설명해서는 안 됩니다. 해당 업무에서 confidence 구간별 실제 정답률을 측정해야 합니다. [Confidence](https://docs.typesafe.ai/confidence)

[![Choice의 확률분포가 한 보기에 모인 경우와 여러 보기에 퍼진 경우를 비교합니다.](/ai-tech-blog/images/jev-system-one-deep-dive/probability-confidence.png)](/ai-tech-blog/images/jev-system-one-deep-dive/probability-confidence.png)

*같은 선택지에 대한 두 가지 확률분포를 비교한 예시입니다. 실제 측정값은 아닙니다. 그림을 누르면 크게 볼 수 있습니다. 출처: [Score](https://docs.typesafe.ai/primitives/score), [Confidence](https://docs.typesafe.ai/confidence).*

여기서 <strong>확률 보정(calibration)</strong>을 구분해야 합니다. 예측 확률이 0.8인 사례들을 많이 모았을 때 해당 결과가 약 80% 발생한다면, 그 구간에서 확률이 실제 빈도와 잘 맞습니다. 개별 예측의 정답을 보장하는 성질은 아닙니다. 신경망의 정확도와 확률 보정은 별도로 평가해야 한다는 점은 [Guo 등의 ICML 2017 연구](https://proceedings.mlr.press/v70/guo17a.html)에서도 다뤘습니다.

TypeSafe는 확률 보정을 학습 목표로 내세웁니다. 그렇더라도 새로운 도메인, 한국어 입력, 새로운 모델 버전에서 같은 품질이 유지되는지는 따로 확인해야 합니다. 자동 처리할 범위를 정할 때는 “confidence가 높은가”와 “그 구간에서 실제로 얼마나 틀리는가”를 함께 봐야 합니다.

## 여러 질문을 한 번에 평가하는 방식

일반적인 자기회귀 언어 모델은 앞서 생성한 토큰에 이어 다음 토큰을 생성합니다. 구조화된 출력을 요청해도, 문자열을 생성하는 방식에서는 출력 길이만큼 순차적인 생성 단계가 이어집니다. TypeSafe는 Jev가 state를 한 번 받아 각 질문을 병렬로 평가한다고 설명합니다. 아래 그림은 이 처리 방식의 차이를 설명합니다. GPU 내부 구조나 실제 지연 시간의 비율을 나타낸 그림은 아닙니다.

[![언어 모델의 순차적인 토큰 생성과 Jev의 공유 state에 대한 여러 질문 평가를 비교합니다.](/ai-tech-blog/images/jev-system-one-deep-dive/flow-comparison.png)](/ai-tech-blog/images/jev-system-one-deep-dive/flow-comparison.png)

*토큰을 순서대로 생성하는 방식과 여러 질문을 함께 평가하는 방식을 비교했습니다. 출처: [System One](https://docs.typesafe.ai/concepts/system-one), [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out).*

같은 문의에서 요청 유형, 취소 의사, 업무 영향도를 판단한다면 세 질문을 한 요청에 넣을 수 있습니다. state를 반복해서 보내는 비용도 줄어듭니다. 답이 필요한 분기가 아직 정해지지 않았더라도, 동일한 자료만 있으면 판단할 수 있는 질문은 미리 묶어 보낼 수 있습니다. 이를 문서는 speculative fan-out이라고 설명합니다. [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)

단, <strong>같은 요청 안의 질문은 서로의 답을 읽지 않습니다.</strong> “첫 질문에서 고른 상품을 두 번째 질문에서 평가해 달라”는 의존 관계를 이름만으로 연결할 수 없습니다. 첫 답을 이용해 데이터베이스를 조회하거나 다음 선택지를 만들어야 한다면, 그 결과로 새 state를 구성하고 다음 요청을 보내야 합니다. [질문 사이의 의존 관계](https://docs.typesafe.ai/primitives#when-one-question-depends-on-another)

이 점은 시스템을 설계할 때 유용합니다. 먼저 코드로 계산 가능한 항목을 처리하고, 같은 자료로 판단할 수 있는 질문을 묶습니다. 앞 단계 결과가 꼭 필요한 작업만 다음 단계로 남깁니다. 이 방식의 성능은 모델뿐 아니라 질문 분해와 데이터 준비 방식에도 영향을 받습니다.

## RLCD에서 공개된 내용과 아직 모르는 내용

RLCD는 Reinforcement Learning for Calibrated Decisions의 약자입니다. TypeSafe는 이를 판단값과 보정된 확률을 반환하도록 학습하는 방식으로 설명합니다. RLHF가 사람의 선호를, RLVR이 검증 가능한 보상을 강조하는 것과 학습 목표를 비교합니다. 이는 TypeSafe가 제시하는 구분이며, 실제 모델의 학습 과정 전체를 세 이름만으로 설명할 수 있다는 뜻은 아닙니다. [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)

소프트웨어에서 틀린 판단의 비용은 행동마다 다릅니다. 낮은 확신으로 문의 분류를 추천하는 일과 결제를 확정하는 일에 같은 기준을 쓰기는 어렵습니다. 확률이 실제 결과와 잘 맞는다면, 프로그램은 예상 오류와 검토 비용을 비교해 자동화 범위를 정할 수 있습니다. 이것이 보정된 판단을 학습 목표로 삼는 이유입니다.

공개 자료로 확인할 수 있는 것은 출력 계약, 병렬 평가 방식, 학습 목표, 업체가 보고한 결과입니다. 이번에 확인한 자료에서는 파라미터 수, 상세 신경망 구성, 재현 가능한 학습 절차와 손실함수를 확인하지 못했습니다. ‘트랜스포머를 완전히 대체했다’거나 특정한 분류 헤드를 쓴다고 단정할 근거도 확보하지 못했습니다. <strong>이 글의 그림은 공개 인터페이스와 프로그램의 실행 흐름을 설명합니다.</strong>

JSON Schema를 따르는 출력과도 구분해서 볼 필요가 있습니다. 스키마는 값의 형식을 제한합니다. 판단이 현실에 맞는지, 확률이 실제 빈도와 맞는지, 얼마의 비용과 지연으로 얻는지는 각각 별도의 평가 대상입니다. Jev의 주장은 이러한 항목을 판단 업무에 맞춰 함께 최적화했다는 데 있습니다.

## API 요청에서 프로그램의 분기까지

다음은 문서의 현재 요청 형식을 따라 작성한 예시입니다. `POST https://api.typesafe.ai/v1/systemone`에 보내는 JSON이며, 실제 추론은 실행하지 않았습니다. 한국어 설명과 대응하기 쉽도록 입력도 한국어로 작성했습니다. 한국어 품질은 별도 평가가 필요합니다. 결과를 비교할 수 있도록 모델 버전을 고정했습니다. `jev-latest` 같은 별칭은 새 버전이 나오면 가리키는 모델이 바뀔 수 있습니다. [모델 버전과 별칭](https://docs.typesafe.ai/models)

```json
{
  "model": "jev-1.13.0",
  "state": {
    "message": "어제 주문한 물건을 취소하고 싶습니다.",
    "order": {"status": "배송 준비", "cancellable": true}
  },
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "message에 담긴 고객의 주된 요청을 분류하세요.",
      "criteria": {
        "cancel": "주문 취소를 요청합니다.",
        "delivery": "배송 상태나 도착 일정을 묻습니다.",
        "other": "다른 요청이거나 요청을 명확히 파악할 수 없습니다."
      }
    },
    "explicit_cancel": {
      "type": "noul",
      "instructions": "message에서 고객이 주문 취소 의사를 명시했습니까?"
    },
    "frustration": {
      "type": "score",
      "instructions": "message에 드러난 고객의 불만 표현을 평가하세요.",
      "criteria": [
        "불만을 표현하지 않고 요청이나 사실을 전달합니다.",
        "불편이나 불만을 표현하지만 공격적인 표현은 없습니다.",
        "강한 분노나 모욕, 이용 중단 의사를 표현합니다."
      ]
    }
  }
}
```

`intent` 같은 질문 ID는 결과를 찾아 읽는 키입니다. 모델에는 ID가 전달되지 않으므로 질문의 의미를 `instructions`에 적어야 합니다. `criteria`의 설명도 질문과 같은 기준을 사용해야 합니다. [Primitives](https://docs.typesafe.ai/primitives), [HTTP API](https://docs.typesafe.ai/api)

응답을 받은 코드는 유형과 필요한 필드를 검증하고, 주문 시스템에서 취소 가능 여부를 다시 확인합니다. 고객의 의도가 취소라고 판단되면 취소 확인 화면을 보여줄 수 있습니다. confidence가 낮거나 문의 내용이 모호하면 사람에게 넘깁니다. <strong>모델의 확신이 높아도 사용자 권한이나 주문 정책을 대신할 수는 없습니다.</strong>

위 예제의 불만 점수는 상담 우선순위에 활용할 수 있습니다. 취소 허용 여부는 주문 시스템의 규칙으로 계산합니다. 그림에서 두 판단을 한 화면에 보여주더라도 두 값의 역할은 다릅니다.

## 속도와 비용 수치를 읽는 기준

2026년 9월 22일 공식 모델 페이지에 표시된 버전은 `jev-1.13.0`입니다. 입력 100만 토큰당 가격은 0.042달러이고 출력 토큰은 무료입니다. 요청 전체는 64k 토큰, state와 가장 긴 질문의 합은 32k 토큰이라는 두 제한이 함께 적용됩니다. 영어가 주된 학습 언어이며 다른 언어의 품질은 직접 확인하도록 안내합니다. [Models](https://docs.typesafe.ai/models)

가격을 단순 계산하면 입력 1,000토큰은 0.000042달러입니다. 그런 요청 100만 건은 입력 요금만 42달러입니다. 질문과 선택지 설명도 입력에 포함되고, 전처리, 다른 모델 호출, 재시도, 사람의 검토 비용은 이 계산에 들어 있지 않습니다. 이는 공개 단가를 이용한 산술 예시입니다.

홈페이지의 ‘193.6배 빠름, 444.6배 저렴함’은 업체의 워크플로 평가에서 나온 수치입니다. 출시 글은 이 배수가 실제 개선 폭의 높은 쪽일 수 있다고 밝힙니다. 평가를 주로 미국 서부에서 실행했고, 비교 LLM에는 확률까지 생성하도록 하는 래퍼를 적용했습니다. 따라서 모든 업무, 모든 리전, 모든 출력 방식에서 동일한 배수가 나온다고 해석하기 어렵습니다. [출시 발표의 평가 조건](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

[![Jev와 비교 모델의 비용, 지연, 참조 답안 일치율을 함께 비교한 업체 자체 평가입니다.](/ai-tech-blog/images/jev-system-one-deep-dive/benchmark.png)](/ai-tech-blog/images/jev-system-one-deep-dive/benchmark.png)

*공식 평가의 다섯 모델을 비교한 차트입니다. 각 수치는 4개 워크플로를 같은 비중으로 평균한 값입니다. 비용과 시간은 케이스당 값입니다. 원문의 비용과 시간 축은 로그이며, 여기서는 0을 기준으로 한 선형 축으로 그렸습니다. 모델명과 반올림된 값은 원문 표기를 따릅니다. 출처: [Workflow evals](https://evals.typesafe.ai/), 2026년 9월 22일 확인.*

이 평가의 참조 답안은 GPT-6 Astra와 Claude Fable 5.1을 높은 추론 설정으로 실행한 결과를 평균해 만들었습니다. 다른 모델은 제공자의 기본 추론 설정을 사용합니다. 그래프의 ‘accuracy’는 이 참조 답안을 기준으로 한 값이므로, 사람이 검증한 실제 업무 정답률과 구분해야 합니다. 보안 사고, 에이전트 실행 기록, 청구서, 고객 서비스의 네 워크플로가 같은 비중으로 반영됩니다. [평가 방법](https://evals.typesafe.ai/)

TypeSafe는 이 참조 답안 구성에 OpenAI와 Anthropic 모델에 유리한 편향이 있다고 밝히며, 자사와 DeepSeek 모델의 상대 성능을 과소평가할 수 있다고 덧붙입니다. 따라서 그래프에서 가까운 수치 몇 개만으로 모델의 일반적인 지능 순위를 정할 수는 없습니다. [참조 답안의 편향에 관한 설명](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

발췌한 그래프에서는 Jev의 지연과 비용이 작지만, 참조 답안 일치율이 가장 높지는 않습니다. 실제 도입에서는 같은 업무 결과를 얻는 데 필요한 전체 비용을 비교해야 합니다. 판단이 어려운 사례를 더 큰 모델이나 사람에게 넘겼을 때의 비용까지 포함하면 비교 결과가 달라질 수 있습니다.

## 타입 보장과 판단 오류를 구분해야 합니다

TypeSafe는 출시 글에서 ‘환각 없음’을 강조하면서, 해당 0% 수치가 스키마 일치 보장에 근거한다고 설명합니다. 허용된 선택지가 `cancel`, `delivery`, `other`라면 목록 밖의 답을 생성하지 않는다는 성질입니다. `delivery`가 정답인 상황에서 `cancel`을 고르는 오류는 별도로 남습니다. [타입 보장에 관한 설명](https://typesafe.ai/blog/introducing-system-one-models-and-jev)

공식 ‘Jev 1.13 jaggedness’ 문서는 실제 적용에서 주의할 실패 형태를 공개합니다. 특히 다음 항목은 글을 읽고 바로 실험해 볼 만합니다. [알려진 한계](https://docs.typesafe.ai/model-jaggedness/jev-1.13)

| 입력이나 작업 | 문서에 나온 한계 | 프로그램에서 처리할 부분 |
|---|---|---|
| 숫자 계산, 개수 세기, 날짜 비교 | 정확한 수치 처리에 약합니다. | 파서와 계산 코드로 처리합니다. |
| 무관한 자료가 많은 긴 state | 관련 없는 내용이 정확도를 떨어뜨릴 수 있습니다. | 검색과 필터링으로 필요한 자료를 고릅니다. |
| 입력에 숨은 지시, 오도하는 문장 | state를 기본적으로 적대적 입력으로 취급하지 않습니다. | 공격 사례를 평가하고 실행 권한을 별도로 제한합니다. |
| 같은 의미를 다르게 물은 질문 | 질문 사이의 논리적 항등식을 보장하지 않습니다. | 중복 질문을 줄이고 필요한 관계를 코드로 강제합니다. |

마지막 항목은 확률을 조합할 때 중요합니다. “취소를 요청했는가?”와 “취소가 아닌 다른 것을 요청했는가?”를 별도 질문으로 보내고 두 응답의 합이 반드시 1이라고 가정하면 안 됩니다. 상호 배타적인 분류가 필요하다면 하나의 Choice로 정의하는 방법을 검토할 수 있습니다. 모델 출력이 프로그램에서 어떤 관계를 만족해야 하는지 명시해야 합니다.

## 작게 검증할 때 측정할 것

첫 실험으로는 업무 하나를 고르는 편이 좋습니다. 예를 들어 고객 문의를 취소, 배송, 기타로 분류하되 실제 주문은 변경하지 않고 기존 담당자의 판단과 비교합니다. 다음 절차는 이 글에서 제안하는 평가 방법입니다.

1. 사람이 검토한 정답 데이터에 모호한 문의, 복합 요청, 한국어 표현, 입력에 숨은 지시를 포함합니다.
2. 질문 문구와 임계값을 조정할 데이터와 최종 평가 데이터를 분리합니다.
3. 정답률과 함께 확률 구간별 실제 빈도, 자동 처리한 비율, 자동 처리한 사례 중 오답 비율을 측정합니다.
4. 모델 호출뿐 아니라 검색, 재시도, 대체 모델, 검토 대기까지 포함한 비용과 지연을 기록합니다.
5. 모델 버전과 질문, 처리 규칙을 고정하고 변경 때마다 같은 평가를 다시 실행합니다.

자동 처리 비율을 올리면 더 어려운 사례까지 받아들일 수 있습니다. 반대로 확신이 낮은 사례를 많이 보류하면 자동화로 줄이는 업무량이 작아집니다. 이 관계를 보면서 원하는 오류 수준에서 얼마나 많은 요청을 처리할 수 있는지 정해야 합니다. confidence 임계값 하나를 인터넷 예제에서 가져오는 것으로는 이 결정을 대신할 수 없습니다.

Jev를 적용할 부분은 프로그램이 반복해서 수행하는 좁은 의미 판단에서 찾을 수 있습니다. 질문의 답이 어떤 행동에 쓰이는지부터 적고, 그 행동에 필요한 데이터와 검증 기준을 정하면 실험 범위가 명확해집니다. 실제 지불, 삭제, 권한 변경을 시작하기 전에는 기존 처리와 나란히 실행해 결과 차이를 확인할 수 있습니다.

## References

- [TypeSafe AI, Introducing System One Models & Jev, 2026-09-15](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [TypeSafe AI, System One](https://docs.typesafe.ai/concepts/system-one)
- [TypeSafe AI, State](https://docs.typesafe.ai/concepts/state)
- [TypeSafe AI, Primitives](https://docs.typesafe.ai/primitives)
- [TypeSafe AI, Choice](https://docs.typesafe.ai/primitives/choice)
- [TypeSafe AI, Noul](https://docs.typesafe.ai/primitives/noul)
- [TypeSafe AI, Score](https://docs.typesafe.ai/primitives/score)
- [TypeSafe AI, Confidence](https://docs.typesafe.ai/confidence)
- [TypeSafe AI, AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)
- [TypeSafe AI, Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)
- [TypeSafe AI, How to build with TypeSafe](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)
- [TypeSafe AI, API reference](https://docs.typesafe.ai/api)
- [TypeSafe AI, Models](https://docs.typesafe.ai/models)
- [TypeSafe AI, Workflow evals](https://evals.typesafe.ai/)
- [TypeSafe AI, Jev 1.13 jaggedness, 2026-09-17 검토본](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- [Guo, Pleiss, Sun, Weinberger, On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)
