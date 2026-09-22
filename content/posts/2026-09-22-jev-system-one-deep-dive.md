---
title: "Jev: 판단 전용 AI의 원리와 활용"
slug: "jev-system-one-deep-dive"
date: 2026-09-22T09:38:04+09:00
lastmod: 2026-09-22T13:19:40+09:00
tocopen: false
draft: false
author: "Jesam Kim"
description: "고객 문의 예제로 Jev의 주요 개념을 차근차근 설명합니다. 입력 자료와 질문, 세 가지 응답 타입, 확률과 confidence, 멀티에이전트에서의 활용까지 살펴봅니다."
categories: ["AI/ML 기술 심층분석"]
tags: ["Jev", "TypeSafe AI", "System One", "RLCD", "Calibration", "AI Agent"]
cover:
  image: "/ai-tech-blog/images/jev-system-one-deep-dive/cover.png"
  alt: "들어온 물건을 여러 센서가 함께 살펴보고 정해진 경로로 분류하는 복고풍 로봇"
  relative: false
---

고객이 “어제 주문한 물건을 취소하고 싶습니다”라고 썼습니다. 프로그램에는 몇 가지 판단이 필요합니다. 주문 취소 요청인지, 이미 배송한 주문인지, 바로 처리할 수 있는지 확인해야 합니다. 고객에게 보낼 답장을 쓰기 전에도 이런 결정이 여러 번 일어납니다.

TypeSafe AI가 2026년 9월 15일 early access로 공개한 <strong>Jev는 프로그램이 사용할 판단값을 반환하는 AI 모델</strong>입니다. 개발자가 상황과 질문을 보내면, 정해진 선택지와 점수, 확률로 결과를 돌려줍니다. 회사는 이 모델 계열을 ‘System One Model’이라고 부릅니다. 이 글은 2026년 9월 22일 확인한 공식 문서와 업체 평가를 기준으로 작성했습니다.

<small>참조: [출시 발표](https://typesafe.ai/blog/introducing-system-one-models-and-jev)</small>

## Jev가 맡는 일부터 구분하기

AI 모델을 처음 접할 때는 대화창을 떠올리기 쉽습니다. 질문을 쓰면 모델이 문장으로 답합니다. 그런데 소프트웨어가 다음 행동을 정할 때는 긴 설명보다 작은 판단값이 필요한 경우가 많습니다. “취소 요청입니다”라는 문장을 읽는 대신, `cancel`이라는 값을 받아 취소 처리 코드를 선택하는 식입니다.

고객의 취소 요청을 처리할 때 필요한 작업을 나누어 보겠습니다. 아래에서 LLM은 Large Language Model의 약자로, 대규모 언어 모델을 뜻합니다.

| 필요한 작업 | 이 예시에서 맡길 곳 | 결과 |
|---|---|---|
| 고객이 무엇을 원하는지 파악하기 | Jev 같은 판단 모델 | `cancel`, `delivery`, `other` 중 선택 |
| 주문이 실제로 발송됐는지 확인하기 | 주문 시스템을 조회하는 코드 | 현재 주문 상태 |
| 취소 정책에 따라 처리 가능 여부 계산하기 | 애플리케이션의 규칙 코드 | 허용 또는 검토 필요 |
| 고객에게 자연스러운 안내문 쓰기 | 정해진 문구 또는 생성형 LLM | 고객이 읽을 답장 |

LLM도 문의 분류에 사용할 수 있습니다. 위 표는 판단, 사실 조회, 규칙 적용, 문장 작성을 나누어 구성한 예시입니다. Jev가 판단을 잘하더라도 주문 데이터베이스를 조회하고 취소를 실행하는 코드는 여전히 필요합니다.

<small>참조: [소프트웨어에서 역할을 나누는 방법](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)</small>

Jev는 자연어 입력을 이해하지만 자유로운 답장이나 코드를 작성하지는 않습니다. 코드 작성을 돕는 코딩 에이전트에는 계속 LLM이 필요합니다. 공식 문서는 코딩 에이전트가 애플리케이션 코드에 Jev 호출을 넣도록 안내합니다. 예를 들어 개발한 상담 앱에서 Jev가 담당자를 고르고, 선택된 담당자용 LLM이 답장을 작성하게 할 수 있습니다.

<small>참조: [Jev와 코딩 에이전트의 관계](https://docs.typesafe.ai/introduction/coding-agents)</small>

### System One이라는 이름의 의미

System One이라는 이름은 대니얼 카너먼이 널리 알린 System 1과 System 2의 구분에서 가져왔습니다. System 1은 빠르고 직관적인 판단, System 2는 시간을 들여 생각하는 과정을 가리킵니다. TypeSafe는 이 중 빠르고 범위가 좁은 판단을 강조해 모델 계열의 이름으로 사용합니다.

<small>참조: [System One](https://docs.typesafe.ai/concepts/system-one)</small>

이 글의 고객 문의에서도 “취소 의사를 밝혔는가?”는 한정된 질문입니다. 반면 여러 계약을 해석해 새로운 처리 방침을 만들고 설명문까지 작성하는 일은 훨씬 넓은 작업입니다. Jev를 이해할 때는 <strong>큰 업무 안에서 반복되는 작은 판단을 담당한다</strong>는 점부터 보면 됩니다. System One이라는 이름만으로 인간의 사고 과정을 그대로 재현했다고 이해할 필요는 없습니다.

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

*고객 문의에 대한 판단을 코드의 처리 규칙과 연결한 예시입니다.*

<small>참조: [TypeSafe의 소프트웨어 설계 가이드](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)</small>

그림에서 Jev는 문의 유형을 고르고, 환불 요청 여부와 불만 정도를 판단합니다. 아래쪽의 애플리케이션 코드는 그 결과를 받아 다음 행동을 정합니다. 그림의 화살표를 따라갈 때는 “모델이 무엇을 판단했는지”와 “프로그램이 무엇을 실행했는지”를 구분하면 됩니다.

## 입력 자료와 질문을 따로 정합니다

프로그램은 API를 통해 Jev에 요청을 보냅니다. API는 프로그램이 다른 프로그램에 요청을 보내고 응답을 받는 접점입니다. Jev에 보내는 요청은 <strong>판단할 자료</strong>와 <strong>그 자료에 대해 물을 질문</strong>으로 나뉩니다. 자료를 담는 필드 이름이 `state`, 질문을 담는 필드 이름이 `questions`입니다. `state`라는 이름이 낯설다면 우선 “이번 판단에 사용할 자료”로 읽어도 됩니다.

<small>참조: [State](https://docs.typesafe.ai/concepts/state)</small>

다음은 취소 요청을 다루는 상황을 글로 정리한 예시입니다.

| 구분 | 넣을 내용 |
|---|---|
| 자료 `state` | 고객의 원문, 주문번호, 현재 배송 상태, 적용할 취소 정책 |
| 질문의 지시문 `instructions` | 고객이 어떤 요청을 하는지 분류하세요. |
| 판단 기준 `criteria` | 취소 요청, 배송 문의, 기타에 각각 어떤 문장이 해당하는지 설명 |
| 모델 결과를 받은 뒤 할 일 | 주문 상태와 사용자 권한을 확인하고, 확인 화면을 보여주거나 검토 담당자에게 전달 |

자료와 질문을 나누면 같은 고객 문의를 여러 관점에서 살펴볼 수 있습니다. “주된 요청은 무엇인가?”와 “불만을 표현했는가?”는 같은 문장을 보더라도 서로 다른 질문입니다. 무엇을 판단할지 바뀌면 질문을 바꾸고, 새로운 주문 정보가 필요하면 자료를 보강합니다.

### state는 애플리케이션이 준비하는 자료입니다

간단한 분류라면 고객 문장 하나만 보내도 됩니다. 앞선 대화나 주문 정책을 함께 봐야 한다면 관련 내용을 묶어 보냅니다. JSON은 이름과 값을 짝지어 이런 데이터를 주고받는 형식입니다. 예를 들어 `message`에는 고객 문장을, `order`에는 주문 정보를 넣어 어느 자료인지 구분할 수 있습니다. 현재 Jev 입력은 텍스트이며, 이미지나 녹음은 별도 처리로 텍스트 또는 구조화된 필드로 바꿔야 합니다.

<small>참조: [State](https://docs.typesafe.ai/concepts/state)</small>

Jev를 호출하는 프로그램은 필요한 대화 기록과 최신 주문 정보를 직접 준비해야 합니다. 예제에서 `order`를 전달했다는 이유로 Jev가 주문 시스템에 접속해 최신 상태를 확인하는 것은 아닙니다. 실제 조회와 자료 구성은 애플리케이션에서 수행합니다. 따라서 오래된 주문 정보를 전달하면 출력 형식이 정확하더라도 잘못된 자료를 바탕으로 판단할 수 있습니다.

<small>참조: [코드가 맡는 자료 준비와 실행](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)</small>

자료가 복잡하면 질문에 `order.status`처럼 확인할 필드를 적어 범위를 명확히 할 수 있습니다. 이 표기는 order 안의 status 값을 가리킨다는 뜻입니다.

<small>참조: [질문에서 자료를 가리키는 방법](https://docs.typesafe.ai/primitives)</small>

### typed output은 답의 종류와 범위를 정한다는 뜻입니다

자료와 질문을 준비했다면, 어떤 형태로 답을 받을지도 정해야 합니다. `typed output`은 돌려받을 값의 종류와 범위가 정해져 있다는 뜻입니다. 문의 유형이라면 정해진 선택지 하나, 여부를 묻는 질문이라면 0부터 1 사이의 확률, 정도를 묻는 질문이라면 기준에 따른 점수를 받습니다. 프로그램은 미리 정한 형식에 따라 결과를 읽습니다.

<small>참조: [Primitives](https://docs.typesafe.ai/primitives)</small>

예를 들어 `cancel`이 오면 취소 요청용 코드를, `delivery`가 오면 배송 조회용 코드를 선택할 수 있습니다. 고객 표현이 “취소해 주세요”에서 “이 주문은 진행하지 않겠습니다”로 달라져도, 프로그램이 받는 값의 종류는 그대로 유지됩니다. 다만 두 문장을 실제로 올바르게 분류하는지는 모델 품질을 평가해야 할 문제입니다.

Jev가 자유로운 문장을 생성하지 않는다고 해도 API로 주고받는 JSON 표현은 있습니다. 개발자는 그 JSON 안의 정해진 필드를 읽습니다. 이때 답의 형식이 정해졌다는 사실과, 그 답이 상황에 맞는다는 사실은 따로 확인해야 합니다. LLM도 구조화된 출력으로 같은 형태를 반환할 수 있습니다. 따라서 JSON 모양만 보고 어느 방식이 더 정확하고 저렴한지 판단할 수는 없습니다.

## 판단을 표현하는 세 가지 타입

Jev에서는 답의 형태에 따라 Choice, Noul, Score 중 하나를 선택합니다. 표에 나오는 confidence는 확률분포의 모양을 요약한 값으로, 뒤에서 확률과 구분해 살펴보겠습니다.

| 질문 | 사용할 타입 | 반환값의 의미 |
|---|---|---|
| 취소, 배송 문의, 기타 중 어떤 요청입니까? | Choice | 선택지 하나, 선택지별 확률, confidence |
| 고객이 취소 의사를 명시했습니까? | Noul | ‘그렇다’의 확률 |
| 업무가 얼마나 중단됐습니까? | Score | 설명한 수준들의 위치를 확률로 가중한 점수, 수준별 확률, confidence |

### Choice: 정해진 보기 중 하나를 고릅니다

Choice는 개발자가 제시한 선택지 안에서 고릅니다. 문서상 `choice`는 가장 높은 확률의 선택지입니다. 그 선택지만 반환하는 데 그치지 않고, 다른 선택지에도 얼마의 확률을 두었는지 `probabilities`에 담아 줍니다.

<small>참조: [Choice](https://docs.typesafe.ai/primitives/choice)</small>

보기의 이름과 설명은 요청을 만드는 개발자가 정합니다. 취소, 배송, 기타로 의도를 구분할 수도 있고, 영상처럼 결제, 배송, 기타로 담당 업무를 구분할 수도 있습니다. 어떤 분류가 필요한지에 따라 같은 Choice 타입에 다른 보기를 전달하는 것입니다.

가령 취소 0.85, 배송 0.10, 기타 0.05라는 가상 결과라면 선택은 취소입니다. 세 값의 합은 1이고, 취소에 가장 많은 확률을 두고 있습니다. 여기서 기타를 빼고 취소와 배송만 제시하면, 그 두 보기 중에서 판단하도록 문제를 바꾸는 셈입니다. 입력이 어느 항목에도 맞지 않을 수 있다면 ‘기타’나 ‘정보 부족’을 미리 포함해야 합니다.

선택지는 서로 구분되게 설명해야 합니다. “취소 문의”와 “주문 문의”는 겹칠 수 있으므로, 무엇을 우선할지 기준에 적는 편이 좋습니다. 고객이 “배송일도 알려 주시고, 늦으면 취소해 주세요”라고 썼다면 주된 요청 하나를 고를지, 배송과 취소 의사를 각각 확인할지부터 정해야 합니다.

### Noul: 한 문장이 맞는지 확률로 답합니다

Noul은 yes의 확률입니다. `0.9`는 yes 쪽으로 판단했다는 뜻이고, `0.1`은 no 쪽으로 판단했다는 뜻입니다. `0.5` 부근은 판단이 불확실한 상태입니다. <strong>Noul에는 별도 confidence 필드가 없습니다.</strong> “업무가 중단됐는가?”의 `0.5`를 “업무가 절반 중단됐다”로 읽으면 의미가 달라집니다.

<small>참조: [Noul](https://docs.typesafe.ai/primitives/noul)</small>

한 문의가 여러 항목에 해당할 수 있다면 Noul을 나누어 물을 수 있습니다. “배송일을 묻는가?”와 “취소 의사를 표현했는가?”를 별도로 판단하는 식입니다. 두 질문은 동시에 yes일 수 있습니다. 따라서 각 질문의 확률을 더해 1이 되어야 한다고 요구하지 않습니다. Choice는 보기 사이에서 하나를 선택하는 문제이고, 여러 Noul은 각각의 조건이 성립하는지 묻는 문제입니다.

<small>참조: [질문 타입 선택](https://docs.typesafe.ai/primitives)</small>

반대로 두 값이 모두 낮을 수도 있습니다. 모델은 두 조건에 모두 해당하지 않는 쪽으로 판단한 것입니다. 이 경우에 어느 처리로 보낼지 정해 두면, 억지로 한 담당을 고르는 대신 기타 문의 처리나 추가 확인으로 연결할 수 있습니다.

### Score: 말로 정의한 수준에 점수를 붙입니다

Score는 정도를 구분할 기준이 있을 때 씁니다. 먼저 낮은 수준부터 높은 수준까지 말로 설명합니다. 배열의 위치가 0, 1, 2가 되고, 점수는 각 위치에 확률을 곱해 더한 값입니다. 이 값을 기댓값, 즉 확률로 가중한 평균이라고 부릅니다.

<small>참조: [Score](https://docs.typesafe.ai/primitives/score)</small>

업무 영향도를 평가하는 가상 결과를 계산해 보겠습니다.

| 수준 | 기준 | 가상 확률 | 점수에 더하는 값 |
|---|---|---|---|
| 0 | 업무에 영향이 없습니다. | 0.20 | 0 × 0.20 = 0 |
| 1 | 불편하지만 대체 방법이 있습니다. | 0.60 | 1 × 0.60 = 0.60 |
| 2 | 대체 방법 없이 업무가 중단됩니다. | 0.20 | 2 × 0.20 = 0.40 |

마지막 열을 더하면 점수는 1.0입니다. 확률은 주로 수준 1에 모여 있지만 수준 0과 2에도 나뉘어 있습니다. 점수가 소수로 나오더라도 그 숫자는 설명한 수준 사이의 위치입니다. 1.5를 ‘업무가 75% 중단됐다’로 환산해서는 안 됩니다.

기준을 만들 때는 ‘조금’, ‘중간’, ‘많이’보다 각 수준을 구분할 상황을 적습니다. 불만 정도와 업무 영향도를 한 점수에 함께 넣으면, 화는 났지만 업무는 정상인 고객을 어디에 놓을지 모호해집니다. 두 판단이 필요하다면 별도 질문으로 나누는 편이 해석하기 쉽습니다.

<small>참조: [평가 수준을 작성하는 방법](https://docs.typesafe.ai/primitives/score)</small>

### 같은 평균에도 다른 분포가 들어갑니다

수준이 0, 1, 2인 경우를 생각해 보겠습니다. 확률이 `[0, 1, 0]`이면 점수는 1입니다. 확률이 `[0.5, 0, 0.5]`여도 점수는 1입니다. 전자는 중간 수준 하나에 확률이 모여 있고, 후자는 양 끝으로 나뉘어 있습니다. 평균만 저장하면 이 차이를 잃습니다.

[![수준 1에 확률이 모인 분포와 수준 0과 2에 절반씩 나뉜 분포는 모두 평균 점수가 1.0입니다.](/ai-tech-blog/images/jev-system-one-deep-dive/score-distributions.png)](/ai-tech-blog/images/jev-system-one-deep-dive/score-distributions.png)

*같은 평균이 서로 다른 분포에서 나오는 예시입니다. 실제 모델 출력은 아닙니다.*

<small>참조: [Score의 해석](https://docs.typesafe.ai/primitives/score)</small>

이 차이는 실제 처리에도 영향을 줍니다. 앞의 기준에서 후자는 ‘대체 방법 없이 업무가 중단되는 수준 2’에 확률을 0.5 두고 있습니다. 평균이 같다고 두 사례를 똑같이 취급하기보다, 심각한 결과에 얼마나 확률을 두었는지도 읽어야 합니다.

## 확률, confidence, 실제 정답률을 구분하기

### 확률분포는 답마다 가능성을 나눠 적습니다

확률 하나는 특정 답에 부여한 가능성입니다. 확률분포는 가능한 답들에 그 가능성이 어떻게 나뉘는지 보여 줍니다. 영상 예시의 결제, 배송, 기타 보기로 돌아가 보겠습니다. 설명용 값인 결제 0.80, 배송 0.15, 기타 0.05가 한 분포입니다. 아래 그림처럼 0.36, 0.34, 0.30으로 나뉘었다면 결제가 가장 높기는 해도 다른 보기와 차이가 작습니다.

두 경우 모두 `choice`만 읽으면 결제라는 동일한 값을 얻습니다. 분포까지 읽으면 ‘한 답에 판단이 모인 경우’와 ‘보기 사이에서 판단이 나뉜 경우’를 구분할 수 있습니다.

### confidence는 분포의 모양을 요약한 값입니다

Choice와 Score가 반환하는 `confidence`는 확률분포 모양에서 계산한 0부터 1 사이의 통계량입니다. 공식 문서에는 세 선택지용 데모 근사식이 있지만, 이를 실제 API가 쓰는 정확한 계산식으로 명시하지는 않습니다. 따라서 `confidence`를 최고 확률과 같은 값으로 가정하거나, `0.9`를 “이 답이 정답일 확률 90%”로 바꾸어 설명해서는 안 됩니다. 해당 업무에서 confidence 구간별 실제 정답률을 측정해야 합니다.

<small>참조: [Confidence](https://docs.typesafe.ai/confidence)</small>

[![Choice의 확률분포가 한 보기에 모인 경우와 여러 보기에 퍼진 경우를 비교합니다.](/ai-tech-blog/images/jev-system-one-deep-dive/probability-confidence.png)](/ai-tech-blog/images/jev-system-one-deep-dive/probability-confidence.png)

*같은 선택지에 대한 두 가지 확률분포를 비교한 예시입니다. 실제 측정값은 아닙니다. 그림을 누르면 크게 볼 수 있습니다.*

<small>참조: [Score](https://docs.typesafe.ai/primitives/score), [Confidence](https://docs.typesafe.ai/confidence)</small>

‘모델의 판단이 한쪽으로 모였다’는 것과 ‘현실에서 그 판단이 맞았다’는 것은 다른 정보입니다. 잘못된 자료나 잘못 정의한 기준을 사용하면 모델이 한 답에 확률을 많이 두고도 틀릴 수 있습니다. 그래서 confidence는 자동 처리할 사례를 고르는 데 쓸 수 있는 신호이며, 실제 정답과 대조하는 평가도 필요합니다.

### calibration은 예측값과 실제 결과를 대조합니다

<strong>확률 보정(calibration)</strong>은 모델이 말한 확률이 실제 발생 빈도와 얼마나 맞는지 보는 개념입니다. 예를 들어 서로 다른 문의 중 “취소를 명시적으로 요청했다”에 약 0.8의 확률을 준 사례 100건을 모았다고 하겠습니다. 사람이 확인한 결과 약 80건이 실제 취소 요청이라면, 그 구간에서 예측과 관측이 잘 맞습니다. 50건만 맞았다면 0.8이라는 예측은 실제 빈도보다 높았던 셈입니다. 이 숫자들은 개념 설명을 위한 가상 예입니다.

이 평가는 다수의 사례를 모아 확인합니다. 다음 문의 한 건이 반드시 맞는다는 보장은 얻을 수 없습니다. 또 전체 정답률이 높더라도, 0.8이라고 예측한 구간과 0.6이라고 예측한 구간이 실제 빈도에 맞는지는 별도로 봐야 합니다. Guo 등의 연구도 신경망의 정확도와 확률 보정을 구분해 평가하는 문제를 다뤘습니다.

<small>참조: [Guo 등의 ICML 2017 연구](https://proceedings.mlr.press/v70/guo17a.html)</small>

| 값 | 읽을 때의 질문 | 주의할 점 |
|---|---|---|
| `probabilities` 또는 `noul` | 어떤 답에 어느 정도의 가능성을 두었습니까? | 업무의 완료 비율이나 심각도를 뜻하지 않습니다. |
| `confidence` | Choice나 Score의 분포를 어떻게 요약했습니까? | 그 숫자를 그대로 실제 정답률로 읽지 않습니다. |
| 평가 데이터의 정답률 | 정답과 비교했을 때 실제로 몇 건 맞았습니까? | 평가한 데이터와 조건이 바뀌면 결과도 달라질 수 있습니다. |

TypeSafe는 확률 보정을 학습 목표로 내세웁니다. 그렇더라도 새로운 도메인, 한국어 입력, 새로운 모델 버전에서 같은 품질이 유지되는지는 따로 확인해야 합니다. 자동 처리할 범위를 정할 때는 “confidence가 높은가”와 “그 구간에서 실제로 얼마나 틀리는가”를 함께 봐야 합니다.

## 여러 질문을 한 번에 평가하는 방식

여러 질문에 대한 답을 계산하는 방식도 살펴보겠습니다. 먼저 토큰은 모델이 텍스트를 처리할 때 사용하는 단위입니다. 단어나 단어의 일부가 토큰이 될 수 있습니다. 일반적인 자기회귀 언어 모델은 앞서 생성한 토큰을 참고해 다음 토큰을 생성합니다. ‘자기회귀’라는 말이 낯설다면, 여기서는 앞서 쓴 내용에 이어 다음 내용을 만드는 방식으로 이해하면 됩니다.

구조화된 출력을 요청해도, 문자열을 생성하는 방식에서는 출력 길이만큼 순차적인 생성 단계가 이어집니다. TypeSafe는 Jev가 state를 한 번 받아 각 질문을 병렬로 평가한다고 설명합니다. 아래 그림은 이 처리 방식의 차이를 설명합니다. GPU 내부 구조나 실제 지연 시간의 비율을 나타낸 그림은 아닙니다.

[![언어 모델의 순차적인 토큰 생성과 Jev의 공유 state에 대한 여러 질문 평가를 비교합니다.](/ai-tech-blog/images/jev-system-one-deep-dive/flow-comparison.png)](/ai-tech-blog/images/jev-system-one-deep-dive/flow-comparison.png)

*토큰을 순서대로 생성하는 방식과 여러 질문을 함께 평가하는 방식을 비교했습니다.*

<small>참조: [System One](https://docs.typesafe.ai/concepts/system-one), [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)</small>

같은 문의에서 요청 유형, 취소 의사, 업무 영향도를 판단한다면 세 질문을 한 요청에 넣을 수 있습니다. state를 반복해서 보내는 비용도 줄어듭니다. 답이 필요한 분기가 아직 정해지지 않았더라도, 동일한 자료만 있으면 판단할 수 있는 질문은 미리 묶어 보낼 수 있습니다. 이를 문서는 speculative fan-out이라고 설명합니다.

<small>참조: [Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)</small>

예를 들어 취소 요청일 때만 사용할 추가 판단도 같은 자료로 답할 수 있다면 함께 물을 수 있습니다. 결과를 받은 코드가 취소 요청에 해당하지 않는다고 판단하면 그 추가 답을 사용하지 않으면 됩니다. 미리 수행하는 것은 질문의 평가이며, 실제 주문 취소는 선택된 처리 코드에서만 실행합니다.

단, <strong>같은 요청 안의 질문은 서로의 답을 읽지 않습니다.</strong> “첫 질문에서 고른 상품을 두 번째 질문에서 평가해 달라”는 의존 관계를 이름만으로 연결할 수 없습니다. 첫 답을 이용해 데이터베이스를 조회하거나 다음 선택지를 만들어야 한다면, 그 결과로 새 state를 구성하고 다음 요청을 보내야 합니다.

<small>참조: [질문 사이의 의존 관계](https://docs.typesafe.ai/primitives#when-one-question-depends-on-another)</small>

| 필요한 판단 | 한 요청에 묶을 수 있는지 |
|---|---|
| 같은 문의의 유형, 불만 정도, 명시적 취소 의사 | 같은 자료로 각각 답할 수 있으므로 함께 보낼 수 있습니다. |
| 선택한 주문을 조회한 뒤 그 주문에 적용할 판단 | 조회 결과를 얻은 다음 요청에서 판단합니다. |
| 한 에이전트가 작업한 뒤 그 결과를 검토하는 판단 | 작업 결과가 생긴 뒤 새 요청을 보냅니다. |

애플리케이션에서는 먼저 코드로 계산 가능한 항목을 처리하고, 같은 자료로 판단할 수 있는 질문을 묶습니다. 앞 단계 결과가 꼭 필요한 작업만 다음 단계로 남깁니다. 이 방식의 성능은 모델뿐 아니라 질문 분해와 데이터 준비 방식에도 영향을 받습니다.

## RLCD에서 공개된 내용과 아직 모르는 내용

앞의 병렬 평가는 답을 계산하는 방식에 관한 설명입니다. 모델이 그런 판단을 하도록 가르치는 학습 단계에서 TypeSafe가 내세우는 방식은 RLCD입니다. 애플리케이션이 API를 호출할 때는 이미 학습된 모델로 판단을 얻습니다.

강화학습은 모델이 낸 결과에 보상을 주어 원하는 결과를 내도록 학습하는 방식입니다. RLCD는 Reinforcement Learning for Calibrated Decisions의 약자로, TypeSafe는 판단값과 보정된 확률을 반환하도록 학습하는 방식으로 설명합니다.

비교 대상으로 나오는 RLHF는 Reinforcement Learning from Human Feedback의 약자로, 사람의 피드백을 활용합니다. RLVR은 Reinforcement Learning with Verifiable Rewards의 약자로, 검증 가능한 보상을 활용합니다. 이는 학습 목표의 차이를 설명하는 구분이며, 세 이름만으로 실제 모델의 학습 과정 전체를 알 수는 없습니다.

<small>참조: [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)</small>

여기서 학습 목표는 “모델이 어떤 결과를 내면 좋은 성적을 주는가”로 이해할 수 있습니다. 사람에게 설명하는 답변이라면 사람이 선호하는 답을 만드는 것이 목표가 될 수 있습니다. 정답을 계산으로 확인할 수 있는 문제라면 그 검사를 통과하는 결과를 보상할 수 있습니다. TypeSafe가 RLCD에서 강조하는 목표는 판단값과 함께 반환하는 확률을 실제 결과에 맞추는 것입니다.

예를 들어 두 모델이 모두 취소 요청이라는 답을 냈다고 하겠습니다. 한 모델이 모든 입력에 거의 1.0의 확률을 부여하고, 다른 모델이 정보가 모호할 때는 확률을 나누어 준다면, 프로그램이 사람에게 검토를 맡길 사례를 고르는 데 쓸 수 있는 정보가 다릅니다. 다만 두 번째 모델의 확률이 더 유용한지는 정답 데이터와 비교해 확인해야 합니다. 자신 없어 보이는 숫자를 반환한다는 사실만으로 잘 보정된 모델이 되지는 않습니다.

소프트웨어에서 틀린 판단의 비용은 행동마다 다릅니다. 낮은 확신으로 문의 분류를 추천하는 일과 결제를 확정하는 일에 같은 기준을 쓰기는 어렵습니다. 확률이 실제 결과와 잘 맞는다면, 프로그램은 예상 오류와 검토 비용을 비교해 자동화 범위를 정할 수 있습니다.

공개 자료에서 확인할 수 있는 범위는 출력 형식, 병렬 평가 방식, 학습 목표, 업체가 보고한 결과입니다. 이번에 확인한 자료에서는 파라미터 수, 상세 신경망 구성, 재현 가능한 학습 절차와 손실함수를 확인하지 못했습니다. ‘트랜스포머를 완전히 대체했다’거나 특정한 분류 헤드를 쓴다고 단정할 근거도 확보하지 못했습니다. <strong>이 글의 그림은 공개 인터페이스와 프로그램의 실행 흐름을 설명합니다.</strong>

JSON Schema는 JSON에 담을 값의 형식을 정의하는 규칙입니다. 이 규칙을 지키는지와 판단의 정확도, 확률 보정, 비용과 지연은 각각 별도로 평가해야 합니다. TypeSafe는 이러한 항목을 판단 업무에 맞춰 함께 최적화했다고 주장합니다.

## 멀티에이전트에서 판단 단계를 분리하는 활용

지금까지 살펴본 판단 기능을 여러 에이전트가 협업하는 프로그램에 적용해 보겠습니다. 여기서 에이전트는 AI 모델을 이용해 주어진 작업을 수행하고 필요한 도구를 사용하는 프로그램을 뜻합니다. 멀티에이전트는 여러 에이전트가 역할을 나누어 작업하는 구성입니다.

예를 들어 주문을 조회하는 에이전트, 결제 내역을 확인하는 에이전트, 고객에게 답장을 작성하는 에이전트를 둘 수 있습니다. 이들을 조정하는 프로그램은 어느 에이전트를 호출할지 정하고 결과를 모읍니다. 이런 조정 프로그램을 오케스트레이터라고 부릅니다.

Jev는 그 안에서 반복되는 분류와 평가에 사용할 수 있습니다. 공식 intent routing 문서도 요청을 정해진 처리 코드, 전문 LLM, 사람에게 나누어 보내는 패턴을 설명합니다. Jev가 담당자를 고르는 판단을 반환하면, 오케스트레이터가 실제 호출을 수행하는 구성입니다.

<small>참조: [Intent routing](https://docs.typesafe.ai/patterns/intent-routing)</small>

### 문의 분류에서 담당 에이전트 호출까지

“같은 주문이 두 번 결제됐고 배송도 늦어지고 있습니다”라는 가상의 문의를 처리한다고 하겠습니다. 아래는 이 글에서 제안하는 구성 예시이며, 실제 운영 사례나 측정 결과는 아닙니다.

[![Jev가 담당 필요 여부를 판단하면 코드가 전문 에이전트를 호출하고, 실제 작업 결과를 모아 새 입력으로 검토합니다.](/ai-tech-blog/images/jev-system-one-deep-dive/multiagent-flow.png)](/ai-tech-blog/images/jev-system-one-deep-dive/multiagent-flow.png)

*담당 선택과 작업 결과 검토를 서로 다른 시점에 수행하는 구성 예시입니다.*

<small>참조: [Intent routing](https://docs.typesafe.ai/patterns/intent-routing), [Agent Trace Observability](https://evals.typesafe.ai/agent_trace_observability)</small>

| 단계 | 담당 | 하는 일 |
|---|---|---|
| 1. 자료 준비 | 조정 코드 | 고객 문의와 필요한 계정 정보를 모아 state를 만듭니다. |
| 2. 필요한 담당 판단 | Jev | 결제 담당이 필요한지, 배송 담당이 필요한지 각각 판단합니다. |
| 3. 작업 호출 | 조정 코드 | 판단 결과와 권한을 확인해 해당 에이전트를 호출합니다. |
| 4. 조회와 처리 | 전문 에이전트 | 허용된 도구로 실제 결제와 배송 상태를 확인합니다. |
| 5. 결과 확인 | 코드와 필요시 Jev | 코드가 도구 실행 결과를 검사하고, Jev는 답변이 요청을 다뤘는지 같은 의미 판단을 맡습니다. |
| 6. 답변 또는 검토 | 조정 코드와 답변 담당 | 확인한 결과를 바탕으로 안내하거나 검토 담당자에게 넘깁니다. |

주담당 하나만 고르려면 Choice로 결제, 배송, 일반 상담 중 하나를 선택할 수 있습니다. 두 담당이 모두 필요할 수 있다면 각각의 필요 여부를 Noul로 묻는 편이 문제에 맞습니다. Score는 명시한 기준에 따라 문의의 긴급성 등을 평가하는 데 사용할 수 있습니다. 이렇게 얻은 값들을 어떤 호출과 연결할지는 코드가 정합니다.

<strong>담당 선택과 작업 결과 검토는 서로 다른 시점의 판단입니다.</strong> 단계 2에서는 전문 에이전트의 실행 결과가 아직 없습니다. 단계 5에서 그 결과를 검토하려면 실제 도구 응답과 작업 기록을 새로운 state에 넣어 다시 호출해야 합니다.

### 작업 검토에 필요한 근거

에이전트가 “취소를 완료했습니다”라고 썼다는 사실만으로 주문이 취소됐다고 확정할 수는 없습니다. 코드가 주문 시스템의 실제 상태와 도구 반환값을 확인해야 합니다. 반면 “최종 답변이 고객의 두 가지 요청을 모두 설명했는가?”처럼 의미를 읽어야 하는 판단은 Jev에 맡겨 볼 수 있습니다.

TypeSafe의 Agent Trace Observability 평가는 이런 사후 검토와 관련된 예를 제공합니다. 에이전트가 끝낸 작업을 보고 사람이 검토할 필요가 있는지 판단하는 워크플로입니다. 입력에는 지시사항과 대화, 도구에 전달한 인자와 실제 반환값, 최종 답변 등이 포함됩니다. 이는 업체 평가 사례이며, Jev가 실시간으로 모든 에이전트를 안전하게 감독한다는 보장으로 읽으면 안 됩니다.

<small>참조: [Agent Trace Observability](https://evals.typesafe.ai/agent_trace_observability)</small>

판단과 실행을 나누면 어느 단계에서 문제가 생겼는지 살펴보기 쉽습니다. 담당자를 잘못 골랐는지, 담당 에이전트가 조회를 잘못했는지, 최종 답변이 내용을 빠뜨렸는지 구분할 수 있습니다. 속도와 비용이 실제로 개선되는지는 호출 횟수, state 크기, 재시도와 추가 검토까지 포함해 측정해야 합니다.

여러 판단을 동시에 받더라도 오류가 서로 독립적이라는 뜻은 아닙니다. 같은 잘못된 자료에 영향을 받으면 여러 판단이 함께 틀릴 수 있습니다. confidence가 높거나 여러 에이전트가 동의한다는 이유만으로 권한 검사를 생략하거나 성공을 확정해서는 안 됩니다.

<small>참조: [질문 간 관계와 알려진 한계](https://docs.typesafe.ai/model-jaggedness/jev-1.13)</small>

## API 요청에서 프로그램의 분기까지

실제 API 요청 형식으로 돌아와, 취소 문의 하나를 처리하는 예제를 보겠습니다. 애플리케이션이 Jev에 자료와 질문을 보내고, 응답에서 필요한 값을 읽습니다.

요청의 최상위에는 다음 필드가 들어갑니다.

| 최상위 필드 | 이 예제에서의 의미 |
|---|---|
| `model` | 사용할 Jev 버전 |
| `state` | 고객 문장과 주문 정보 |
| `questions` | 이번 요청에서 평가할 질문들의 묶음 |

`questions` 안에는 `intent` 같은 이름으로 각 질문을 등록합니다. 각 질문 안에서 사용할 필드는 다음과 같습니다.

| 각 질문 안의 필드 | 이 예제에서의 의미 |
|---|---|
| `type` | `choice`, `noul`, `score` 중 어떤 형태의 답을 받을지 지정 |
| `instructions` | 각 질문에서 판단할 내용 |
| `criteria` | Choice의 선택지 설명 또는 Score의 수준 설명 |

Noul에도 yes와 no를 어떤 기준으로 판단할지 설명하는 선택 항목으로 `criteria`를 넣을 수 있습니다. 이 예제의 Noul은 `instructions`에 적은 질문만 사용합니다.

다음은 문서의 현재 요청 형식을 따라 작성한 예시입니다. `POST https://api.typesafe.ai/v1/systemone`에 보내는 JSON이며, 실제 추론은 실행하지 않았습니다. 한국어 설명과 대응하기 쉽도록 입력도 한국어로 작성했습니다. 한국어 품질은 별도 평가가 필요합니다. 결과를 비교할 수 있도록 모델 버전을 고정했습니다. `jev-latest` 같은 별칭은 새 버전이 나오면 가리키는 모델이 바뀔 수 있습니다.

<small>참조: [모델 버전과 별칭](https://docs.typesafe.ai/models)</small>

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

`intent` 같은 질문 ID는 결과를 찾아 읽는 키입니다. 모델에는 ID가 전달되지 않으므로 질문의 의미를 `instructions`에 적어야 합니다. `criteria`의 설명도 질문과 같은 기준을 사용해야 합니다.

<small>참조: [Primitives](https://docs.typesafe.ai/primitives), [HTTP API](https://docs.typesafe.ai/api)</small>

응답에는 같은 질문 ID 아래에 결과가 들어갑니다. 다음 표는 응답 중 네 값만 추려 뜻을 보여 주는 가상 예입니다. 실제 응답에는 사용 모델을 나타내는 `model`, 토큰 사용량인 `usage`, 각 답의 `type`, Score의 수준 설명인 `legend` 같은 필드도 있습니다. 점으로 이어진 이름은 응답 데이터 안에서 그 값을 찾는 위치를 나타냅니다.

<small>참조: [응답 구조](https://docs.typesafe.ai/api)</small>

| 읽는 위치 | 가상 값 | 프로그램에서 해석할 내용 |
|---|---|---|
| `answers.intent.choice` | `cancel` | 선택한 문의 유형이 취소입니다. |
| `answers.intent.probabilities` | 취소 0.90, 배송 0.07, 기타 0.03 | 취소 쪽에 확률을 가장 많이 두었습니다. |
| `answers.explicit_cancel.noul` | 0.92 | 취소 의사를 명시했다는 답에 부여한 확률입니다. |
| `answers.frustration.score` | 0.30 | 예를 들어 수준별 확률이 0.75, 0.20, 0.05라면 기대 위치는 0.30입니다. |

실제 Choice와 Score 응답에는 confidence도 있습니다. 위 표에서는 임의로 만든 숫자를 실제 API 공식으로 계산한 값처럼 보이지 않도록 confidence 수치를 넣지 않았습니다. 또한 취소라는 Choice 결과와 취소 의사를 묻는 Noul은 서로 다른 질문이므로 두 확률이 정확히 같아야 하는 것은 아닙니다.

응답을 받은 코드는 유형과 필요한 필드를 검증하고, 주문 시스템에서 취소 가능 여부를 다시 확인합니다. 고객의 의도가 취소라고 판단되면 취소 확인 화면을 보여줄 수 있습니다. confidence가 낮거나 문의 내용이 모호하면 사람에게 넘깁니다. <strong>모델의 확신이 높아도 사용자 권한이나 주문 정책을 대신할 수는 없습니다.</strong>

위 예제의 불만 점수는 상담 우선순위에 활용할 수 있고, 취소 허용 여부는 주문 시스템의 규칙으로 계산합니다.

## 속도와 비용 수치를 읽는 기준

2026년 9월 22일 공식 모델 페이지에 표시된 버전은 `jev-1.13.0`입니다. 입력 100만 토큰당 가격은 0.042달러이고 출력 토큰은 무료입니다. 요청 전체는 64k 토큰, state와 가장 긴 질문의 합은 32k 토큰이라는 두 제한이 함께 적용됩니다. 영어가 주된 학습 언어이며 다른 언어의 품질은 직접 확인하도록 안내합니다.

<small>참조: [Models](https://docs.typesafe.ai/models)</small>

가격을 단순 계산하면 입력 1,000토큰은 0.000042달러입니다. 그런 요청 100만 건은 입력 요금만 42달러입니다. 질문과 선택지 설명도 입력에 포함되고, 전처리, 다른 모델 호출, 재시도, 사람의 검토 비용은 이 계산에 들어 있지 않습니다. 이는 공개 단가를 이용한 산술 예시입니다.

홈페이지의 ‘193.6배 빠름, 444.6배 저렴함’은 업체의 워크플로 평가에서 나온 수치입니다. 출시 글은 이 배수가 실제 개선 폭의 높은 쪽일 수 있다고 밝힙니다. 평가를 주로 미국 서부에서 실행했고, 비교 LLM에는 확률까지 생성하도록 하는 래퍼를 적용했습니다. 따라서 모든 업무, 모든 리전, 모든 출력 방식에서 동일한 배수가 나온다고 해석하기 어렵습니다.

<small>참조: [출시 발표의 평가 조건](https://typesafe.ai/blog/introducing-system-one-models-and-jev)</small>

[![Jev와 비교 모델의 비용, 지연, 참조 답안 일치율을 함께 비교한 업체 자체 평가입니다.](/ai-tech-blog/images/jev-system-one-deep-dive/benchmark.png)](/ai-tech-blog/images/jev-system-one-deep-dive/benchmark.png)

*공식 평가의 다섯 모델을 비교한 차트입니다. 각 수치는 4개 워크플로를 같은 비중으로 평균한 값입니다. 비용과 시간은 케이스당 값입니다. 원문의 비용과 시간 축은 로그이며, 여기서는 0을 기준으로 한 선형 축으로 그렸습니다. 모델명과 반올림된 값은 원문 표기를 따릅니다.*

<small>참조: [Workflow evals](https://evals.typesafe.ai/). 2026년 9월 22일에 확인했습니다.</small>

이 평가의 참조 답안은 GPT-6 Astra와 Claude Fable 5.1을 높은 추론 설정으로 실행한 결과를 평균해 만들었습니다. 다른 모델은 제공자의 기본 추론 설정을 사용합니다. 그래프의 ‘accuracy’는 이 참조 답안을 기준으로 한 값이므로, 사람이 검증한 실제 업무 정답률과 구분해야 합니다. 보안 사고, 에이전트 실행 기록, 청구서, 고객 서비스의 네 워크플로가 같은 비중으로 반영됩니다.

<small>참조: [평가 방법](https://evals.typesafe.ai/)</small>

TypeSafe는 이 참조 답안 구성에 OpenAI와 Anthropic 모델에 유리한 편향이 있다고 밝히며, 자사와 DeepSeek 모델의 상대 성능을 과소평가할 수 있다고 덧붙입니다. 따라서 그래프에서 가까운 수치 몇 개만으로 모델의 일반적인 지능 순위를 정할 수는 없습니다.

<small>참조: [참조 답안의 편향에 관한 설명](https://typesafe.ai/blog/introducing-system-one-models-and-jev)</small>

발췌한 그래프에서는 Jev의 지연과 비용이 작지만, 참조 답안 일치율이 가장 높지는 않습니다. 실제 도입에서는 같은 업무 결과를 얻는 데 필요한 전체 비용을 비교해야 합니다. 판단이 어려운 사례를 더 큰 모델이나 사람에게 넘겼을 때의 비용까지 포함하면 비교 결과가 달라질 수 있습니다.

## 타입 보장과 판단 오류를 구분해야 합니다

TypeSafe는 출시 글에서 ‘환각 없음’을 강조하면서, 해당 0% 수치가 스키마 일치 보장에 근거한다고 설명합니다. 허용된 선택지가 `cancel`, `delivery`, `other`라면 목록 밖의 답을 생성하지 않는다는 성질입니다. `delivery`가 정답인 상황에서 `cancel`을 고르는 오류는 별도로 남습니다.

<small>참조: [타입 보장에 관한 설명](https://typesafe.ai/blog/introducing-system-one-models-and-jev)</small>

공식 ‘Jev 1.13 jaggedness’ 문서는 실제 적용에서 주의할 실패 형태를 공개합니다. 아래 항목을 평가에 포함하면 모델이 어려워하는 조건을 확인할 수 있습니다.

<small>참조: [알려진 한계](https://docs.typesafe.ai/model-jaggedness/jev-1.13)</small>

| 입력이나 작업 | 문서에 나온 한계 | 프로그램에서 처리할 부분 |
|---|---|---|
| 숫자 계산, 개수 세기, 날짜 비교 | 정확한 수치 처리에 약합니다. | 파서와 계산 코드로 처리합니다. |
| 무관한 자료가 많은 긴 state | 관련 없는 내용이 정확도를 떨어뜨릴 수 있습니다. | 검색과 필터링으로 필요한 자료를 고릅니다. |
| 입력에 숨은 지시, 오도하는 문장 | state를 기본적으로 적대적 입력으로 취급하지 않습니다. | 공격 사례를 평가하고 실행 권한을 별도로 제한합니다. |
| 같은 의미를 다르게 물은 질문 | 질문 사이의 논리적 항등식을 보장하지 않습니다. | 중복 질문을 줄이고 필요한 관계를 코드로 강제합니다. |

마지막 항목은 확률을 조합할 때 중요합니다. “취소를 요청했는가?”와 “취소가 아닌 다른 것을 요청했는가?”를 별도 질문으로 보내고 두 응답의 합이 반드시 1이라고 가정하면 안 됩니다. 상호 배타적인 분류가 필요하다면 하나의 Choice로 정의하는 방법을 검토할 수 있습니다. 모델 출력이 프로그램에서 어떤 관계를 만족해야 하는지 명시해야 합니다.

## 작게 검증할 때 측정할 것

첫 실험으로는 업무 하나를 고르는 편이 좋습니다. 예를 들어 고객 문의를 취소, 배송, 기타로 분류하되 실제 주문은 변경하지 않고 기존 담당자의 판단과 비교합니다. 다음 절차는 이 글에서 제안하는 평가 방법입니다.

임계값은 자동 처리 여부를 나눌 기준값입니다. confidence 등에 이 기준을 적용해 자동으로 진행할 사례와 검토할 사례를 구분합니다.

1. 사람이 검토한 정답 데이터에 모호한 문의, 복합 요청, 한국어 표현, 입력에 숨은 지시를 포함합니다.
2. 질문 문구와 임계값을 조정할 데이터와 최종 평가 데이터를 분리합니다.
3. 정답률과 함께 확률 구간별 실제 빈도, 자동 처리한 비율, 자동 처리한 사례 중 오답 비율을 측정합니다.
4. 모델 호출뿐 아니라 검색, 재시도, 대체 모델, 검토 대기까지 포함한 비용과 지연을 기록합니다.
5. 모델 버전과 질문, 처리 규칙을 고정하고 변경 때마다 같은 평가를 다시 실행합니다.

자동 처리 비율을 올리면 더 어려운 사례까지 받아들일 수 있습니다. 반대로 확신이 낮은 사례를 많이 보류하면 자동화로 줄이는 업무량이 작아집니다. 이 관계를 보면서 원하는 오류 수준에서 얼마나 많은 요청을 처리할 수 있는지 정해야 합니다. confidence 임계값 하나를 인터넷 예제에서 가져오는 것으로는 이 결정을 대신할 수 없습니다.

자동 처리 기준을 조정할 때 정답률과 자동화율을 따로 보는 이유를 가상 결과로 살펴보겠습니다. 같은 문의 100건을 두 가지 기준으로 처리했다고 가정합니다.

| 가상 처리 기준 | 자동 처리 | 사람 검토 | 자동 처리한 사례 중 오답 |
|---|---|---|---|
| 넓게 자동 처리하는 기준 | 60건 | 40건 | 6건, 자동 처리 60건의 10% |
| 엄격하게 선별하는 기준 | 20건 | 80건 | 1건, 자동 처리 20건의 5% |

첫 번째 설정의 자동화율은 60%, 두 번째는 20%입니다. 두 번째가 자동 처리한 사례의 오류 비율은 낮지만, 사람이 처리할 건수는 늘었습니다. 이는 설명용 결과이며 실제 Jev 성능을 뜻하지 않습니다. 기준을 엄격하게 하면 원하는 오류 수준에 도달하는지도 직접 평가해야 합니다. 업무에서 허용할 오류와 검토 가능한 물량을 함께 정하면 어떤 설정을 채택할지 판단할 수 있습니다.

Jev를 적용할 부분은 프로그램이 반복해서 수행하는 좁은 의미 판단에서 찾을 수 있습니다. 질문의 답이 어떤 행동에 쓰이는지부터 적고, 그 행동에 필요한 데이터와 검증 기준을 정하면 실험 범위가 명확해집니다. 실제 지불, 삭제, 권한 변경을 시작하기 전에는 기존 처리와 나란히 실행해 결과 차이를 확인할 수 있습니다.

## References

- [TypeSafe AI, Introducing System One Models & Jev, 2026-09-15](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [TypeSafe AI, System One](https://docs.typesafe.ai/concepts/system-one)
- [TypeSafe AI, Jev with coding agents](https://docs.typesafe.ai/introduction/coding-agents)
- [TypeSafe AI, State](https://docs.typesafe.ai/concepts/state)
- [TypeSafe AI, Primitives](https://docs.typesafe.ai/primitives)
- [TypeSafe AI, Choice](https://docs.typesafe.ai/primitives/choice)
- [TypeSafe AI, Noul](https://docs.typesafe.ai/primitives/noul)
- [TypeSafe AI, Score](https://docs.typesafe.ai/primitives/score)
- [TypeSafe AI, Confidence](https://docs.typesafe.ai/confidence)
- [TypeSafe AI, AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer)
- [TypeSafe AI, Speculative fan-out](https://docs.typesafe.ai/patterns/fan-out)
- [TypeSafe AI, How to build with TypeSafe](https://docs.typesafe.ai/concepts/how-to-build-with-system-one)
- [TypeSafe AI, Intent routing](https://docs.typesafe.ai/patterns/intent-routing)
- [TypeSafe AI, API reference](https://docs.typesafe.ai/api)
- [TypeSafe AI, Models](https://docs.typesafe.ai/models)
- [TypeSafe AI, Workflow evals](https://evals.typesafe.ai/)
- [TypeSafe AI, Agent Trace Observability](https://evals.typesafe.ai/agent_trace_observability)
- [TypeSafe AI, Jev 1.13 jaggedness, 2026-09-17 검토본](https://docs.typesafe.ai/model-jaggedness/jev-1.13)
- [Guo, Pleiss, Sun, Weinberger, On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)
