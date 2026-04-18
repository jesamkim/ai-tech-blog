---
title: "Stanford AI Index 2026 심층 해부: 숫자로 읽는 2026년 AI 지형"
date: 2026-04-19T10:00:00+09:00
draft: false
cover:
  image: "/ai-tech-blog/images/2026-04-19-stanford-ai-index-2026-deep-dive/cover.png"
  alt: "Stanford AI Index 2026"
  relative: false
categories: ["논문 리뷰"]
tags: ["AI Index", "Stanford HAI", "AI 산업 동향", "벤치마크", "AI 정책"]
author: "Jesam Kim"
---

Stanford HAI가 2017년부터 매년 발간하는 [AI Index Report](https://hai.stanford.edu/ai-index/2026-ai-index-report)는 AI 분야의 현황을 수치로 고정시키는 몇 안 되는 기준점이다. 기술 성능, 투자 흐름, 연구 출판, 일자리 변화, 대중 인식을 한 곳에서 다루는 보고서는 거의 없다. 올해로 아홉 번째를 맞은 2026년판은 4월 13일 공개되었다.

숫자부터 결론을 말하자면, 이번 리포트는 편안하지 않다.

2026년 리포트가 던지는 핵심 메시지는 세 방향으로 정리된다. <strong>첫째, 성능의 폭발</strong>이다. 지난 1년 사이 AI 에이전트가 소프트웨어 엔지니어링, 사이버보안, 수학 올림피아드 문제를 다루는 성공률이 10퍼센트대에서 90퍼센트대로 뛰었다. 벤치마크가 포화되는 속도가 너무 빠르다 보니 측정 도구 자체가 따라가지 못하는 상황이 되었다. <strong>둘째, 미중 격차 소멸</strong>이다. 2025년 2월 DeepSeek-R1이 미국 최상위 모델과 일시적으로 동률을 이뤘고, 2026년 3월 기준 Anthropic의 최상위 모델이 앞서는 폭은 단 [2.7%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)다. <strong>셋째, 투명성과 신뢰의 붕괴</strong>다. Foundation Model Transparency Index 평균 점수가 58점에서 40점으로 떨어졌고, 미국에서 AI 규제를 신뢰한다는 응답은 31%로 조사 대상국 중 최하를 기록했다.

이 글은 [Stanford HAI의 12 Takeaways 블로그](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)를 한국 엔터프라이즈 관점에서 재구성한다. 원 리포트 전문은 [AI Index 2026 공식 페이지](https://hai.stanford.edu/ai-index/2026-ai-index-report)에서 내려받을 수 있다.

---

## 성능의 폭발: 측정 한계에 다다른 모델들

숫자부터 보자.

[AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report)에 따르면 <strong>SWE-bench Verified</strong> 점수는 1년 사이 60%에서 거의 100% 수준으로 올랐다. 실제 GitHub 이슈를 자율적으로 해결하는 능력을 측정하는 이 벤치마크는, 2024년까지만 해도 "60%면 이미 상당하다"는 평가였다. 지금은 포화 직전이다.

에이전트가 실제 터미널 환경에서 복잡한 태스크를 수행하는 능력을 측정하는 <strong>Terminal-Bench</strong>는 더 드라마틱하다. [HAI 블로그](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)는 2025년 초 20%였던 성공률이 현재 77.3%로 올랐다고 보고한다. <strong>사이버보안 에이전트</strong>의 경우 2024년 15%였던 문제 해결률이 2026년 93%로 뛰었다. 2년 만에 여섯 배다.

수학 쪽도 마찬가지다. [The Decoder의 분석](https://the-decoder.com/stanfords-ai-index-2026-shows-rapid-progress-growing-safety-concerns-and-declining-public-trust/)에 따르면 Gemini Deep Think가 국제수학올림피아드(IMO)에서 <strong>금메달</strong>에 해당하는 성적을 기록했다. AI Index 2026은 AI가 박사급 과학 문제에서 인간 베이스라인을 넘어섰다는 점도 언급한다.

![AI 에이전트 벤치마크 성능 도약 (2024/25 → 2026)](/ai-tech-blog/images/2026-04-19-stanford-ai-index-2026-deep-dive/benchmarks.png)

*AI 에이전트 주요 벤치마크 점수 변화. 출처: [Stanford HAI, AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report)*

이 수치들을 그대로 받아들이기 전에 한 가지 맥락이 필요하다. 리포트가 직접 언급하는 <strong>"들쭉날쭉한 프런티어(Jagged Frontier)"</strong>다. 올림피아드 금메달을 딴 모델이 아날로그 시계를 읽는 정확도는 [50.1%](https://the-decoder.com/stanfords-ai-index-2026-shows-rapid-progress-growing-safety-concerns-and-declining-public-trust/)에 불과하다. 로봇이 가사 태스크를 수행하는 성공률은 [12%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)다. 특정 영역에서는 인간을 넘어섰지만, 아무 어린이나 해내는 일을 아직 못 한다.

이 비대칭이 실무에서 중요한 이유는 명확하다. 에이전트에게 코드 디버깅이나 취약점 분석을 맡기는 건 현실적인 선택이 되었다. 반면 로봇 자동화나 물리적 환경 인식이 필요한 작업은 여전히 인간 감독이 필수다. 성능 곡선이 고른 게 아니라 "어떤 일을 맡길 수 있는가"를 더 면밀히 따져야 하는 구조라는 뜻이다.

벤치마크 포화 문제도 짚어야 한다. SWE-bench가 100%에 가까워지면, 더 어려운 벤치마크가 등장한다. 이미 Terminal-Bench나 각종 PhD-level 테스트들이 그 역할을 맡기 시작했다. "AI가 벤치마크를 깼다"는 뉴스가 나올 때마다 실제 업무 적용 가능성을 구분해서 읽는 습관이 필요한 시점이다. 특히 사이버보안 에이전트가 93% 성공률을 기록했다는 수치는, 방어 측면에서도 같은 도구를 쓸 수 있다는 신호로 읽어야 한다.

---

## 미중 구도의 재편: 2.7%의 좁은 격차

2025년 2월, DeepSeek-R1이 공개된 직후 주요 벤치마크에서 미국 최상위 모델과 [일시적으로 동률](https://siliconangle.com/2026/04/13/stanford-hais-2026-ai-index-reveals-china-u-s-now-neck-neck-race-global-dominance/)을 이뤘다. 이 사건은 AI 업계의 기존 전제를 바꿨다. "중국이 따라잡는 것은 시간문제"라는 분위기를 "이미 따라잡았다"는 분위기로.

2026년 3월 기준 Anthropic의 최상위 모델이 중국을 앞서는 격차는 [단 2.7%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)다. [SiliconANGLE](https://siliconangle.com/2026/04/13/stanford-hais-2026-ai-index-reveals-china-u-s-now-neck-neck-race-global-dominance/)은 이를 "neck-and-neck"으로 표현했다.

그렇다고 전선이 단순하지는 않다. 미국이 앞서는 영역과 중국이 앞서는 영역이 명확히 다르다. 리포트는 미국이 최상위 AI 모델 수와 고영향 특허에서 우위를 유지한다고 본다. 반면 중국은 논문 발표 볼륨, 피인용 수, 특허 출원 건수, 산업용 로봇 설치 규모에서 앞선다.

![미국 vs 중국: AI 경쟁 구도 (Stanford AI Index 2026)](/ai-tech-blog/images/2026-04-19-stanford-ai-index-2026-deep-dive/us-vs-china.png)

*미국 vs 중국 민간 AI 투자 비교 및 영역별 우위. 출처: [Stanford HAI, AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report)*

투자 격차는 여전히 크다. 2025년 민간 AI 투자 기준 미국은 [$285.9B](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)로, 다음으로 높은 중국($12.4B)의 23.1배다. 그러나 중국 정부 가이던스 펀드의 2000년부터 2023년까지 누적 추정치는 [$912B](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)에 달한다. 민간과 정부의 투자 구조 자체가 다르다.

더 주목해야 할 수치는 인재 흐름이다. 미국으로 유입되는 AI 연구자 수가 [2017년 대비 -89%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)로 줄었고, 지난 1년만으로 좁히면 [감소폭이 -80%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)다. 이 추세가 가속되고 있다는 점이 리포트의 표현이다.

기술 격차와 투자 격차를 동시에 보면 미묘한 그림이 나온다. 미국이 모델 품질 최선단에서는 여전히 앞서지만 그 폭이 급격히 좁아졌고, 인재 흐름은 반전되고 있다. 중국은 학술 규모와 정부 주도 산업 자동화에서 미국보다 앞서 나간다. "누가 이기고 있냐"는 질문에 단답형 답이 나오지 않는 구조가 된 것이다.

---

## 환경 비용: 측정 가능해진 AI의 대가

AI의 에너지 비용이 이번처럼 구체적인 수치로 기록된 적은 없었다.

[HAI 블로그](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)에 따르면 Grok 4 한 모델의 학습 과정에서 발생한 탄소 배출량 추정치는 <strong>72,816톤 CO2 equivalent</strong>다. 원문 표현을 그대로 옮기면 "차 17,000대를 1년간 운행하는 것과 같다". 모델 한 개 학습의 탄소 발자국이 중소 도시 교통 배출량 수준이다.

전력 소비도 압도적이다. AI 데이터센터의 전력 용량은 현재 [29.6 GW](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)인데, 리포트는 이를 "뉴욕주 피크 전력 수요 전체에 맞먹는 수준"으로 표현한다.

물 문제도 기록되었다. [GPT-4o의 연간 추론 물 사용량](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)은 1,200만 명의 식수 수요를 초과할 수 있다. 데이터센터 냉각수 소비가 서비스 규모에 따라 국가 단위 자원 수요로 올라선다는 뜻이다. 누적 AI 시스템 전력 수요는 스위스 또는 오스트리아의 [국가 전력 소비와 비슷한 수준](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)으로 추산된다.

이 수치들이 중요한 이유는 두 가지다.

첫째, 규제 압력의 방향이다. 탄소 공시 의무화가 확산되는 흐름에서, AI 서비스 제공자와 이를 도입하는 기업 모두 간접 배출량 계산에 AI 학습 및 추론 비용을 포함해야 하는 시점이 가까워지고 있다. 실제로 EU 탄소 경계 조정 메커니즘(CBAM) 같은 정책들은 공급망 내 에너지 집약 공정을 직접 겨냥한다.

둘째, 인프라 설계 결정이다. 기업이 AI를 도입할 때 퍼블릭 클라우드 API 호출과 온프레미스 추론 사이에서 선택을 해야 한다면, 에너지 비용은 이미 경제적 변수다. 칩 효율성 향상이 소비 증가를 상쇄하지 못하고 있는 현재 추세에서, 이 논쟁은 앞으로 더 자주 등장할 것이다.

---

## 투명성의 역설: 강력할수록 닫힌다

[Foundation Model Transparency Index(FMTI)](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)는 주요 AI 기업들이 자사 모델에 대해 얼마나 공개하는지를 측정하는 지표다. 훈련 데이터, 모델 구조, 평가 방식, 사용 정책 등 100여 개 항목을 점수화한다.

[AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report)에 따르면 FMTI 평균 점수는 이전 58점에서 <strong>40점</strong>으로 내려갔다. 리포트가 직접 지적하는 패턴은 명확하다. "가장 유능한 모델이 가장 적게 공개한다."

이 역설은 왜 일어날까. 경쟁이 치열해질수록 핵심 기술을 공개하는 인센티브가 줄어든다. 오픈소스 모델은 FMTI에서 상대적으로 높은 점수를 받는 경향이 있지만, 상업적 최전선 모델들은 점점 더 블랙박스화된다. 보안, 저작권, 경쟁 우위라는 세 압력이 동시에 투명성에 반하는 방향으로 작용한다.

이 흐름이 기업 입장에서 의미하는 바는 구체적이다. 모델이 어떤 데이터로 훈련되었는지, 어떤 안전 필터가 적용되었는지, 실패 케이스가 어떻게 기록되는지를 모르는 채로 비즈니스 프로세스를 위탁하는 일이 늘어난다. 감사 가능성(auditability)이 낮아지는 환경에서, 그 공백을 채우는 외부 거버넌스 수요는 커질 수밖에 없다. FMTI 점수 하락은 규제 기관과 기업 감사팀 양쪽 모두에게 경보 신호다.

[The Decoder](https://the-decoder.com/stanfords-ai-index-2026-shows-rapid-progress-growing-safety-concerns-and-declining-public-trust/)는 이를 "declining public trust"의 맥락에서 다룬다. 투명성이 낮아질수록 검증 비용이 외부로 전가된다는 구조적 문제가 있다.

---

## 사회적 파장: 일자리, 대중 인식, 교육, 의료

### 일자리: 재분배의 시작

[AI Index 2026](https://hai.stanford.edu/ai-index/2026-ai-index-report)은 소프트웨어 개발 직군에서 뚜렷한 연령대별 분화를 포착했다. 22-25세 소프트웨어 개발자 고용이 2024년 이후 약 [-20%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report) 줄어든 반면, 시니어와 중견 개발자의 헤드카운트는 늘었다. AI 코드 생성 도구가 주니어의 역할 일부를 흡수한 결과로 읽을 수 있다.

생산성 수치는 다른 결을 보여준다. [The Decoder](https://the-decoder.com/stanfords-ai-index-2026-shows-rapid-progress-growing-safety-concerns-and-declining-public-trust/)가 정리한 리포트 내 데이터에 따르면 고객지원과 소프트웨어 개발 직무에서 생산성이 14-26% 향상되었고, 마케팅 분야에서는 최대 72%까지 올랐다는 연구들이 인용된다. 그러나 기업의 <strong>에이전트 도입률</strong>은 거의 모든 부서에서 한 자릿수에 머문다. 생산성 증거는 쌓이는데, 조직 차원의 배포는 아직 초기 단계다.

### 채택 속도: 개인과 기업의 간극

[GenAI 인구 채택률](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)은 출시 3년 만에 53%에 달했다. PC와 인터넷보다 빠른 속도다. 나라별로는 싱가포르 [61%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report), UAE [54%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report), 미국은 24위로 [28.3%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)다.

미국 소비자가 GenAI에서 얻는 연간 가치는 2026년 초 기준 [$172B](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)로 추산되며, 사용자 1인당 중위 가치는 1년 사이 [3배](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)가 되었다. 개인이 체감하는 가치는 빠르게 커지고 있다.

### 대중 인식: 낙관과 불안이 동시에

AI에 대한 글로벌 낙관론은 [59%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)로 이전 52% 대비 7%p 올랐다. 불안감은 [52%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)로 2%p 증가했다. 두 수치가 동시에 오를 수 있는 건, 같은 기술에 기대와 걱정이 공존하기 때문이다.

전문가와 일반 대중의 온도 차이는 더 극적이다. [The Decoder](https://the-decoder.com/stanfords-ai-index-2026-shows-rapid-progress-growing-safety-concerns-and-declining-public-trust/)에 따르면 미국 전문가의 73%가 낙관적인 반면, 일반 대중의 낙관 비율은 23%로 <strong>50%p 격차</strong>가 있다. 기술을 만드는 사람과 그 결과를 살아가는 사람 사이의 인식 차이가 이 정도라면, 소통 방식 자체를 다시 생각해야 하는 수준이다.

AI 규제에 대한 신뢰도에서는 미국이 [31%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)로 조사 대상국 중 최하다. EU가 미국과 중국보다 높은 규제 신뢰도를 보인다. 규제 속도와 방식이 신뢰와 반드시 정비례하지 않는다는 점, 그리고 빠른 배포가 신뢰를 깎아먹을 수 있다는 점을 시사한다.

### 교육: 사용은 넘쳤고, 정책은 부족하다

미국 고등학생과 대학생의 5명 중 4명이 학업에 AI를 사용한다. 중고교의 [50%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)가 AI 정책을 보유하고 있지만, 그 정책이 명확하다고 답한 교사는 [6%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)에 불과하다. 정책이 있다는 것과 그 정책이 현장에서 작동한다는 것은 전혀 다른 문제다.

### 의료: 기대와 근거의 간극

임상 노트 자동 생성 도구를 사용하는 의사들은 노트 작성 시간이 최대 [-83%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report) 줄었다고 보고한다. 번아웃 감소를 경험한다는 보고도 있다.

그런데 리포트는 다른 숫자도 함께 제시한다. 500편 이상의 임상 AI 연구를 검토한 결과, 거의 절반이 시험지 스타일의 문제에 의존했으며 실제 임상 데이터를 사용한 연구는 [5%](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)에 불과했다. 효과를 주장하는 논문은 많지만, 실제 환자 데이터로 검증된 것은 극히 드물다는 뜻이다.

한편 디지털 트윈(digital twins) 분야에서는 의미 있는 변화가 있었다. 데이터 트윈 관련 출판이 2015년 거의 없던 것에서 2025년 [372편](https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report)으로 늘었다. 이 분야에서 AI와 의료 데이터의 결합을 탐구하는 연구자들이 증가하고 있다.

---

## 시사점: 한국 엔터프라이즈에 주는 메시지

리포트를 읽는 실무자라면 "그래서 우리 조직은?"이라는 질문으로 자연스럽게 이어진다. 세 가지를 짚어본다.

### 1. 채택의 비대칭: 개인 53% vs 기업 에이전트 한 자릿수

GenAI 인구 채택률이 53%인데 기업의 에이전트 도입률이 거의 모든 부서에서 한 자릿수라는 사실은 무엇을 의미할까. 개인이 본인 업무에서 AI를 쓰는 것과, 조직이 AI 에이전트를 프로세스에 공식 통합하는 것 사이에 커다란 간극이 있다. 이 간극은 제거해야 할 지연(delay)이 아니라, 거버넌스 설계를 완료하기 전까지 유지해야 할 완충 구간으로 읽는 게 맞을 수 있다.

그러나 경쟁 맥락에서 보면 위험이기도 하다. 같은 기술을 개인 수준에서 쓰는 조직과 프로세스 수준에서 통합한 조직은 1-2년 안에 실행력 격차가 벌어진다. "우리도 관심은 있다"는 답이 "우리는 이미 운영 중"이 되는 데 걸리는 시간이 짧아지고 있다. 파일럿이 아닌 프로덕션 배포를 고민해야 하는 시점이 다가오고 있다는 것, 이 리포트는 그 타임라인을 좁혀서 보여준다.

### 2. 투명성 하락이 만드는 거버넌스 수요

FMTI 점수가 58점에서 40점으로 떨어졌다. 이건 단순히 "AI 기업들이 정보를 덜 공개한다"는 이야기가 아니다. 도입 기업 입장에서는 "우리가 사용하는 모델의 한계와 위험을 스스로 검증해야 한다"는 의미이기도 하다.

이 맥락에서 관리형 AI 플랫폼의 역할이 재조명된다. 훈련 데이터 계보, 안전 필터 구성, 모델 버전 관리, 사용 로그를 체계적으로 관리하는 플랫폼은, 투명성이 낮아지는 환경에서 내부 감사와 규제 대응을 가능하게 하는 인프라가 된다. 거버넌스가 선택 사항에서 필수 요건으로 전환되는 속도가 빨라지고 있다.

이 흐름은 ESG 공시, 개인정보 보호, AI 안전 규제가 맞물리는 지점에서 더 강해질 것이다. AI 적용 프로젝트를 추진할 때 처음부터 감사 가능성을 설계에 포함시키는 조직과 그렇지 않은 조직의 차이가 규제 대응 비용으로 나타날 것이다.

### 3. 인재 이동 정체: 내재화에 투자해야 한다

미국으로의 AI 연구자 유입이 2017년 대비 -89%, 지난 1년만으로는 -80% 줄었다. 이 수치는 미국 중심의 AI 생태계에 구멍이 뚫리고 있음을 보여주는 동시에, 다른 국가들에게는 기회이기도 하다.

그러나 한국 기업 입장에서 더 직접적인 교훈은 다른 데 있다. 외부 AI 전문가를 영입하는 방식은 공급 자체가 조여드는 시장에서 한계가 있다. 결국 내부 인재를 AI 실무자로 키우는 역량 내재화 전략이 더 현실적이다. 이미 도메인 지식을 가진 사람에게 AI 도구 활용 능력을 더하는 것이, 외부 AI 전문가가 도메인을 새로 배우는 것보다 빠른 경우가 많다.

교육 데이터도 이 방향을 지지한다. UAE, 칠레, 남아프리카가 AI 엔지니어링 스킬 습득 속도 상위를 차지한다는 사실은, 지리적 거점보다 학습 구조가 중요하다는 것을 보여준다.

---

## 마치며

Stanford AI Index 2026의 숫자들은 확실히 드라마틱하다. SWE-bench 거의 100%, 사이버보안 에이전트 93%, GenAI 채택 53%. 하지만 같은 리포트에 아날로그 시계 50.1%, 로봇 가사 12%, 임상 데이터 기반 연구 5%도 있다.

실무자에게 중요한 건 평균값이나 최대값이 아니라, 어느 지표가 자신의 조직 의사결정을 바꾸는가다. AI Index는 방향을 알려주는 지도이지, 무엇을 만들지 알려주는 설계도가 아니다.

이 리포트를 읽고 나서 해볼 만한 질문은 하나다: 지금 우리 조직에서 파일럿 단계로 묶여 있는 AI 프로젝트가 있다면, 그것이 파일럿으로 남아 있는 이유가 기술의 한계인지 거버넌스의 부재인지를 구분해 보는 것. 대부분의 경우 답은 후자다.

---

## References

1. Shana Lynch, "Inside the AI Index: 12 Takeaways from the 2026 Report," Stanford HAI, April 13, 2026. https://hai.stanford.edu/news/inside-the-ai-index-12-takeaways-from-the-2026-report
2. Stanford HAI, "The 2026 AI Index Report." https://hai.stanford.edu/ai-index/2026-ai-index-report
3. Maximilian Schreiner, "Stanford's AI Index 2026 shows rapid progress, growing safety concerns, and declining public trust," The Decoder, April 14, 2026. https://the-decoder.com/stanfords-ai-index-2026-shows-rapid-progress-growing-safety-concerns-and-declining-public-trust/
4. IEEE Spectrum, "Stanford's AI Index for 2026 Shows the State of AI." https://spectrum.ieee.org/state-of-ai-index-2026
5. SiliconANGLE, "China has erased the US lead in AI, Stanford HAI's 2026 AI index reveals," April 13, 2026. https://siliconangle.com/2026/04/13/stanford-hais-2026-ai-index-reveals-china-u-s-now-neck-neck-race-global-dominance/
