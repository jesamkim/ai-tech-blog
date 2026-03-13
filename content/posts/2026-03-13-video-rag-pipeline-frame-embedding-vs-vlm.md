---
title: "프레임 임베딩 vs 비디오 언어 모델: 비디오 RAG 파이프라인, 어떻게 만들어야 할까"
date: 2026-03-13T10:00:00+09:00
author: "Jesam Kim"
categories: ["AI/ML 기술 심층분석"]
tags: ["Video RAG", "Twelve Labs", "Amazon Bedrock", "Pegasus", "Marengo", "Video Understanding", "Embedding"]
description: "비디오 RAG를 구축할 때 프레임 임베딩과 비디오 언어 모델 기반 파이프라인의 성능 차이를 실험으로 비교합니다. Twelve Labs Pegasus v1.2와 Marengo Embed 3.0을 Amazon Bedrock에서 테스트한 결과, VLM 기반 파이프라인이 5~10배 높은 유사도 스코어를 보여줍니다."
cover:
  image: "images/cover-video-rag-pipeline.png"
---

## 1. 왜 비디오 RAG인가: 텍스트 RAG를 넘어서

텍스트 RAG(Retrieval-Augmented Generation)는 이미 성숙한 기술입니다. 문서를 청크로 나누고, 임베딩하고, 벡터 데이터베이스에 저장한 뒤, 쿼리와 유사한 청크를 검색해 LLM의 응답을 보강하는 패턴이 확립되어 있습니다.

하지만 기업 데이터의 상당 부분은 텍스트가 아닌 <strong>비디오</strong>입니다. CCTV 녹화, 회의 녹화, 교육 콘텐츠, 마케팅 영상 등 비디오 형태로 축적된 정보는 방대합니다. Cisco의 예측에 따르면 [2025년 기준 인터넷 트래픽의 82%가 비디오](https://www.cisco.com/c/en/us/solutions/collateral/executive-perspectives/annual-internet-report/white-paper-c11-741490.html)가 될 것이라고 합니다.

이러한 배경에서 [VideoRAG 논문](https://arxiv.org/abs/2501.05874)(Jeong et al., 2025, ACL Findings)이 비디오 RAG 프레임워크를 제안했습니다. 기존 접근법은 비디오를 텍스트로 변환할 때 멀티모달 정보가 손실되거나, 쿼리 기반 검색 없이 사전에 정의된 비디오만 사용하는 한계가 있었습니다.

비디오 RAG는 이러한 문제를 해결하고, 텍스트처럼 <strong>검색 가능한 비디오 지식 베이스</strong>를 구축하는 것을 목표로 합니다.

## 2. 실전 유즈케이스: 비디오 RAG가 풀 수 있는 문제들

비디오 RAG가 실제로 어떤 문제를 해결할 수 있을까요?

### 산업 현장 안전 관리 (건설/제조)

건설 현장이나 제조 공장에서는 CCTV 아카이브가 방대하게 쌓입니다. 과거 사고 사례를 검색하거나 안전 교육 자료를 추출하는 데 비디오 RAG를 활용할 수 있습니다.

예를 들어 "지난달 크레인 근접 작업 위반 영상 찾아줘"라는 쿼리에 대해, 과거 몇 달치 CCTV 아카이브에서 관련 장면을 검색하고 타임스탬프를 제공할 수 있습니다.

<strong>주의:</strong> 실시간 안전 감지(안전모 미착용 감지 등)는 스트리밍 추론을 사용하며, RAG와는 별개의 아키텍처입니다. 비디오 RAG는 <strong>사후 검색</strong>에 적합합니다.

### 미디어/방송 아카이브

방송사는 수만 시간의 방송 아카이브를 보유하고 있습니다. 특정 장면, 인물, 발언을 검색할 때 비디오 RAG가 유용합니다.

텍스트 자막 검색만으로는 부족한 경우가 많습니다. "붉은 드레스를 입은 진행자가 스튜디오에서 인터뷰하는 장면"처럼 비주얼 요소와 텍스트를 함께 검색하는 멀티모달 검색의 시너지가 필요합니다.

### 교육/이러닝

온라인 강의 플랫폼은 수천 개의 강의 영상을 보유하고 있습니다. 학습자가 특정 개념을 검색하면, 해당 개념을 설명하는 강의 구간을 정확히 찾아주는 것이 가능합니다.

"미분의 기하학적 의미"를 검색하면, 관련 강의 영상과 타임스탬프를 반환해 학습자 맞춤 콘텐츠 큐레이션을 제공할 수 있습니다.

## 3. 비디오를 벡터로 만드는 두 가지 접근법

비디오 RAG를 구축하려면, 비디오를 벡터로 변환해야 합니다. 두 가지 접근법이 있습니다.

![프레임 임베딩 vs VLM 파이프라인 비교 다이어그램](/ai-tech-blog/images/video-rag-two-approaches.png)

### 접근법 A: 프레임 임베딩 (Frame-level Embedding)

가장 단순한 방법은 비디오에서 N개의 프레임을 추출하고, 각 프레임을 이미지 임베딩 모델로 변환한 뒤, 벡터들의 <strong>평균</strong>을 구하는 것입니다.

<strong>장점:</strong>
- 구현이 단순합니다
- 기존 이미지 임베딩 인프라를 재사용할 수 있습니다

<strong>단점:</strong>
- <strong>시간적 맥락이 소실</strong>됩니다. "사람이 문을 열고 들어온다"와 "사람이 문을 닫고 나간다"는 프레임 평균으로는 구분하기 어렵습니다
- 오디오와 대화를 무시합니다
- 프레임 간 의미가 희석됩니다. 서로 다른 장면의 프레임들을 평균내면, 의미적으로 중심점이 애매해집니다

<strong>이론적 분석:</strong> 벡터 평균은 기하학적 중심점을 계산합니다. 예를 들어 "열기구" 프레임(v<sub>1</sub>)과 "사람 대화" 프레임(v<sub>2</sub>)의 평균 (v<sub>1</sub> + v<sub>2</sub>) / 2는 두 의미의 중간 어딘가를 가리키지만, 비디오의 실제 의미를 대표하지 못할 수 있습니다.

### 접근법 B: VLM(Video Language Model) 기반 파이프라인

비디오 전체를 Video Language Model에 입력하고, 텍스트 설명을 생성한 뒤, 그 텍스트를 임베딩하는 방식입니다.

<strong>장점:</strong>
- <strong>시간적 맥락과 관계를 포착</strong>합니다. "처음에는 A가 일어나고, 그 다음 B가 일어난다"는 인과 관계를 이해합니다
- 오디오와 대화를 반영할 수 있습니다
- 이벤트와 행동을 자연어로 표현합니다

<strong>단점:</strong>
- 추가 추론 비용이 발생합니다
- 파이프라인이 복잡해집니다

실험을 통해 두 접근법의 성능 차이를 비교해 보겠습니다.

## 4. Twelve Labs 모델: Pegasus v1.2와 Marengo Embed 3.0

### Twelve Labs 소개

[Twelve Labs](https://www.twelvelabs.io/)는 비디오 이해 전문 AI 스타트업입니다. [AWS SageMaker HyperPod](https://aws.amazon.com/blogs/aws/twelvelabs-video-understanding-models-are-now-available-in-amazon-bedrock/)에서 모델을 학습하며, 2025년 1월부터 Amazon Bedrock에서 사용할 수 있습니다.

### Pegasus v1.2 (비디오 언어 모델)

[Pegasus v1.2](https://www.twelvelabs.io/blog/introducing-pegasus-1-2)는 Twelve Labs의 최신 비디오 언어 모델입니다.

<strong>아키텍처:</strong> video encoder + video-language alignment + language decoder로 구성되어 있으며, [Pegasus-1 Technical Report](https://arxiv.org/abs/2404.14687)에 따르면 약 80B 파라미터를 가지고 있습니다.

<strong>성능:</strong>
- 최대 1시간 비디오 지원
- 15분까지 일정한 TTFT(Time To First Token) 레이턴시
- [VideoMME-Long 벤치마크](https://www.twelvelabs.io/blog/introducing-pegasus-1-2)에서 GPT-4o, Gemini 1.5 Pro 대비 SOTA 성능 달성

<strong>특징:</strong>
- 타임스탬프 기반 시간 이해가 탁월합니다. 예를 들어 미식축구 경기에서 "3쿼터 7분 20초에 무슨 일이 일어났나요?"라는 질문에 정확히 답변합니다
- 한국어 응답을 자연스럽게 생성합니다

<strong>Bedrock 설정:</strong>
- Inference Profile: `us.twelvelabs.pegasus-1-2-v1:0`
- 리전: US West (Oregon), Europe (Ireland) (cross-region inference)
- [파라미터 문서](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html)

### Marengo Embed 3.0 (멀티모달 임베딩)

[Marengo Embed 3.0](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html)은 512차원 벡터를 출력하는 멀티모달 임베딩 모델입니다.

<strong>입력:</strong> 텍스트 + 이미지

<strong>주의:</strong> Bedrock에서 Marengo Embed 3.0은 <strong>비디오 직접 임베딩을 지원하지 않습니다</strong>. 텍스트와 이미지만 입력으로 받습니다. 비디오를 직접 임베딩하려면 Marengo 2.7 모델과 StartAsyncInvoke API를 사용해야 합니다.

<strong>Bedrock 설정:</strong>
- Inference Profile: `us.twelvelabs.marengo-embed-3-0-v1:0`
- 리전: US East (N. Virginia)

## 5. 실전 실험: Bedrock에서 직접 비교해 보니

실험 코드는 [GitHub 레포지토리](https://github.com/jesamkim/bedrock-twelvelabs)에서 확인할 수 있습니다.

### 실험 환경

3개의 샘플 비디오를 준비했습니다. 모두 Pexels에서 CC0 라이선스로 제공되는 영상입니다.

- <strong>nature.mp4:</strong> 카파도키아 열기구 (19.3초)
- <strong>city.mp4:</strong> 강변 대화 (37.9초)
- <strong>cooking.mp4:</strong> 실험실 피펫 작업 (14.0초)

재미있는 점은 <strong>cooking.mp4</strong>는 파일명과 실제 내용이 일치하지 않는다는 것입니다. 실제로는 실험실에서 피펫을 사용하는 장면이지만, 파일명은 cooking입니다. Pegasus가 이것을 정확히 파악하는지 확인할 수 있습니다.

### 실험 1: Pegasus 비디오 QA

먼저 Pegasus v1.2가 비디오를 얼마나 잘 이해하는지 테스트했습니다.

3개 비디오 × 4개 프롬프트 = 총 12건의 질의를 수행했고, <strong>모두 성공</strong>했습니다(100% 성공률).

<strong>주요 발견:</strong>
- 한국어 응답이 자연스럽습니다. 다만 일부 hallucination이 존재합니다(예: "페가수스 공원" 같은 존재하지 않는 장소 언급)
- 타임스탬프 추출이 정확합니다. [MM:SS-MM:SS] 형식으로 구간별 설명을 제공합니다
- <strong>cooking.mp4의 실제 내용(실험실 피펫 작업)을 정확히 파악</strong>했습니다. 파일명에 속지 않고 비디오 내용을 정확히 분석했습니다

### 실험 2: 임베딩 비교

두 가지 방식으로 비디오 임베딩을 생성하고, 쿼리와의 유사도를 비교했습니다.

<strong>Frame-avg:</strong>
1. 비디오에서 4개 프레임 추출
2. 각 프레임을 Marengo Embed 3.0으로 이미지 임베딩
3. 4개 벡터의 평균 계산

<strong>Pegasus+Marengo:</strong>
1. Pegasus v1.2로 비디오 전체를 텍스트 설명으로 변환
2. 텍스트를 Marengo Embed 3.0으로 임베딩

### 결과 데이터

3개의 쿼리에 대해 각 비디오와의 cosine similarity를 측정했습니다.

| 쿼리 | 방식 | nature | city | cooking |
|------|------|--------|------|---------|
| "hot air balloons in cappadocia" | Frame-avg | 0.043 | -0.051 | -0.058 |
|  | Pegasus+Marengo | <strong>0.421</strong> | 0.176 | 0.117 |
| "people having conversation outdoors" | Frame-avg | 0.006 | 0.069 | -0.028 |
|  | Pegasus+Marengo | 0.290 | <strong>0.616</strong> | 0.158 |
| "science experiment in lab" | Frame-avg | -0.032 | -0.043 | 0.068 |
|  | Pegasus+Marengo | 0.222 | 0.254 | <strong>0.562</strong> |

![파이프라인 유사도 비교 차트](/ai-tech-blog/images/video-rag-pipeline-comparison.png)

<strong>핵심 발견:</strong>

1. <strong>두 방식 모두 정확한 1위 매칭</strong>을 했습니다(3/3). 즉, "열기구" 쿼리는 nature 비디오를, "대화" 쿼리는 city 비디오를, "실험" 쿼리는 cooking 비디오를 각각 최고 점수로 검색했습니다.

2. 하지만 <strong>마진이 극적으로 다릅니다</strong>. Frame-avg의 최고 스코어는 0.069인 반면, Pegasus+Marengo는 0.616으로 <strong>약 9배 높습니다</strong>.

3. Frame-avg는 음수 스코어가 많습니다. 이는 프레임 평균 벡터가 쿼리와 거의 직교하거나 반대 방향을 가리킨다는 의미입니다.

### 프레임 간 유사도 문제

Frame-avg 방식의 근본적인 문제는 <strong>프레임끼리 너무 비슷하다</strong>는 것입니다.

![프레임 간 유사도 히트맵](/ai-tech-blog/images/video-rag-similarity-matrix.png)

3개 비디오에서 추출한 12개 프레임의 cosine similarity를 계산하면, 대부분 0.74~0.79 범위에 분포합니다. 즉, <strong>서로 다른 비디오의 프레임끼리도 유사도가 높습니다</strong>.

이는 이미지 임베딩이 저수준 시각적 특징(색상, 질감, 구도)에 민감하기 때문입니다. "열기구"와 "대화"는 의미적으로 전혀 다르지만, 프레임 레벨에서는 "야외 풍경"이라는 공통점 때문에 비슷하게 보입니다.

결국 프레임 평균으로는 <strong>의미적 차이를 구분하기 어렵습니다</strong>.

## 6. 비디오 RAG 파이프라인 아키텍처: 권장 패턴

실험 결과를 바탕으로, 권장하는 비디오 RAG 아키텍처는 다음과 같습니다.

![비디오 RAG 아키텍처 다이어그램](/ai-tech-blog/images/video-rag-architecture.png)

### Phase 1: 인덱싱

1. <strong>비디오 저장:</strong> S3에 비디오를 업로드합니다
2. <strong>텍스트 설명 생성:</strong> Pegasus v1.2로 비디오 전체를 텍스트로 요약합니다
3. <strong>임베딩 생성:</strong> Marengo Embed 3.0으로 텍스트를 512차원 벡터로 변환합니다
4. <strong>저장:</strong> 벡터와 메타데이터(video_key, description, timestamps)를 Vector DB에 저장합니다

### Phase 2: 검색

1. <strong>쿼리 임베딩:</strong> 사용자 쿼리를 Marengo Embed 3.0으로 임베딩합니다
2. <strong>벡터 검색:</strong> Vector DB에서 cosine similarity 기반으로 Top-K 비디오를 검색합니다
3. <strong>메타데이터 반환:</strong> 검색된 비디오의 메타데이터(S3 key, 설명)를 가져옵니다

### Phase 3: 생성

1. <strong>상세 QA:</strong> 검색된 비디오를 Pegasus v1.2에 다시 입력하고, 사용자 쿼리에 대한 상세 답변을 생성합니다
2. <strong>타임스탬프 제공:</strong> 답변과 함께 관련 장면의 타임스탬프를 제공합니다

### AWS 서비스 구성

- <strong>S3:</strong> 비디오 저장
- <strong>Amazon Bedrock:</strong> Pegasus v1.2 + Marengo Embed 3.0
- <strong>Vector DB:</strong> Amazon OpenSearch Service 또는 Pinecone

[AWS 케이스 스터디](https://aws.amazon.com/bedrock/twelvelabs/)에서는 TwelveLabs와 S3 Vectors 통합이 소개되어 있습니다. S3 Vectors를 사용하면 별도의 Vector DB 없이 S3에서 직접 벡터 검색이 가능합니다.

### 스케일링 고려사항

- <strong>비동기 인덱싱:</strong> 수천 개의 비디오를 인덱싱할 때는 비동기 워크플로우(Step Functions 등)를 사용합니다
- <strong>캐시 전략:</strong> 자주 검색되는 비디오는 Pegasus 응답을 캐싱해 비용을 절감합니다
- <strong>메타데이터 필터링:</strong> 날짜, 카테고리 등 메타데이터로 사전 필터링하면 검색 정확도가 향상됩니다

## 7. 비용, 한계, 그리고 전망

### 비용 분석

10초 비디오를 Pegasus로 요약하면:
- 입력 비용: $0.00049 × 10초 = $0.0049
- 출력 비용: 상세 설명 생성 시(약 1,000토큰) $0.0075 × 1 = $0.0075
- <strong>총 약 $0.02 수준</strong>

30초 평균 비디오 1,000개를 인덱싱해도 약 $60 수준으로, 비용 부담이 크지 않습니다.

### 한계점

1. <strong>Marengo Embed 3.0의 비디오 직접 임베딩 미지원:</strong> Bedrock에서는 텍스트와 이미지만 임베딩할 수 있습니다. 비디오를 직접 임베딩하려면 Marengo 2.7 모델과 비동기 API를 사용해야 합니다. 향후 Embed 3.0도 지원 확대가 예상됩니다.

2. <strong>Hallucination:</strong> Pegasus가 한국어 요약 시 일부 환각을 생성합니다. 예를 들어 "페가수스 공원" 같은 존재하지 않는 장소를 언급하는 경우가 있습니다.

3. <strong>리전 제한:</strong> Pegasus는 US/EU만, Marengo 3.0은 us-east-1만 지원합니다. cross-region inference를 사용하면 latency가 증가할 수 있습니다.

### 전망

비디오 RAG는 아직 초기 단계이지만, 텍스트 RAG가 걸어온 길을 빠르게 따라올 것입니다.

- <strong>VAST Data + TwelveLabs 파트너십:</strong> [VAST Data와 TwelveLabs가 파트너십을 체결](https://www.vastdata.com/press-releases/vast-data-and-twelvelabs-partner-to-expand-video-intelligence)해 on-premise 환경에서도 비디오 인텔리전스를 구축할 수 있게 되었습니다.

- <strong>비디오 검색의 민주화:</strong> 몇 년 내에 비디오가 텍스트만큼 검색 가능해질 것입니다. "지난주 회의에서 누가 예산 삭감을 언급했지?"라는 질문에, 회의 녹화 아카이브에서 정확한 장면과 타임스탬프를 찾아주는 시대가 옵니다.

- <strong>멀티모달 이해의 진화:</strong> 텍스트 + 비주얼 + 오디오를 통합 이해하는 모델이 계속 발전하면, "말투가 불안해 보이는 발표자"처럼 감정과 맥락까지 검색할 수 있게 될 것입니다.

## 결론

비디오 RAG를 구축할 때, <strong>프레임 임베딩 평균 방식</strong>은 구현이 간단하지만 시간적 맥락을 잃고 의미적 구분이 약합니다. 반면 <strong>VLM 기반 파이프라인</strong>(Pegasus + Marengo)은 5~10배 높은 유사도 스코어를 달성하며, 비디오의 의미를 훨씬 잘 포착합니다.

Amazon Bedrock에서 Twelve Labs 모델을 사용하면, 복잡한 인프라 구축 없이도 프로덕션 수준의 비디오 RAG를 빠르게 구축할 수 있습니다.

지금이 비디오 지식 베이스를 구축하기 좋은 시점입니다. 텍스트 RAG처럼, 비디오 RAG도 곧 표준이 될 것입니다.

---

## References

- Twelve Labs Pegasus 1.2 블로그: https://www.twelvelabs.io/blog/introducing-pegasus-1-2
- Pegasus-1 Technical Report: https://arxiv.org/abs/2404.14687 (Jung et al., 2024)
- AWS Blog - TwelveLabs on Bedrock: https://aws.amazon.com/blogs/aws/twelvelabs-video-understanding-models-are-now-available-in-amazon-bedrock/
- VideoRAG: https://arxiv.org/abs/2501.05874 (Jeong et al., 2025, ACL Findings)
- TwelveLabs on Bedrock: https://aws.amazon.com/bedrock/twelvelabs/
- Bedrock Pegasus Documentation: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-pegasus.html
- Bedrock Marengo Documentation: https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-marengo.html
- Twelve Labs Pegasus-1 블로그: https://www.twelvelabs.io/blog/introducing-pegasus-1
- VAST Data + TwelveLabs Partnership: https://www.vastdata.com/press-releases/vast-data-and-twelvelabs-partner-to-expand-video-intelligence
- 실험 코드 GitHub 레포: https://github.com/jesamkim/bedrock-twelvelabs
- Cisco Annual Internet Report: https://www.cisco.com/c/en/us/solutions/collateral/executive-perspectives/annual-internet-report/white-paper-c11-741490.html
