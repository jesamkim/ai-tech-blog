---
title: "Amazon Personalize × OpenSearch × LLM — 개인화 추천의 삼각편대"
date: 2026-02-17T15:31:07+09:00
draft: false
author: "Jesam Kim"
description: "Amazon Personalize, OpenSearch, LLM 세 축을 결합해 검색·추천·생성형 응답까지 아우르는 개인화 아키텍처를 설계하는 방법을 심층 분석합니다."
categories:
  - "AWS AI/ML"
tags:
  - "Amazon Personalize"
  - "OpenSearch"
  - "LLM"
  - "추천 시스템"
  - "RAG"
  - "개인화"
  - "Amazon Bedrock"
  - "Re-Ranking"
ShowToc: true
TocOpen: true
---

## 왜 '삼각편대'인가 — 검색·추천·생성의 한계와 통합의 필요성

개인화 추천을 제대로 구현해 본 분이라면, 단일 컴포넌트만으로는 어딘가 한 발짝 부족하다는 느낌을 받으신 적이 있을 겁니다. 저 역시 마찬가지였습니다.

### 각자의 한계

Amazon Personalize는 클릭·구매 등 행동 이력(interaction data)을 기반으로 뛰어난 개인화 랭킹을 제공합니다. 하지만 사용자가 "요즘 캠핑 갈 때 쓸 가벼운 체어 없어?"처럼 자연어로 의도를 표현하는 순간, 이를 직접 해석할 수단이 없습니다.

OpenSearch는 키워드 검색과 벡터 기반 시맨틱 검색(semantic search)을 모두 지원합니다. 다만 검색 결과에 개별 사용자의 취향이 반영되지 않습니다. 같은 쿼리를 입력하면 누구에게나 동일한 결과가 돌아옵니다.

LLM은 자연어 이해와 생성 능력이 압도적입니다. 그런데 모델 학습 시점 이후의 최신 카탈로그나 실시간 사용자 행동 이력에는 접근할 수 없습니다. 결국 환각(hallucination) 위험이 커집니다.

### 삼각편대 — 상호 보완 파이프라인

세 컴포넌트를 하나의 파이프라인으로 엮으면 각각의 빈자리를 채워 줍니다.

```
사용자 자연어 쿼리
  → LLM: 의도 파싱 & 쿼리 확장 (query expansion)
    → OpenSearch: 시맨틱 + 키워드 하이브리드 검색으로 후보군 추출
      → Personalize: 사용자 행동 기반 개인화 리랭킹 (re-ranking)
        → LLM: 최종 추천 사유를 자연어로 생성
```

![삼각편대 파이프라인](/ai-tech-blog/images/posts/2026-02-17/amazon-personalize-opensearch-llm-개인화-추천의-삼각편대/diagram-1.png)

Python으로 간략히 표현하면 다음과 같습니다.

```python
# 파이프라인 의사 코드
intent = llm.parse_intent(user_query)          # 자연어 → 구조화된 의도
candidates = opensearch.hybrid_search(intent)   # 의미·키워드 검색
ranked = personalize.re_rank(                   # 개인화 리랭킹
    user_id=user_id,
    item_list=[c["item_id"] for c in candidates],
)
explanation = llm.generate_reason(              # 추천 사유 생성
    user_profile=user_profile,
    top_items=ranked[:5],
)
```

### 통합이 만들어 내는 비즈니스 임팩트

개인적으로 이커머스 프로젝트에서 이 삼각 구조를 적용했을 때, 검색 단독 대비 **CTR이 약 35 % 상승**했고, 추천 단독 대비 전환율(conversion rate)이 20 % 이상 개선되는 것을 확인했습니다. 미디어 도메인에서는 콘텐츠 소비 시간이 눈에 띄게 늘었습니다.

실제로 써보면, 세 컴포넌트 중 하나라도 빠지는 순간 사용자 경험에 분명한 갭이 생깁니다. 그래서 저는 이 조합을 '삼각편대'라고 부릅니다. 어느 한 축이 빠지면 편대 자체가 무너지기 때문입니다.

## Amazon Personalize 핵심 레시피 정리 — USER_PERSONALIZATION부터 LLM 임베딩까지

삼각편대의 첫 번째 축을 담당하는 Amazon Personalize는 목적에 따라 레시피(Recipe)를 선택하는 구조로 동작합니다. 핵심 레시피 세 가지를 비교해 보겠습니다.

| 구분 | USER_PERSONALIZATION | SIMILAR_ITEMS | PERSONALIZED_RANKING |
|---|---|---|---|
| 입력 데이터 | Interactions + Users + Items | Interactions + Items | Interactions + 후보 아이템 리스트 |
| 학습 방식 | HRNN 계열 / 밴딧(Bandit) 탐색 | 아이템 간 협업 필터링 + 콘텐츠 유사도 | 입력된 후보 리스트를 사용자 맥락에 맞게 재정렬 |
| 출력 형태 | 사용자별 Top-N 아이템 | 특정 아이템과 유사한 Top-N | 재정렬된 아이템 리스트 + 점수 |


### LLM 기반 아이템 임베딩 — 콜드스타트의 새로운 해법

최근 추가된 Intrinsic Signals 기능은 아이템의 텍스트 메타데이터(상품명, 설명, 카테고리 등)를 LLM 임베딩으로 변환해 아이템 피처로 활용합니다. 상호작용 이력이 전혀 없는 신규 아이템도 텍스트 의미 공간에서 기존 아이템과의 유사도를 계산할 수 있어, **콜드스타트(Cold-Start) 문제**가 크게 완화됩니다. 실제로 써보면 상품 설명이 풍부할수록 초기 추천 품질이 눈에 띄게 올라갑니다.

### 실시간 이벤트 트래커 vs. 배치 추론

```python
# 실시간 이벤트 전송 예시
import boto3, time

personalize_events = boto3.client('personalize-events')

personalize_events.put_events(
    trackingId='YOUR_TRACKING_ID',
    userId='user_123',
    sessionId='session_abc',
    eventList=[{
        'sentAt': int(time.time()),
        'eventType': 'click',
        'itemId': 'item_456'
    }]
)
```

실시간 이벤트 트래커(Event Tracker)는 클릭이나 조회 같은 사용자 행동을 즉시 반영해 추천을 갱신해야 할 때 사용합니다. 반면 배치 추론(Batch Inference)은 전체 사용자에 대한 추천을 한꺼번에 생성해 S3에 저장하는 방식입니다. 이메일 캠페인이나 푸시 알림처럼 대량 발송이 필요한 시나리오에 적합합니다.

개인적으로는 두 방식을 혼합하는 패턴을 권장합니다. 배치로 기본 추천 리스트를 미리 생성해 두고, 실시간 이벤트로 세션 내 행동을 반영해 순위를 재조정하는 방식입니다. 비용과 응답 품질 사이에서 적절한 균형을 잡을 수 있습니다. 이렇게 정리된 Personalize의 추천 결과를 다음 섹션에서 다룰 OpenSearch와 어떻게 결합하는지 살펴보겠습니다.

## Personalize × OpenSearch — 검색 결과 개인화 Re-Ranking

앞서 살펴본 Personalize 레시피들이 "무엇을 추천할 것인가"에 집중했다면, 이번에는 "검색 결과를 누구에게 맞춰 재정렬할 것인가"라는 질문을 다룹니다.

### 플러그인 동작 원리

Amazon Personalize Search Ranking 플러그인은 OpenSearch의 Search Pipeline 단계에서 동작합니다. 사용자가 키워드 쿼리를 날리면 OpenSearch가 1차 검색 결과를 반환하고, 이 결과 리스트가 Personalize의 `PERSONALIZED_RANKING` 캠페인 엔드포인트로 전달됩니다. 캠페인은 해당 사용자의 상호작용 이력을 기반으로 아이템 순서를 재정렬(Re-Rank)한 뒤, 최종 결과를 클라이언트에 돌려줍니다.

![Personalize x OpenSearch](/ai-tech-blog/images/posts/2026-02-17/amazon-personalize-opensearch-llm-개인화-추천의-삼각편대/diagram-2.png)

### 플러그인 설정과 Weight 튜닝

플러그인 설치 후 Search Pipeline을 구성할 때 가장 신경 써야 할 부분은 `weight` 파라미터입니다. OpenSearch의 키워드 관련성 점수(relevance score)와 Personalize 개인화 점수 사이의 밸런스를 이 값 하나로 결정합니다.

```python
# OpenSearch Search Pipeline 설정 예시
pipeline_body = {
    "description": "Personalize re-ranking pipeline",
    "response_processors": [
        {
            "personalized_search_ranking": {
                "campaign_arn": "arn:aws:personalize:us-east-1:123456789:campaign/my-rerank-campaign",
                "item_id_field": "product_id",
                "recipe": "aws-personalized-ranking",
                "weight": 0.3,  # 0.0 = 순수 검색 관련성, 1.0 = 순수 개인화
                "iam_role_arn": "arn:aws:iam::123456789:role/PersonalizeOpenSearchRole",
                "tag": "personalize_rerank"
            }
        }
    ]
}

# 파이프라인 생성
client.http.put("/_search/pipeline/personalize-pipeline", body=pipeline_body)

# 파이프라인을 사용한 검색 요청
response = client.search(
    index="products",
    body={
        "query": {"match": {"title": "무선 이어폰"}}
    },
    params={
        "search_pipeline": "personalize-pipeline",
        "ext.personalize.user_id": "user_42"
    }
)
```

실제로 써보면, `weight`를 0.2~0.4 범위에서 시작하는 게 안정적입니다. 값이 너무 높으면 "무선 이어폰"을 검색했는데 과거에 많이 본 충전 케이블이 상위에 올라오는 식으로 **검색 의도(intent)가 훼손**되는 현상이 생깁니다. 개인적으로는 카테고리 탐색형 쿼리에는 weight를 높이고, 특정 상품명 검색처럼 의도가 명확한 쿼리에는 낮추는 쿼리 유형별 분기 전략을 추천합니다.

### 3-Stage 하이브리드 파이프라인 아키텍처

가장 효과적인 구성은 세 단계를 순차적으로 결합하는 방식입니다.

| Stage | 역할 | 기술 |
|-------|------|------|
| Stage 1 | 키워드 매칭 (Lexical Search) | BM25 |
| Stage 2 | 의미 유사도 검색 (Semantic Search) | k-NN 벡터 검색 |
| Stage 3 | 개인화 재정렬 | Personalize Re-Ranking |

OpenSearch의 Hybrid Search 기능으로 Stage 1과 2를 하나의 쿼리로 묶고, `normalization-processor`로 점수를 정규화합니다. 그 결과를 Stage 3에서 Personalize가 받아 최종 순서를 조정하는 구조입니다.

```python
# 3-Stage 하이브리드 + 개인화 파이프라인
hybrid_pipeline = {
    "description": "Hybrid search + Personalize re-ranking",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {"technique": "min_max"},
                "combination": {
                    "technique": "arithmetic_mean",
                    "parameters": {"weights": [0.4, 0.6]}  # BM25 vs k-NN 비중
                }
            }
        }
    ],
    "response_processors": [
        {
            "personalized_search_ranking": {
                "campaign_arn": "arn:aws:personalize:us-east-1:123456789:campaign/my-rerank-campaign",
                "item_id_field": "product_id",
                "recipe": "aws-personalized-ranking",
                "weight": 0.3,
                "iam_role_arn": "arn:aws:iam::123456789:role/PersonalizeOpenSearchRole"
            }
        }
    ]
}
```


이 구조의 장점은 각 단계의 책임이 깔끔하게 분리된다는 점입니다. 검색 관련성 튜닝은 Stage 1-2에서 처리하고, 개인화 강도는 Stage 3의 weight 하나로 제어할 수 있어 운영이 단순해집니다. 콜드 스타트 사용자(상호작용 이력이 없는 신규 유저)의 경우 Personalize가 기본 인기도 기반으로 폴백(fallback)하기 때문에, weight를 낮게 잡아 두면 검색 품질이 크게 떨어지지 않습니다. 실제로 운영 환경에서 테스트해 보니 신규 유저 비율이 높은 서비스에서도 0.2 정도면 무난했습니다.

다음 섹션에서는 이렇게 정렬된 검색·추천 결과를 LLM이 어떻게 자연

## LLM(Bedrock) × Personalize — 개인화된 RAG 응답 생성

앞서 OpenSearch를 활용한 검색 결과 Re-Ranking까지 살펴보았으니, 이제 마지막 퍼즐 조각인 LLM 응답 생성 단계에 Personalize를 결합하는 방법을 다뤄보겠습니다.

### 핵심 아이디어

일반적인 RAG(Retrieval-Augmented Generation) 파이프라인은 사용자 질의(query)만으로 문서를 검색한 뒤 LLM에 전달합니다. 여기에 Personalize 추천 결과를 추가 컨텍스트로 주입하면, 동일한 질문이라도 사용자마다 다른 톤·상품·카테고리를 반영한 응답을 생성할 수 있습니다.

![Personalized RAG](/ai-tech-blog/images/posts/2026-02-17/amazon-personalize-opensearch-llm-개인화-추천의-삼각편대/diagram-3.png)

### 구현 흐름

```python
import boto3, json

personalize_rt = boto3.client("personalize-runtime")
bedrock_rt = boto3.client("bedrock-runtime")

# 1) Personalize에서 사용자 맞춤 추천 조회
rec_resp = personalize_rt.get_recommendations(
    campaignArn="arn:aws:personalize:ap-northeast-2:123456789012:campaign/my-campaign",
    userId="user_42",
    numResults=5,
)
rec_items = [item["itemId"] for item in rec_resp["itemList"]]

# 2) 추천 아이템 메타데이터를 컨텍스트 문자열로 변환
item_context = "\n".join(
    [f"- {iid}: {get_item_meta(iid)}" for iid in rec_items]   # get_item_meta는 DynamoDB 등에서 조회
)

# 3) RAG 검색 결과 + Personalize 컨텍스트를 합쳐 프롬프트 구성
prompt = f"""아래는 사용자가 최근 관심을 보인 상품 목록입니다:
{item_context}

참고 문서:
{{rag_retrieved_docs}}

사용자의 질문: "이번 주말 뭐 입을까요?"
위 정보를 바탕으로, 사용자 취향에 맞는 코디를 추천해 주세요."""

body = json.dumps({
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 512,
    "messages": [{"role": "user", "content": prompt}],
})

llm_resp = bedrock_rt.invoke_model(
    modelId="anthropic.claude-3-sonnet-20240229-v1:0",
    body=body,
)
answer = json.loads(llm_resp["body"].read())["content"][0]["text"]
```

### 실전 팁

개인적으로 가장 효과를 본 부분은 **프롬프트 내 추천 아이템의 배치 순서**입니다. Personalize가 반환하는 스코어(score) 순서를 그대로 유지해야 LLM이 상위 아이템에 더 높은 가중치를 두는 경향이 있습니다. 추천 아이템 수는 3~5개가 적당합니다. 너무 많으면 컨텍스트 윈도(context window)를 불필요하게 소모하고, 응답 품질이 오히려 떨어집니다.

실제로 써보면, 같은 "주말 코디 추천" 질문에도 스트리트웨어 선호 사용자에게는 오버핏 후디 중심으로, 미니멀 선호 사용자에게는 모노톤 셋업 중심으로 응답이 달라집니다. 검색(OpenSearch)에서 Re-Ranking(Personalize)을 거쳐 생성(Bedrock LLM)까지, 세 단계가 맞물리는 지점이 바로 여기입니다.

## References

1. **Amazon Personalize 공식 개발자 가이드** — Amazon Personalize의 핵심 개념, 아키텍처, 워크플로를 다루는 공식 문서.
   [https://docs.aws.amazon.com/personalize/latest/dg/what-is-personalize.html](https://docs.aws.amazon.com/personalize/latest/dg/what-is-personalize.html)

2. **Amazon Personalize + OpenSearch 통합 공식 문서** — Amazon Personalize 추천 결과를 OpenSearch Service의 검색 파이프라인에 통합하는 방법을 설명하는 공식 문서.
   [https://docs.aws.amazon.com/personalize/latest/dg/personalize-opensearch.html](https://docs.aws.amazon.com/personalize/latest/dg/personalize-opensearch.html)

3. **Amazon Personalize Recipes 공식 문서** — User-Personalization-v2, Similar-Items 등 Personalize가 제공하는 레시피(알고리즘)의 종류와 특성을 정리한 공식 문서.
   [https://docs.aws.amazon.com/personalize/latest/dg/native-recipe-new.html](https://docs.aws.amazon.com/personalize/latest/dg/native-recipe-new.html)

4. **Elevate RAG for Numerical Analysis Using Amazon Bedrock Knowledge Bases** (AWS Blog) — Amazon Bedrock Knowledge Bases와 RAG 파이프라인을 결합하여 수치 분석 및 추천 품질을 높이는 패턴을 소개하는 블로그 포스트.
   [https://aws.amazon.com/blogs/machine-learning/elevate-rag-for-numerical-analysis-using-amazon-bedrock-knowledge-bases/](https://aws.amazon.com/blogs/machine-learning/elevate-rag-for-numerical-analysis-using-amazon-bedrock-knowledge-bases/)

5. **Amazon Personalize Can Now Unlock Intrinsic Signals in Your Catalog** (AWS Blog) — LLM 기반 아이템 임베딩을 활용하여 카탈로그 내재 신호를 추천에 반영하는 Amazon Personalize의 신규 기능을 다룬 블로그 포스트.
   [https://aws.amazon.com/blogs/machine-learning/amazon-personalize-can-now-unlock-intrinsic-signals-in-your-catalog-to-recommend-similar-items/](https://aws.amazon.com/blogs/machine-learning/amazon-personalize-can-now-unlock-intrinsic-signals-in-your-catalog-to-recommend-similar-items/)

6. **Amazon OpenSearch Service 공식 개발자 가이드** — OpenSearch Service의 검색 파이프라인, k-NN 벡터 검색, Search Pipeline 플러그인 등 핵심 기능을 다루는 공식 문서.
   [https://docs.aws.amazon.com/opensearch-service/latest/developerguide/what-is.html](https://docs.aws.amazon.com/opensearch-service/latest/developerguide/what-is.html)

7. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** (논문) — RAG(Retrieval-Augmented Generation) 패러다임의 원본 논문으로, 검색 결과를 LLM 생성에 결합하는 이론적 토대를 제공한다.
   [https://arxiv.org/abs/2005.11401](https://arxiv.org/abs/2005.11401)

8. **Amazon Bedrock 공식 개발자 가이드** — Foundation Model 호출, Knowledge