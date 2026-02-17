#!/usr/bin/env python3
"""
AI Tech Blog - 자동 주제 선정 모듈
매주 cron에서 호출되어 주제 후보 3개를 생성하고 Jay에게 확인 요청.
"""

import json
import os
import re
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta

import yaml
import boto3
import feedparser
import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("topic_selector")

# ===== 주제 선정 기준 (Jay 피드백 기반) =====
TOPIC_CRITERIA = """
## 주제 선정 기준

### 1. Jay의 SA 업무와 연결
- AWS 서비스(Bedrock, SageMaker, Personalize, Neptune, IoT 등)와 실제 고객사에 적용 가능한 주제
- 담당 고객: 삼성물산 4개 부문
  - **건설**: 가장 비중 큼. 건설 현장 AI, 디지털 트윈, BIM, 안전관리, 스마트빌딩, HVAC 최적화
  - **리조트(에버랜드)**: 어트랙션 설비 예방정비(Predictive Maintenance), 동물(펭귄/홍학 등) identity 관리 및 트래킹(Computer Vision, Re-ID)
  - **패션(SSF몰)**: 리테일 AI, 개인화 추천, 수요 예측, 비주얼 검색, 트렌드 분석
  - **상사**: 공급망 최적화, 무역 데이터 분석

### 2. 이론 + 실용 밸런스
- 논문 리뷰도 "그래서 AWS에서 어떻게 구현하나"로 연결
- 코드 예시, 아키텍처 다이어그램 포함 가능한 주제

### 3. 카테고리 로테이션
- 기존 카테고리: "AI/ML 기술 심층분석", "MLOps & Platform", "Physical AI", "논문 리뷰", "AWS AI/ML"
- 최근 발행 카테고리와 겹치지 않도록 로테이션

### 4. 트렌디한 키워드
- 업계에서 관심 높은 주제 (최근 arxiv, AWS 블로그, 기술 뉴스)
- SA로서 발표/공유할 만한 수준

### 5. 도메인 접점
- 건설/제조/에너지 + AI 교차점
- MCP(Model Context Protocol), 에이전트, RAG 등 Jay 관심 분야

### 6. 기존 포스트와 차별화
- 이미 다룬 주제 반복하지 않기
- 시리즈물 가능 (e.g., 프루닝 → 양자화, GraphRAG → 에이전트 RAG)
"""


def load_config(config_path: str = "scripts/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_published_posts(content_dir: str = "content/posts") -> list:
    """기존 발행 포스트 목록 (제목, 카테고리, 날짜)"""
    posts = []
    content_path = Path(content_dir)
    if not content_path.exists():
        return posts

    for md_file in sorted(content_path.glob("*.md"), reverse=True):
        text = md_file.read_text()
        title_m = re.search(r'^title:\s*"(.+?)"', text, re.MULTILINE)
        cat_m = re.search(r'categories:\s*\n\s*-\s*"(.+?)"', text, re.MULTILINE)
        date_m = re.search(r'^date:\s*(\S+)', text, re.MULTILINE)

        posts.append({
            "title": title_m.group(1) if title_m else md_file.stem,
            "category": cat_m.group(1) if cat_m else "unknown",
            "date": date_m.group(1)[:10] if date_m else "unknown",
            "file": md_file.name,
        })
    return posts


def fetch_trending_topics(config: dict) -> list:
    """RSS + arxiv에서 최근 트렌드 수집"""
    topics = []

    # RSS feeds
    for feed_conf in config.get("sources", {}).get("rss_feeds", []):
        try:
            feed = feedparser.parse(feed_conf["url"])
            for entry in feed.entries[:10]:
                topics.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:200],
                    "source": feed_conf["name"],
                    "url": entry.get("link", ""),
                    "date": entry.get("published", ""),
                })
        except Exception as e:
            logger.warning(f"RSS fetch failed: {feed_conf['name']}: {e}")

    # Brave Search for recent AI trends
    try:
        brave_key = os.environ.get("BRAVE_SEARCH_API_KEY", "")
        if brave_key:
            headers = {"Accept": "application/json", "Accept-Encoding": "gzip", "X-Subscription-Token": brave_key}
            queries = [
                "latest AI ML research breakthrough 2026",
                "AWS new AI service announcement 2026",
                "construction AI digital twin 2026",
            ]
            for q in queries:
                resp = requests.get(f"https://api.search.brave.com/res/v1/web/search?q={q}&count=5", headers=headers, timeout=10)
                if resp.status_code == 200:
                    for r in resp.json().get("web", {}).get("results", []):
                        topics.append({
                            "title": r.get("title", ""),
                            "summary": r.get("description", "")[:200],
                            "source": "Brave Search",
                            "url": r.get("url", ""),
                        })
    except Exception as e:
        logger.warning(f"Brave search failed: {e}")

    return topics


def generate_topic_candidates(config: dict, published_posts: list, trending: list) -> str:
    """Bedrock Claude로 주제 후보 3개 생성"""

    posts_summary = "\n".join([
        f"- [{p['date']}] [{p['category']}] {p['title']}"
        for p in published_posts[:15]
    ])

    trending_summary = "\n".join([
        f"- [{t['source']}] {t['title']}"
        for t in trending[:20]
    ])

    # 최근 카테고리 분석
    recent_cats = [p["category"] for p in published_posts[:5]]
    cat_counts = {}
    for c in recent_cats:
        cat_counts[c] = cat_counts.get(c, 0) + 1

    prompt = f"""당신은 AWS Solutions Architect가 운영하는 AI 기술 블로그의 주제 선정 어시스턴트입니다.

{TOPIC_CRITERIA}

## 기존 발행 포스트
{posts_summary}

## 최근 카테고리 분포 (최근 5개)
{json.dumps(cat_counts, ensure_ascii=False)}

## 최근 트렌드 (RSS + 검색)
{trending_summary}

## 요청
위 기준과 기존 포스트, 트렌드를 종합해서 **다음 주 블로그 주제 후보 3개**를 제안해주세요.

각 후보에 대해:
1. **주제명** (블로그 제목 수준, 간결하게)
2. **카테고리** (기존 카테고리 중 택1)
3. **선정 이유** (2-3문장)
4. **핵심 키워드** (5개 이내)
5. **참고 논문/자료** (있다면)

JSON 형식으로 응답하세요:
```json
[
  {{
    "title": "주제명",
    "category": "카테고리",
    "reason": "선정 이유",
    "keywords": ["kw1", "kw2"],
    "references": ["url1"]
  }},
  ...
]
```
"""

    client = boto3.client("bedrock-runtime", region_name=config["bedrock"]["region"])
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 2000,
        "temperature": 0.8,
        "messages": [{"role": "user", "content": prompt}]
    })

    response = client.invoke_model(
        modelId=config["bedrock"]["model_id"],
        body=body,
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]


def select_topics(config_path: str = "scripts/config.yaml") -> dict:
    """메인 함수: 주제 후보 생성 + 결과 반환"""
    config = load_config(config_path)

    logger.info("기존 포스트 분석...")
    published = get_published_posts()

    logger.info("트렌드 수집...")
    trending = fetch_trending_topics(config)
    logger.info(f"트렌드 {len(trending)}건 수집")

    logger.info("주제 후보 생성 (Bedrock)...")
    candidates_text = generate_topic_candidates(config, published, trending)

    # JSON 추출
    json_match = re.search(r'```json\s*(.*?)```', candidates_text, re.DOTALL)
    if json_match:
        candidates = json.loads(json_match.group(1))
    else:
        # try raw JSON
        try:
            candidates = json.loads(candidates_text)
        except:
            candidates = [{"raw": candidates_text}]

    result = {
        "generated_at": datetime.now().isoformat(),
        "published_count": len(published),
        "trending_count": len(trending),
        "candidates": candidates,
    }

    # 결과 저장
    output_path = Path("logs/topic_candidates.json")
    output_path.parent.mkdir(exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info(f"주제 후보 저장: {output_path}")

    return result


if __name__ == "__main__":
    result = select_topics()
    print(json.dumps(result, ensure_ascii=False, indent=2))
