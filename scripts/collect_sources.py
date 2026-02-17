#!/usr/bin/env python3
"""RSS/arXiv/Papers with Code에서 AI/ML 트렌딩 콘텐츠 수집"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import feedparser
import requests
import yaml
from dateutil import parser as dateparser

# 프로젝트 루트
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR = BASE_DIR / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("collect_sources")


def load_config() -> dict:
    config_path = SCRIPTS_DIR / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def collect_rss(feeds: list, lookback: timedelta) -> list:
    """RSS 피드에서 최근 글 수집"""
    cutoff = datetime.now(timezone.utc) - lookback
    items = []
    for feed_cfg in feeds:
        url = feed_cfg["url"]
        logger.info("RSS 파싱: %s", feed_cfg["name"])
        try:
            d = feedparser.parse(url)
            for entry in d.entries[:feed_cfg.get("max", 20)]:
                pub = entry.get("published_parsed") or entry.get("updated_parsed")
                if pub:
                    pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
                    if pub_dt < cutoff:
                        continue
                else:
                    pub_dt = datetime.now(timezone.utc)
                items.append({
                    "source": "rss",
                    "feed": feed_cfg["name"],
                    "category": feed_cfg.get("category", ""),
                    "title": entry.get("title", ""),
                    "link": entry.get("link", ""),
                    "summary": entry.get("summary", "")[:500],
                    "published": pub_dt.isoformat(),
                    "weight": feed_cfg.get("weight", 1.0),
                })
        except Exception as e:
            logger.error("RSS 실패 (%s): %s", url, e)
    logger.info("RSS 수집 완료: %d건", len(items))
    return items


def collect_arxiv(cfg: dict, lookback: timedelta) -> list:
    """arXiv API로 최신 논문 수집"""
    categories = cfg.get("categories", ["cs.AI", "cs.LG"])
    max_results = cfg.get("max_results", 50)
    cat_query = "+OR+".join(f"cat:{c}" for c in categories)
    url = (
        f"http://export.arxiv.org/api/query?search_query={cat_query}"
        f"&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    )
    logger.info("arXiv 검색: %s", categories)
    items = []
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        import xml.etree.ElementTree as ET
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(resp.text)
        cutoff = datetime.now(timezone.utc) - lookback
        for entry in root.findall("atom:entry", ns):
            published = entry.findtext("atom:published", "", ns)
            pub_dt = dateparser.parse(published)
            if pub_dt and pub_dt < cutoff:
                continue
            title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
            summary = entry.findtext("atom:summary", "", ns).strip()[:500]
            link_el = entry.find("atom:id", ns)
            link = link_el.text if link_el is not None else ""
            cats = [c.attrib.get("term", "") for c in entry.findall("atom:category", ns)]
            items.append({
                "source": "arxiv",
                "title": title,
                "link": link,
                "summary": summary,
                "published": pub_dt.isoformat() if pub_dt else "",
                "categories": cats,
                "weight": cfg.get("weight", 1.0),
            })
    except Exception as e:
        logger.error("arXiv 실패: %s", e)
    logger.info("arXiv 수집 완료: %d건", len(items))
    return items


def collect_papers_with_code(cfg: dict) -> list:
    """Papers with Code 트렌딩 수집"""
    if not cfg.get("enabled", True):
        return []
    url = cfg.get("trending_url", "https://paperswithcode.com/latest")
    logger.info("Papers with Code 수집: %s", url)
    items = []
    try:
        from bs4 import BeautifulSoup
        resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for row in soup.select(".row.paper-card.infinite-item")[:20]:
            title_el = row.select_one("h1 a")
            if not title_el:
                continue
            title = title_el.get_text(strip=True)
            link = "https://paperswithcode.com" + title_el["href"] if title_el.get("href") else ""
            abstract_el = row.select_one(".item-strip-abstract")
            summary = abstract_el.get_text(strip=True)[:500] if abstract_el else ""
            items.append({
                "source": "papers_with_code",
                "title": title,
                "link": link,
                "summary": summary,
                "published": datetime.now(timezone.utc).isoformat(),
                "weight": cfg.get("weight", 1.3),
            })
    except Exception as e:
        logger.error("Papers with Code 실패: %s", e)
    logger.info("Papers with Code 수집 완료: %d건", len(items))
    return items


def rank_sources(items: list) -> list:
    """가중치 기반 정렬"""
    for item in items:
        item["score"] = item.get("weight", 1.0)
    return sorted(items, key=lambda x: x["score"], reverse=True)


def main():
    config = load_config()
    lookback = timedelta(days=config.get("collection", {}).get("lookback_days", 7))
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    all_items = []
    all_items.extend(collect_rss(config.get("sources", {}).get("rss_feeds", []), lookback))
    all_items.extend(collect_arxiv(config.get("sources", {}).get("arxiv", {}), lookback))
    all_items.extend(collect_papers_with_code(config.get("sources", {}).get("papers_with_code", {})))

    ranked = rank_sources(all_items)
    today = datetime.now().strftime("%Y-%m-%d")
    output_path = LOGS_DIR / f"sources_{today}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f, ensure_ascii=False, indent=2)
    logger.info("결과 저장: %s (%d건)", output_path, len(ranked))
    return ranked


if __name__ == "__main__":
    main()
