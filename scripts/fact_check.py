#!/usr/bin/env python3
"""블로그 본문 팩트체크 (강화 버전: 웹 검색 + AI 판단 + Dead Link 자동 수정)"""

import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import boto3
import requests
import yaml
from botocore.config import Config as BotoConfig

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
LOGS_DIR = BASE_DIR / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fact_check")


def load_config() -> dict:
    with open(SCRIPTS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


# ── 주장 추출 패턴 ──────────────────────────────

CLAIM_PATTERNS = [
    (r"[^.]*\d+[\d,.]*\s*(%|퍼센트|배|억|만|천|B|M|K|GB|TB|MB)[^.]*\.", "numeric"),
    (r"[^.]*(?:20\d{2}[-년/]\s*\d{0,2})[^.]*\.", "date"),
    (r"[^.]*(?:Google|OpenAI|Meta|Microsoft|NVIDIA|Amazon|AWS|Anthropic|DeepMind|LeCun|Fei-Fei Li)\S*\s+(?:발표|출시|공개|도입|개발|제안|설립|주장|논문|발표했|주도)[^.]*\.", "entity_claim"),
    (r"[^.]*(?:논문|연구|실험|벤치마크|결과)에서[^.]*\.", "research_claim"),
]


def extract_claims(text: str) -> list:
    claims = []
    seen = set()
    for pattern, claim_type in CLAIM_PATTERNS:
        for match in re.finditer(pattern, text):
            claim = match.group(0).strip()
            if claim not in seen and len(claim) > 20 and "![" not in claim and "```" not in claim:
                seen.add(claim)
                claims.append({"text": claim, "type": claim_type})
    return claims


def extract_references(text: str) -> list:
    urls = re.findall(r'https?://[^\s)\]"\']+', text)
    return list(set(urls))


# ── URL 접근성 확인 ──────────────────────────────

def check_url_accessible(url: str) -> dict:
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True,
                             headers={"User-Agent": "Mozilla/5.0"})
        # Some sites block HEAD, try GET for 4xx/5xx
        if resp.status_code >= 400:
            resp = requests.get(url, timeout=10, allow_redirects=True,
                                headers={"User-Agent": "Mozilla/5.0"}, stream=True)
        final_url = str(resp.url)
        # Detect suspicious redirects (service page → generic product/home page)
        redirected_to_generic = False
        if resp.status_code < 400 and final_url != url:
            generic_pages = ["/products/", "/products", "/index.html", "aws.amazon.com/?nc"]
            redirected_to_generic = any(g in final_url for g in generic_pages)
            if redirected_to_generic:
                logger.warning("🔀 Suspicious redirect: %s → %s", url, final_url)
        accessible = resp.status_code < 400 and not redirected_to_generic
        return {"url": url, "accessible": accessible, "status": resp.status_code,
                "final_url": final_url, "redirected_to_generic": redirected_to_generic}
    except Exception as e:
        return {"url": url, "accessible": False, "error": str(e)}


# ── Dead Link 자동 대체 (Perplexity 검색) ────────

def find_replacement_url(dead_url: str, context: str) -> str | None:
    """Dead link에 대해 Perplexity로 대체 URL 검색"""
    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not api_key:
        key_path = Path.home() / ".config" / "perplexity" / "api_key"
        if key_path.exists():
            api_key = key_path.read_text().strip()
    if not api_key:
        return None

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": "You find working replacement URLs for dead links. Reply with ONLY a single valid HTTPS URL on one line, nothing else. No markdown, no brackets, no citations. If you cannot find one, reply NONE."},
                    {"role": "user", "content": f"This URL returns 404: {dead_url}\nIt was referenced in this context: {context[:200]}\nFind a working official page, blog post, or documentation URL for the same topic. Reply with ONLY the URL."}
                ],
                "max_tokens": 100,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"].strip()
            # Extract URL from response
            url_match = re.search(r'https?://[^\s\)]+', content)
            if url_match and "NONE" not in content.upper():
                candidate = re.sub(r'\[\d+\]$', '', url_match.group(0).rstrip('.'))
                # Verify the replacement URL actually works
                check = check_url_accessible(candidate)
                if check.get("accessible"):
                    return candidate
                else:
                    logger.warning("  대체 URL도 접근 불가: %s (status: %s)", candidate, check.get("status", "?"))
    except Exception as e:
        logger.warning("  대체 URL 검색 실패: %s", e)
    return None


def find_replacement_url_bedrock(dead_url: str, context: str, config: dict) -> str | None:
    """Bedrock Claude로 대체 URL 추천 (Perplexity 실패 시 fallback)"""
    bedrock_cfg = config.get("bedrock", {})
    boto_config = BotoConfig(read_timeout=60, connect_timeout=10)
    client = boto3.client("bedrock-runtime", region_name=bedrock_cfg.get("region", "us-west-2"), config=boto_config)

    prompt = f"""다음 URL이 404 에러입니다: {dead_url}
블로그 문맥: {context[:300]}

같은 주제의 공식 페이지/블로그/문서 중 확실히 존재하는 대체 URL을 1개만 추천하세요.
추측하지 말고, 실제로 존재하는 URL만 제시하세요.
URL만 한 줄로 출력하세요. 모르면 NONE이라고만 답하세요."""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = client.invoke_model_with_response_stream(
            modelId=bedrock_cfg.get("model_id", "global.anthropic.claude-opus-4-6-v1"),
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        chunks = []
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                chunks.append(chunk["delta"]["text"])
        result_text = "".join(chunks).strip()

        url_match = re.search(r'https?://[^\s\)]+', result_text)
        if url_match and "NONE" not in result_text.upper():
            candidate = url_match.group(0).rstrip('.')
            check = check_url_accessible(candidate)
            if check.get("accessible"):
                return candidate
            else:
                logger.warning("  Bedrock 추천 URL도 접근 불가: %s", candidate)
    except Exception as e:
        logger.warning("  Bedrock 대체 URL 검색 실패: %s", e)
    return None


def fix_dead_links(post_path: str, url_results: list, config: dict) -> list:
    """Dead link를 자동으로 대체하고 파일 수정"""
    dead_links = [u for u in url_results if not u.get("accessible")]
    if not dead_links:
        return []

    with open(post_path, encoding="utf-8") as f:
        content = f.read()

    fixes = []
    for dl in dead_links:
        dead_url = dl["url"]
        # URL 주변 문맥 추출
        idx = content.find(dead_url)
        if idx < 0:
            continue
        context = content[max(0, idx - 200):idx + len(dead_url) + 200]

        logger.info("🔗 Dead link 대체 시도: %s", dead_url)

        # 1차: Perplexity 웹 검색
        replacement = find_replacement_url(dead_url, context)

        # 2차: Bedrock AI (Perplexity 실패 시)
        if not replacement:
            replacement = find_replacement_url_bedrock(dead_url, context, config)

        if replacement:
            content = content.replace(dead_url, replacement)
            fixes.append({"dead": dead_url, "replacement": replacement, "status": "fixed"})
            logger.info("  ✅ 대체 완료: %s → %s", dead_url, replacement)
        else:
            # 대체 실패 → 해당 URL을 아예 제거하지는 않음, 대신 로그에 경고
            fixes.append({"dead": dead_url, "replacement": None, "status": "unfixed"})
            logger.error("  ❌ 대체 URL을 찾을 수 없음: %s — 수동 확인 필요!", dead_url)

    if any(f["status"] == "fixed" for f in fixes):
        with open(post_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("📝 포스트 파일 업데이트 완료 (dead link %d건 수정)", sum(1 for f in fixes if f["status"] == "fixed"))

    unfixed = [f for f in fixes if f["status"] == "unfixed"]
    if unfixed:
        logger.error("🚨 수정 못 한 dead link %d건 — 발행 전 수동 확인 필수!", len(unfixed))

    return fixes


# ── 소스 기반 키워드 매칭 ────────────────────────

def check_claim_with_source(claim: str, urls: list) -> dict:
    result = {"claim": claim, "verified": False, "source": None, "confidence": 0.0}
    keywords = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)*|\d+[\d,.]*\s*(?:%|B|M|K|억|만)', claim)
    if not keywords:
        return result

    for url in urls[:3]:
        try:
            resp = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code >= 400:
                continue
            text = resp.text[:10000].lower()
            matched = sum(1 for kw in keywords if kw.lower() in text)
            confidence = matched / len(keywords) if keywords else 0
            if confidence > result["confidence"]:
                result["confidence"] = confidence
                result["source"] = url
                if confidence >= 0.5:
                    result["verified"] = True
        except Exception:
            continue
    return result


# ── Perplexity 웹 검색 검증 ─────────────────────

def search_verify_claim(claim: str) -> dict:
    api_key = os.environ.get("PERPLEXITY_API_KEY", "")
    if not api_key:
        key_path = Path.home() / ".config" / "perplexity" / "api_key"
        if key_path.exists():
            api_key = key_path.read_text().strip()
    if not api_key:
        return {"method": "search", "verified": None, "reason": "API key not found"}

    try:
        resp = requests.post(
            "https://api.perplexity.ai/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "sonar",
                "messages": [
                    {"role": "system", "content": "You are a fact-checker. Verify the following claim. Reply in JSON: {\"verified\": true/false/null, \"reason\": \"brief explanation\", \"confidence\": 0.0-1.0}"},
                    {"role": "user", "content": f"Verify this claim: {claim}"}
                ],
                "max_tokens": 300,
            },
            timeout=30,
        )
        if resp.status_code == 200:
            content = resp.json()["choices"][0]["message"]["content"]
            json_match = re.search(r'\{.*?\}', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result["method"] = "search"
                return result
        return {"method": "search", "verified": None, "reason": f"API error {resp.status_code}"}
    except Exception as e:
        return {"method": "search", "verified": None, "reason": str(e)}


# ── Bedrock AI 판단 ──────────────────────────────

def ai_judge_claims(claims: list, config: dict) -> list:
    if not claims:
        return []

    bedrock_cfg = config.get("bedrock", {})
    boto_config = BotoConfig(read_timeout=120, connect_timeout=10)
    client = boto3.client("bedrock-runtime", region_name=bedrock_cfg.get("region", "us-west-2"), config=boto_config)

    claims_text = "\n".join([f"{i+1}. [{c['type']}] {c['text']}" for i, c in enumerate(claims)])

    prompt = f"""당신은 AI/ML 기술 블로그의 팩트체커입니다.
아래 주장들의 사실 여부를 판단하세요.

## 주장 목록
{claims_text}

## 판단 기준
- 날짜, 수치, 인물, 기관명이 정확한지
- 기술적 설명이 올바른지
- 인과관계가 논리적인지

## 출력 형식 (JSON 배열)
[
  {{"id": 1, "verdict": "correct|incorrect|uncertain", "issue": "문제가 있으면 간략히 설명, 없으면 null"}}
]

JSON 배열만 출력하세요."""

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.0,
        "messages": [{"role": "user", "content": prompt}],
    }

    try:
        response = client.invoke_model_with_response_stream(
            modelId=bedrock_cfg.get("model_id", "global.anthropic.claude-opus-4-6-v1"),
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        chunks = []
        for event in response["body"]:
            chunk = json.loads(event["chunk"]["bytes"])
            if chunk["type"] == "content_block_delta":
                chunks.append(chunk["delta"]["text"])
        result_text = "".join(chunks)

        json_match = re.search(r'\[.*\]', result_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        logger.warning("AI 팩트체크 실패: %s", e)

    return []


# ── 메인 ─────────────────────────────────────────

def fact_check_post(post_path: str, config: dict = None, auto_fix: bool = True) -> dict:
    if config is None:
        config = load_config()
    fc_cfg = config.get("fact_check", {})
    if not fc_cfg.get("enabled", True):
        logger.info("팩트체크 비활성화")
        return {"enabled": False}

    with open(post_path, encoding="utf-8") as f:
        content = f.read()

    body = re.sub(r"^---.*?---", "", content, count=1, flags=re.DOTALL).strip()

    # 1. 주장 추출
    claims = extract_claims(body)
    max_claims = fc_cfg.get("max_claims_to_check", 10)
    claims = claims[:max_claims]
    logger.info("팩트체크 대상: %d건", len(claims))

    # 2. URL 접근성
    urls = extract_references(content)
    logger.info("참조 URL: %d건", len(urls))
    url_results = [check_url_accessible(u) for u in urls]
    dead_links = [u for u in url_results if not u.get("accessible")]
    if dead_links:
        logger.warning("⚠️ Dead links: %d건", len(dead_links))

    # 2.3 Reference 라벨↔URL 종류 불일치 검사
    ref_mismatches = []
    ref_pattern = re.findall(r'(\S+(?:공식 문서|Blog|논문|GitHub)[^
]*)(https?://[^\s\)\]]+)', body)
    for label_line, url in ref_pattern:
        is_docs = "docs.aws" in url or "documentation" in url
        is_blog = "/blogs/" in url or "/blog/" in url
        is_arxiv = "arxiv.org" in url
        is_github = "github.com" in url
        mismatch = None
        if "공식 문서" in label_line and not is_docs:
            mismatch = f"'공식 문서' label but URL is not docs: {url}"
        if "Blog" in label_line and not is_blog:
            mismatch = f"'Blog' label but URL is not a blog: {url}"
        if mismatch:
            ref_mismatches.append(mismatch)
            logger.warning("⚠️ 라벨 불일치: %s", mismatch)

    # 2.5 Dead Link 자동 수정
    link_fixes = []
    if dead_links and auto_fix:
        link_fixes = fix_dead_links(post_path, url_results, config)
        # 수정 못 한 dead link가 있으면 발행 차단
        unfixed = [f for f in link_fixes if f["status"] == "unfixed"]
        if unfixed:
            logger.error("🚨 Dead link %d건 수정 실패 — 발행을 중단합니다!", len(unfixed))

    # 3. 소스 기반 키워드 매칭
    claim_results = []
    threshold = fc_cfg.get("confidence_threshold", 0.6)
    for c in claims:
        r = check_claim_with_source(c["text"], urls) if urls else {"claim": c["text"], "verified": False, "confidence": 0.0}
        r["type"] = c["type"]
        claim_results.append(r)

    # 4. Perplexity 웹 검색 (미확인 주장만, 최대 5건)
    unverified = [c for c in claim_results if not c.get("verified")]
    search_results = []
    for c in unverified[:5]:
        sr = search_verify_claim(c["claim"])
        search_results.append({"claim": c["claim"][:80], **sr})
        if sr.get("verified") is True:
            c["verified"] = True
            c["confidence"] = max(c.get("confidence", 0), sr.get("confidence", 0.7))
            c["verification_method"] = "web_search"
    logger.info("웹 검색 검증: %d건 시도", len(search_results))

    # 5. Bedrock AI 판단 (여전히 미확인인 주장)
    still_unverified = [c for c in claim_results if not c.get("verified")]
    if still_unverified:
        logger.info("AI 판단: %d건", len(still_unverified))
        ai_results = ai_judge_claims(
            [{"text": c["claim"], "type": c.get("type", "")} for c in still_unverified],
            config
        )
        for ai_r in ai_results:
            idx = ai_r.get("id", 0) - 1
            if 0 <= idx < len(still_unverified):
                c = still_unverified[idx]
                verdict = ai_r.get("verdict", "uncertain")
                c["ai_verdict"] = verdict
                c["ai_issue"] = ai_r.get("issue")
                if verdict == "correct":
                    c["verified"] = True
                    c["confidence"] = max(c.get("confidence", 0), 0.7)
                    c["verification_method"] = "ai_judge"
                elif verdict == "incorrect":
                    c["flagged"] = True
                    c["flag_reason"] = ai_r.get("issue", "AI가 오류로 판단")

    # 최종 플래그
    for c in claim_results:
        if "flagged" not in c:
            c["flagged"] = not c.get("verified") and c.get("confidence", 0) < threshold

    flagged = [c for c in claim_results if c.get("flagged")]

    result = {
        "post": str(post_path),
        "timestamp": datetime.now().isoformat(),
        "total_claims": len(claims),
        "verified_claims": sum(1 for c in claim_results if c.get("verified")),
        "flagged_claims": len(flagged),
        "dead_links": len(dead_links),
        "dead_links_fixed": sum(1 for f in link_fixes if f["status"] == "fixed"),
        "dead_links_unfixed": sum(1 for f in link_fixes if f["status"] == "unfixed"),
        "link_fixes": link_fixes,
        "claims": claim_results,
        "search_results": search_results,
        "urls": url_results,
    }

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    out = LOGS_DIR / f"fact_check_{today}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("팩트체크 결과: %d/%d 확인, %d건 플래그, %d dead links (%d fixed, %d unfixed)",
                result["verified_claims"], result["total_claims"], result["flagged_claims"],
                result["dead_links"], result["dead_links_fixed"], result["dead_links_unfixed"])
    if flagged:
        for c in flagged:
            reason = c.get("flag_reason", c.get("ai_issue", "미확인"))
            logger.warning("⚠️ [%s] %s — %s", c.get("ai_verdict", "?"), c["claim"][:60], reason)
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="블로그 팩트체크")
    parser.add_argument("--post", required=True, help="포스트 파일 경로")
    parser.add_argument("--no-fix", action="store_true", help="Dead link 자동 수정 비활성화")
    args = parser.parse_args()

    result = fact_check_post(args.post, auto_fix=not args.no_fix)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    # Dead link 미수정 있으면 exit code 2
    if result.get("dead_links_unfixed", 0) > 0:
        sys.exit(2)
    # 플래그된 주장 있으면 exit code 1
    if result.get("flagged_claims", 0) > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
