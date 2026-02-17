#!/usr/bin/env python3
"""qa_published.py — 배포된 블로그 포스트 최종 검수

검수 항목:
1. 텍스트 검사: LaTeX, SVG, [DIAGRAM] 플레이스홀더, 섹션 수
2. 이미지 접근 검사: 다이어그램 PNG URL HTTP 200 확인
3. Dead link 검사: 본문 내 외부 링크 접근성
4. AI 문체 검사: humanizer 패턴 (과장, 반복, 반말 등)
5. 스크린샷: Playwright로 실제 렌더링 캡처 (선택)

Usage:
  python3 qa_published.py --url <published_url> [--post <local_md_path>] [--screenshot]
"""

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, unquote

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("qa_published")

# ── 검수 결과 ────────────────────────────────────────────

class QAResult:
    def __init__(self):
        self.checks = []
        self.errors = []
        self.warnings = []

    def error(self, category, msg):
        self.errors.append({"category": category, "message": msg})
        self.checks.append({"category": category, "status": "ERROR", "message": msg})
        logger.error("❌ [%s] %s", category, msg)

    def warn(self, category, msg):
        self.warnings.append({"category": category, "message": msg})
        self.checks.append({"category": category, "status": "WARN", "message": msg})
        logger.warning("⚠️ [%s] %s", category, msg)

    def ok(self, category, msg):
        self.checks.append({"category": category, "status": "OK", "message": msg})
        logger.info("✅ [%s] %s", category, msg)

    @property
    def passed(self):
        return len(self.errors) == 0

    def summary(self):
        total = len(self.checks)
        errors = len(self.errors)
        warnings = len(self.warnings)
        oks = total - errors - warnings
        return f"검수 완료: {total}건 ({oks} OK, {warnings} WARN, {errors} ERROR)"


# ── 1. 텍스트 검사 ──────────────────────────────────────

def check_text(content: str, result: QAResult):
    """배포된 페이지 텍스트에서 잔존 코드/마크업 검사"""

    # LaTeX $...$ 잔존
    latex_matches = re.findall(r"\$[^$]{3,}\$", content)
    if latex_matches:
        result.error("TEXT_LATEX", f"LaTeX 수식 {len(latex_matches)}개 잔존: {latex_matches[0][:50]}...")
    else:
        result.ok("TEXT_LATEX", "LaTeX 수식 없음")

    # <svg> 태그 잔존
    svg_count = content.lower().count("<svg")
    if svg_count > 0:
        result.error("TEXT_SVG", f"<svg> 태그 {svg_count}개 잔존")
    else:
        result.ok("TEXT_SVG", "SVG 태그 없음")

    # [DIAGRAM: ...] 플레이스홀더 잔존
    diagram_placeholders = re.findall(r"\[DIAGRAM:[^\]]*\]", content)
    if diagram_placeholders:
        result.error("TEXT_DIAGRAM", f"[DIAGRAM] 플레이스홀더 {len(diagram_placeholders)}개 잔존")
    else:
        result.ok("TEXT_DIAGRAM", "[DIAGRAM] 플레이스홀더 없음")

    # ```svg 또는 ```mermaid 코드 블록 잔존
    code_blocks = re.findall(r"```(?:svg|mermaid)", content)
    if code_blocks:
        result.error("TEXT_CODEBLOCK", f"svg/mermaid 코드 블록 {len(code_blocks)}개 잔존")
    else:
        result.ok("TEXT_CODEBLOCK", "svg/mermaid 코드 블록 없음")

    # 섹션 수 확인 (최소 3개 = 2 본문 + References)
    sections = re.findall(r"^#{2}\s+.+", content, re.MULTILINE)
    # HTML headings fallback
    if not sections:
        sections = re.findall(r"<h[23][^>]*>(.+?)</h[23]>", content, re.IGNORECASE)
    if len(sections) < 3:
        result.warn("TEXT_SECTIONS", f"섹션 {len(sections)}개 (최소 3개 권장)")
    else:
        result.ok("TEXT_SECTIONS", f"섹션 {len(sections)}개")

    # 분량 확인 (최소 3000자)
    if len(content) < 3000:
        result.warn("TEXT_LENGTH", f"분량 {len(content)}자 (최소 3000자 권장)")
    else:
        result.ok("TEXT_LENGTH", f"분량 {len(content)}자")


# ── 2. 이미지 접근 검사 ─────────────────────────────────

def check_images(content: str, base_url: str, result: QAResult):
    """본문 내 이미지 URL 접근성 확인"""
    import requests

    # 마크다운 이미지 참조 추출
    img_refs = re.findall(r"!\[.*?\]\((.*?)\)", content)
    if not img_refs:
        result.warn("IMG_COUNT", "이미지 0개 (다이어그램 없음?)")
        return

    result.ok("IMG_COUNT", f"이미지 {len(img_refs)}개 발견")

    for img_url in img_refs:
        # 상대 경로 → 절대 URL
        if img_url.startswith("/"):
            full_url = "https://jesamkim.github.io" + img_url
        elif not img_url.startswith("http"):
            full_url = urljoin(base_url, img_url)
        else:
            full_url = img_url

        try:
            resp = requests.head(full_url, timeout=10, allow_redirects=True)
            if resp.status_code == 200:
                result.ok("IMG_ACCESS", f"이미지 OK: {img_url.split('/')[-1]}")
            else:
                result.error("IMG_ACCESS", f"이미지 접근 실패 ({resp.status_code}): {img_url.split('/')[-1]}")
        except Exception as e:
            result.error("IMG_ACCESS", f"이미지 요청 실패: {img_url.split('/')[-1]} ({e})")


# ── 3. Dead link 검사 ───────────────────────────────────

def check_links(content: str, result: QAResult):
    """외부 링크 접근성 확인"""
    import requests

    # 마크다운 링크 추출 (이미지 제외)
    links = re.findall(r"(?<!!)\[.*?\]\((https?://[^\)]+)\)", content)
    if not links:
        result.ok("LINKS", "외부 링크 0개")
        return

    dead = 0
    for url in links[:20]:  # 최대 20개만 검사
        try:
            resp = requests.head(url, timeout=10, allow_redirects=True,
                                 headers={"User-Agent": "Mozilla/5.0 QA-Bot"})
            if resp.status_code >= 400:
                # GET으로 재시도 (일부 서버는 HEAD 거부)
                resp = requests.get(url, timeout=10, allow_redirects=True,
                                    headers={"User-Agent": "Mozilla/5.0 QA-Bot"})
            if resp.status_code >= 400:
                result.error("LINKS_DEAD", f"Dead link ({resp.status_code}): {url[:80]}")
                dead += 1
        except Exception as e:
            result.warn("LINKS_ERR", f"링크 확인 실패: {url[:60]} ({type(e).__name__})")

    if dead == 0:
        result.ok("LINKS", f"외부 링크 {len(links)}개 모두 접근 가능")


# ── 4. AI 문체 검사 ─────────────────────────────────────

AI_PATTERNS = [
    # 과장/빈 수식어
    (r"핵심적인 역할을 합니다", "과장 표현"),
    (r"매우 중요한", "과장 표현"),
    (r"혁신적인 접근", "과장 표현"),
    (r"획기적인", "과장 표현"),
    (r"살펴보도록 하겠습니다", "AI 패턴"),
    (r"알아보도록 하겠습니다", "AI 패턴"),
    (r"다루어 보겠습니다", "AI 패턴"),

    # AI 특유 도입부
    (r"급변하는.*시대에", "AI 클리셰"),
    (r"디지털 전환의 시대", "AI 클리셰"),
    (r"없어서는 안 될", "AI 클리셰"),

    # 과도한 강조
    (r"무엇보다도? 중요한 것은", "과도한 강조"),
    (r"결코 과언이 아닙니다", "과도한 강조"),

    # 반말 체크 (존댓말 통일)
    (r"(?<![가-힣])이다\.", "반말 (존댓말 통일 위반)"),
    (r"(?<![가-힣])한다\.", "반말 (존댓말 통일 위반)"),
    (r"(?<![가-힣])된다\.", "반말 (존댓말 통일 위반)"),
    (r"(?<![가-힣])없다\.", "반말 (존댓말 통일 위반)"),
    (r"(?<![가-힣])있다\.", "반말 (존댓말 통일 위반)"),
]

def check_ai_style(content: str, result: QAResult):
    """AI가 쓴 티가 나는 패턴 검사"""
    issues = []
    for pattern, label in AI_PATTERNS:
        matches = re.findall(pattern, content)
        if matches:
            issues.append(f"{label}: '{matches[0]}' ({len(matches)}회)")

    if issues:
        for issue in issues[:5]:  # 상위 5개만 리포트
            result.warn("AI_STYLE", issue)
        if len(issues) > 5:
            result.warn("AI_STYLE", f"...외 {len(issues) - 5}건")
    else:
        result.ok("AI_STYLE", "AI 문체 패턴 미발견")


# ── 5. 스크린샷 (Playwright) ────────────────────────────

def take_screenshot(url: str, output_path: str, result: QAResult):
    """Playwright로 페이지 스크린샷 촬영"""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1280, "height": 900})
            page.goto(url, wait_until="networkidle", timeout=30000)
            page.screenshot(path=output_path, full_page=True)
            browser.close()
            result.ok("SCREENSHOT", f"스크린샷 저장: {output_path}")
    except ImportError:
        result.warn("SCREENSHOT", "Playwright 미설치 — 스크린샷 생략")
    except Exception as e:
        result.warn("SCREENSHOT", f"스크린샷 실패: {e}")


# ── 메인 ────────────────────────────────────────────────

def fetch_page(url: str) -> str:
    """배포된 페이지 텍스트 가져오기"""
    import requests
    resp = requests.get(url, timeout=15,
                        headers={"User-Agent": "Mozilla/5.0 QA-Bot"})
    resp.raise_for_status()
    # HTML에서 텍스트 추출 (간단)
    from html.parser import HTMLParser

    class TextExtractor(HTMLParser):
        def __init__(self):
            super().__init__()
            self.text = []
            self.skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style", "nav", "header", "footer"):
                self.skip = True

        def handle_endtag(self, tag):
            if tag in ("script", "style", "nav", "header", "footer"):
                self.skip = False

        def handle_data(self, data):
            if not self.skip:
                self.text.append(data)

    extractor = TextExtractor()
    extractor.feed(resp.text)
    return "\n".join(extractor.text)


def read_local_post(post_path: str) -> str:
    """로컬 마크다운 파일 읽기"""
    with open(post_path, encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="배포된 블로그 포스트 QA 검수")
    parser.add_argument("--url", required=True, help="배포된 포스트 URL")
    parser.add_argument("--post", help="로컬 마크다운 파일 경로 (이미지 참조 추출용)")
    parser.add_argument("--screenshot", action="store_true", help="Playwright 스크린샷 촬영")
    parser.add_argument("--screenshot-path", default="/tmp/qa_screenshot.png", help="스크린샷 저장 경로")
    parser.add_argument("--wait", type=int, default=0, help="검수 전 대기 시간(초)")
    args = parser.parse_args()

    if args.wait > 0:
        logger.info("⏳ 배포 완료 대기 %d초...", args.wait)
        time.sleep(args.wait)

    result = QAResult()

    # 페이지 fetch
    logger.info("🔍 검수 시작: %s", args.url)
    try:
        page_text = fetch_page(args.url)
    except Exception as e:
        result.error("FETCH", f"페이지 접근 실패: {e}")
        print(json.dumps({"passed": False, "summary": result.summary(), "checks": result.checks}, ensure_ascii=False, indent=2))
        sys.exit(2)

    # 로컬 마크다운 (이미지 참조용)
    local_content = None
    if args.post and Path(args.post).exists():
        local_content = read_local_post(args.post)

    # 1. 텍스트 검사 (배포된 페이지)
    logger.info("📝 텍스트 검사...")
    check_text(page_text, result)

    # 2. 이미지 접근 검사 (로컬 마크다운에서 경로 추출)
    if local_content:
        logger.info("🖼️ 이미지 접근 검사...")
        check_images(local_content, args.url, result)
    else:
        result.warn("IMG_CHECK", "로컬 마크다운 없음 — 이미지 검사 생략")

    # 3. Dead link 검사
    if local_content:
        logger.info("🔗 Dead link 검사...")
        check_links(local_content, result)

    # 4. AI 문체 검사
    logger.info("✍️ AI 문체 검사...")
    check_ai_style(page_text, result)

    # 5. 스크린샷
    if args.screenshot:
        logger.info("📸 스크린샷 촬영...")
        take_screenshot(args.url, args.screenshot_path, result)

    # 결과 출력
    print()
    print("=" * 60)
    print(result.summary())
    print("=" * 60)
    for check in result.checks:
        icon = {"OK": "✅", "WARN": "⚠️", "ERROR": "❌"}[check["status"]]
        print(f"  {icon} [{check['category']}] {check['message']}")

    # JSON 로그 저장
    log_path = Path("/tmp/qa_result.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump({
            "url": args.url,
            "passed": result.passed,
            "summary": result.summary(),
            "errors": len(result.errors),
            "warnings": len(result.warnings),
            "checks": result.checks,
        }, f, ensure_ascii=False, indent=2)
    logger.info("📋 결과 저장: %s", log_path)

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
