#!/usr/bin/env python3
"""통합 파이프라인 실행"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
HUGO_DIR = BASE_DIR / "hugo-site"
LOGS_DIR = BASE_DIR / "logs"

# 로깅 설정
LOGS_DIR.mkdir(parents=True, exist_ok=True)
today = datetime.now().strftime("%Y-%m-%d")
log_file = LOGS_DIR / f"publish_{today}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("publish")


def load_config() -> dict:
    with open(SCRIPTS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def step_collect(config: dict) -> list:
    """1단계: 소스 수집"""
    logger.info("=" * 50)
    logger.info("1단계: 소스 수집")
    from collect_sources import main as collect_main
    sources = collect_main()
    logger.info("수집 완료: %d건", len(sources))
    return sources


def step_select_topic(sources: list, topic: str = None) -> tuple:
    """자동 주제 선정"""
    if topic:
        logger.info("수동 주제: %s", topic)
        relevant = [s for s in sources if any(
            kw.lower() in (s.get("title", "") + s.get("summary", "")).lower()
            for kw in topic.lower().split()
        )]
        return topic, relevant[:5] if relevant else sources[:5]

    if not sources:
        logger.warning("소스 없음, 기본 주제 사용")
        return "최신 AI/ML 트렌드 분석", []

    # 가장 높은 score의 소스를 주제로
    top = sources[0]
    topic = top["title"]
    logger.info("자동 주제 선정: %s (score: %.2f)", topic, top.get("score", 0))
    return topic, sources[:5]


def step_generate(topic: str, sources: list, config: dict) -> Path:
    """2단계: 포스트 생성"""
    logger.info("=" * 50)
    logger.info("2단계: 포스트 생성")
    from generate_post import generate_post
    post_path = generate_post(topic, sources, config)
    logger.info("포스트 생성: %s", post_path)
    return post_path


def step_qa_images(config: dict) -> list:
    """3단계: 이미지 QA"""
    logger.info("=" * 50)
    logger.info("3단계: 이미지 QA")
    from qa_images import qa_images_for_date
    results = qa_images_for_date(config=config)
    failed = [r for r in results if not r["valid"]]
    if failed:
        logger.warning("이미지 QA 실패: %d건", len(failed))
    else:
        logger.info("이미지 QA 통과: %d건", len(results))
    return results


def step_fact_check(post_path: Path, config: dict) -> dict:
    """4단계: 팩트체크"""
    logger.info("=" * 50)
    logger.info("4단계: 팩트체크")
    from fact_check import fact_check_post
    result = fact_check_post(str(post_path), config)
    return result


def step_hugo_build() -> bool:
    """5단계: Hugo 빌드 확인"""
    logger.info("=" * 50)
    logger.info("5단계: Hugo 빌드")
    try:
        result = subprocess.run(
            ["hugo", "--minify"],
            cwd=str(HUGO_DIR),
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            logger.info("Hugo 빌드 성공")
            return True
        logger.error("Hugo 빌드 실패: %s", result.stderr)
        return False
    except FileNotFoundError:
        logger.warning("Hugo 미설치, 빌드 스킵")
        return True
    except Exception as e:
        logger.error("Hugo 빌드 에러: %s", e)
        return False


def step_git_push(config: dict) -> bool:
    """6단계: Git commit & push"""
    pub_cfg = config.get("publish", {})
    logger.info("=" * 50)
    logger.info("6단계: Git push")

    if not pub_cfg.get("auto_commit", True):
        logger.info("자동 커밋 비활성화")
        return True

    try:
        subprocess.run(["git", "add", "-A"], cwd=str(HUGO_DIR), check=True, capture_output=True)
        msg = f"📝 Auto-publish: {today}"
        result = subprocess.run(
            ["git", "commit", "-m", msg],
            cwd=str(HUGO_DIR), capture_output=True, text=True,
        )
        if "nothing to commit" in result.stdout:
            logger.info("변경사항 없음")
            return True
        logger.info("커밋 완료: %s", msg)

        if pub_cfg.get("auto_push", False):
            subprocess.run(["git", "push"], cwd=str(HUGO_DIR), check=True, capture_output=True, timeout=60)
            logger.info("푸시 완료")
        else:
            logger.info("자동 푸시 비활성화 (수동 push 필요)")
        return True
    except Exception as e:
        logger.error("Git 에러: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="AI Tech Blog 통합 파이프라인")
    parser.add_argument("--topic", help="블로그 주제 (미지정 시 자동 선정)")
    parser.add_argument("--auto", action="store_true", help="자동 주제 선정 모드")
    parser.add_argument("--skip-collect", action="store_true", help="소스 수집 스킵")
    parser.add_argument("--skip-build", action="store_true", help="Hugo 빌드 스킵")
    parser.add_argument("--skip-push", action="store_true", help="Git push 스킵")
    parser.add_argument("--dry-run", action="store_true", help="실제 생성 없이 테스트")
    args = parser.parse_args()

    config = load_config()
    logger.info("🚀 AI Tech Blog 파이프라인 시작 (%s)", today)

    # 1. 수집
    sources = []
    if not args.skip_collect:
        sources = step_collect(config)

    # 2. 주제 선정
    topic = args.topic
    if not topic and args.auto:
        topic = None  # 자동 선정
    elif not topic:
        logger.error("--topic 또는 --auto 필요")
        sys.exit(1)
    topic, relevant_sources = step_select_topic(sources, topic)

    if args.dry_run:
        logger.info("🏁 드라이런 완료. 주제: %s, 소스: %d건", topic, len(relevant_sources))
        return

    # 3. 포스트 생성
    post_path = step_generate(topic, relevant_sources, config)

    # 4. 이미지 QA
    step_qa_images(config)

    # 5. 팩트체크
    fc_result = step_fact_check(post_path, config)
    if fc_result.get("dead_links_unfixed", 0) > 0:
        logger.error("🚨 Dead link %d건 수정 실패 — 발행 중단!", fc_result["dead_links_unfixed"])
        sys.exit(2)
    if fc_result.get("dead_links_fixed", 0) > 0:
        logger.info("🔗 Dead link %d건 자동 수정 완료", fc_result["dead_links_fixed"])
    if fc_result.get("flagged_claims", 0) > 0:
        logger.warning("⚠️ 미확인 주장 %d건 — 수동 검토 권장", fc_result["flagged_claims"])

    # 6. Hugo 빌드
    if not args.skip_build:
        if not step_hugo_build():
            logger.error("❌ Hugo 빌드 실패, 중단")
            sys.exit(1)

    # 6.5 최종 검수 (Final QA)
    logger.info("=" * 50)
    logger.info("6.5단계: 최종 검수")
    import re as _re
    with open(post_path, encoding="utf-8") as _f:
        _content = _f.read()
    _body = _re.sub(r"^---.*?---", "", _content, count=1, flags=_re.DOTALL)
    _sections = _body.split("\n## ")
    qa_issues = []
    
    # 문체 통일 검사
    for i, sec in enumerate(_sections[1:], 1):
        title = sec.split("\n")[0][:30]
        if "References" in title:
            continue
        casual = len(_re.findall(r"(?:이다|한다|된다|있다|없다|않다|왔다|간다)[.]", sec))
        if casual > 2:
            qa_issues.append(f"문체 불일치: 섹션 '{title}' 반말 {casual}건")
    
    # 잘림 검사
    code_blocks = _content.split("```")
    if len(code_blocks) % 2 == 0:
        qa_issues.append("미닫힌 코드 블록")
    if _content.rstrip().endswith(("(", "[", "](")):
        qa_issues.append("마지막 줄 잘림")
    
    # 이미지 경로 검사
    for img in _re.findall(r'!\[.*?\]\((.*?)\)', _content):
        if not img.startswith("/ai-tech-blog/"):
            qa_issues.append(f"이미지 prefix 누락: {img}")
    
    if qa_issues:
        for iss in qa_issues:
            logger.error("🚨 최종 검수 실패: %s", iss)
        logger.error("❌ 최종 검수 미통과 — 발행 중단!")
        sys.exit(3)
    else:
        logger.info("✅ 최종 검수 통과")

    # 7. Git push
    if not args.skip_push:
        step_git_push(config)

    # 8. 배포 후 QA 검수
    if not args.skip_push:
        logger.info("⏳ GitHub Actions 배포 대기 (90초)...")
        import time
        time.sleep(90)

        # 포스트 URL 생성
        with open(post_path, encoding="utf-8") as _f:
            _post_content = _f.read()
        _title_m = _re.search(r'^title:\s*"(.+?)"', _post_content, _re.MULTILINE)
        if _title_m:
            from generate_post import slugify
            _slug = slugify(_title_m.group(1))
            _date_m = _re.search(r'^date:\s*(\d{4}-\d{2}-\d{2})', _post_content, _re.MULTILINE)
            _date = _date_m.group(1) if _date_m else ""
            _post_url = f"https://jesamkim.github.io/ai-tech-blog/posts/{_date}-{_slug}/"

            logger.info("🔍 배포 후 QA 검수: %s", _post_url)
            try:
                import subprocess
                qa_cmd = [
                    sys.executable, "scripts/qa_published.py",
                    "--url", _post_url,
                    "--post", str(post_path),
                ]
                qa_result = subprocess.run(qa_cmd, capture_output=True, text=True, timeout=120)
                print(qa_result.stdout)
                if qa_result.returncode != 0:
                    logger.warning("⚠️ 배포 후 QA에서 이슈 발견 (수동 확인 필요)")
                    if qa_result.stderr:
                        logger.warning(qa_result.stderr[-500:])
                else:
                    logger.info("✅ 배포 후 QA 통과")
            except Exception as e:
                logger.warning("⚠️ QA 검수 실행 실패: %s", e)

    logger.info("✅ 파이프라인 완료: %s", post_path)


if __name__ == "__main__":
    main()
