#!/usr/bin/env python3
"""이미지 품질 검증"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
HUGO_DIR = BASE_DIR / "hugo-site"
LOGS_DIR = BASE_DIR / "logs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("qa_images")


def load_config() -> dict:
    with open(SCRIPTS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def validate_image(image_path: Path, config: dict) -> dict:
    """단일 이미지 검증. 결과 dict 반환."""
    qa_cfg = config.get("image_qa", {})
    min_w = qa_cfg.get("min_width", 600)
    min_h = qa_cfg.get("min_height", 400)
    min_size_kb = qa_cfg.get("min_file_size_kb", 1)
    max_size_mb = qa_cfg.get("max_file_size_mb", 5)

    result = {"path": str(image_path), "valid": True, "issues": []}

    if not image_path.exists():
        result["valid"] = False
        result["issues"].append("파일 없음")
        return result

    file_size = image_path.stat().st_size
    if file_size < min_size_kb * 1024:
        result["valid"] = False
        result["issues"].append(f"파일 크기 너무 작음: {file_size}B < {min_size_kb}KB")

    if file_size > max_size_mb * 1024 * 1024:
        result["valid"] = False
        result["issues"].append(f"파일 크기 너무 큼: {file_size/(1024*1024):.1f}MB > {max_size_mb}MB")

    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            w, h = img.size
            if w < min_w:
                result["valid"] = False
                result["issues"].append(f"너비 부족: {w}px < {min_w}px")
            if h < min_h:
                result["valid"] = False
                result["issues"].append(f"높이 부족: {h}px < {min_h}px")
            result["width"] = w
            result["height"] = h
            result["format"] = img.format
    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"이미지 손상: {e}")

    result["file_size_bytes"] = file_size
    return result


def qa_images_for_date(date_str: str = None, config: dict = None) -> list:
    """특정 날짜의 모든 포스트 이미지 검증"""
    if config is None:
        config = load_config()
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    img_dir = HUGO_DIR / "static" / "images" / "posts" / date_str
    if not img_dir.exists():
        logger.info("이미지 디렉토리 없음: %s", img_dir)
        return []

    results = []
    for img_file in img_dir.iterdir():
        if img_file.suffix.lower() in (".png", ".jpg", ".jpeg", ".svg", ".webp"):
            r = validate_image(img_file, config)
            if not r["valid"]:
                logger.warning("이미지 QA 실패: %s — %s", img_file.name, r["issues"])
            else:
                logger.info("이미지 QA 통과: %s (%dx%d)", img_file.name, r.get("width", 0), r.get("height", 0))
            results.append(r)

    # 결과 저장
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    out = LOGS_DIR / f"qa_images_{date_str}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("QA 결과 저장: %s", out)
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="이미지 품질 검증")
    parser.add_argument("--date", default=None, help="날짜 (YYYY-MM-DD)")
    parser.add_argument("--file", default=None, help="단일 이미지 파일 경로")
    args = parser.parse_args()

    config = load_config()
    if args.file:
        r = validate_image(Path(args.file), config)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        sys.exit(0 if r["valid"] else 1)
    else:
        results = qa_images_for_date(args.date, config)
        failed = [r for r in results if not r["valid"]]
        print(f"검증 완료: {len(results)}건 중 {len(failed)}건 실패")
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
