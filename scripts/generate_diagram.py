#!/usr/bin/env python3
"""Mermaid 다이어그램 생성 + SVG→PNG 변환"""

import logging
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
HUGO_DIR = BASE_DIR / "hugo-site"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_diagram")

WATERMARK_TEXT = "https://jesamkim.github.io/ai-tech-blog"


def add_watermark(png_path: str):
    """PNG 이미지에 워터마크 추가 (좌측 하단, 반투명)"""
    try:
        from PIL import Image, ImageDraw, ImageFont

        img = Image.open(png_path).convert("RGBA")
        w, h = img.size

        # 워터마크 레이어
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # 폰트 크기: 이미지 너비의 ~2.5% (최소 10, 최대 16)
        font_size = max(10, min(16, int(w * 0.025)))
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", font_size)
            except (OSError, IOError):
                font = ImageFont.load_default()

        # 텍스트 크기 계산
        bbox = draw.textbbox((0, 0), WATERMARK_TEXT, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 좌측 하단, 약간의 패딩
        x = 8
        y = h - th - 8

        # 반투명 배경 + 텍스트
        padding = 4
        draw.rectangle([x - padding, y - padding, x + tw + padding, y + th + padding],
                       fill=(255, 255, 255, 160))
        draw.text((x, y), WATERMARK_TEXT, fill=(80, 80, 80, 200), font=font)

        # 합성 후 저장
        result = Image.alpha_composite(img, overlay)
        result = result.convert("RGB")
        result.save(png_path, "PNG")
        logger.info("워터마크 추가: %s", png_path)
    except Exception as e:
        logger.warning("워터마크 실패 (무시): %s", e)


def load_config() -> dict:
    with open(SCRIPTS_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


def generate_diagram(
    mermaid_text: str,
    output_name: str = "diagram",
    date_str: str = None,
    config: dict = None,
) -> str:
    """Mermaid 텍스트 → PNG 이미지 생성, 마크다운 경로 반환"""
    if config is None:
        config = load_config()
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    diag_cfg = config.get("diagrams", {})
    img_dir = HUGO_DIR / "static" / "images" / "posts" / date_str
    img_dir.mkdir(parents=True, exist_ok=True)
    output_png = img_dir / f"{output_name}.png"
    md_path = f"/ai-tech-blog/images/posts/{date_str}/{output_name}.png"

    # .mmd 파일 생성
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mmd", delete=False) as f:
        f.write(mermaid_text)
        mmd_path = f.name

    try:
        # 방법 1: mmdc (mermaid-cli)
        if shutil.which("mmdc"):
            cmd = [
                "mmdc", "-i", mmd_path, "-o", str(output_png),
                "-t", diag_cfg.get("theme", "default"),
                "-b", diag_cfg.get("background", "white"),
                "-w", str(diag_cfg.get("min_width", 600)),
                "-H", str(diag_cfg.get("min_height", 400)),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and output_png.exists():
                logger.info("mmdc로 다이어그램 생성: %s", output_png)
                add_watermark(str(output_png))
                return md_path
            logger.warning("mmdc 실패: %s", result.stderr)

        # 방법 2: mmdc via npx
        if shutil.which("npx"):
            cmd = [
                "npx", "-y", "@mermaid-js/mermaid-cli",
                "-i", mmd_path, "-o", str(output_png),
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and output_png.exists():
                logger.info("npx mmdc로 다이어그램 생성: %s", output_png)
                add_watermark(str(output_png))
                return md_path
            logger.warning("npx mmdc 실패: %s", result.stderr)

        # 방법 3: SVG via mermaid.ink API + cairosvg
        try:
            import base64
            import requests
            encoded = base64.urlsafe_b64encode(mermaid_text.encode()).decode()
            svg_url = f"https://mermaid.ink/svg/{encoded}"
            resp = requests.get(svg_url, timeout=30)
            resp.raise_for_status()
            svg_content = resp.content

            try:
                import cairosvg
                cairosvg.svg2png(bytestring=svg_content, write_to=str(output_png))
                logger.info("cairosvg로 다이어그램 생성: %s", output_png)
                add_watermark(str(output_png))
                return md_path
            except ImportError:
                # SVG로 저장
                svg_path = img_dir / f"{output_name}.svg"
                with open(svg_path, "wb") as f:
                    f.write(svg_content)
                logger.info("SVG로 저장 (cairosvg 미설치): %s", svg_path)
                return f"/images/posts/{date_str}/{output_name}.svg"
        except Exception as e:
            logger.error("mermaid.ink 실패: %s", e)

        logger.error("모든 다이어그램 생성 방법 실패")
        return ""

    finally:
        os.unlink(mmd_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mermaid 다이어그램 생성")
    parser.add_argument("--input", required=True, help="Mermaid .mmd 파일 경로")
    parser.add_argument("--name", default="diagram", help="출력 파일명")
    parser.add_argument("--date", default=None, help="날짜 (YYYY-MM-DD)")
    args = parser.parse_args()

    with open(args.input) as f:
        mermaid_text = f.read()

    result = generate_diagram(mermaid_text, args.name, args.date)
    if result:
        print(f"생성 완료: {result}")
    else:
        print("생성 실패", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    import sys
    main()
