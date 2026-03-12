# 블로그 포스트 작성 태스크

## 메타데이터
- 파일명: content/posts/2026-03-12-vlm-fine-tuning-guide-lora-qlora.md
- 카테고리: AI/ML 기술 심층분석
- date: 2026-03-12T10:00:00+09:00
- author: Jesam Kim
- tags: [VLM, Fine-tuning, LoRA, QLoRA, Qwen2.5-VL, Amazon Bedrock, SageMaker]
- description: 오픈소스 Vision Language Model을 LoRA/QLoRA로 파인튜닝해서 도메인 특화 비전 AI를 만드는 실전 가이드입니다. Qwen2.5-VL, InternVL 같은 최신 모델부터 AWS Bedrock Nova 파인튜닝까지 정리합니다.
- cover: images/cover-vlm-finetuning.png

## 검증된 팩트 (수치는 이것만 사용!)

### LoRA (Hu et al., 2021.06, arXiv:2106.09685)
- 저자: Edward J. Hu, Yelong Shen 등 (Microsoft Research)
- GPT-3 175B 대비: trainable parameters 10,000배 감소, GPU 메모리 3배 감소
- RoBERTa, DeBERTa, GPT-2, GPT-3에서 full fine-tuning과 동등 이상 성능
- 추론 시 추가 latency 없음 (adapter와 다름)
- ICLR 2022 accepted

### QLoRA (Dettmers et al., 2023.05, arXiv:2305.14314)
- 저자: Tim Dettmers, Artidoro Pagnoni 등
- 핵심: 4-bit NormalFloat (NF4) + Double Quantization
- 65B 모델을 단일 48GB GPU에서 파인튜닝 가능
- NeurIPS 2023 accepted

### Qwen2.5-VL (Alibaba Cloud Qwen Team)
- 모델 크기: 3B, 7B, 72B parameters
- 72B: MMBench-EN 88.6점
- ViT에 window attention + SwiGLU + RMSNorm 적용
- 기술 보고서: 2025.02.20 공개
- HuggingFace에서 QLoRA 파인튜닝 레시피 공식 제공

### InternVL3.5 (OpenGVLab, Shanghai AI Lab)
- WindowsAgentArena에서 Qwen2.5-VL-72B 대비 +8.3%
- 241B-A28B (MoE) 모델 포함

### Amazon Nova 파인튜닝
- Nova Pro, Lite: multimodal data(텍스트+이미지)로 Bedrock에서 SFT 파인튜닝 지원
- Nova 2 Lite: Bedrock + SageMaker 둘 다 지원
- Nova Micro: 텍스트 only 파인튜닝
- 기술 보고서: arxiv 2506.12103 (2025.03.17)
- 공식 문서: https://docs.aws.amazon.com/nova/latest/userguide/customize-fine-tune.html

### HuggingFace TRL + QLoRA 파인튜닝
- SFTTrainer로 VLM 파인튜닝 지원
- 공식 cookbook: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl
- Qwen2-VL-7B 예제 포함

### 일반 수치 (검색 결과 기반)
- 오픈소스 VLM: 5,000~50,000 예시로 GPT-4o급 성능 달성 가능 (Label Your Data 2026 가이드)
- LoRA 파인튜닝 컴퓨트 비용: $100~$5,000 (모델 크기, 데이터 양에 따라)

## 아웃라인 (7섹션)

1. 왜 VLM 파인튜닝인가 - 범용 VLM의 한계, 도메인 특화의 필요성
2. 오픈소스 VLM 현황 (2025~2026) - Qwen2.5-VL, InternVL3.5, Gemma 3 비교
3. LoRA와 QLoRA의 핵심 원리 - 왜 full fine-tuning 대신 쓰는지, 수학적 직관
4. 실전 파인튜닝 파이프라인 - 데이터 준비(이미지+대화쌍) -> HuggingFace TRL + QLoRA -> 학습 -> 평가
5. AWS에서의 VLM 파인튜닝 - Bedrock Nova Pro/Lite 멀티모달 + SageMaker 오픈소스 VLM
6. 도메인별 응용 시나리오 - 건설 현장 안전, 패션 상품 분류, 제조 불량 검출 (설계 가능성 톤)
7. 주의사항과 베스트 프랙티스 - 데이터 품질, 과적합, 평가 메트릭

## 규칙
- CJK bold: 모든 **한글**을 <strong>한글</strong>으로 변환
- 인라인 사이테이션 금지 ([1] 번호 없음)
- 구체적 수치에 [출처명](URL) 인라인 링크
- 수식: HTML sub/sup (LaTeX $ 금지)
- 이미지 경로: /ai-tech-blog/images/...
- description 존댓말
- em dash 최소화, AI 어투 금지
- 고객사명 비노출 (삼성물산->건설사, SSF몰->패션 리테일 등)
- 체험담 사칭 금지 - 설계 가능성 톤으로
- 제공된 수치만 사용. 출처 없는 수치 삽입 절대 금지!
- front matter에 cover 추가: cover: { image: "/ai-tech-blog/images/cover-vlm-finetuning.png" }
