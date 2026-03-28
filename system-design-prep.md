# Cohere SA — System Design Interview Prep

> 📅 **다음 라운드**: System Design Interview — **2026-03-25 (수) 19:30~20:15 KST** (Seungmin Lee, Google Meet)
> 📋 **형식**: Role-play (면접관=고객, Jay=SA). LLM 기반 솔루션 설계. 다이어그램 사용 권장.
> 🎯 **핵심 평가**: "Ensuring the application is **reliable and accurate** in a production setting"

---

## Answer Framework (모든 문제에 적용)

1. **Clarify Requirements** — 고객 요구사항 확인 (규모, SLA, 보안, 예산)
2. **High-Level Architecture** — 주요 컴포넌트 다이어그램
3. **Deep Dive** — 핵심 컴포넌트 상세 설계
4. **Trade-offs** — latency vs accuracy, cost vs quality
5. **Production Readiness** — 모니터링, fallback, 에러 핸들링, 평가

---

## Question 1 (기출): Post-Cutoff Knowledge System

### 🇺🇸 Problem
> "Design a mechanism for an LLM-based system that allows it to answer questions about events after its training cutoff, while maintaining reliability and transparency."

### Architecture

```
User Query
    ↓
[Query Analyzer] — classify: factual/opinion/temporal
    ↓
[Retrieval Router]
    ├── Real-time Search API (web, news)
    ├── Internal Knowledge Base (RAG)
    └── Structured DB (SQL, APIs)
    ↓
[Retrieved Documents] → [Reranker (Cohere Rerank v3)] → Top-K
    ↓
[LLM (Command R+)] — grounded generation with citations
    ↓
[Response Validator]
    ├── Citation check: every claim → source
    ├── Freshness check: source date vs query intent
    └── Confidence scoring
    ↓
[User Response] + source citations + confidence indicator
```

### Key Design Decisions

**Retrieval Module:**
- Hybrid retrieval: dense (Cohere Embed v4) + sparse (BM25) for best recall
- Real-time web search for breaking news (Tavily/Bing API)
- AWS: OpenSearch (hybrid) + Lambda (orchestration) + S3 (document store)

**Reasoning Module:**
- Command R+ with `grounded generation` mode — forces citation from retrieved docs
- System prompt: "Only answer using provided sources. If unsure, say so."
- Chain-of-Thought for multi-hop reasoning

**Response Validation:**
- NLI (Natural Language Inference) check: response entailed by sources?
- Temporal consistency: "Who is the US president?" needs 2026 source, not 2023
- Confidence threshold: below 0.7 → "I found relevant information but cannot confirm with high confidence"

**Communicating Uncertainty to Users:**
- ✅ High confidence: direct answer + citations
- ⚠️ Medium: "Based on available sources..." + citations
- ❌ Low: "I don't have reliable information about this. Here's what I found..."

### 🇰🇷 한국어 요약
학습 컷오프 이후 이벤트에 답변하는 시스템. 핵심은 3모듈: 검색(실시간 웹+RAG+DB) → 추론(Command R+ grounded generation) → 검증(인용 확인+시간 일관성+신뢰도 점수). 유저에게 불확실성을 신뢰도 수준별로 다르게 전달. 프로덕션에서는 NLI 기반 hallucination 탐지 + human feedback loop.

---

## Question 2: Enterprise RAG for Internal Documents

### 🇺🇸 Problem
> "A financial services company has 50,000 internal documents (policies, contracts, reports). They want employees to get accurate answers. How would you design this?"

### Architecture

```
[Document Ingestion Pipeline]
    PDF/DOCX → Chunking (semantic, 512 tokens) → Embed (Cohere Embed v4, 1024d)
    → Vector DB (OpenSearch Serverless / Pinecone)
    + Metadata index (department, date, doc_type)

[Query Pipeline]
    User Query → Embed (Embed v4, input_type=search_query)
    → ANN search (top-50) → Rerank (Cohere Rerank v3, top-5)
    → Command R+ (grounded generation, max_tokens=1024)
    → Citation extraction → Response

[Production Layer]
    Guardrails → PII detection → Access control (per-department)
    Monitoring → latency, retrieval quality, user feedback
    A/B testing → chunk size, reranker threshold, prompt variants
```

### Key Points

**Chunking Strategy:**
- Semantic chunking (not fixed-size): respects paragraph/section boundaries
- Overlap 50 tokens for context continuity
- Metadata enrichment: source document, page, section heading

**Why Reranking Matters:**
- Embedding search (bi-encoder): fast but approximate — good for recall
- Reranking (cross-encoder): slow but precise — good for precision
- 2-stage: retrieve 50 → rerank to 5 = best of both worlds
- Cohere Rerank v3 specifically trained for this

**Access Control:**
- Document-level ACL stored as metadata in vector DB
- Query-time filtering: user's department → only sees authorized docs
- No cross-tenant data leakage

**Evaluation:**
- Retrieval: nDCG@10, Recall@10 on test query set
- Generation: faithfulness (NLI), answer relevance, completeness
- Production: user thumbs up/down, escalation rate

### 🇰🇷 한국어 요약
5만건 내부 문서 RAG. 문서 → 시맨틱 청킹 → Embed v4 임베딩 → 벡터DB. 쿼리 시 ANN 검색(50건) → Rerank v3(5건) → Command R+ 답변 생성. 핵심은 접근 제어(부서별 문서 격리), 2단계 검색(recall→precision), 평가 파이프라인(nDCG, faithfulness, 유저 피드백).

---

## Question 3: Tool Calling + MCP Agent System

### 🇺🇸 Problem
> "A customer wants an LLM agent that can interact with internal systems — CRM, database, email. How do you design this?"

### Architecture

```
[User Request] → [Agent Orchestrator (Command R+ / North)]
    ↓
[Planning Module] — decompose task into steps
    ↓
[Tool Router] — select tools via MCP
    ├── CRM Tool (Salesforce API) — read/write contacts, deals
    ├── DB Tool (PostgreSQL) — read-only SQL queries
    ├── Email Tool (SMTP/Graph API) — compose, send (human approval)
    └── Search Tool (RAG) — internal knowledge base
    ↓
[Execution] — tool call → result → next step
    ↓
[Response Synthesis] — summarize results for user
    ↓
[Audit Log] — every tool call logged with input/output
```

### MCP Design

**Tool Registration:**
```json
{
  "name": "crm_lookup",
  "description": "Look up customer information by name or ID",
  "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
  "authentication": "oauth2",
  "permissions": ["read:contacts", "read:deals"]
}
```

**Authentication & Authorization:**
- OAuth2 per tool — user's identity propagated (not service account)
- Least privilege: read-only by default, write requires explicit grant
- Human-in-the-loop for destructive actions (delete, send email)

**Error Handling:**
- Tool timeout: 30s default, retry 1x, then graceful degradation
- Tool failure: "I couldn't access the CRM right now. Let me try another way."
- Rate limiting: per-user, per-tool quotas

**Production Reliability:**
- Circuit breaker pattern: 3 consecutive tool failures → disable tool + alert
- Audit trail: every tool invocation logged (compliance)
- Rollback: write operations are staged, not committed until user confirms

### 🇰🇷 한국어 요약
LLM 에이전트가 CRM, DB, 이메일 등 내부 시스템과 상호작용. MCP로 도구 등록/인증/권한 관리. 핵심은 OAuth2 사용자 신원 전파(서비스 계정 아님), human-in-the-loop(파괴적 작업), circuit breaker(연속 실패 시 도구 비활성화), 감사 로그.

---

## Question 4: SaaS vs PaaS vs On-Premise

### 🇺🇸 Problem
> "A Korean bank wants to use Cohere models but has strict data sovereignty requirements. What deployment options do you recommend?"

### Comparison

| Factor | SaaS (Cohere API) | PaaS (AWS Bedrock) | On-Premise (VPC/Private) |
|--------|-------------------|--------------------|----|
| **Setup time** | Hours | Days | Weeks |
| **Data residency** | Cohere's infra | AWS region (Seoul) | Customer's DC |
| **Compliance** | SOC2, GDPR | + ISMS, CSAP | Full control |
| **Cost** | Pay-per-token | Similar + AWS infra | High upfront, lower marginal |
| **Customization** | Fine-tuning API | Bedrock custom models | Full model access |
| **Latency** | Variable | Regional | Lowest |
| **Ops burden** | Zero | Low | High |

### Recommendation for Korean Bank

**Phase 1: PoC on AWS Bedrock** (PaaS)
- Cohere models available on Bedrock in ap-northeast-2 (Seoul)
- Data stays in AWS Seoul region → ISMS compliance
- Quick start, pay-per-use

**Phase 2: Production on Cohere Private Deployment**
- VPC deployment within customer's AWS account
- No data leaves their environment
- Cohere manages model updates

**Why not pure on-premise?**
- Bank doesn't need to run their own GPU cluster
- Cohere Private Deployment gives data sovereignty without ops burden
- Cost: ~30% more than SaaS, but compliance requirement justifies it

### 🇰🇷 한국어 요약
한국 은행의 데이터 주권 요구. SaaS(빠르지만 데이터 외부)/PaaS(AWS 서울 리전, ISMS 가능)/On-premise(완전 통제). 추천: PoC는 Bedrock(서울), 프로덕션은 Cohere Private Deployment(고객 VPC 안에 모델 배포). 순수 on-premise는 GPU 클러스터 운영 부담 대비 이점 적음.

---

## Question 5: Hallucination Detection Pipeline

### 🇺🇸 Problem
> "Your production LLM generates incorrect information. How do you detect and prevent this?"

### Pipeline

```
[Pre-Generation]
    Query → Retrieval Quality Check (are sources relevant?)
    → Query Rewriting (ambiguous → specific)
    → Source freshness validation

[Generation]
    Command R+ with grounded generation mode
    System prompt: "Only use provided sources. Cite every claim."
    Temperature: 0.0~0.3 for factual tasks

[Post-Generation]
    Response → [Faithfulness Checker]
        NLI model: does response entail from sources?
        Claim extraction → per-claim verification
    → [Factual Consistency]
        Cross-reference multiple sources
        Flag contradictions
    → [Confidence Scorer]
        Aggregate: retrieval quality + NLI score + claim coverage
        < 0.7 → add disclaimer
        < 0.4 → block response, escalate

[Runtime Monitoring]
    User feedback (👍/👎) → flagged responses → human review queue
    Drift detection: hallucination rate trending up?
    A/B test: prompt variants, temperature, retrieval params
```

### Cohere-Specific Features
- **Command R+ grounded mode**: model trained to cite sources inline
- **Rerank v3**: ensures high-quality retrieval (garbage in → garbage out prevention)
- Combine with custom NLI classifier for post-hoc verification

### 🇰🇷 한국어 요약
3단계 파이프라인: 생성 전(검색 품질 확인, 쿼리 재작성), 생성 중(grounded generation, 낮은 temperature), 생성 후(NLI 기반 faithfulness 체크, 주장별 검증, 신뢰도 점수). 신뢰도 0.7 미만이면 면책 조항 추가, 0.4 미만이면 응답 차단. 런타임에서 유저 피드백 + drift detection.

---

## Question 6: Embedding Model Selection & Optimization

### 🇺🇸 Problem
> "A customer asks which embedding model to use and what dimension/quantization settings. How do you advise?"

### Decision Framework

| Factor | Recommendation |
|--------|---------------|
| **Use case: search/RAG** | Cohere Embed v4 (input_type hints improve retrieval) |
| **Use case: clustering/classification** | Either works, Embed v4 slightly better |
| **Budget-sensitive** | 512 dimensions + int8 quantization (90% quality, 75% less storage) |
| **Max quality** | 1024 or 1536 dimensions + float32 |
| **Throughput** | Embed v4 (batch API, 96 texts/call) = 13x faster than Titan v2 |
| **Long documents** | Titan v2 (8K tokens) if docs > 2K tokens; else Embed v4 |
| **Korean-heavy** | Both good; Titan v2 slight edge on Korean retrieval per my benchmark |

### Dimension Ablation (from Jay's benchmark)

| Model | Dim | STS | Retrieval (nDCG@10) |
|-------|-----|-----|-----|
| Cohere v4 | 256 | 0.463 | 0.981 |
| Cohere v4 | 512 | 0.411 | **0.987** |
| Cohere v4 | 1024 | 0.449 | 0.981 |
| Cohere v4 | 1536 | **0.485** | **0.987** |

→ **For retrieval, 512d is enough** (nDCG 0.987 = near perfect). STS needs higher dims.

### Cost Impact
- 512d int8 vs 1536d float32: ~6x storage reduction
- For 1M documents: $50/month vs $300/month on vector DB
- **Recommendation: Start with 512d float, optimize to int8 if cost matters**

### 🇰🇷 한국어 요약
임베딩 모델 선택: 검색/RAG는 Embed v4(input_type 힌트, 배치 API), 예산 민감하면 512차원+int8. 내 벤치마크에서 512차원이 이미 nDCG 0.987 달성(1536과 동일). 저장비용 6배 절감. 한국어 무거운 경우 Titan v2도 고려.

---

## Question 7: Multi-Tenant LLM Service

### 🇺🇸 Problem
> "You're building a shared LLM platform for multiple enterprise customers. How do you handle data isolation and routing?"

### Architecture

```
[API Gateway] — per-tenant API key, rate limiting
    ↓
[Tenant Router] — route to tenant-specific config
    ├── Tenant A: Command R+ (custom fine-tuned), RAG from their docs only
    ├── Tenant B: Command R (standard), no RAG
    └── Tenant C: Embed v4 only (search use case)
    ↓
[Shared Inference Layer] — model serving (shared GPU pool)
    ↓
[Data Isolation Layer]
    Vector DB: namespace per tenant (no cross-query)
    Fine-tuned models: tenant-specific adapters (LoRA)
    Logs: tenant-partitioned, encrypted at rest
```

### Key Design Points
- **Data isolation**: vector DB namespaces, not separate clusters (cost-efficient)
- **Model routing**: tenant config → model + parameters + RAG sources
- **Rate limiting**: per-tenant token quotas + burst allowance
- **Cost attribution**: per-tenant usage metering → chargeback

### 🇰🇷 한국어 요약
멀티테넌트 LLM 서비스. API Gateway에서 테넌트별 키/속도 제한. 테넌트별 모델 라우팅(fine-tuned/표준). 벡터DB 네임스페이스로 데이터 격리(별도 클러스터 아닌 비용 효율적 방식). LoRA 어댑터로 테넌트별 맞춤. 사용량 측정 + 비용 배분.

---

## Question 8: Customer Support Bot with Grounding

### 🇺🇸 Problem
> "An e-commerce company wants a conversational AI for customer support. It must be accurate and never make up policies."

### Architecture

```
[User Message] → [Intent Classifier]
    ├── FAQ / Policy question → RAG pipeline
    ├── Order status → Tool call (Order DB)
    ├── Return request → Tool call (Return API) + Human approval
    └── Complex / Angry → Escalate to human agent

[RAG Pipeline]
    Query → Embed v4 → Policy KB (vector DB) → Rerank v3
    → Command R+ (grounded, system: "Only answer from policy docs")

[Conversation Memory]
    Short-term: last 10 turns in context window
    Long-term: customer profile + past interactions (summary)

[Safety Layer]
    PII redaction (before LLM)
    No price promises without DB lookup
    Escalation triggers: sentiment < -0.5, 3+ failed attempts
```

### 🇰🇷 한국어 요약
이커머스 고객지원봇. 의도 분류 → FAQ는 RAG, 주문조회는 도구호출, 반품은 사람 승인, 복잡한 건 에스컬레이션. 핵심: 정책 문서에서만 답변(grounded), PII 제거, 가격 약속 금지(DB 조회 필수), 감정 분석 기반 에스컬레이션.

---

## Question 9: LLM Evaluation in Production

### 🇺🇸 Problem
> "How do you evaluate your LLM system's quality in production?"

### Evaluation Stack

| Layer | Method | Metric |
|-------|--------|--------|
| **Offline** | Test set evaluation | nDCG, faithfulness, BLEU |
| **Pre-deploy** | LLM-as-Judge (Command R+ evaluates responses) | pass rate |
| **Online** | User feedback (👍/👎) | satisfaction rate |
| **Online** | Implicit signals (copy, follow-up questions) | engagement |
| **Periodic** | Human expert review (sample 100/week) | accuracy, safety |

### LLM-as-Judge Setup
```
System: You are an expert evaluator. Rate the response on:
1. Faithfulness (0-5): Does it only use information from sources?
2. Relevance (0-5): Does it answer the user's question?
3. Completeness (0-5): Does it cover all aspects?

Provide scores and brief justification.
```

### 🇰🇷 한국어 요약
5계층 평가: 오프라인(테스트셋), 배포 전(LLM-as-Judge), 온라인(유저 피드백), 온라인(암시적 신호), 주기적(전문가 리뷰). LLM-as-Judge는 Command R+로 faithfulness/relevance/completeness 평가.

---

## Question 10: Migration from OpenAI to Cohere

### 🇺🇸 Problem
> "A customer is using OpenAI GPT-4 API. They want to evaluate switching to Cohere. How do you approach this?"

### Migration Framework

**Phase 1: Assessment (1 week)**
- Current usage: API calls/day, tokens, use cases, cost
- Pain points: latency? cost? data privacy? vendor lock-in?
- Requirements: must-haves vs nice-to-haves

**Phase 2: Parallel Evaluation (2 weeks)**
- Run same queries through both APIs
- Compare: quality (LLM-as-Judge), latency, cost
- Embed v4 vs text-embedding-3: retrieval benchmark
- Rerank v3 vs no reranking: impact on RAG quality

**Phase 3: PoC Migration (2 weeks)**
- Swap API endpoint (OpenAI → Cohere)
- Prompt adaptation (system prompts may need tuning)
- Test edge cases: long context, multi-turn, tool calling

**Phase 4: Production Cutover**
- Canary deployment: 10% traffic → 50% → 100%
- Rollback plan: keep OpenAI API key active for 30 days

### Cohere Advantages to Highlight
- **Private deployment** (OpenAI doesn't offer)
- **Rerank v3** (OpenAI has no native reranker)
- **Embed v4 batch API** (13x throughput)
- **Cost**: Command R+ typically cheaper than GPT-4

### 🇰🇷 한국어 요약
OpenAI→Cohere 마이그레이션 4단계: 평가(현재 사용량/문제점) → 병렬 비교(같은 쿼리 양쪽 실행) → PoC(API 교체+프롬프트 튜닝) → 카나리 배포(10%→100%). Cohere 차별점: 프라이빗 배포, Rerank(OpenAI 없음), Embed v4 처리량, 비용.

---

# Appendix A: 핵심 용어 해설

## RAG (Retrieval-Augmented Generation)
**EN:** A technique where the LLM retrieves relevant documents from an external knowledge base before generating a response. This grounds the answer in actual data rather than relying solely on training knowledge. Reduces hallucination and enables up-to-date answers.

**KR:** LLM이 응답 생성 전에 외부 지식 베이스에서 관련 문서를 검색하는 기법. 학습된 지식에만 의존하지 않고 실제 데이터에 기반한 답변을 생성. 환각 감소 + 최신 정보 답변 가능.

## Embedding & Dimensions (임베딩 차원의 역할)
**EN:** Embeddings convert text into dense numerical vectors. The dimension (256, 512, 1024, 1536) determines how much information each vector carries. Higher dimensions capture more nuance but cost more storage and compute. For retrieval, 512 dimensions often suffice (nDCG > 0.98). For fine-grained similarity (STS), higher dimensions help.

**KR:** 텍스트를 밀집 수치 벡터로 변환. 차원 수(256~1536)는 각 벡터가 담는 정보량을 결정. 높을수록 뉘앙스 포착 좋지만 저장/연산 비용 증가. 검색용은 512차원이면 충분(nDCG 0.98+). 유사도 비교는 높은 차원이 유리.

## Quantization (양자화)
**EN:** Reducing the precision of embedding values to save storage. float32 (full precision) → int8 (256 levels) → binary (0 or 1). Each step roughly halves storage. Quality loss is minimal for retrieval tasks. Cohere Embed v4 supports all three; Titan v2 only supports float.

**KR:** 임베딩 값의 정밀도를 낮춰 저장 공간 절약. float32(전정밀) → int8(256단계) → binary(0 또는 1). 단계마다 저장공간 ~절반. 검색 작업에서 품질 손실 미미. Embed v4는 세 가지 모두 지원, Titan v2는 float만.

## Bi-Encoder vs Cross-Encoder (Reranking의 원리)
**EN:** **Bi-encoder**: encodes query and document independently into embeddings, then compares via cosine similarity. Fast (pre-computed doc embeddings) but approximate. **Cross-encoder** (reranker): takes query+document pair as input, produces a relevance score. Slow but much more accurate. Best practice: bi-encoder retrieves 50 candidates → cross-encoder reranks to top 5.

**KR:** **Bi-encoder**: 쿼리와 문서를 독립적으로 임베딩 → 코사인 유사도 비교. 빠르지만 대략적. **Cross-encoder**(리랭커): 쿼리+문서 쌍을 입력 → 관련성 점수 출력. 느리지만 훨씬 정확. 최적: bi-encoder로 50개 후보 검색 → cross-encoder로 5개 재순위.

## Vector Database & ANN (벡터 검색)
**EN:** Stores embeddings and enables fast similarity search. ANN (Approximate Nearest Neighbor) algorithms like HNSW and IVF trade a tiny accuracy loss for massive speed gains. HNSW: graph-based, good for < 10M vectors. IVF: cluster-based, scales to billions. AWS: OpenSearch Serverless (HNSW), also Pinecone, Weaviate.

**KR:** 임베딩 저장 + 빠른 유사도 검색. ANN(근사 최근접 이웃) 알고리즘: HNSW(그래프 기반, 1천만 벡터 이하 적합), IVF(클러스터 기반, 수십억 스케일). 아주 약간의 정확도를 희생하고 대규모 속도 확보.

## Grounded Generation / Faithfulness (근거 기반 생성)
**EN:** A generation approach where the LLM is constrained to only use information from provided sources. Cohere Command R+ has a built-in grounded generation mode that automatically cites sources. Faithfulness measures whether the response is factually supported by the retrieved documents.

**KR:** LLM이 제공된 출처의 정보만 사용하도록 제한하는 생성 방식. Command R+에 내장된 grounded generation 모드가 자동으로 출처 인용. Faithfulness는 응답이 검색된 문서에 사실적으로 뒷받침되는지 측정.

## Hallucination & Detection (환각 탐지)
**EN:** When an LLM generates plausible but incorrect information not supported by its sources or training data. Detection methods: NLI (Natural Language Inference) classifiers check if the response is entailed by sources, claim-level extraction and verification, confidence scoring. Prevention: grounded generation, low temperature, explicit "I don't know" instructions.

**KR:** LLM이 출처나 학습 데이터에 없는 그럴듯하지만 틀린 정보를 생성하는 현상. 탐지: NLI 분류기(응답이 출처에 함의되는지), 주장별 추출+검증, 신뢰도 점수. 방지: grounded generation, 낮은 temperature, "모르겠습니다" 명시적 지시.

## MCP (Model Context Protocol)
**EN:** An open protocol (by Anthropic) for connecting LLMs to external tools and data sources. Standardizes how agents discover, authenticate, and invoke tools. Think of it as "USB for AI" — any tool that speaks MCP can be plugged into any LLM that supports it. Cohere North also supports tool integration.

**KR:** LLM을 외부 도구/데이터에 연결하는 개방형 프로토콜(Anthropic 개발). 에이전트가 도구를 발견, 인증, 호출하는 방식을 표준화. "AI의 USB"라고 생각하면 됨 — MCP를 쓰는 도구는 MCP 지원 LLM에 플러그인 가능.

## Tool Calling / Function Calling (도구 호출)
**EN:** The LLM's ability to generate structured API calls instead of text. The model outputs a JSON with function name and parameters, the system executes it, and the result is fed back. Critical for agents that interact with databases, CRMs, APIs. Cohere Command R+ supports native tool use.

**KR:** LLM이 텍스트 대신 구조화된 API 호출을 생성하는 능력. 모델이 함수명+파라미터 JSON 출력 → 시스템이 실행 → 결과를 피드백. DB, CRM, API와 상호작용하는 에이전트에 필수.

## Pre-training vs Fine-tuning vs RLHF vs PEFT/LoRA
**EN:**
- **Pre-training**: Training from scratch on massive text corpus. Months, millions of dollars.
- **Fine-tuning**: Further training on domain-specific data. Days, thousands of dollars.
- **RLHF**: Reinforcement Learning from Human Feedback. Aligns model with human preferences.
- **PEFT/LoRA**: Parameter-Efficient Fine-Tuning. Only trains ~1% of parameters. Hours, hundreds of dollars. Best for enterprise customization.

**KR:**
- **Pre-training**: 대규모 텍스트로 처음부터 학습. 수개월, 수백만 달러.
- **Fine-tuning**: 도메인 데이터로 추가 학습. 수일, 수천 달러.
- **RLHF**: 인간 피드백 강화학습. 모델을 사람 선호에 맞춤.
- **PEFT/LoRA**: 파라미터 효율적 미세조정. 파라미터 ~1%만 학습. 수시간, 수백 달러. 엔터프라이즈 맞춤에 최적.

## Retrieval Metrics (검색 평가 지표)
**EN:**
- **nDCG@k** (Normalized Discounted Cumulative Gain): Measures ranking quality — are the most relevant docs at the top? Range 0~1, 1 is perfect.
- **MRR@k** (Mean Reciprocal Rank): Where does the first relevant result appear? 1/rank averaged.
- **Recall@k**: What fraction of all relevant docs are in the top-k results?

**KR:**
- **nDCG@k**: 순위 품질 측정 — 가장 관련 높은 문서가 상위에 있는가? 0~1, 1이 완벽.
- **MRR@k**: 첫 번째 관련 결과가 몇 번째에 나타나는가? 1/순위의 평균.
- **Recall@k**: 전체 관련 문서 중 top-k에 포함된 비율.

## Agentic AI / Agent Orchestration (에이전틱 AI)
**EN:** AI systems that can plan, use tools, and take actions autonomously to complete tasks. An orchestrator (like Cohere North) manages the agent's planning loop: understand goal → plan steps → execute tools → evaluate results → iterate. Key challenges: reliability, error recovery, human oversight.

**KR:** 계획하고, 도구를 사용하고, 자율적으로 행동해서 작업을 완수하는 AI 시스템. 오케스트레이터(North 등)가 계획 루프 관리: 목표 이해 → 단계 계획 → 도구 실행 → 결과 평가 → 반복. 핵심 과제: 안정성, 에러 복구, 사람 감독.

## Confidence Scoring / Uncertainty Quantification (신뢰도 점수)
**EN:** Estimating how confident the system is in its answer. Methods: retrieval quality score + generation probability + NLI faithfulness score. Used to decide: answer directly (high), add disclaimer (medium), or refuse to answer (low). Critical for production LLMs in regulated industries.

**KR:** 시스템이 답변에 얼마나 확신하는지 추정. 방법: 검색 품질 점수 + 생성 확률 + NLI faithfulness 점수. 용도: 직접 답변(높음), 면책 조항 추가(중간), 답변 거부(낮음). 규제 산업의 프로덕션 LLM에 필수.

---

# Appendix B: Cohere Product Cheat Sheet

| Product | Purpose | Competitor | Jay's Experience |
|---------|---------|------------|-----------------|
| **Command R+** | Enterprise LLM (RAG-optimized, grounded generation, tool use) | GPT-4, Claude 3.5 | Bedrock에서 사용, 고객 PoC |
| **Command A** | Lightweight model (faster, cheaper) | GPT-4o-mini, Haiku | Bedrock에서 테스트 |
| **Embed v4** | Text embeddings (search, RAG, clustering) | text-embedding-3, Titan v2 | **직접 벤치마크 수행!** |
| **Rerank v3** | Cross-encoder reranking for search | No direct competitor (unique strength) | 아키텍처에서 활용 |
| **North** | Agentic AI platform (enterprise agents) | OpenAI Assistants, Bedrock Agents | 관심 있는 제품 |
| **Aya** | Multilingual model family (23+ languages) | No direct competitor | 다국어 고객에 활용 가능 |

### Cohere's Key Differentiators
1. **Private Deployment**: VPC, on-prem, air-gapped — OpenAI can't match
2. **Rerank**: Only major provider with native cross-encoder reranker
3. **Enterprise DNA**: Built for enterprise from day one (not consumer-first)
4. **Grounded Generation**: Built-in citation in Command R+
5. **Deployment Flexibility**: API / AWS Bedrock / Azure / GCP / Private

---

# Appendix C: Interview Day Checklist

- [ ] Excalidraw 열어두기 (draw.excalidraw.com)
- [ ] 이 문서 마지막 복습
- [ ] Embed v4 벤치마크 탭 열어두기 (github.com/jesamkim/embed-bench)
- [ ] Cohere 최신 뉴스 확인
- [ ] 물, 이어폰, 조용한 환경
- [ ] 깔끔한 상의 (화상)
- [ ] 10개 문제 아키텍처 다이어그램 미리 한번 그려보기
- [ ] Manager-level 질문 대비 (Appendix D 복습)
- [ ] "Tell me about a time..." STAR 답변 3개 준비

---

# Appendix D: Manager-Level Interview Prep

> Chris Campbell = SA **Manager**. Staff SA (Seungmin Lee)와 다른 점:
> - System design 실력은 기본, **고객 대응 + 비즈니스 판단 + 팀 협업**도 봄
> - "How would you explain this to a non-technical VP?" 류 질문 가능
> - Trade-off 설명 시 **비용/시간/리스크** 비즈니스 관점 강조

---

## D1. Communication & Customer-Facing Phrases

### Explaining Architecture to Non-Technical Stakeholders
(비기술 임원에게 아키텍처 설명할 때)

**Opening the conversation:**
- "Let me walk you through how this works at a high level."
  (전체적으로 어떻게 동작하는지 설명드릴게요.)
- "Think of it this way — the system has three main parts."
  (이렇게 생각하시면 됩니다 — 시스템이 크게 세 부분으로 나뉩니다.)
- "Before I go into details, let me explain the big picture."
  (세부사항 전에 큰 그림부터 설명드리겠습니다.)

**Simplifying technical concepts:**
- "The search part finds the right documents. The AI part writes the answer using those documents."
  (검색 부분이 맞는 문서를 찾고, AI 부분이 그 문서를 써서 답변을 만듭니다.)
- "This is like a filter. It removes bad results and keeps good ones."
  (이건 필터 같은 겁니다. 나쁜 결과를 빼고 좋은 결과만 남깁니다.)
- "We added this step because without it, the AI sometimes makes things up."
  (이 단계를 넣은 이유는, 없으면 AI가 가끔 지어내기 때문입니다.)

### Discussing Trade-offs with Customers
(고객과 트레이드오프 논의할 때)

**Presenting options:**
- "There are two ways to do this. Option A is faster but costs more. Option B takes longer but saves money."
  (두 가지 방법이 있습니다. A는 더 빠르지만 비용이 높고, B는 더 오래 걸리지만 비용을 절약합니다.)
- "It depends on what matters most to you — speed, cost, or accuracy."
  (뭐가 가장 중요한지에 따라 다릅니다 — 속도, 비용, 정확도.)
- "My recommendation is to start with the simple approach, then add more as needed."
  (제 추천은 단순한 방식으로 시작하고, 필요에 따라 확장하는 겁니다.)

**Handling pushback:**
- "That's a good point. Let me think about that."
  (좋은 지적입니다. 생각해 보겠습니다.)
- "I understand your concern. Here's how we can address it."
  (우려 사항 이해합니다. 이렇게 해결할 수 있습니다.)
- "You're right that this adds cost. But without it, we risk [X]."
  (비용이 추가되는 건 맞습니다. 하지만 없으면 [X] 리스크가 있습니다.)

### Admitting Uncertainty (Honestly)
(모를 때 솔직하게)

- "I'm not 100% sure about that number. Let me check and get back to you."
  (그 수치는 확실하지 않습니다. 확인해서 다시 알려드리겠습니다.)
- "That's a great question. I haven't seen that use case before, but my best guess is..."
  (좋은 질문입니다. 그런 사례는 본 적 없지만, 제 판단으로는...)
- "I'd want to test that before I give a definite answer."
  (확실한 답을 드리기 전에 테스트해 보고 싶습니다.)

---

## D2. Business-Aware System Design Phrases

### Scoping the Problem
(문제 범위 정하기)

- "Before I start designing, can I ask a few questions about the requirements?"
  (설계 시작 전에 요구사항 몇 가지 여쭤봐도 될까요?)
- "How many users are we talking about? Hundreds or millions?"
  (사용자가 어느 정도인가요? 수백 명인지, 수백만 명인지?)
- "What's the budget for this? That changes the approach a lot."
  (예산이 어느 정도인가요? 그에 따라 접근 방식이 많이 달라집니다.)
- "Is there a deadline? If we need this in two weeks, the design looks different from a six-month plan."
  (기한이 있나요? 2주 안에 필요하면 6개월 계획과는 설계가 다릅니다.)

### Explaining Why You Chose Something
(왜 이 설계를 선택했는지)

- "I chose this approach because it's the simplest way to meet the requirements."
  (이 방법을 선택한 이유는 요구사항을 충족하는 가장 단순한 방법이기 때문입니다.)
- "This component is added because of [reason]." ← Jay 스타일!
  (이 컴포넌트는 [이유] 때문에 추가했습니다.)
- "We could make this more complex, but I don't think it's needed at this stage."
  (더 복잡하게 만들 수 있지만, 이 단계에서는 필요 없다고 봅니다.)
- "The main reason for this design is cost. Running GPUs 24/7 is expensive."
  (이 설계의 가장 큰 이유는 비용입니다. GPU를 24시간 돌리면 비쌉니다.)

### Talking About Production Readiness
(프로덕션 준비도)

- "This works in a PoC, but for production we need to add monitoring and error handling."
  (PoC에서는 동작하지만, 프로덕션에서는 모니터링과 에러 처리를 추가해야 합니다.)
- "The first thing I'd do in production is set up alerts. If the system goes down, we need to know right away."
  (프로덕션에서 가장 먼저 할 일은 알림 설정입니다. 시스템이 다운되면 바로 알아야 합니다.)
- "We should plan for failure. What happens when the database is down? We need a fallback."
  (장애를 대비해야 합니다. DB가 다운되면? 대안이 필요합니다.)

---

## D3. Manager-Specific Behavioral Questions

> STAR format: Situation → Task → Action → Result

### Q: "Tell me about a time you helped a customer make a difficult technical decision."
(고객의 어려운 기술 결정을 도운 경험)

**Jay's Answer (Samsung C&T AIPEX 사례):**
- **Situation**: "Samsung C&T wanted to process 1.9 petabytes of construction documents using AI. They had never used external cloud AI before — Samsung Group had no case of using outside GenAI in production."
  (삼성물산이 1.9PB 건설 문서를 AI로 처리하려 했습니다. 삼성그룹에서 외부 클라우드 GenAI를 프로덕션에 쓴 적이 없었습니다.)
- **Task**: "I needed to help them pick the right approach — build from scratch, use open source, or use a managed service like Bedrock with Claude."
  (직접 만들지, 오픈소스를 쓸지, Bedrock+Claude 같은 매니지드 서비스를 쓸지 결정을 도와야 했습니다.)
- **Action**: "I ran a PoC with their actual documents. I showed them the accuracy numbers — 96 out of 100 on their test set. I also showed the cost comparison: building their own would take 6 months and a team of 5, versus 3 months with 2 people using Bedrock."
  (실제 문서로 PoC를 진행했습니다. 정확도 96/100을 보여줬고, 비용 비교도 했습니다. 직접 만들면 6개월+5명, Bedrock 쓰면 3개월+2명.)
- **Result**: "They chose Bedrock. It went to production and saved them about 40 billion won per year. Processing time dropped from 10 days to 3 days. It became a reference case for the whole Samsung Group."
  (Bedrock을 선택했고, 프로덕션 투입 후 연 400억원 절감. 처리 시간 10일→3일. 삼성그룹 전체 레퍼런스가 됐습니다.)

### Q: "How do you handle it when a customer wants something that's not the right solution?"
(고객이 잘못된 솔루션을 원할 때 어떻게 하는지)

**Jay's Answer (삼성물산 패션부문 사례):**
- **Situation**: "Samsung C&T Fashion wanted to migrate from Snowflake. The Snowflake team was fighting hard to keep them."
  (삼성물산 패션부문이 Snowflake에서 이전하려 했습니다. Snowflake 팀이 고객을 지키려 강하게 대응했습니다.)
- **Task**: "The customer initially wanted a simple lift-and-shift, but their real problem was latency — their Aurora PostgreSQL had a different issue."
  (고객은 단순 이전을 원했지만, 진짜 문제는 레이턴시 — Aurora PostgreSQL 쪽에 다른 이슈가 있었습니다.)
- **Action**: "Instead of just doing what they asked, I looked at the actual problem first. I found the Aurora performance issue, fixed it with DMS CDC — latency dropped 80%. Then I showed them why their Snowflake migration plan needed changes too."
  (요청대로 하지 않고 실제 문제부터 봤습니다. Aurora 성능 이슈를 찾아서 DMS CDC로 해결 — 레이턴시 80% 감소. 그 다음 Snowflake 이전 계획도 왜 수정이 필요한지 보여줬습니다.)
- **Result**: "The customer trusted me more because I solved their real problem, not just the one they asked about. We won against Snowflake."
  (고객이 물어본 것만 해결하지 않고 실제 문제를 해결해서 신뢰가 높아졌습니다. Snowflake 경쟁에서 이겼습니다.)

### Q: "How do you stay up to date with a fast-moving field like AI?"
(빠르게 변하는 AI 분야에서 어떻게 최신 상태를 유지하는지)

**Simple answer:**
- "I read papers, but I also build things. I have a personal trading bot that uses LLMs — that teaches me more than reading papers."
  (논문도 읽지만, 직접 만들기도 합니다. LLM을 사용하는 개인 트레이딩 봇을 운영 중인데, 이게 논문보다 더 많이 가르쳐줍니다.)
- "I'm doing a Master's in AI at Yonsei right now. That helps me understand the theory behind the tools."
  (현재 연세대 AI 대학원 석사 과정입니다. 도구 뒤의 이론을 이해하는 데 도움이 됩니다.)
- "I also write technical blog posts and give talks at AWS Summit. Teaching is the best way to learn."
  (기술 블로그도 쓰고 AWS Summit에서 발표도 합니다. 가르치는 게 가장 좋은 학습 방법입니다.)

### Q: "Why Cohere? Why leave AWS?"
(왜 Cohere? 왜 AWS를 떠나는지)

**Simple answer:**
- "I've been at AWS for 4 years and learned a lot. But I want to be closer to the model side — not just the infrastructure."
  (AWS에서 4년간 많이 배웠습니다. 하지만 인프라가 아닌 모델 쪽에 더 가까이 가고 싶습니다.)
- "Cohere builds great models and focuses on enterprise. That matches what I do — I help enterprises use AI."
  (Cohere는 좋은 모델을 만들고 엔터프라이즈에 집중합니다. 제가 하는 일 — 기업의 AI 활용을 돕는 것과 맞습니다.)
- "I like that Cohere offers private deployment. In Korea, data sovereignty is a big deal, and I've seen customers need that option."
  (Cohere의 프라이빗 배포가 좋습니다. 한국에서 데이터 주권은 큰 이슈이고, 고객들이 이 옵션을 필요로 하는 걸 봤습니다.)
- "Also, I used Cohere's models on Bedrock — Embed v4, Rerank v3. I've benchmarked them. I know the product because I've used it."
  (또한 Bedrock에서 Cohere 모델을 써봤습니다 — Embed v4, Rerank v3. 벤치마크도 했고요. 제품을 써봤기 때문에 잘 압니다.)

---

## D4. Useful Transition Phrases During System Design
(시스템 설계 중 유용한 전환 표현)

### Moving between sections:
- "OK, so that's the high-level design. Let me go deeper into the search part."
  (네, 이게 전체 설계입니다. 검색 부분을 더 깊이 들어가겠습니다.)
- "Let me step back for a moment. I want to make sure we agree on the requirements."
  (잠깐 한 발 물러서겠습니다. 요구사항에 동의하는지 확인하고 싶습니다.)
- "Before I move on, does this part make sense? Any questions so far?"
  (넘어가기 전에, 이 부분 이해가 되시나요? 지금까지 질문 있으신가요?)

### When drawing a diagram:
- "Let me draw this out so it's easier to follow."
  (그림으로 그려서 보여드리겠습니다.)
- "Here's the flow: user sends a query, it goes to the router, then..."
  (흐름은 이렇습니다: 유저가 쿼리를 보내면, 라우터로 가고, 그 다음...)
- "I'll add the database here. This stores the embeddings."
  (여기에 데이터베이스를 추가할게요. 여기에 임베딩이 저장됩니다.)

### Handling hard questions:
- "That's a good question. Let me think about it for a second."
  (좋은 질문입니다. 잠깐 생각해 보겠습니다.)
- "I haven't faced that exact case, but I think the approach would be..."
  (그 정확한 사례를 겪어보진 않았지만, 접근 방식은 이럴 것 같습니다...)
- "Honestly, I'm not sure about the best approach here. But if I had to choose, I'd go with [X] because..."
  (솔직히 여기서 최적의 방법을 모르겠습니다. 하지만 선택해야 한다면 [X]를 고르겠습니다. 왜냐하면...)

### Wrapping up:
- "So to summarize: the system has three parts — search, generation, and validation."
  (정리하면: 시스템은 세 부분 — 검색, 생성, 검증입니다.)
- "If I had more time, I'd also add [X]. But for a first version, this covers the main requirements."
  (시간이 더 있다면 [X]도 추가하겠습니다. 하지만 첫 버전으로는 주요 요구사항을 충족합니다.)
- "The key trade-off here is [X] versus [Y]. I chose [X] because of [reason]."
  (여기서 핵심 트레이드오프는 [X] 대 [Y]입니다. [이유] 때문에 [X]를 선택했습니다.)
