# LinkedIn Article Draft

## Post Text (EN) - Feed intro
AI agents are no longer just chatbots. They send emails, edit documents, query databases. But when I talk to enterprise customers, the response is always the same: "Sounds great. We can't use it."

Data sovereignty, network isolation, audit trails — the real barriers aren't technical. They're about governance.

I wrote about how to build the same AI agent capabilities inside AWS Private networks, where data never leaves your infrastructure.

## Post Text (KR) - Feed intro
AI 에이전트가 이메일을 보내고, 문서를 편집하고, DB를 조회하는 시대입니다. 하지만 엔터프라이즈 고객들의 반응은 늘 같습니다. "좋은데, 우리는 못 써요."

데이터 주권, 네트워크 격리, 감사 추적 — 진짜 장벽은 기술이 아니라 거버넌스입니다.

AWS Private 네트워크 안에서 같은 AI 에이전트 역량을 구축하는 방법을 정리했습니다.

---

## Article (EN)

# Enterprise AI agents: building securely on AWS Private networks

AI agents crossed a threshold in early 2026. They stopped being conversational tools and started doing real work — sending emails, editing documents, managing calendars, querying databases. SaaS AI agents now connect to Gmail, Google Drive, DocuSign, and Salesforce through plugin marketplaces. Spotify automated code migration across thousands of microservices using AI coding agents. The technology works.

But as a Solutions Architect at AWS, I hear a different story from enterprise customers. The reaction is almost universal: "The tech is impressive, but we can't use it."

## The enterprise gap

The barriers aren't technical — they're about governance.

**Data sovereignty** comes first. Financial institutions and government agencies don't allow internal data to flow to external SaaS platforms. Using external SaaS AI agents means sending documents to third-party servers, which can violate security policies.

**Network isolation** is another wall. Healthcare, defense, and some financial sectors operate systems that cannot communicate with the public internet at all. Calling external SaaS APIs is simply not an option in these environments.

**Audit and compliance** requirements are strict. GDPR, HIPAA, and financial regulations demand detailed records of who accessed what data and when. External SaaS platforms rarely provide the granularity enterprises need.

The technology direction is right. But for regulated industries, the deployment model needs to evolve — data sovereignty, audit trails, and network isolation aren't optional.

## Bring the agent to the data

The key insight is simple: instead of sending data to the agent, bring the agent to the data.

Most enterprises already store their critical data in AWS — documents in S3, transactions in RDS, analytics in Redshift. If an AI agent needs to work with this data, running the agent inside the same AWS environment is the most natural and secure approach.

Here's the architecture:

**Amazon Bedrock** handles model inference entirely within AWS. When you call Claude Sonnet 4.6 through Bedrock, both the request and response stay within the AWS network. Customer data is never used for model training. Data sovereignty requirements are met by design.

**Claude Code on Bedrock** extends this to coding agents. Setting  routes all Claude Code activity through Amazon Bedrock. Code generation, refactoring, and test writing all happen within AWS infrastructure. Source code never leaves your network boundary.

**Amazon Lightsail** provides a simple runtime for self-hosted AI agents like OpenClaw. A single Lightsail instance with Docker runs a fully autonomous agent for about 2/month. It executes terminal commands, makes Git commits, runs code reviews — capabilities similar to SaaS AI agents, but fully self-hosted.

**VPC + PrivateLink** keeps all traffic on the AWS backbone. Lightsail connects to the VPC via peering. Bedrock calls go through VPC Endpoints. No internet traversal.

**AWS Direct Connect** gives corporate users access through dedicated private circuits. From the network topology, the agent looks like an internal resource.

**MCP (Model Context Protocol)** connects the agent to internal systems — S3 for documents, RDS for structured data, OpenSearch for RAG-powered knowledge bases, DynamoDB for application state. Same concept as Cowork's plugins, but connecting to your own systems instead of external SaaS.

## Security and observability

Architecture alone isn't enough. Enterprises need traceability.

- **IAM** enforces least-privilege per agent. HR agent accesses only HR databases; finance agent accesses only accounting systems.
- **KMS** encrypts all conversation logs and tool call records at rest.
- **CloudTrail** records every Bedrock API call, S3 access, and RDS query with timestamps and IAM identity.
- **CloudWatch** monitors agent behavior in real-time — response times, error rates, API call frequency.
- **GuardDuty** detects anomalies — unusual API patterns or unexpected network connections.
- **AWS Config** tracks resource configuration changes over time.

When a compliance auditor asks "who accessed this customer data?", you have the CloudTrail logs ready. That level of traceability is difficult to achieve with external SaaS platforms.

## Start small, scale with confidence

AI agent adoption in the enterprise isn't a technology problem — it's a governance problem. The capabilities that SaaS AI agents have demonstrated are achievable within AWS Private networks, with the added benefits of complete data control, network isolation, and comprehensive audit trails.

OpenClaw on Lightsail is a practical starting point. Deploy an agent for a few dollars a month, connect it to internal tools via MCP, and validate the use case before scaling. The infrastructure for secure AI agents already exists. The question isn't whether to adopt AI agents — it's how to adopt them safely.

