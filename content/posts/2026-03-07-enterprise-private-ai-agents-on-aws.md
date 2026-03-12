---
title: "엔터프라이즈 AI 에이전트, AWS Private 환경에서 시큐어하게 구축하기"
date: 2026-03-07T10:00:00+09:00
description: "Claude Cowork가 보여준 AI 에이전트 트렌드는 맞지만, 엔터프라이즈는 보안과 거버넌스 요구사항이 있습니다. AWS Private 네트워크 환경에서 같은 컨셉을 옵저버빌리티와 함께 구축하는 방법을 살펴봅니다."
categories: ["AWS AI/ML"]
tags: ["AI Agent", "Claude Cowork", "OpenClaw", "Amazon Bedrock", "Amazon Lightsail", "MCP", "Enterprise AI", "Security"]
author: "Jesam Kim"
cover:
  image: "/ai-tech-blog/images/cover-enterprise-private-ai-agents-on-aws.png"
---

## 2026년, AI 에이전트가 도구를 쓰기 시작했다

AI가 질문에 답하는 걸 넘어 실제 업무 도구를 사용하기 시작했습니다. 이메일을 보내고, 문서를 편집하고, 캘린더를 관리합니다. Anthropic의 Claude Cowork는 Gmail, Google Drive, DocuSign 같은 서비스를 플러그인으로 연결해서 반복 작업을 자동화합니다. Spotify는 고객 지원 티켓을 AI 에이전트로 처리하고 있고, Novo Nordisk는 내부 문서 검색과 요약에 활용합니다.

AWS가 2026년 3월 발표한 OpenClaw on Lightsail도 비슷한 방향입니다. 터미널 명령을 실행하고, Git 커밋을 만들고, 코드 리뷰를 진행하는 자율 AI 에이전트를 Lightsail 인스턴스 하나로 띄울 수 있습니다. 한 달에 몇 달러면 팀 전용 AI 개발자를 둘 수 있는 셈입니다.

<strong>AI 에이전트가 단순 대화를 넘어 완성된 결과물을 만들어낸다</strong>는 게 핵심입니다.

## Claude Cowork이 보여준 가능성

Claude Cowork는 2026년 2월에 발표됐습니다. 플러그인 마켓플레이스가 있어서 Gmail, Slack, Google Drive, DocuSign, Salesforce 같은 도구를 바로 연결할 수 있습니다. Model Context Protocol(MCP)이라는 표준 인터페이스로 외부 시스템과 통신합니다.

Anthropic이 공개한 실제 활용 사례를 보면:
- <strong>Spotify</strong>: 수천 개 마이크로서비스의 코드 마이그레이션을 Claude Agent SDK로 자동화했습니다. 엔지니어가 자연어로 "이 라이브러리를 최신 버전으로 업데이트해줘"라고 요청하면, Claude가 코드베이스 전체를 분석하고 수정합니다.
- <strong>Salesforce</strong>: 내부 업무 자동화에 Claude를 활용하고 있습니다.
- <strong>금융 서비스 분야</strong>: 계약서 초안 작성, 컴플라이언스 문서 검토, 재무 분석 등에 플러그인 템플릿을 적용합니다.

Claude Cowork의 Recurring Tasks 기능은 특히 유용합니다. "매주 월요일 오전 9시에 지난주 Jira 이슈를 요약해서 Slack에 올려줘" 같은 작업을 한 번 설정하면 계속 돌아갑니다.

VentureBeat 기사에 따르면 Anthropic은 Claude Code로 개발 도구 시장을, Cowork로 엔터프라이즈 자동화 시장을 노리고 있습니다. 기술적으로는 가능한 이야기입니다.

## 엔터프라이즈 현실은 다르다

하지만 실제 엔터프라이즈 고객들과 얘기하면 다른 반응이 나옵니다. "좋은데, 우리는 못 써요."

<strong>데이터 주권</strong>이 첫 번째 문제입니다. 금융권이나 공공 기관은 내부 데이터가 외부 SaaS로 나가는 걸 허용하지 않습니다. Claude Cowork를 쓰려면 문서를 Anthropic 서버로 보내야 하는데, 그 자체가 보안 정책 위반입니다.

<strong>네트워크 격리</strong>도 있습니다. 특정 산업(의료, 국방, 일부 금융)은 민감한 시스템이 인터넷과 직접 통신하지 못하게 막습니다. Claude Cowork 같은 외부 API를 호출하는 것 자체가 불가능한 환경입니다.

<strong>감사와 컴플라이언스</strong> 요구도 까다롭습니다. GDPR, HIPAA, 금융감독 규정 등은 "누가 언제 어떤 데이터에 접근했는지" 상세히 기록하도록 요구합니다. 외부 SaaS는 이런 로그를 제공하지 않거나 세밀한 제어가 어렵습니다.

<strong>커스텀 거버넌스</strong>도 필요합니다. 부서마다 다른 권한을 줘야 하고, 특정 모델만 사용하도록 제한해야 하고, 비용 한도를 설정해야 합니다. SaaS 플랫폼은 이런 세밀한 정책을 지원하지 않는 경우가 많습니다.

SaaS AI 에이전트의 기술 방향은 맞습니다. 다만 엔터프라이즈 환경에서는 데이터 주권, 감사 로그, 네트워크 격리 같은 요구사항을 추가로 충족해야 합니다.

## AWS Private 환경에서의 AI 에이전트 아키텍처

같은 컨셉을 AWS 안에서 구현하면 어떨까요? 데이터가 외부로 나가지 않고, 모든 통신이 Private 네트워크에서 완결되고, 감사 로그가 자동으로 남는 환경 말입니다.

사실 많은 엔터프라이즈가 이미 <strong>핵심 데이터를 AWS에 저장</strong>하고 있습니다. S3에 문서를 보관하고, RDS에 트랜잭션 데이터를 넣고, Redshift로 분석합니다. 이 데이터를 AI 에이전트가 활용하려면, <strong>데이터가 있는 곳에서 에이전트를 운영하는 게 가장 자연스럽고 안전</strong>합니다.

데이터를 외부 SaaS로 보내는 대신, <strong>에이전트를 데이터 옆에 놓는 발상</strong>입니다. 이미 AWS 안에 있는 자산을 그대로 활용하면서, Private 네트워크 경계를 넘지 않는 구조입니다.

AWS Private 환경에서 AI 에이전트를 구축하면 다음과 같은 구조가 됩니다:

![AWS Private AI Agent Architecture Overview](/ai-tech-blog/images/saas-vs-private-comparison.png)

컨셉추얼 아키텍처는 다음과 같습니다:

![Enterprise Private AI Agent Architecture](/ai-tech-blog/images/enterprise-ai-agent-architecture.png)

### Amazon Bedrock: 모델 추론이 AWS 내에서 완결

Amazon Bedrock은 Claude, Nova2, Llama 같은 Foundation 모델을 API로 제공하는 관리형 서비스입니다. 중요한 점은 <strong>추론이 AWS 인프라 내에서 이뤄지고, 고객 데이터가 모델 학습에 사용되지 않는다</strong>는 겁니다.

Claude Sonnet 4.6을 Bedrock으로 호출하면 요청과 응답이 모두 AWS 네트워크 안에 머뭅니다. Anthropic 외부 API를 거치지 않습니다. 데이터 주권 요구사항을 만족시킬 수 있는 구조입니다.

### Claude Code + Bedrock: 코딩 에이전트도 Private 환경에서

대화형 AI뿐 아니라 <strong>코딩 에이전트</strong>도 같은 원리로 Private 환경에서 운영할 수 있습니다. Anthropic의 Claude Code는 터미널에서 직접 코드를 작성하고, 테스트를 실행하고, Git 커밋까지 하는 자율 코딩 에이전트입니다.

핵심은 `CLAUDE_CODE_USE_BEDROCK=1` 환경 변수 하나로 <strong>Amazon Bedrock을 백엔드로 전환</strong>할 수 있다는 점입니다. 이렇게 설정하면 코드 생성, 리팩토링, 테스트 작성 같은 모든 개발 작업이 AWS 인프라 안에서 처리됩니다. 소스 코드가 Anthropic 외부 API를 거치지 않으므로, 민감한 코드베이스를 다루는 기업에서도 안심하고 사용할 수 있습니다.

EC2나 Lightsail 인스턴스에 Claude Code를 설치하고 Bedrock 연동하면, 개발팀 전용 AI 코딩 어시스턴트를 VPC 내부에서 운영할 수 있습니다. IAM Role로 Bedrock 접근 권한을 제어하고, CloudTrail로 모든 API 호출을 감사하는 것도 동일합니다.

### Amazon Lightsail: 에이전트 런타임

OpenClaw는 Claude Code와 비슷한 자율 AI 에이전트인데, 셀프 호스팅이 가능합니다. Lightsail 인스턴스 하나에 Docker 컨테이너로 띄우면 됩니다. 월 $12 정도로 시작할 수 있습니다.

EC2로 띄워도 되지만, Lightsail이 더 간단합니다. AWS 블로그에서 권장하는 4GB 메모리 플랜 기준 월 $22 정도입니다. 고정 IP, SSH 접속, 방화벽 설정이 콘솔 몇 번 클릭으로 끝납니다. 프로덕션 규모가 커지면 그때 ECS나 EKS로 마이그레이션해도 됩니다.

### VPC + PrivateLink: 모든 통신이 AWS 내부망에서 완결

Lightsail 인스턴스를 VPC와 연결(VPC Peering)하고, Bedrock은 VPC Endpoint로 호출하면 인터넷을 거치지 않습니다. 모든 트래픽이 AWS 백본 네트워크 안에서만 움직입니다.

PrivateLink를 쓰면 네트워크 격리 요구사항도 만족시킬 수 있습니다. S3, RDS, 사내 API 모두 Private 엔드포인트로 접근하면 됩니다.

<strong>사내 사용자는 AWS Direct Connect를 통해 에이전트에 접근</strong>할 수 있습니다. Public 인터넷을 경유하지 않고, 전용 회선으로 기업 데이터센터와 AWS VPC를 직접 연결합니다. 이미 Direct Connect를 사용 중인 기업이라면 <strong>추가 인프라 없이 바로 연결</strong>할 수 있습니다. 네트워크 토폴로지상 에이전트는 사내 리소스처럼 보입니다.

### MCP(Model Context Protocol): 사내 도구 연동

Claude Cowork가 Gmail, DocuSign을 플러그인으로 쓰는 것처럼, OpenClaw도 MCP로 외부 도구를 연동합니다. 차이는 연동 대상이 사내 시스템이라는 점입니다.

다음과 같은 AWS 데이터 소스를 MCP 서버로 만들어서 에이전트에 연결할 수 있습니다:

- <strong>Amazon S3</strong>: 문서 저장소에서 계약서, 매뉴얼, 보고서 등을 읽어올 수 있습니다.
- <strong>Amazon RDS</strong>: 트랜잭션 데이터베이스에서 고객 정보, 주문 내역, 재고 현황 등을 쿼리할 수 있습니다.
- <strong>Amazon OpenSearch Service</strong>: Bedrock Knowledge Base와 연동하여 RAG(Retrieval-Augmented Generation) 구현이 가능합니다. 대량의 기업 문서를 벡터 검색으로 찾아내고, 관련 컨텍스트를 LLM에 전달하여 더 정확한 답변을 생성할 수 있습니다.
- <strong>Amazon DynamoDB</strong>: 애플리케이션 데이터(사용자 프로필, 세션 정보, 캐시 데이터 등)에 빠르게 접근할 수 있습니다.
- <strong>사내 REST API</strong>: 기존 마이크로서비스나 레거시 시스템도 MCP로 래핑하면 AI 에이전트가 호출할 수 있습니다.

OpenClaw 공식 문서에 MCP 서버 작성 가이드가 있습니다. 동작 방식은 간단합니다. 에이전트가 Bedrock(Claude)에 질문을 보냅니다. Claude가 "OpenSearch Knowledge Base에서 관련 문서를 검색해야겠네"라고 판단하면 MCP 호출을 요청합니다. 에이전트는 MCP 서버를 통해 OpenSearch API를 호출하고 검색 결과를 Claude에 다시 보냅니다. 모든 통신이 VPC 안에서만 일어납니다.

Claude Cowork와 로직은 같습니다. 차이는 데이터가 밖으로 나가지 않는다는 점입니다.

## 시큐리티와 옵저버빌리티

아키텍처만 Private으로 만든다고 끝이 아닙니다. 엔터프라이즈가 요구하는 건 <strong>추적 가능성</strong>입니다. 무엇이 언제 어떻게 일어났는지 감사할 수 있어야 합니다.

### IAM: 에이전트별 최소 권한 원칙

OpenClaw 인스턴스에 IAM Role을 붙입니다. 이 Role은 Bedrock 호출 권한, S3 특정 버킷 읽기 권한, RDS 특정 테이블 접근 권한만 가집니다. 필요 없는 권한은 주지 않습니다.

에이전트가 여러 개면 각각 다른 Role을 씁니다. HR 에이전트는 인사 데이터베이스만, 재무 에이전트는 회계 시스템만 접근하도록 격리할 수 있습니다.

### KMS: 대화 로그와 도구 호출 기록 암호화

에이전트가 처리한 대화 내용과 도구 호출 기록을 S3에 저장할 때 KMS로 암호화합니다. 키 접근 권한도 IAM으로 제어하면 특정 사람만 로그를 볼 수 있습니다.

### CloudTrail: API 호출 감사 추적

Bedrock API 호출, S3 접근, RDS 쿼리 모두 CloudTrail에 기록됩니다. "2026년 3월 6일 15:32에 어떤 IAM Role이 어떤 Bedrock 모델을 호출했는지" 추적할 수 있습니다.

금융감독 요구사항이나 GDPR 감사에서 "이 고객 데이터에 누가 접근했나요?"라는 질문을 받으면 CloudTrail 로그를 제출하면 됩니다.

### CloudWatch: 에이전트 동작 모니터링

OpenClaw는 로그를 표준 출력으로 내보냅니다. 이걸 CloudWatch Logs로 보내면 에이전트가 뭘 하고 있는지 실시간으로 볼 수 있습니다. 응답 시간, 에러율, API 호출 빈도 같은 메트릭도 수집할 수 있습니다.

"Bedrock API 응답 시간이 2초 넘으면 알림"처럼 CloudWatch Alarm을 설정하면 성능 이슈를 조기에 감지할 수 있습니다.

### Amazon GuardDuty: 위협 탐지

GuardDuty는 VPC 트래픽, CloudTrail 로그, DNS 쿼리를 분석해서 이상 징후를 찾습니다. 에이전트가 갑자기 평소와 다른 API를 대량으로 호출하거나, 알 수 없는 외부 IP와 통신하려 하면 알림이 옵니다.

### AWS Config: 리소스 구성 변경 추적

Config는 리소스 설정 변경을 기록합니다. "누가 언제 IAM Role 권한을 수정했는지", "S3 버킷 정책이 언제 바뀌었는지" 타임라인으로 볼 수 있습니다. 컴플라이언스 규칙을 설정하면 자동으로 체크도 해줍니다.

이런 도구들을 조합하면 Claude Cowork 같은 SaaS보다 훨씬 세밀한 제어와 추적이 가능합니다.

## AI 에이전트 도입은 기술이 아니라 거버넌스의 문제

Claude Cowork가 보여준 방향은 맞습니다. AI 에이전트가 반복 작업을 자동화하고, 여러 도구를 엮어서 복잡한 워크플로우를 처리하는 시대입니다.

하지만 엔터프라이즈 도입의 열쇠는 기술 구현이 아니라 <strong>안전한 운영 환경</strong>입니다. 데이터 주권, 네트워크 격리, 감사 추적, 거버넌스 같은 요구사항을 만족시켜야 실제로 쓸 수 있습니다.

AWS는 그 기반을 이미 갖추고 있습니다. Bedrock으로 모델을 Private 환경에서 호출하고, Lightsail/EC2로 에이전트를 셀프 호스팅하고, VPC+PrivateLink로 네트워크를 격리하고, CloudTrail/CloudWatch/GuardDuty로 추적과 모니터링을 할 수 있습니다.

OpenClaw on Lightsail은 좋은 시작점입니다. 월 몇 달러로 AI 에이전트를 띄워보고, MCP로 사내 도구를 연동해보고, 실제 업무에 적용할 수 있는지 테스트해볼 수 있습니다. 작게 시작해서 점진적으로 확장하는 게 현실적인 접근입니다.

AI 에이전트 시대는 이미 시작됐습니다. 이제 중요한 건 어떻게 안전하게 도입하느냐입니다.

## References

- AWS Blog: Introducing OpenClaw on Amazon Lightsail to run your autonomous private AI agents
  https://aws.amazon.com/ko/blogs/aws/introducing-openclaw-on-amazon-lightsail-to-run-your-autonomous-private-ai-agents/

- AWS Lightsail OpenClaw Quick Start Guide
  https://docs.aws.amazon.com/lightsail/latest/userguide/amazon-lightsail-quick-start-guide-openclaw.html

- VentureBeat: Anthropic says Claude Code transformed programming, now Claude Cowork is
  https://venturebeat.com/orchestration/anthropic-says-claude-code-transformed-programming-now-claude-cowork-is

- TechCrunch: Anthropic launches new push for enterprise agents with plugins for finance, engineering and design
  https://techcrunch.com/2026/02/24/anthropic-launches-new-push-for-enterprise-agents-with-plugins-for-finance-engineering-and-design/

- Model Context Protocol Official Documentation
  https://modelcontextprotocol.io/

- Amazon Bedrock User Guide
  https://docs.aws.amazon.com/bedrock/latest/userguide/

- OpenClaw Official Documentation
  https://docs.openclaw.ai/

- OpenClaw on Lightsail Security Guide
  https://docs.openclaw.ai/gateway/security

- Spotify Case Study: Claude Agent SDK
  https://claude.com/customers/spotify

- Claude Code on Amazon Bedrock
  https://code.claude.com/docs/en/amazon-bedrock

- Claude Cowork Safety Guide (Audit Logs limitation)
  https://support.claude.com/en/articles/13345190-get-started-with-cowork
