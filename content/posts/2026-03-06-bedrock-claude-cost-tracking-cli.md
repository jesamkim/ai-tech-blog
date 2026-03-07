---
title: "Amazon Bedrock Claude 비용 추적 — CloudWatch 메트릭으로 만드는 경량 CLI"
date: 2026-03-06T10:00:00+09:00
description: "AWS Cost Explorer는 서비스 레벨만 보여주고, Bedrock은 토큰 단위 과금인데 모델별 세분화가 안 됩니다. CloudWatch에 숨어있는 Bedrock 메트릭을 Python으로 직접 조회하여 일별/모델별 비용을 추적하는 경량 CLI를 만들었습니다."
categories: ["AWS AI/ML"]
tags: ["Amazon Bedrock", "Claude", "Cost Tracking", "CloudWatch", "Prompt Caching", "Python CLI", "FinOps"]
author: "Jesam Kim"
---

## 1. 왜 Bedrock 비용 추적이 어려운가

Claude 같은 Foundation Model을 프로덕션에서 쓰면 비용이 얼마나 나올지 궁금해집니다. 특히 Prompt Caching을 켜면 캐시 히트율에 따라 비용 구조가 복잡해지는데, AWS Cost Explorer는 이 정도 세분화를 지원하지 않습니다.

### AWS Cost Explorer의 한계

Cost Explorer는 서비스 레벨만 보여줍니다. "Amazon Bedrock"으로 필터링하면 전체 합계는 나오지만, 어떤 모델에 얼마를 썼는지, 일별 트렌드가 어떤지, 캐시 히트율은 몇 퍼센트인지 알 수 없습니다.

```
Cost Explorer 조회 결과:
- Amazon Bedrock: $5,724.02 (지난 7일)

... 그래서 어떤 모델에 얼마 썼나요?
```

Bedrock은 토큰 단위로 과금합니다. Opus 4.6과 Sonnet 4.6의 Input 토큰 가격은 5배 차이($15 vs $3)입니다. 모델별로 얼마나 썼는지 모르면 최적화할 수 없습니다.

### Prompt Caching 도입 후 복잡도 증가

Prompt Caching을 켜면 비용 구조가 4개로 나뉩니다.

- <strong>Input Tokens</strong>: 일반 입력 ($15/1M for Opus)
- <strong>Output Tokens</strong>: 생성된 응답 ($75/1M for Opus)
- <strong>Cache Read</strong>: 캐시 히트 ($1.50/1M for Opus, 90% 절약)
- <strong>Cache Write</strong>: 캐시 생성 ($18.75/1M for Opus, 25% 추가)

캐시 히트율이 99%면 비용이 1/10로 줄어들지만, 캐시가 자주 미스나면 오히려 비싸집니다. 이 수치들을 실시간으로 모니터링해야 하는데, Cost Explorer는 이 레벨까지 들어가지 못합니다.

---

## 2. CloudWatch에 숨어있는 Bedrock 메트릭

사실 AWS는 Bedrock 사용량 메트릭을 CloudWatch에 자동으로 보내고 있습니다. 단지 콘솔에서 눈에 잘 띄지 않을 뿐입니다.

### AWS/Bedrock 네임스페이스

CloudWatch → Metrics → AWS/Bedrock으로 가면 다음 메트릭들이 보입니다.

![CloudWatch 메트릭 조회 플로우](/ai-tech-blog/images/bedrock-cost-tracking/cloudwatch-flow.svg)
*CloudWatch에서 Bedrock 메트릭을 조회하는 3단계 플로우*

<strong>InputTokenCount</strong>: 프롬프트 입력 토큰 수 (Cache Write 제외)

<strong>OutputTokenCount</strong>: 응답 생성 토큰 수

<strong>CacheReadInputTokenCount</strong>: 캐시에서 읽은 토큰 수 (캐시 히트)

<strong>CacheWriteInputTokenCount</strong>: 캐시에 쓴 토큰 수 (캐시 생성)

<strong>Invocations</strong>: API 호출 횟수

각 메트릭은 ModelId 디멘션으로 필터링 가능합니다. `global.anthropic.claude-opus-4-6-v1` 같은 정확한 모델 ID로 조회하면 모델별 토큰 사용량이 나옵니다.

### boto3가 아닌 AWS CLI 직접 호출

Python boto3 CloudWatch 클라이언트를 쓸 수도 있지만, 의존성을 줄이기 위해 AWS CLI를 subprocess로 호출하는 방법을 택했습니다.

```python
cmd = [
    "aws", "cloudwatch", "get-metric-statistics",
    "--namespace", "AWS/Bedrock",
    "--metric-name", "InputTokenCount",
    "--start-time", "2026-02-27T00:00:00Z",
    "--end-time", "2026-03-06T23:59:59Z",
    "--period", "86400",  # 1일 단위
    "--statistics", "Sum",
    "--dimensions", "Name=ModelId,Value=global.anthropic.claude-opus-4-6-v1",
]
result = subprocess.run(cmd, capture_output=True, text=True)
data = json.loads(result.stdout)
```

반환 형식은 다음과 같습니다.

```json
{
  "Datapoints": [
    {
      "Timestamp": "2026-03-03T00:00:00Z",
      "Sum": 1234567.0,
      "Unit": "None"
    }
  ]
}
```

Sum 값이 토큰 수입니다. 이걸 모델별 가격표에 곱하면 비용이 나옵니다.

---

## 3. Python CLI 구현 — claude-cost

설계 철학은 간단합니다. Python 표준 라이브러리 + boto3만 쓰고, Docker나 인프라는 필요 없습니다. 터미널에서 바로 실행 가능해야 합니다.

### 핵심 로직 3단계

![비용 계산 로직 플로우](/ai-tech-blog/images/bedrock-cost-tracking/cost-calculation-flow.svg)
*토큰 집계 → 가격표 적용 → 비용 산출 3단계 플로우*

<strong>1단계: 모델별 토큰 집계</strong>

```python
def cw_list_bedrock_models(profile=None, region=None):
    """CloudWatch에서 사용 중인 Bedrock 모델 목록 조회"""
    cmd = [
        "aws", "cloudwatch", "list-metrics",
        "--namespace", "AWS/Bedrock",
        "--metric-name", "Invocations",
    ]
    if profile:
        cmd.extend(["--profile", profile])
    if region:
        cmd.extend(["--region", region])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    data = json.loads(result.stdout)

    models = set()
    for m in data.get("Metrics", []):
        for d in m.get("Dimensions", []):
            if d["Name"] == "ModelId":
                models.add(d["Value"])
    return sorted(models)
```

먼저 `list-metrics`로 사용 중인 모델 목록을 가져옵니다. 그 다음 각 모델에 대해 5개 메트릭(Input, Output, CacheRead, CacheWrite, Invocations)을 조회합니다.

```python
token_metrics = [
    "InputTokenCount",
    "OutputTokenCount",
    "CacheReadInputTokenCount",
    "CacheWriteInputTokenCount"
]

for model_id in anthropic_models:
    totals = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}

    for metric in token_metrics:
        datapoints = cw_get_metric(model_id, metric, start_str, end_str)
        for dp in datapoints:
            val = int(dp.get("Sum", 0))
            if metric == "InputTokenCount":
                totals["input"] += val
            elif metric == "OutputTokenCount":
                totals["output"] += val
            # ... (나머지 메트릭)
```

<strong>2단계: 가격표 적용</strong>

Anthropic 공식 가격을 하드코딩합니다. (2026년 3월 기준)

```python
PRICING = {
    "claude-opus-4-6": {
        "input": 15.0, "output": 75.0,
        "cache_read": 1.5, "cache_creation": 18.75,
    },
    "claude-sonnet-4-6": {
        "input": 3.0, "output": 15.0,
        "cache_read": 0.30, "cache_creation": 3.75,
    },
    "claude-haiku-4-5": {
        "input": 0.80, "output": 4.0,
        "cache_read": 0.08, "cache_creation": 1.0,
    },
}
```

Bedrock ModelId(`global.anthropic.claude-opus-4-6-v1`)를 짧은 이름으로 매핑하여 가격을 조회합니다.

<strong>3단계: 비용 계산</strong>

```python
def calc_cost(model, totals):
    p = get_pricing(model)
    cost = (
        totals["input"] * p["input"] / 1_000_000
        + totals["output"] * p["output"] / 1_000_000
        + totals["cache_read"] * p["cache_read"] / 1_000_000
        + totals["cache_write"] * p["cache_creation"] / 1_000_000
    )
    return cost
```

### 실행 예시

```bash
claude-cost aws 7
```

출력:

```
======================================================================
  AWS Bedrock Usage - All Environments (last 7 days)
  Region: us-west-2
======================================================================
  Total Estimated Cost:  $   5724.02 USD
======================================================================

  claude-opus-4-6  (global.anthropic.claude-opus-4-6-v1)
  ──────────────────────────────────────────────────────────────────
    Cost:           $   5679.68  (99%)
    Invocations:        10,462
    Input:           6,885,109 tokens
    Output:          2,962,258 tokens
    Cache Read:     1,229,148,119 tokens  (99% hit)
    Cache Write:    187,227,315 tokens

  claude-sonnet-4-6  (global.anthropic.claude-sonnet-4-6-v1)
  ──────────────────────────────────────────────────────────────────
    Cost:           $     44.34  (1%)
    Invocations:          234
    Input:             421,045 tokens
    Output:            186,712 tokens
    Cache Read:      8,012,334 tokens  (95% hit)
    Cache Write:     1,234,567 tokens

  Daily Trend (all models)
  ──────────────────────────────────────────────────────────────────
  02-27  $  891.74   1882 calls  ████████████████████
  03-03  $ 1292.76   2253 calls  ██████████████████████████████
  03-06  $  383.78    766 calls  ████████
```

---

## 4. 실전 활용 — 비용 인사이트

### 일별 트렌드로 비용 급증 감지

Daily Trend 섹션은 비용이 갑자기 튀는 날을 찾는 데 유용합니다. 03-03에 $1,292로 치솟았다면, 그날 무슨 작업을 했는지 역추적할 수 있습니다.

```python
daily_cost = defaultdict(float)
for model_id in model_daily:
    p = get_pricing(model_id)
    for day, d in model_daily[model_id].items():
        cost = (
            d["input"] * p["input"] / 1_000_000
            + d["output"] * p["output"] / 1_000_000
            + d["cache_read"] * p["cache_read"] / 1_000_000
            + d["cache_write"] * p["cache_creation"] / 1_000_000
        )
        daily_cost[day] += cost
```

### 모델별 비용 비중: Opus vs Sonnet

위 예시에서 Opus가 99%, Sonnet이 1%를 차지합니다. Opus Input 토큰 가격은 $15/1M이고, Sonnet은 $3/1M입니다. 5배 차이입니다.

만약 워크로드의 50%를 Sonnet으로 옮기면:

```
Before: Opus 100% → $5,679
After:  Opus 50% + Sonnet 50% → ~$1,140 (Opus) + ~$228 (Sonnet) = $1,368
절약액: $4,311 (76% 감소)
```

물론 Sonnet이 Opus만큼 성능을 낼 수 있는 태스크에 한정됩니다. 단순 요약이나 데이터 추출은 Sonnet으로 충분하지만, 복잡한 추론은 Opus가 필요합니다.

### Cache Hit Rate 분석

Cache Read가 1,229M이고, Input이 6.9M이면:

```
Cache Hit Rate = 1,229 / (1,229 + 6.9) = 99.4%
```

99% 히트율은 Prompt Caching이 제대로 작동한다는 뜻입니다. 캐시 Read 가격($1.50)이 일반 Input($15)보다 10배 싸기 때문에, 캐시를 쓸수록 비용이 줄어듭니다.

만약 히트율이 50% 미만으로 떨어지면, 프롬프트가 매번 바뀌어서 캐시가 무용지물이거나, TTL이 너무 짧아서 자주 만료되는 것입니다. 이 경우 Prompt Caching을 끄는 게 나을 수 있습니다. Cache Write 비용($18.75)이 일반 Input($15)보다 비싸기 때문입니다.

### Profile별 조회로 계정/환경 분리

AWS CLI는 `--profile` 옵션으로 여러 계정을 관리합니다.

```bash
claude-cost aws 30 --profile dev
claude-cost aws 30 --profile prod
```

dev 환경에서 실험하고, prod 환경에서 실제 비용을 추적할 수 있습니다. 리전도 `--region`으로 지정 가능합니다.

---

## 5. 확장 가능성

### cron + Slack 알림으로 일일 비용 리포트 자동화

cron에 등록하여 매일 아침 비용을 Slack으로 보낼 수 있습니다.

```bash
# crontab -e
0 9 * * * /usr/local/bin/claude-cost aws 1 | /usr/local/bin/send-to-slack
```

Slack Incoming Webhook으로 보내면 팀원들이 실시간으로 비용을 확인할 수 있습니다.

```python
import requests
import subprocess

result = subprocess.run(
    ["claude-cost", "aws", "1"],
    capture_output=True,
    text=True
)
requests.post(SLACK_WEBHOOK_URL, json={"text": result.stdout})
```

### 예산 임계치 초과 시 경고

일일 비용이 $1,000를 넘으면 경고를 보내는 로직을 추가할 수 있습니다.

```python
daily_cost = parse_output(result.stdout)
if daily_cost > 1000:
    send_alert("Bedrock 비용 초과: $" + str(daily_cost))
```

### 팀 단위 비용 분배 (태그 기반)

현재 CloudWatch 메트릭에는 사용자 정보가 없어서 계정 전체 합산만 가능합니다. 팀별로 나누려면 Bedrock API 호출 시 태그를 붙이고, CloudTrail 로그를 파싱해야 합니다.

이 경우 AWS 공식 샘플의 ADOT + Athena 파이프라인이 필요합니다. claude-cost는 개인용 경량 도구이므로, 팀 단위 분석이 필요하면 더 큰 인프라를 고려해야 합니다.

---

## References

- claude-cost CLI GitHub: https://github.com/jesamkim/claude-cost-cli
- Amazon Bedrock Pricing: https://aws.amazon.com/bedrock/pricing/
- Anthropic Claude Models: https://docs.anthropic.com/en/docs/about-claude/models
- Amazon CloudWatch Metrics for Bedrock: https://docs.aws.amazon.com/bedrock/latest/userguide/monitoring-cw.html
- AWS CLI CloudWatch Reference: https://awscli.amazonaws.com/v2/documentation/api/latest/reference/cloudwatch/index.html
