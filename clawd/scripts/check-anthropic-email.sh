#!/bin/bash
# Anthropic Email Alert Script

# Gmail API 토큰 가져오기
ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>&1)

# 마지막 체크 시간 파일
LAST_CHECK_FILE="$HOME/.anthropic-email-last-check"

# 마지막 체크 시간 로드 (없으면 현재 시간 사용)
if [ -f "$LAST_CHECK_FILE" ]; then
    LAST_CHECK=$(cat "$LAST_CHECK_FILE")
else
    LAST_CHECK=$(date -u +"%Y/%m/%d")
fi

# 새 메일 검색 (마지막 체크 이후)
RESPONSE=$(curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://gmail.googleapis.com/gmail/v1/users/me/messages?q=from:anthropic.com+OR+from:sujin@anthropic.com+after:${LAST_CHECK}&maxResults=5")

# 메시지 개수 확인
MSG_COUNT=$(echo "$RESPONSE" | python3 -c "import json, sys; data=json.load(sys.stdin); print(len(data.get('messages', [])))" 2>/dev/null)

if [ "$MSG_COUNT" -gt 0 ]; then
    echo "🔔 Found $MSG_COUNT new Anthropic email(s)!"
    
    # 각 메일 상세 정보 가져오기
    echo "$RESPONSE" | python3 -c "
import json, sys
data = json.load(sys.stdin)
if 'messages' in data:
    for msg in data['messages']:
        print(msg['id'])
" | while read MSG_ID; do
        # 메일 헤더 가져오기
        curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
          "https://gmail.googleapis.com/gmail/v1/users/me/messages/$MSG_ID?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date" | \
          python3 -c "
import json, sys
data = json.load(sys.stdin)
headers = {h['name']: h['value'] for h in data['payload']['headers']}
print('FROM:' + headers.get('From', 'N/A'))
print('SUBJECT:' + headers.get('Subject', 'N/A'))
print('DATE:' + headers.get('Date', 'N/A'))
"
    done
    
    # 현재 시간을 마지막 체크 시간으로 저장
    date -u +"%Y/%m/%d" > "$LAST_CHECK_FILE"
    
    exit 1  # 새 메일이 있음을 나타냄
else
    echo "✅ No new Anthropic emails"
    exit 0
fi
