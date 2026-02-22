#!/bin/bash
# Anthropic Recruiting Email Alert (Exclude Product/Marketing emails)

# Gmail API 토큰 가져오기
ACCESS_TOKEN=$(gcloud auth application-default print-access-token 2>&1)

# 마지막 체크 시간 파일
LAST_CHECK_FILE="$HOME/.anthropic-email-last-check"

# 마지막 체크 시간 로드
if [ -f "$LAST_CHECK_FILE" ]; then
    LAST_CHECK=$(cat "$LAST_CHECK_FILE")
else
    # 첫 실행 시 오늘부터
    LAST_CHECK=$(date -u +"%Y/%m/%d")
    echo "$LAST_CHECK" > "$LAST_CHECK_FILE"
fi

# 채용 관련 메일만 검색 (제품/마케팅 메일 제외)
# - 포함: sujin@anthropic.com, recruiting@anthropic.com, interview 관련
# - 제외: no-reply@email.claude.com, Claude Team
RESPONSE=$(curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
  "https://gmail.googleapis.com/gmail/v1/users/me/messages?q=from:sujin@anthropic.com+OR+(from:anthropic.com+AND+(subject:interview+OR+subject:recruiting+OR+subject:position+OR+subject:next+steps))+after:${LAST_CHECK}&maxResults=5")

# 메시지 ID 추출
MSG_IDS=$(echo "$RESPONSE" | python3 -c "import json, sys; data=json.load(sys.stdin); print('\n'.join([m['id'] for m in data.get('messages', [])]))" 2>/dev/null)

if [ -n "$MSG_IDS" ]; then
    # 새 메일이 있음!
    ALERT_MSG="🔔 *Anthropic 새 메일 도착!*\n\n"
    
    # 각 메일 정보 수집
    while IFS= read -r MSG_ID; do
        if [ -n "$MSG_ID" ]; then
            MSG_INFO=$(curl -s -H "Authorization: Bearer $ACCESS_TOKEN" \
              "https://gmail.googleapis.com/gmail/v1/users/me/messages/$MSG_ID?format=metadata&metadataHeaders=From&metadataHeaders=Subject&metadataHeaders=Date" | \
              python3 -c "
import json, sys
data = json.load(sys.stdin)
headers = {h['name']: h['value'] for h in data['payload']['headers']}
print('*From:* ' + headers.get('From', 'N/A'))
print('*Subject:* ' + headers.get('Subject', 'N/A'))
print('*Date:* ' + headers.get('Date', 'N/A'))
" 2>/dev/null)
            
            ALERT_MSG="${ALERT_MSG}${MSG_INFO}\n\n---\n\n"
        fi
    done <<< "$MSG_IDS"
    
    # 텔레그램으로 알림 전송
    echo -e "$ALERT_MSG"
    
    # 현재 날짜로 업데이트
    date -u +"%Y/%m/%d" > "$LAST_CHECK_FILE"
    
    exit 0
else
    # 새 메일 없음 - 조용히 종료
    exit 0
fi
