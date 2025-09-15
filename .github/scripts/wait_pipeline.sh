#!/usr/bin/env bash
set -euo pipefail

PROJECT=$1
PIPELINE_ID=$2
TOKEN=$3
MAX_WAIT_MINUTES=${MAX_WAIT_MINUTES:-1440}

SUCCESS_STATUSES=("success")
FAILURE_STATUSES=("failed" "canceled" "skipped")

for i in $(seq 1 $MAX_WAIT_MINUTES); do
  status=$(curl -s \
    --header "PRIVATE-TOKEN: $TOKEN" \
    "https://${LRZ_HOST}/api/v4/projects/${LRZ_GROUP}%2F${PROJECT}/pipelines/$PIPELINE_ID" \
    | jq -r '.status')

  echo "[$i] $PROJECT pipeline status: $status"

  case "$status" in
    ${SUCCESS_STATUSES[*]})
      echo "$PROJECT CI pipeline succeeded"
      exit 0
      ;;
    ${FAILURE_STATUSES[*]})
      echo "$PROJECT CI pipeline finished with status: $status"
      exit 1
      ;;
  esac

  sleep 60
done

echo "Timed out after $MAX_WAIT_MINUTES minutes waiting for $PROJECT CI pipeline"
exit 1
