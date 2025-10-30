#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------
# This script monitors the status of a LRZ GitLab CI pipeline on TUM COMA cluster
# for a specified project and pipeline ID. It checks the status every minute until
# the pipeline succeeds, fails, or a maximum wait time is reached.
#----------------------------------------------------------------------------------------
set -euo pipefail

PROJECT=$1
PIPELINE_ID=$2
TOKEN=$3
MAX_WAIT_MINUTES=${MAX_WAIT_MINUTES:-1440}

pipeline_url="https://${LRZ_HOST}/${LRZ_GROUP}/${PROJECT}/-/pipelines/${PIPELINE_ID}"
echo "Monitoring LRZ GitLab CI pipeline: $pipeline_url"

for i in $(seq 1 "$MAX_WAIT_MINUTES"); do
  # Get response and HTTP code
  response=$(curl -s -w "%{http_code}" \
    --header "PRIVATE-TOKEN: $TOKEN" \
    "https://${LRZ_HOST}/api/v4/projects/${LRZ_GROUP}%2F${PROJECT}/pipelines/${PIPELINE_ID}")

  http_code="${response: -3}"
  body="${response::-3}"

  if [[ "$http_code" != "200" ]]; then
    echo "Failed to fetch pipeline status, HTTP code: $http_code"
    echo "Response: $body"
    exit 1
  fi

  # Parse JSON safely
  status=$(echo "$body" | jq -r '.status // empty')
  if [[ -z "$status" ]]; then
    echo "Failed to parse pipeline status from response:"
    echo "$body"
    exit 1
  fi

  echo "[$i] $PROJECT pipeline status: $status"

  case "$status" in
    success)
      echo "$PROJECT CI pipeline succeeded"
      exit 0
      ;;
    failed|canceled|skipped)
      echo "$PROJECT CI pipeline finished with status: $status"
      exit 1
      ;;
  esac

  sleep 60
done

echo "Timed out after $MAX_WAIT_MINUTES minutes waiting for $PROJECT CI pipeline"
exit 1
