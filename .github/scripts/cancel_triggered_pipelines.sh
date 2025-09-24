#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# This script cancels all running or pending LRZ GitLab CI pipelines on TUM COMA cluster
# for a specified project and branch. Only pipelines triggered via the trigger token
# (i.e., from NeoN GitHub CI) are considered.
#----------------------------------------------------------------------------------------

set -euo pipefail

# Arguments
PROJECT=$1        # GitLab project name, e.g., "NeoN" or "FoamAdapter"
BRANCH=$2         # Branch/ref to filter pipelines
TOKEN=$3

# Read environment variables defined in GitHub workflow
LRZ_GROUP="${LRZ_GROUP:?LRZ_GROUP is not set in environment}"
LRZ_HOST="${LRZ_HOST:-gitlab-ce.lrz.de}"

if [ -z "$PROJECT" ] || [ -z "$BRANCH" ] || [ -z "$TOKEN" ]; then
  echo "Usage: $0 <project> <branch> <token>"
  exit 1
fi

project_path="${LRZ_GROUP}%2F${PROJECT}"

echo "Fetching pipelines for project '$PROJECT' (path: ${LRZ_GROUP}/${PROJECT}) on branch '$BRANCH' triggered via NeoN GitHub CI..."

# Fetch pipelines
response=$(curl -s -w "%{http_code}" -o response.json \
  --header "PRIVATE-TOKEN: $TOKEN" \
  "https://${LRZ_HOST}/api/v4/projects/${project_path}/pipelines?ref=${BRANCH}&order_by=id&sort=desc")

http_code="${response:(-3)}"
if [[ "$http_code" != "200" ]]; then
  echo "Failed to fetch pipelines (HTTP $http_code)"
  cat response.json
  exit 1
fi

# Filter running/pending pipelines triggered via trigger token (NeoN GitHub CI)
pipeline_ids=$(jq -r '.[] | select((.status=="running" or .status=="pending") and .source=="trigger") | .id' response.json)

if [ -z "$pipeline_ids" ]; then
  echo "No running/pending pipelines triggered by NeoN GitHub CI found."
  exit 0
fi

# Cancel each pipeline only if NEON_BRANCH variable is set
for id in $pipeline_ids; do
  echo "Checking pipeline $id for NEON_BRANCH..."

  vars=$(curl -s --header "PRIVATE-TOKEN: $TOKEN" \
    "https://${LRZ_HOST}/api/v4/projects/${project_path}/pipelines/${id}/variables")

  neon_branch=$(echo "$vars" | jq -r '.[] | select(.key=="NEON_BRANCH") | .value' || true)

  if [[ -n "$neon_branch" && "$neon_branch" != "null" ]]; then
    echo "Cancelling pipeline $id (NEON_BRANCH=$neon_branch)..."
    curl -s --request POST \
      --header "PRIVATE-TOKEN: $TOKEN" \
      "https://${LRZ_HOST}/api/v4/projects/${project_path}/pipelines/${id}/cancel"
  else
    echo "Skipping pipeline $id (NEON_BRANCH not set)."
  fi
done

echo "All applicable pipelines cancelled."
