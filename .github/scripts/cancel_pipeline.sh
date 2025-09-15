#!/usr/bin/env bash
set -euo pipefail

# Arguments
PROJECT=$1          # GitLab project, e.g., "NeoN" or "FoamAdapter"
BRANCH=$2           # Branch/ref to filter pipelines
TOKEN=$3            # GitLab private token
VARIABLES="$@"      # Optional extra variables in the form: "variables[KEY]=VALUE"

if [ -z "$PROJECT" ] || [ -z "$BRANCH" ] || [ -z "$TOKEN" ]; then
  echo "Usage: $0 <project> <branch> <token> [optional variables]"
  exit 1
fi

echo "Fetching pipelines for project '$PROJECT' on branch '$BRANCH' triggered by NeoN GitHub CI..."

# Fetch pipelines
response=$(curl -s -w "%{http_code}" -o response.json \
  --header "PRIVATE-TOKEN: $TOKEN" \
  "https://${LRZ_HOST}/api/v4/projects/${PROJECT//\//%2F}/pipelines?ref=$BRANCH&order_by=id&sort=desc")

http_code="${response:(-3)}"
if [[ "$http_code" != "200" ]]; then
  echo "Failed to fetch pipelines (HTTP $http_code)"
  cat response.json
  exit 1
fi

# Filter running/pending pipelines triggered via "trigger" source
pipeline_ids=$(jq -r '.[] | select((.status=="running" or .status=="pending") and .source=="trigger") | .id' response.json)

if [ -z "$pipeline_ids" ]; then
  echo "No running/pending pipelines triggered by NeoN GitHub CI found."
  exit 0
fi

# Cancel each pipeline
for id in $pipeline_ids; do
  echo "Cancelling pipeline $id..."
  curl -s --request POST \
    --header "PRIVATE-TOKEN: $TOKEN" \
    "https://${LRZ_HOST}/api/v4/projects/${PROJECT//\//%2F}/pipelines/$id/cancel"
done

echo "All applicable pipelines cancelled."
