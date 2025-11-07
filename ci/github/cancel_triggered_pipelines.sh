#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------
# This script cancels all running or pending LRZ GitLab CI pipelines on TUM COMA cluster
# for a specified project and branch. Only pipelines triggered via the trigger token
# (i.e., from NeoN GitHub CI) are considered.
#----------------------------------------------------------------------------------------
set -euo pipefail

# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
PROJECT=$1        # GitLab project name, e.g., "NeoN"
BRANCH=$2         # Branch/ref to filter pipelines
TOKEN=$3

# -----------------------------------------------------------------------------
# Environment variables
# -----------------------------------------------------------------------------
LRZ_GROUP="${LRZ_GROUP:?LRZ_GROUP is not set in environment}"
LRZ_HOST="${LRZ_HOST:-gitlab-ce.lrz.de}"

if [ -z "$PROJECT" ] || [ -z "$BRANCH" ] || [ -z "$TOKEN" ]; then
  echo "Usage: $0 <project> <branch> <token>"
  exit 1
fi

project_path="${LRZ_GROUP}%2F${PROJECT}"

echo "Fetching pipelines for project '$PROJECT' (path: ${LRZ_GROUP}/${PROJECT}) on branch '$BRANCH'..."

# -----------------------------------------------------------------------------
# Fetch pipelines
# -----------------------------------------------------------------------------
response=$(curl -s -w "%{http_code}" -o response.json \
  --header "PRIVATE-TOKEN: $TOKEN" \
  "https://${LRZ_HOST}/api/v4/projects/${project_path}/pipelines?ref=${BRANCH}&order_by=id&sort=desc")

http_code="${response:(-3)}"
if [[ "$http_code" != "200" ]]; then
  echo "Failed to fetch pipelines (HTTP $http_code)"
  cat response.json
  exit 1
fi

# -----------------------------------------------------------------------------
# Select candidate pipelines
# -----------------------------------------------------------------------------
pipeline_ids=$(jq -r '.[] | select((.status=="running" or .status=="pending")) | .id' response.json)

if [ -z "$pipeline_ids" ]; then
  echo "No running/pending pipelines found on branch '$BRANCH'."
  exit 0
fi

echo "Found the following pipelines to inspect: $pipeline_ids"

# -----------------------------------------------------------------------------
# Cancel pipelines based on project type
# -----------------------------------------------------------------------------
for id in $pipeline_ids; do
  echo "Inspecting pipeline $id..."

  if [[ "$PROJECT" == "NeoN" ]]; then
    # Case 1: NeoN -> cancel all running/pending pipelines on the branch
    echo "Cancelling pipeline $id (PROJECT=NeoN)..."
    curl -s --request POST \
      --header "PRIVATE-TOKEN: $TOKEN" \
      "https://${LRZ_HOST}/api/v4/projects/${project_path}/pipelines/${id}/cancel" >/dev/null
    continue
  fi
done

echo "All applicable pipelines cancelled."
