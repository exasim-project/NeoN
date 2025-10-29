#!/usr/bin/env bash
#----------------------------------------------------------------------------------------
# SPDX-FileCopyrightText: 2023 - 2025 NeoN authors
#
# SPDX-License-Identifier: Unlicense
#----------------------------------------------------------------------------------------
# This script triggers a LRZ GitLab CI pipeline on TUM COMA cluster for a specified project and branch.
# Optionally, extra variables can be passed in the form: "variables[KEY]=VALUE".
#----------------------------------------------------------------------------------------

set -euo pipefail

# Arguments
PROJECT=$1
BRANCH=$2
CHECK_TOKEN=$3     # read_repository scope
TRIGGER_TOKEN=$4   # LRZ GitLab trigger token
shift 4
VARIABLES="$@"     # Optional extra variables in the form: "variables[KEY]=VALUE"

if [ -z "$PROJECT" ] || [ -z "$BRANCH" ] || [ -z "$CHECK_TOKEN" ] || [ -z "$TRIGGER_TOKEN" ]; then
  echo "Usage: $0 <project> <branch> <check_token> <trigger_token> [optional variables]"
  exit 1
fi

# Default LRZ host and group must be set in environment
: "${LRZ_HOST:?Need to set LRZ_HOST}"
: "${LRZ_GROUP:?Need to set LRZ_GROUP}"

# URL-encode branch name
BRANCH_ENC=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$BRANCH")

# Check if branch exists in GitLab using CHECK_TOKEN
branch_exists=$(curl -s --header "PRIVATE-TOKEN: $CHECK_TOKEN" \
  "https://${LRZ_HOST}/api/v4/projects/${LRZ_GROUP}%2F${PROJECT}/repository/branches/${BRANCH_ENC}" \
  | jq -r '.name // empty')

if [ -n "$branch_exists" ]; then
  echo "Branch '$BRANCH' exists in $PROJECT. Using it for pipeline trigger."
else
  echo -e "\033[31mError: Branch '$BRANCH' does not exist in $PROJECT. Exiting workflow.\033[0m"
  exit 1
fi

# Prepare curl form data for variables
FORM_DATA="--form ref=$BRANCH --form token=$TRIGGER_TOKEN"
for var in $VARIABLES; do
  FORM_DATA="$FORM_DATA --form $var"
done

# Trigger pipeline
response=$(curl -s --request POST $FORM_DATA \
  "https://${LRZ_HOST}/api/v4/projects/${LRZ_GROUP}%2F${PROJECT}/trigger/pipeline")

echo "$response" | jq .

pipeline_id=$(echo "$response" | jq -r '.id')
if [ -z "$pipeline_id" ] || [ "$pipeline_id" = "null" ]; then
  echo "Failed to trigger pipeline for project $PROJECT on branch $BRANCH"
  exit 1
fi

echo "Triggered pipeline $pipeline_id on branch $BRANCH"
# Set GitHub Actions output
echo "pipeline_id=$pipeline_id" >> "$GITHUB_OUTPUT"
