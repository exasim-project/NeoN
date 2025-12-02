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

# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Environment setup
# -----------------------------------------------------------------------------
: "${LRZ_HOST:?Need to set LRZ_HOST}"
: "${LRZ_GROUP:?Need to set LRZ_GROUP}"

# URL-encode branch name
BRANCH_ENC=$(python3 -c "import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$BRANCH")

# -----------------------------------------------------------------------------
# Function to check if a branch exists on a given project
# -----------------------------------------------------------------------------
check_branch_exists() {
  local project=$1
  local branch_enc=$2
  local token=$3

  curl -s --header "PRIVATE-TOKEN: $token" \
    "https://${LRZ_HOST}/api/v4/projects/${LRZ_GROUP}%2F${project}/repository/branches/${branch_enc}" \
    | jq -r '.name // empty'
}

# -----------------------------------------------------------------------------
# Determine branch to use for pipeline trigger
# -----------------------------------------------------------------------------
branch_exists=$(check_branch_exists "$PROJECT" "$BRANCH_ENC" "$CHECK_TOKEN")

if [ "$PROJECT" = "NeoN" ]; then
  if [ -n "$branch_exists" ]; then
    echo "NeoN branch '$BRANCH' exists on LRZ GitLab. Using it for pipeline trigger."
    TARGET_BRANCH="$BRANCH"
  else
    echo -e "\033[31m Error: Branch '$BRANCH' does not exist in NeoN. Exiting workflow.\033[0m"
    exit 1
  fi

elif [ "$PROJECT" = "NeoFOAM" ]; then
  if [ -n "$branch_exists" ]; then
    echo "NeoFOAM branch '$BRANCH' exists on LRZ GitLab. Using it for pipeline trigger."
    TARGET_BRANCH="$BRANCH"
  else
    echo -e "\033[33m️ Branch '$BRANCH' does not exist in NeoFOAM. Falling back to 'develop'.\033[0m"
    TARGET_BRANCH="develop"
  fi

else
  echo -e "\033[31m Error: Unknown project '$PROJECT'. Supported: NeoN, NeoFOAM.\033[0m"
  exit 1
fi

# -----------------------------------------------------------------------------
# Prepare curl form data for variables
# -----------------------------------------------------------------------------
FORM_DATA="--form ref=$TARGET_BRANCH --form token=$TRIGGER_TOKEN"
for var in $VARIABLES; do
  FORM_DATA="$FORM_DATA --form $var"
done

# -----------------------------------------------------------------------------
# Trigger pipeline
# -----------------------------------------------------------------------------
echo "Triggering pipeline for project '$PROJECT' on branch '$TARGET_BRANCH'..."
response=$(curl -s --request POST $FORM_DATA \
  "https://${LRZ_HOST}/api/v4/projects/${LRZ_GROUP}%2F${PROJECT}/trigger/pipeline")

echo "$response" | jq .

pipeline_id=$(echo "$response" | jq -r '.id')
if [ -z "$pipeline_id" ] || [ "$pipeline_id" = "null" ]; then
  echo -e "\033[31m Failed to trigger pipeline for project '$PROJECT' on branch '$TARGET_BRANCH'.\033[0m"
  exit 1
fi

echo "Triggered pipeline $pipeline_id on branch '$TARGET_BRANCH'."
# Set GitHub Actions output
echo "pipeline_id=$pipeline_id" >> "$GITHUB_OUTPUT"
