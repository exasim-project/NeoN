# This script triggers a LRZ GitLab CI pipeline on TUM COMA cluster for a specified project and branch.
# Optionally, extra variables can be passed in the form: "variables[KEY]=VALUE".

#!/usr/bin/env bash
set -euo pipefail

# Arguments
PROJECT=$1
BRANCH=$2
TOKEN=$3
shift 3
VARIABLES="$@"   # Optional extra variables in the form: "variables[KEY]=VALUE"

if [ -z "$PROJECT" ] || [ -z "$BRANCH" ] || [ -z "$TOKEN" ]; then
  echo "Usage: $0 <project> <branch> <token> [optional variables]"
  exit 1
fi

# Default LRZ host and group must be set in environment
: "${LRZ_HOST:?Need to set LRZ_HOST}"
: "${LRZ_GROUP:?Need to set LRZ_GROUP}"

# Check if branch exists in LRZ GitLab
branch_exists=$(curl -s --header "PRIVATE-TOKEN: $TOKEN" \
  "https://${LRZ_HOST}/api/v4/projects/${LRZ_GROUP}%2F${PROJECT}/repository/branches/${BRANCH}" \
  | jq -r '.name // empty')

if [ -n "$branch_exists" ]; then
  echo "Branch '$BRANCH' exists. Using it for pipeline trigger."
else
  echo "Branch '$BRANCH' does not exist. Falling back to 'main'."
  BRANCH="main"
fi

# Prepare curl form data for variables
FORM_DATA="--form ref=$BRANCH --form token=$TOKEN"
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

echo "pipeline_id=$pipeline_id"
# Set GitHub Actions output
echo "pipeline_id=$pipeline_id" >> "$GITHUB_OUTPUT"
