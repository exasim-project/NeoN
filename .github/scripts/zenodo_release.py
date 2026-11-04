# SPDX-FileCopyrightText: 2026 NeoN authors
#
# SPDX-License-Identifier: MIT

"""Create a new Zenodo version for a NeoN release.

Metadata (title, authors, license, abstract) is taken from CITATION.cff,
funding and other fields are carried over from the previous Zenodo version.

Environment:
    ZENODO_TOKEN       personal access token (deposit:write, deposit:actions)
    ZENODO_CONCEPT_ID  concept record id covering all versions
    TAG                release tag, e.g. v0.2.0
    RELEASE_DATE       publication date, YYYY-MM-DD
    ARCHIVE            path to the source archive to upload
    PUBLISH            "true" to publish, otherwise the draft is left for review
"""

import os
import sys

import requests
import yaml

API = os.environ.get("ZENODO_URL", "https://zenodo.org/api")
TOKEN = os.environ["ZENODO_TOKEN"]
CONCEPT_ID = os.environ["ZENODO_CONCEPT_ID"]
TAG = os.environ["TAG"]
RELEASE_DATE = os.environ["RELEASE_DATE"]
ARCHIVE = os.environ["ARCHIVE"]
PUBLISH = os.environ.get("PUBLISH", "false").lower() == "true"

session = requests.Session()
session.headers.update(
    {
        "Authorization": f"Bearer {TOKEN}",
        "Accept": "application/vnd.inveniordm.v1+json",
    }
)


def call(method, url, **kwargs):
    response = session.request(method, url, **kwargs)
    if not response.ok:
        sys.exit(f"{method} {url} failed ({response.status_code}): {response.text}")
    return response.json() if response.content else {}


def creators(cff):
    result = []
    for author in cff["authors"]:
        person: dict = {
            "type": "personal",
            "given_name": author.get("given-names", ""),
            "family_name": author["family-names"],
        }
        if "orcid" in author:
            orcid = author["orcid"].rsplit("/", 1)[-1]
            person["identifiers"] = [{"scheme": "orcid", "identifier": orcid}]
        entry: dict = {"person_or_org": person}
        if "affiliation" in author:
            entry["affiliations"] = [{"name": author["affiliation"]}]
        result.append(entry)
    return result


def main():
    with open("CITATION.cff", encoding="utf-8") as f:
        cff = yaml.safe_load(f)

    latest = call("GET", f"{API}/records/{CONCEPT_ID}/versions/latest")
    if latest["metadata"].get("version") == TAG:
        sys.exit(f"Zenodo already has version {TAG}: {latest['links']['self_html']}")

    # Returns the pending new-version draft if one already exists
    draft = call("POST", f"{API}/records/{latest['id']}/versions")
    draft_id = draft["id"]
    files_url = f"{API}/records/{draft_id}/draft/files"

    for entry in call("GET", files_url).get("entries", []):
        call("DELETE", f"{files_url}/{entry['key']}")

    key = os.path.basename(ARCHIVE)
    call("POST", files_url, json=[{"key": key}])
    with open(ARCHIVE, "rb") as f:
        call(
            "PUT",
            f"{files_url}/{key}/content",
            data=f,
            headers={"Content-Type": "application/octet-stream"},
        )
    call("POST", f"{files_url}/{key}/commit")

    metadata = draft["metadata"]
    metadata.update(
        {
            "title": cff["title"],
            "version": TAG,
            "publication_date": RELEASE_DATE,
            "creators": creators(cff),
            "rights": [{"id": cff["license"].lower()}],
            "description": cff["abstract"],
        }
    )
    body = {
        key: draft[key]
        for key in ("metadata", "access", "custom_fields", "pids")
        if key in draft
    }
    body["files"] = {"enabled": True}
    updated = call("PUT", f"{API}/records/{draft_id}/draft", json=body)
    if updated.get("errors"):
        print(f"Draft validation warnings: {updated['errors']}")

    if not PUBLISH:
        print(f"Draft ready for review: https://zenodo.org/uploads/{draft_id}")
        return

    published = call("POST", f"{API}/records/{draft_id}/draft/actions/publish")
    print(f"Published {TAG}: {published['links']['self_html']}")


if __name__ == "__main__":
    main()
