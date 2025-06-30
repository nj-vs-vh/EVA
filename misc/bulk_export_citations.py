import argparse
import os

import requests

token = os.environ["ADS_TOKEN"]
auth = {"Authorization": f"Bearer {token}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--output")
    args = parser.parse_args()

    bibtex_entries: list[str] = []
    ads_ids: list[str] = []

    ads_prefix = "#ADS:"  # see extract.py

    for file in args.files:
        try:
            line: str | None = None
            with open(file, "r") as f:
                for line in f:
                    if line.startswith(ads_prefix):
                        break
            if line is None:
                continue
            ads_ids.extend(line.removeprefix(ads_prefix).strip().split(","))
        except Exception:
            pass

    print("Collected ADS ids:")
    print("- " + "\n- ".join(ads_ids))

    print("Exporting citations to BibTeX...")
    resp = requests.post(
        "https://api.adsabs.harvard.edu/v1/export/bibtex",
        headers=auth,
        json={"bibcode": [ads_ids]},
    )
    print("Response:")
    print(resp)
    print(resp.text)
    # print(resp.json())
