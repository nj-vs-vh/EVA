import argparse
import os
from pathlib import Path

import requests  # type: ignore

token = os.environ["ADS_TOKEN"]
auth = {"Authorization": f"Bearer {token}"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    bibtex_entries: list[str] = []
    seen_ads_ids: set[str] = set()

    ads_prefix = "#ADS:"  # see extract.py

    print("Collecting citations from the following files:")
    print("- " + "\n- ".join(args.files))
    input("Press enter to confirm")

    for file in args.files:
        filename = Path(file).stem
        try:
            line: str | None = None
            with open(file, "r") as f:
                for line in f:
                    if line.startswith(ads_prefix):
                        break
            if line is None:
                continue
            file_ads_ids = line.removeprefix(ads_prefix).strip().split(",")

            print("Collected ADS ids:")
            print("- " + "\n- ".join(file_ads_ids))

            print("Exporting citation(s) to BibTeX...")
            for i, ads_id in enumerate(file_ads_ids):
                if ads_id in seen_ads_ids:
                    continue
                else:
                    seen_ads_ids.add(ads_id)
                resp = requests.get(
                    f"https://api.adsabs.harvard.edu/v1/export/bibtex/{ads_id}", headers=auth
                )
                print("Response:")
                print(resp)
                bibtex = resp.text
                bibtex = bibtex.replace(
                    ads_id, filename + (f"_{i}" if len(file_ads_ids) > 1 else ""), 1
                )
                bibtex_entries.append(bibtex)

        except Exception:
            pass

    result = "\n\n".join(bibtex_entries)
    print(result)
    with open(args.output, "w") as out:
        out.write(result)
