# > How do I get a truncated reference, e.g., "Ann Author et al." when there are 5 or more authors?

# > The BibTeX style files distributed with REVTeX 4.1 and 4.2 no longer truncate the author lists
# of references (REVTeX 4's .bst files would truncate the list if there were more than 10 authors).
# APS editors prefer full author lists be used for references with 15 or less authors.
# For longer lists, use the phrase "and others" in place of the authors you want to omit.

# from https://journals.aps.org/revtex/revtex-faq

# So I'm supposed to manually go through my .bib file and change long author list to "and others"?
# Is this some kind of sick joke? If you prefer 15 authors than why don't you set the option
# to 15 in your God forsaken .bst file? I feel like I'm losing my mind.
# I can't believe I have to do this shit in the year 2025.

import argparse
import subprocess
from pathlib import Path

import bibtexparser
import bibtexparser.middlewares as m
from bibtexparser.model import ExplicitComment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nopush", action="store_true")
    args = parser.parse_args()

    article_repo = Path(__file__).parent.parent / "../articles/cr-knees-fit"

    print("\nGetting latest article source...\n")
    subprocess.run(["git", "pull"], cwd=article_repo.absolute())

    print("\nTransforming bibliography...\n")
    bib_source = article_repo / "main.bib"
    library = bibtexparser.parse_file(str(bib_source))
    if len(library.failed_blocks) > 0:
        print("Some blocks failed to parse. Check the entries of `library.failed_blocks`.")
        print(library.failed_blocks)

    print(
        f"Parsed {len(library.blocks)} blocks, including:"
        f"\n\t- {len(library.entries)} entries"
        f"\n\t- {len(library.comments)} comments"
        f"\n\t- {len(library.strings)} strings and"
        f"\n\t- {len(library.preambles)} preambles"
    )

    class EtaliciseMiddleware(m.BlockMiddleware):
        def transform_entry(self, entry, *args, **kwargs):
            if isinstance(entry["author"], list) and len(entry["author"]) > 10:
                print(f"Shortening bib entry: {entry}")
                entry["author"] = entry["author"][:10] + ["others"]
            return entry

    bib_out = article_repo / "shortened.bib"
    print(f"\nWriting shortened bibliography to {bib_out}\n")
    library.add(
        ExplicitComment(
            comment="DO NOT EDIT BY HAND! The file is produced by etalicise.py from main.bib.",
            start_line=0,
        )
    )
    library._blocks.insert(0, library._blocks.pop())  # HACK: insert comment on top of the file
    bibtexparser.write_file(
        str(bib_out),
        library,
        append_middleware=[
            m.SeparateCoAuthors(),
            EtaliciseMiddleware(),
            m.MergeCoAuthors(),
        ],
    )

    if args.nopush:
        print("Not pushing files because of --nopush")
    else:
        print("\nPushing updated files...\n")
        subprocess.run(
            ["git", "commit", "-a", "-m", "bibliography etalicised"], cwd=article_repo.absolute()
        )
        subprocess.run(["git", "push"], cwd=article_repo.absolute())
