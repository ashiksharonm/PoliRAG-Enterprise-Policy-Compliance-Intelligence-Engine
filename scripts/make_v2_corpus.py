#!/usr/bin/env python
"""Create data/versions/v2 from data/raw by applying a fixed set of synthetic policy edits.

The edits are invented for the impact-analysis demo (fictional ACME policies). Each is a
string replacement that must match exactly once, so the script fails loudly if v1 changes.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC, DST = ROOT / "data/raw", ROOT / "data/versions/v2"

EDITS = {
    # numeric change in a clause that several questions ask about
    "privacy_data_subject_rights.txt": [
        (
            "ACME must complete the request within 30 calendar days of receipt.",
            "ACME must complete the request within 45 calendar days of receipt.",
        ),
        ("within the first 30 days.", "within the first 45 days."),
    ],
    "sample_policy.txt": [
        (
            "of seven (7) years from the end of the fiscal year",
            "of ten (10) years from the end of the fiscal year",
        ),
    ],
    "anticorruption_gifts_conflicts.txt": [
        (
            "gifts with a market value of up to $75 from a single",
            "gifts with a market value of up to $50 from a single",
        ),
    ],
    "business_continuity_backup.txt": [
        (
            "Production databases must be backed up every 6 hours.",
            "Production databases must be backed up every 4 hours.",
        ),
        # wording-only edit: same obligations, different phrasing
        ("Backups are retained for 35 days.", "Backups are kept for 35 days."),
    ],
    # new clause inserted before 2.2: later clauses keep their numbers; 2.3 is a new clause
    "software_change_management.txt": [
        (
            "2.2 Branch Protection\n",
            "2.1a Emergency Changes\nEmergency fixes may be merged with one reviewer if a second "
            "review is completed within 24 hours after release.\n\n2.2 Branch Protection\n",
        ),
    ],
}
# a clause removed outright
REMOVE = {"vendor_risk_management.txt": "3.2 Financial Review"}


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for src in sorted(SRC.glob("*.txt")):
        text = src.read_text(encoding="utf-8")
        for old, new in EDITS.get(src.name, []):
            assert text.count(old) == 1, f"{src.name}: {old!r} matched {text.count(old)} times"
            text = text.replace(old, new)
        if src.name in REMOVE:
            head = REMOVE[src.name]
            start = text.index(head)
            end = text.index("\n\n", text.index("\n", start)) + 2
            text = text[:start] + text[end:]
        (DST / src.name).write_text(text, encoding="utf-8")
    print(f"wrote {len(list(DST.glob('*.txt')))} files to {DST}")


if __name__ == "__main__":
    main()
