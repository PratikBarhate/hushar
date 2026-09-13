#!/usr/bin/env python3
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.

"""Reduces one benchmark client log to table rows.

Called by scripts/benchmark/run_client.sh once per rate. Parsing the client's own output
rather than having the client emit two formats keeps one printed result, so the table and
the run it came from cannot disagree.

One row for the rate, then one per model when the server split traffic between two. Both
are wanted: the overall row is what a caller experienced and the per-model rows are what
a roll-out is deciding on, and neither substitutes for the other -- an overall p99 hides
a candidate that is slower, and per-model rows alone do not say what the service as a
whole delivered.

A column that does not apply to a row is `-` rather than blank or zero: `shed` is a
property of the run and not of an arm, and a zero there would read as "none shed".
"""

import re
import sys

# "    bench-named-v1    75.2%     11.3    22.76    24.80    28.67        91"
# MULTILINE, so the anchors bind to each line rather than to the whole log.
PER_MODEL = re.compile(
    r"^ {4}(\S+) +([0-9.]+)% +([0-9.]+) +([0-9.]+) +([0-9.]+) +([0-9.]+) +([0-9]+)\s*$",
    re.MULTILINE,
)


def row(model, offered, achieved, share, p50, p95, p99, shed, failed):
    """One line, in the column order run_client.sh writes the header in."""
    return (
        f"{model:<26} {offered:>7} {achieved:>9} {share:>7} "
        f"{p50:>8} {p95:>8} {p99:>8} {shed:>6} {failed:>7}"
    )


def main():
    if len(sys.argv) != 4:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    tag, rate, path = sys.argv[1:4]
    with open(path) as handle:
        text = handle.read()

    def find(pattern, default="-"):
        found = re.search(pattern, text)
        return found.group(1) if found else default

    lines = [
        row(
            tag,
            rate,
            find(r"achieved *: *([0-9.]+)"),
            "-",
            find(r"p50 *: *([0-9.]+)"),
            find(r"p95 *: *([0-9.]+)"),
            find(r"p99 *: *([0-9.]+)"),
            find(r"shed *: *([0-9]+)"),
            find(r"failed *: *([0-9]+)"),
        )
    ]

    # Indented under the rate they belong to, so a sweep of several rates stays readable
    # as a sequence of blocks rather than needing the tag repeated on every line.
    for match in PER_MODEL.finditer(text):
        model, share, achieved, p50, p95, p99, _requests = match.groups()
        lines.append(
            row(f"  {model}", "-", achieved, f"{share}%", p50, p95, p99, "-", "-")
        )

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
