#!/usr/bin/env python3
# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.

"""Reduces one benchmark client log to a single table row.

Called by scripts/run_benchmark.sh once per rate. Parsing the client's own output rather
than having the client emit two formats keeps one printed result, so the table and the
run it came from cannot disagree.
"""

import re
import sys


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

    print(
        f"{tag:<24} {rate:>8} {find(r'achieved *: *([0-9.]+)'):>10} "
        f"{find(r'p50 *: *([0-9.]+)'):>8} {find(r'p90 *: *([0-9.]+)'):>8} "
        f"{find(r'p99 *: *([0-9.]+)'):>8} {find(r'shed *: *([0-9]+)'):>7} "
        f"{find(r'failed *: *([0-9]+)'):>7}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
