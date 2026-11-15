#!/usr/bin/env python3
"""Masks account identifiers in this run's collected evidence, before it is committed.

    ./scrub.py             # report what would change
    ./scrub.py --apply     # rewrite the files

Run it AFTER both collect.py and BEFORE report.py, so the summary is generated from
already-masked inputs.

WHY THIS EXISTS. This campaign writes inference logs to S3, so its SERVER BANNER and
RESOLVED CONFIGURATION both name a bucket -- and a bucket name embeds an AWS account id.
The sweep masks it where text becomes a cell record, but the archived server logs and
service configurations come from the host untouched.

WHAT IT MASKS, AND WHAT IT DELIBERATELY DOES NOT.

    masked    the bucket name, every literal occurrence
    masked    any >=6-digit fragment of a KNOWN account id, in STRING context only --
              which is what a truncated S3 error line leaves behind:
                  ...FAILED: 45678901-us-west-2/inference-logs/...
              `\\d{12}` cannot see that, and `check_privacy.py` -- which checks the
              accounts in use exactly -- steps over it too.
    NOT       a bare 12-digit run on its own. That is `check_privacy.py`'s backstop over
              tracked files, and applying it here does real damage: `bytes_written` is a
              12-digit byte count in this evidence, and an earlier version of this script
              replaced it with a placeholder and INVALIDATED the JSON. A scrubber that
              corrupts the measurements is worse than the exposure it prevents.

STRUCTURE-AWARE FOR JSON. A `.json` file is parsed and walked, and only string values and
keys are touched, so no numeric measurement can be replaced by a placeholder and no file
can come out unparseable. Text and log files are masked by substitution, where there is no
structure to break.

Idempotent, and it re-parses every JSON it writes, so a scrub that would corrupt a file
fails instead of saving it.
"""
import json
import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent


def benchmark_data(start):
    for parent in [start, *start.parents]:
        if parent.name == "benchmark-data":
            return parent
    raise SystemExit(f"{start} is not inside a benchmark-data directory")


BD = benchmark_data(HERE)
RUN = "minibatch-size-cpu-c9g-10M-20M"
RAW = BD / "results" / RUN / "raw"
APPLY = "--apply" in sys.argv
MIN_FRAGMENT = 6
TEXT_SUFFIXES = {".log", ".txt", ".md", ".pbtxt", ".out"}


def known_accounts():
    """The account ids in use -- same sources as scripts/check_privacy.py."""
    raw = os.environ.get("HUSHAR_ACCOUNT_IDS", "")
    if not raw:
        for candidate in (BD.parent / ".account-ids", pathlib.Path(".account-ids")):
            if candidate.exists():
                raw = candidate.read_text()
                break
    return sorted({t for t in re.split(r"[,\s]+", raw) if t.isdigit()})


def fragments(accounts):
    """Every contiguous >=MIN_FRAGMENT-digit substring of a known id, longest first.

    Longest first matters: masking a 6-digit piece before a 10-digit one would leave the
    remainder of the longer match in place and read as scrubbed.
    """
    out = set()
    for a in accounts:
        for size in range(MIN_FRAGMENT, len(a) + 1):
            for i in range(0, len(a) - size + 1):
                out.add(a[i:i + size])
    return sorted(out, key=len, reverse=True)


def mask_string(text, buckets, frags):
    """Masks one string. Never called on a number, so no measurement can be replaced."""
    for name in buckets:
        text = text.replace(name, "<bucket>")
    for frag in frags:
        # Non-digit boundaries, so a fragment that is part of a longer number -- a byte
        # count, a timestamp -- is left alone even inside a string.
        text = re.sub(rf"(?<![0-9]){re.escape(frag)}(?![0-9])", "<account-fragment>",
                      text)
    return text


def walk(value, buckets, frags):
    """Masks every string in a parsed JSON document, leaving numbers untouched."""
    if isinstance(value, str):
        return mask_string(value, buckets, frags)
    if isinstance(value, list):
        return [walk(v, buckets, frags) for v in value]
    if isinstance(value, dict):
        return {mask_string(k, buckets, frags): walk(v, buckets, frags)
                for k, v in value.items()}
    return value


def main():
    accounts = known_accounts()
    if not accounts:
        sys.exit("no account list; set $HUSHAR_ACCOUNT_IDS or write .account-ids")
    if not RAW.exists():
        sys.exit(f"no evidence at {RAW}; run the lanes' collect.py first")

    buckets = [f"hushar-bench-{a}-us-west-2" for a in accounts]
    frags = fragments(accounts)
    changed, broken = [], []

    for path in sorted(RAW.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in TEXT_SUFFIXES and suffix != ".json":
            continue
        try:
            text = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue

        if suffix == ".json":
            try:
                doc = json.loads(text)
            except ValueError as error:
                broken.append((path.relative_to(RAW), f"already unparseable: {error}"))
                continue
            masked = walk(doc, buckets, frags)
            out = json.dumps(masked, indent=2, sort_keys=isinstance(doc, dict)) + "\n"
            if json.dumps(masked, sort_keys=True) == json.dumps(doc, sort_keys=True):
                continue
            # A scrub that cannot be re-read is a scrub that must not be saved.
            try:
                json.loads(out)
            except ValueError as error:
                broken.append((path.relative_to(RAW), f"would corrupt: {error}"))
                continue
        else:
            out = mask_string(text, buckets, frags)
            if out == text:
                continue

        n = sum(1 for a, b in zip(text.splitlines(), out.splitlines()) if a != b) or 1
        changed.append((path.relative_to(RAW), n))
        if APPLY:
            path.write_text(out)

    print(f"{len(accounts)} account(s), {len(buckets)} bucket name(s), "
          f"{len(frags)} fragment(s) >= {MIN_FRAGMENT} digits")
    for rel, why in broken:
        print(f"  ! {rel}: {why}")
    if not changed:
        print(f"clean: nothing to mask under {RAW}")
    else:
        for rel, n in changed:
            print(f"  {'masked' if APPLY else 'would mask'} {n} line(s) in {rel}")
        print(f"{len(changed)} file(s)" + ("" if APPLY else " -- re-run with --apply"))
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
