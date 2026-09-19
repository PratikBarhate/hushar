#!/usr/bin/env python3
"""Fails if anything tracked by git looks like an account number or a credential.

An AWS account id reached this repository inside an S3 bucket name, which is the easy way for
one to travel -- nobody types an account id, they paste a path that happens to contain it. So
this scans what git actually tracks, not the working tree, and it names the file and the match
rather than only complaining that something was found.

  python3 scripts/check_privacy.py

The accounts in use are checked exactly, and read from `$HUSHAR_ACCOUNT_IDS` (comma separated)
or from an untracked `.account-ids` file, one per line. They are deliberately not written here:
an id committed to teach a scanner about it has still been committed, and a hash would not help
either, since twelve digits brute-force in seconds. A bare twelve-digit run is reported anyway,
so an id from an account nobody listed is still caught.

Exits non-zero on the first category that matches, so it can gate a commit.
"""
import os
import re
import subprocess
import sys

# A bare 12-digit run is an AWS account id. The boundary excludes adjacent letters as well as
# digits, because a 64-character hex checksum contains 12-digit runs by chance and those are
# not identifiers. Timestamps and byte counts of that width are rare enough that a false
# positive is cheaper than a leak, and a real one can be spelt <account-id>.
ACCOUNT_RUN = re.compile(r"(?<![0-9A-Za-z])[0-9]{12}(?![0-9A-Za-z])")

PATTERNS = [
    ("AWS account id", ACCOUNT_RUN),
    ("GitHub token", re.compile(r"gh[pousr]_[A-Za-z0-9]{16,}|github_pat_[A-Za-z0-9_]{20,}")),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY")),
    ("AWS secret access key assignment",
     re.compile(r"aws_secret_access_key\s*[=:]\s*\S{20,}", re.I)),
]

# Paths whose matches are known not to be identifiers. Keep this list short and specific.
ALLOW = (
    "onnxrt-rs/src/bindings",   # generated FFI constants
    "libnrt-rs/src/bindings",
)


def known_accounts():
    """The account ids in use, from the environment or the untracked list beside the repo."""
    raw = os.environ.get("HUSHAR_ACCOUNT_IDS", "")
    if not raw:
        try:
            raw = open(".account-ids").read()
        except OSError:
            return []
    return sorted({t for t in re.split(r"[,\s]+", raw) if t.isdigit()})


def tracked_files():
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True, check=True)
    return [f for f in out.stdout.split("\0") if f]


def main():
    known = known_accounts()
    # A known id must be matched wherever it sits, including inside a longer token, because
    # concatenating it into a name is exactly how it escaped the first time.
    patterns = list(PATTERNS)
    if known:
        patterns.insert(0, ("account in use", re.compile("|".join(re.escape(k) for k in known))))

    files = tracked_files()
    findings = []
    for path in files:
        if any(path.startswith(a) for a in ALLOW):
            continue
        try:
            with open(path, encoding="utf-8", errors="ignore") as fh:
                for lineno, line in enumerate(fh, 1):
                    for label, pat in patterns:
                        m = pat.search(line)
                        if m:
                            findings.append((label, path, lineno, m.group(0)))
                            break   # one finding per line is enough to fail it
        except OSError:
            continue

    scope = (f"{len(known)} account(s) in use checked exactly" if known
             else "no account list supplied, so only the generic patterns ran -- set "
                  "$HUSHAR_ACCOUNT_IDS or write .account-ids")
    if not findings:
        print(f"  no account identifiers or credentials in {len(files)} tracked files")
        print(f"  {scope}")
        return 0

    by_label = {}
    for label, path, lineno, match in findings:
        by_label.setdefault(label, []).append((path, lineno, match))
    print(f"  {len(findings)} PROBLEM(S):  {scope}")
    for label, hits in by_label.items():
        print(f"    {label} -- {len(hits)}:")
        for path, lineno, match in hits[:20]:
            shown = match if label in ("AWS account id", "account in use") else match[:12] + "..."
            print(f"      {path}:{lineno}  {shown}")
        if len(hits) > 20:
            print(f"      ... and {len(hits) - 20} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
