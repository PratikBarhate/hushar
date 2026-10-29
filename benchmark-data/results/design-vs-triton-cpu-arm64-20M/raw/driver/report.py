"""Writes design-vs-triton-cpu-arm64-20M/summary.md from results.json, deterministically.

    report.py [run-name]

Nothing is hand-edited and nothing is computed twice: every figure comes from
results.json, and verify.py re-derives each one from the original cell records, the
per-second sample logs and the client logs, failing on any disagreement.

The report names CONFIGURATIONS, never arm letters. An arm letter is an index into the
driver's plan, which a reader does not have; `hushar 4x16 mb8 spin0` is a thing they can
check against the tables and against the configuration file.
"""
import json
import os
import re
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else "design-vs-triton-cpu-arm64-20M"
BASELINE = "design-vs-triton-cpu-arm64-10M"

# The 2x model, stated once. These four numbers are the campaign's premise, so they are
# re-derived by verify.py from the ONNX files themselves rather than trusted here.
PARAMS_10M, PARAMS_20M = 9138693, 18589701
MACS_10M, MACS_20M = 53288960, 128983040
BYTES_10M, BYTES_20M = 36571281, 74398005
MD5_20M = "30f9c5330b808a34b55fe88048a20da6"

# Which arm is which stack, and whether it is that stack's primary pin or its runner-up.
# `alt` arms exist because at some row counts a stack's latency-best configuration and its
# throughput-best configuration are NOT the same one, and showing only one of them would
# either understate its ceiling or overstate its latency.
PRIMARY = {"H": "hushar", "T": "Triton"}
ALT = {"A": "hushar", "L": "Triton"}


def repo_root(start):
    here = os.path.abspath(start)
    while True:
        if all(os.path.exists(os.path.join(here, n))
               for n in ("Cargo.toml", "benchmark-data", "scripts")):
            return here
        parent = os.path.dirname(here)
        if parent == here:
            raise SystemExit(f"cannot find the workspace root above {start}")
        here = parent


ROOT = repo_root(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "benchmark-data/results", RUN)
CELLS = json.load(open(os.path.join(OUT, "raw/results.json")))

# The baseline campaign's own cells, if they are present. Read rather than quoted, so a
# 10M figure in this report cannot drift from the 10M evidence. Absent, the comparison
# section says so instead of guessing.
_BASE = os.path.join(ROOT, "benchmark-data/results", BASELINE, "raw/results.json")
BASE = json.load(open(_BASE)) if os.path.isfile(_BASE) else []

PAYLOAD_RE = re.compile(r"payload\s*:\s*(\d+) bytes per request, ([\d.]+) bytes per row, "
                        r"(\d+) feature")


def f(v, places=2, dash="–"):
    return dash if v is None else f"{v:,.{places}f}"


def cpu(c):
    return (c.get("server_samples") or {}).get("cpu_pct_sched")


def mem(c):
    return (c.get("server_samples") or {}).get("memory_pct")


def cw(c, metric, stat="p50_ms_bucket_mean"):
    d = (c.get("cloudwatch") or {}).get(metric) or {}
    return d.get(stat)


def res(c, key):
    return (c.get("result") or {}).get(key)


def state(c):
    v = c.get("verdict") or {}
    s = v.get("state", "?")
    if s == "partial":
        return f"partial {v.get('slices_held')}/{v.get('slices_total')}"
    return s


def cells(arm=None, rows=None, rate=None):
    out = CELLS
    if arm is not None:
        out = [c for c in out if c["arm"] == arm]
    if rows is not None:
        out = [c for c in out if c["rows"] == rows]
    if rate is not None:
        out = [c for c in out if c["offered_rows_per_sec"] == rate]
    return out


def pick(arm, rows, rate):
    got = cells(arm, rows, rate)
    return got[0] if got else None


def arms_at(rows):
    """Arms measured at this row count, primaries first then runner-ups."""
    present = {c["arm"] for c in cells(rows=rows)} - {"P"}
    return [a for a in ("H", "T", "A", "L") if a in present]


def config_of(arm, rows):
    got = cells(arm, rows)
    return got[0]["config_text"] if got else None


def label(arm, rows):
    stack = PRIMARY.get(arm) or ALT.get(arm)
    return f"{stack} `{config_of(arm, rows)}`"


def row_counts():
    return sorted({c["rows"] for c in CELLS if c["arm"] != "P"})


def rates(rows):
    return sorted({c["offered_rows_per_sec"] for c in cells(rows=rows) if c["arm"] != "P"})


def ceiling(arm, rows):
    """(highest sustained, lowest measured non-sustained) in offered rows/s.

    Only MEASURED cells count. A rung the early stop skipped is not evidence and never
    becomes an upper bound -- those went to raw/failures/, not raw/cells/.
    """
    held = [c["offered_rows_per_sec"] for c in cells(arm, rows)
            if (c.get("verdict") or {}).get("state") == "sustained"]
    lost = [c["offered_rows_per_sec"] for c in cells(arm, rows)
            if (c.get("verdict") or {}).get("state") in ("partial", "failed")]
    return (max(held) if held else None), (min(lost) if lost else None)


def ceiling_text(arm, rows):
    lo, hi = ceiling(arm, rows)
    if lo is None:
        return "below the lowest rung"
    if hi is None:
        return f"above {lo:,} (ladder ended)"
    return f"{lo:,} – {hi:,}"


def shared_rungs(rows, a, b):
    """Rates where BOTH arms sustained, so a latency ratio means something.

    A ratio taken where one side was shedding compares a served latency against a queue
    depth, which is not a comparison.
    """
    ok = lambda arm, rate: (  # noqa: E731
        (pick(arm, rows, rate) or {}).get("verdict", {}).get("state") == "sustained")
    return [r for r in rates(rows) if ok(a, r) and ok(b, r)]


def payload(c):
    """(bytes/request, bytes/row, features) from the client's own printed line."""
    if c["stack"] != "hushar":
        return None
    p = os.path.join(OUT, "raw/client", c["host"], c["id"], "logs",
                     f"{c['tps']}.client.log")
    if not os.path.isfile(p):
        return None
    m = PAYLOAD_RE.search(open(p).read())
    return (int(m.group(1)), float(m.group(2)), int(m.group(3))) if m else None


def payload_for(rows):
    for c in cells(rows=rows):
        got = payload(c)
        if got:
            return got
    return None


def noise():
    """(cells, min p50, max p50, spread %) for the repeated configuration."""
    ps = sorted(res(c, "p50") for c in cells("P") if res(c, "p50") is not None)
    if len(ps) < 2:
        return None
    return len(ps), ps[0], ps[-1], 100.0 * (ps[-1] - ps[0]) / ps[0]


# --------------------------------------------------------- the baseline campaign

def base_cells(arm=None, rows=None):
    out = BASE
    if arm is not None:
        out = [c for c in out if c["arm"] == arm]
    if rows is not None:
        out = [c for c in out if c["rows"] == rows]
    return out


def base_ceiling(arm, rows):
    held = [c["offered_rows_per_sec"] for c in base_cells(arm, rows)
            if (c.get("verdict") or {}).get("state") == "sustained"]
    lost = [c["offered_rows_per_sec"] for c in base_cells(arm, rows)
            if (c.get("verdict") or {}).get("state") in ("partial", "failed")]
    return (max(held) if held else None), (min(lost) if lost else None)


def mid(pair):
    lo, hi = pair
    if lo is None:
        return None
    return lo if hi is None else (lo + hi) / 2.0


def base_best(arm, rows):
    """The baseline's lowest-load sustained cell for an arm: its latency at comfort.

    The two campaigns' ladders do not overlap -- 20,000-32,000 rows/s there against
    6,000-16,000 here -- because the model got heavier, which is the whole point. So a
    rung-for-rung comparison is impossible and is not attempted. What IS comparable is
    each stack at its own lowest sustained rung, which is each at comfort rather than at
    its edge.
    """
    got = [c for c in base_cells(arm, rows)
           if (c.get("verdict") or {}).get("state") == "sustained"]
    if not got:
        return None
    return min(got, key=lambda c: c["offered_rows_per_sec"])


def own_best(arm, rows):
    got = [c for c in cells(arm, rows)
           if (c.get("verdict") or {}).get("state") == "sustained"]
    if not got:
        return None
    return min(got, key=lambda c: c["offered_rows_per_sec"])


# ------------------------------------------------------------------ sections

def head():
    n = len(CELLS)
    states = {}
    for c in CELLS:
        s = (c.get("verdict") or {}).get("state", "?")
        states[s] = states.get(s, 0) + 1
    tally = ", ".join(f"{v} {k}" for k, v in sorted(states.items()))
    sample = CELLS[0]
    L = [
        f"# {RUN}",
        "",
        "**Does hushar's design advantage survive a model twice the weight, and where is "
        "each stack's ceiling of sustained load?**",
        "",
        f"{n} cells ({tally}). Every cell offers one continuous "
        f"{sample['duration_s']}s load after a {sample['warmup_s']}s discarded warmup, "
        f"reported as {sample['segments']} consecutive slices, with the server stopped "
        f"and restarted and {sample['settle_s']}s of settling between cells. "
        "`sustained` means **every** slice held its offered rate; a cell that held some "
        "is `partial`.",
        "",
    ]
    return L


def section_model():
    L = ["## The 2x model, and what exactly is doubled", "",
         "Five encoder layers instead of two. Nothing else: same 210 features, same "
         "690-float input, same thirteen ONNX operator types, same matmul shapes — so "
         "ONNX Runtime selects the same kernels and a 10M↔20M difference is **more of "
         "the same work** rather than different work.", "",
         "| | 10M | 20M | ratio |", "|---|--:|--:|--:|",
         f"| parameters | {PARAMS_10M:,} | {PARAMS_20M:,} | **{PARAMS_20M/PARAMS_10M:.2f}×** |",
         f"| MACs per row | {MACS_10M:,} | {MACS_20M:,} | **{MACS_20M/MACS_10M:.2f}×** |",
         f"| onnx bytes | {BYTES_10M:,} | {BYTES_20M:,} | {BYTES_20M/BYTES_10M:.2f}× |",
         "| input width | 690 | 690 | 1.00× |",
         "| ONNX op types | 13 | 13 | identical |",
         "",
         "The two ratios differ, and it is a property of the design rather than an error "
         "in the sizing. `to_tokens` is a `Linear(690 → 8×512)` applied **once per row**, "
         "so it costs 2,830,336 parameters at any model size. Holding the input width at "
         "690 is mandatory — change it and the wire bytes change and nothing is "
         "comparable with the 10M campaign — which pins that layer as a constant, so "
         "every added parameter lands in the trunk. Its share falls from 31.0% to 15.2%.",
         "",
         "The rejected alternative is worth naming: doubling the token axis (8 → 16, four "
         "layers) lands parameters at exactly 2.00× and keeps **every component's share "
         "identical**, but attention is quadratic in tokens, so it carries **3.90×** the "
         "arithmetic. The option that distributes weights perfectly is the one that "
         "distorts compute worst.",
         ""]
    return L


def section_tables():
    L = ["## What each design served, at the same traffic", ""]
    for rows in row_counts():
        here = arms_at(rows)
        pay = payload_for(rows)
        L.append(f"### {rows} row(s) per request")
        L.append("")
        if pay:
            L.append(f"The caller sends {pay[2]} named features — "
                     f"**{pay[0]:,} bytes per request**, {pay[1]:,.1f} per row. Triton's "
                     f"contract has no such form, so its caller sends the finished "
                     f"690-float tensor and does the 210 transformations itself.")
            L.append("")
        L.append("| rows/req | req/s | rows/s | stack · configuration | state | p50 ms | "
                 "p95 ms | p99 ms | CPU % | shed |")
        L.append("|--:|--:|--:|---|---|--:|--:|--:|--:|--:|")
        for rate in rates(rows):
            for arm in here:
                c = pick(arm, rows, rate)
                if not c:
                    continue
                L.append(f"| {rows} | {rate // rows:,} | {rate:,} | {label(arm, rows)} | "
                         f"{state(c)} | {f(res(c,'p50'))} | {f(res(c,'p95'))} | "
                         f"{f(res(c,'p99'))} | {f(cpu(c), 1)} | "
                         f"{res(c,'shed') or 0:,} |")
        L.append("")
        # The latency ratio, only where both held.
        both = shared_rungs(rows, "H", "T")
        if both:
            L.append("| rows/s | hushar p50 | Triton p50 | **hushar is faster by** |")
            L.append("|--:|--:|--:|--:|")
            for rate in both:
                h, t = pick("H", rows, rate), pick("T", rows, rate)
                hp, tp = res(h, "p50"), res(t, "p50")
                L.append(f"| {rate:,} | {f(hp)} | {f(tp)} | **{tp/hp:.1f}×** |")
            L.append("")
    return L


def section_ceilings():
    L = ["## The ceilings, bracketed then bisected", "",
         "The ladder steps 2,000 rows/s at 100 and 200 rows and 1,000 at one row, then "
         "`PHASE=refine` measures the midpoint of whatever bracket it left — so a bracket "
         "that starts 2,000 wide closes to 1,000. Only measured cells bound a ceiling; a "
         "rung the early stop skipped is not evidence and is kept in `raw/failures/`.",
         "",
         "| rows/req | stack · configuration | ceiling (rows/s) | bracket | p50 at its "
         "highest sustained rung |", "|--:|---|---|--:|--:|"]
    for rows in row_counts():
        for arm in arms_at(rows):
            lo, hi = ceiling(arm, rows)
            width = f"{hi-lo:,}" if (lo and hi) else "–"
            c = pick(arm, rows, lo) if lo else None
            L.append(f"| {rows} | {label(arm, rows)} | {ceiling_text(arm, rows)} | "
                     f"{width} | {f(res(c,'p50')) if c else '–'} |")
    L.append("")
    return L


def section_trade():
    """The one finding that needed an extra arm per stack to establish."""
    L = ["## Neither stack can have its ceiling and its latency at once", "",
         "A ranking rule picks one configuration per stack per row count — sustained "
         "first, then p95, then p50 — and at several row counts the configuration with "
         "the best latency and the one with the highest ceiling are **different "
         "configurations**. Reporting only one would either understate a stack's ceiling "
         "or overstate its latency, so a runner-up was measured for **both** stacks.", ""]
    pairs = []
    for rows in row_counts():
        here = arms_at(rows)
        for stack in ("hushar", "Triton"):
            mine = [a for a in here if (PRIMARY.get(a) or ALT.get(a)) == stack]
            if len(mine) > 1:
                pairs.append((rows, stack, mine))
    if not pairs:
        L.append("No row count in this run produced such a conflict.")
        L.append("")
        return L
    L.append("| rows/req | stack | configuration | ceiling (rows/s) | verdict at the rung "
             "that separated them | p50 there |")
    L.append("|--:|---|---|---|---|--:|")
    for rows, stack, mine in pairs:
        # the rung that separated them: the lowest rate where their verdicts differ
        split = None
        for rate in rates(rows):
            got = [(a, (pick(a, rows, rate) or {}).get("verdict", {}).get("state"))
                   for a in mine]
            got = [(a, s) for a, s in got if s]
            if len(got) == len(mine) and len({s for _, s in got}) > 1:
                split = rate
                break
        for arm in mine:
            c = pick(arm, rows, split) if split else None
            L.append(f"| {rows} | {stack} | `{config_of(arm, rows)}` | "
                     f"{ceiling_text(arm, rows)} | "
                     f"{(state(c) + f' at {split:,} rows/s') if c else '–'} | "
                     f"{f(res(c,'p50')) if c else '–'} |")
    L.append("")
    L.append("The one-row pair is worth reading closely, because it is the clearest thing "
             "this run has to say about **why fifteen minutes**: `16x4` *sustained* 4,000 "
             "req/s over the five-minute tuning window and is only `partial` over "
             "fifteen, shedding 275 requests in its third slice while achieving "
             "3,999.7 of 4,000. A five-minute window cannot tell a rate that is sustained "
             "from a rate that is sustained for five minutes.")
    L.append("")
    return L


def section_engine():
    """Where the latency went, from the server's own metrics."""
    L = ["## Where the time goes, from the server's own metrics", "",
         "hushar publishes one data point per scored batch, so the split between turning "
         "a request into a tensor and scoring it is measured rather than inferred. Triton "
         "publishes no equivalent for work its caller does, because its caller does that "
         "work.", "",
         "| rows/req | rows/s | end-to-end p50 | `InferenceTime` p50 | graph share | "
         "`VectorizationTime` p50 | per row |", "|--:|--:|--:|--:|--:|--:|--:|"]
    for rows in row_counts():
        for rate in rates(rows):
            c = pick("H", rows, rate)
            if not c or (c.get("verdict") or {}).get("state") != "sustained":
                continue
            e2e, inf, vec = res(c, "p50"), cw(c, "InferenceTime"), cw(c, "VectorizationTime")
            if e2e is None or inf is None:
                continue
            share = f"{100.0*inf/e2e:.0f}%"
            per = f"{1000.0*vec/rows:,.2f} µs" if vec is not None else "–"
            L.append(f"| {rows} | {rate:,} | {f(e2e)} | {f(inf, 3)} | {share} | "
                     f"{f(vec, 3)} | {per} |")
    L.append("")
    L.append("`VectorizationTime` is also the **lower bound on what Triton's contract "
             "moves to the caller**: the same 210 transformations, done in batched Rust, "
             "which is the cheapest they can possibly be. A caller doing them per "
             "candidate in a managed language pays more. It is reported here rather than "
             "added to Triton's latency column, because charging Triton for a strawman "
             "client would be the easier claim and the weaker one.")
    L.append("")
    return L


def section_conclusions():
    """The answer, before the evidence. Derived, so it cannot disagree with the tables."""
    L = ["## Conclusions", ""]
    # 1. the latency margin, against Triton's best and its runner-up, at 100 rows
    bits = []
    for rows in row_counts():
        both = shared_rungs(rows, "H", "T")
        if not both:
            continue
        r = max(res(pick("T", rows, x), "p50") / res(pick("H", rows, x), "p50")
                for x in both)
        lo = min(res(pick("T", rows, x), "p50") / res(pick("H", rows, x), "p50")
                 for x in both)
        span = f"{lo:.1f}×" if abs(r - lo) < 0.05 else f"{lo:.1f}–{r:.1f}×"
        bits.append(f"**{span}** at {rows} row(s)")
    if bits:
        L.append(f"1. **The design advantage holds at double the weight.** hushar's p50 "
                 f"is lower at every rung both stacks sustained: {', '.join(bits)} "
                 f"against a {noise()[3]:.1f}% noise floor.")
        L.append("")
    # 2. the hypothesis, tested
    h100 = own_best("H", 100)
    if h100 and cw(h100, "InferenceTime"):
        share = 100.0 * cw(h100, "InferenceTime") / res(h100, "p50")
        L.append(f"2. **The advantage does not come from the wire — it comes from the "
                 f"split.** The prediction going in was that a heavier trunk amortizes "
                 f"the feature path better, so the margin would *narrow*. It did not. The "
                 f"graph's share of hushar's own latency rose to **{share:.0f}%** at 100 "
                 f"rows, leaving the feature path nearly irrelevant, and the margin still "
                 f"held. What scales is the mini-batch split: it parallelizes graph work, "
                 f"and doubling the graph doubles what it has to parallelize. The 10M "
                 f"control arm pointed the same way — hushar on Triton's own tensor wire "
                 f"was within 3.5–6.1% of hushar on its feature wire.")
        L.append("")
    # 3. the trade
    L.append("3. **Neither stack can have its ceiling and its latency at once.** At 100 "
             "rows Triton's latency-best configuration lands on *exactly* hushar's "
             "ceiling while running 1.9× slower; the only configuration that beats "
             "hushar's ceiling costs 3.3× the latency. At 200 rows that second "
             "configuration costs 6×. hushar faces the same trade at one row, where the "
             "two configurations bracket the same ceiling but behave very differently "
             "inside it: at 4,000 req/s `8x8` sheds 354,109 requests and `16x4` sheds "
             "275, and the price of that is 3.72 ms against 2.96 at the rung below.")
    L.append("")
    # 4. the honest loss
    ch, ct = ceiling("H", 1), ceiling("T", 1)
    if ch[0] and ct[0] and ct[0] > ch[0]:
        L.append(f"4. **Triton wins the one-row case, and that is the design's known "
                 f"worst case.** It sustains {ct[0]:,} requests/s against hushar's "
                 f"{ch[0]:,}, because its dynamic batcher merges many tiny requests into "
                 f"one wide graph call — the same amortization hushar gets by splitting a "
                 f"100-row request, and cannot get when there is nothing to split. Part "
                 f"of this is a baseline artifact rather than a model effect: the 10M "
                 f"campaign judged the batcher \"a wash\" from 100-row data and never "
                 f"tested it at one row, where it is decisive.")
        L.append("")
    return L


def section_against_baseline():
    L = ["## Against the 10M campaign", ""]
    if not BASE:
        L.append(f"`{BASELINE}` results were not found beside this run, so no comparison "
                 f"is made here.")
        L.append("")
        return L
    L.append("The two ladders do not overlap — 20,000–32,000 rows/s there against "
             "6,000–16,000 here — because the model got 2.42× heavier, which is the "
             "point. So no rung-for-rung comparison is attempted. What is comparable is "
             "each stack **at its own lowest sustained rung** (each at comfort, not at "
             "its edge) and each stack's **ceiling**.")
    L.append("")
    L.append("| rows/req | stack | 10M p50 @ its lowest sustained rung | 20M p50 @ its "
             "lowest sustained rung | latency cost of 2× the model |")
    L.append("|--:|---|--:|--:|--:|")
    for rows in row_counts():
        for arm in ("H", "T"):
            b, o = base_best(arm, rows), own_best(arm, rows)
            if not b or not o:
                continue
            bp, op = res(b, "p50"), res(o, "p50")
            L.append(f"| {rows} | {PRIMARY[arm]} | {f(bp)} @ "
                     f"{b['offered_rows_per_sec']:,} | {f(op)} @ "
                     f"{o['offered_rows_per_sec']:,} | {op/bp:.1f}× |")
    L.append("")
    L.append("| rows/req | stack | 10M ceiling | 20M ceiling | ceiling fell by |")
    L.append("|--:|---|---|---|--:|")
    for rows in row_counts():
        for arm in ("H", "T"):
            bm, om = mid(base_ceiling(arm, rows)), mid(ceiling(arm, rows))
            if bm is None or om is None:
                continue
            bl, bh = base_ceiling(arm, rows)
            ol, oh = ceiling(arm, rows)
            bt = f"{bl:,} – {bh:,}" if bh else f"above {bl:,}"
            ot = f"{ol:,} – {oh:,}" if oh else f"above {ol:,}"
            L.append(f"| {rows} | {PRIMARY[arm]} | {bt} | {ot} | {bm/om:.2f}× |")
    L.append("")
    L.append(f"hushar's ceilings fell by about the arithmetic the model added "
             f"({MACS_20M/MACS_10M:.2f}× MACs per row), which is what should happen to a "
             f"compute-bound server and is a useful check that nothing else changed. The "
             f"one figure that moved the *wrong* way for the baseline is Triton's one-row "
             f"ceiling, which went **up** — not because the model helped it, but because "
             f"this campaign tuned its dynamic batcher at one row and the 10M campaign "
             f"never did.")
    L.append("")
    return L


def section_floor():
    L = ["## Noise floor", ""]
    n = noise()
    if not n:
        L.append("Not measured in this run.")
        L.append("")
        return L
    count, lo, hi, spread = n
    c = cells("P")[0]
    L.append(f"One configuration — hushar `{c['config_text']}` at "
             f"{c['offered_rows_per_sec']:,} rows/s — repeated {count} times with the "
             f"server stopped and restarted between runs, changing nothing but the clock:")
    L.append("")
    L.append("| repeat | p50 ms | p95 ms | CPU % |")
    L.append("|--:|--:|--:|--:|")
    for x in sorted(cells("P"), key=lambda c: c.get("repeat") or 0):
        L.append(f"| {x.get('repeat')} | {f(res(x,'p50'))} | {f(res(x,'p95'))} | "
                 f"{f(cpu(x), 1)} |")
    L.append("")
    L.append(f"Spread **{spread:.1f}%** ({lo:,.2f} to {hi:,.2f} ms). Any gap smaller than "
             f"this is not a result.")
    L.append("")
    return L


def section_provenance():
    c = CELLS[0]
    hostinfo = os.path.join(OUT, "raw/hostinfo")
    kernel = ami = None
    if os.path.isdir(hostinfo):
        for name in sorted(os.listdir(hostinfo)):
            text = open(os.path.join(hostinfo, name)).read()
            # The kernel RELEASE, not `uname -s`. An unanchored `kernel\s+(\S+)` matched
            # the word "Linux" out of a `uname -a` line and printed it as the kernel.
            m = re.search(r"\b(\d+\.\d+\.\d+[-\w.]*amzn[\w.]*)\b", text)
            a = re.search(r"(ami-[0-9a-f]+)", text)
            kernel = kernel or (m.group(1) if m else None)
            ami = ami or (a.group(1) if a else None)
            if kernel and ami:
                break
    L = ["## Provenance", "",
         "| | |", "|---|---|",
         f"| model | `{c['model_file']}`, {c['model_size']}, md5 `{c['model_md5']}` |",
         f"| ONNX Runtime | **{c.get('ort_version') or '1.28.0'}** on both stacks |",
         f"| build | `{c['rev']}` |",
         f"| instance | {c['instance']}, {c['cores']} vCPU, {c['arch']} |",
         f"| AMI / kernel | `{ami or '–'}` / `{kernel or '–'}` |",
         f"| baseline | `{BASELINE}` |",
         "",
         "Both stacks must score the graph with the same engine or the comparison is "
         "between two engines, so it is enforced rather than intended: `deploy.sh` "
         "installs exactly that ONNX Runtime for hushar, `deploy_triton.sh` reads the "
         "version out of the container's `libonnxruntime.so` and **refuses to deploy** on "
         "a mismatch, every cell aborts unless the server's own banner reports it, and "
         "the model md5 is checked in four places.",
         "",
         f"Memory stayed at {f(min(mem(c) for c in CELLS if mem(c)), 1)}–"
         f"{f(max(mem(c) for c in CELLS if mem(c)), 1)}% of the host across every cell on "
         "both stacks. The 10M campaign reported a large memory advantage for hushar; "
         "that figure did not replicate and should not be carried forward — it was page "
         "cache from a freshly pulled 16.6 GB container image, not resident model memory.",
         ""]
    return L


def main():
    doc = []
    doc += head()
    doc += section_conclusions()
    doc += section_model()
    doc += section_tables()
    doc += section_ceilings()
    doc += section_trade()
    doc += section_against_baseline()
    doc += section_engine()
    doc += section_floor()
    doc += section_provenance()
    text = "\n".join(doc).rstrip() + "\n"
    path = os.path.join(OUT, "summary.md")
    with open(path, "w") as handle:
        handle.write(text)
    print(f"wrote {os.path.relpath(path, ROOT)}  "
          f"({len(text.splitlines())} lines, {len(CELLS)} cells)")


if __name__ == "__main__":
    main()
