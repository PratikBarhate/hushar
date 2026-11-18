# The trn1 10M/20M FP32 re-run: NOT NEEDED. That round was already FP32.

This file previously laid out a round to re-measure the Trainium cells of
`featurewire-log20-g5-trn1-10M-20M` at "genuine FP32", on the belief that they had been
compiled with BF16 matmuls under an FP32 label. **That belief was wrong.** The round is
already what the re-run would have produced, and running it would spend device hours
reproducing figures that exist.

## What the claim was, and why it was wrong

`neuronx-cc`'s CLI reference gives `--auto-cast` a default of `matmult`:

> `--auto-cast`: Controls how the compiler makes tradeoffs between performance and
> accuracy for FP32 operations. (Default: `matmult`)

`compile_neuron.py` passed only `--model-type=transformer`, so the inference was that the
default applied and every earlier Trainium NEFF had BF16 matmuls. That inference skipped
the step that mattered: **checking the artifacts those rounds recorded.**

They disagreed with it immediately. Their own prime records carry a NEFF size and a
compiled-versus-PyTorch error, and both say FP32:

| | NEFF / FP32-weight bytes | trace error |
|---|--:|--:|
| `featurewire-log20-g5-trn1-10M-20M` (10M, 20M; no flag) | 0.937 - 0.992x | 1.43e-06 - 1.83e-06 |
| this round, `--auto-cast=none` | 0.938 - 1.002x | 1.48e-06 - 2.13e-06 |
| this round, `--auto-cast=matmult` | **0.403 - 0.449x** | **6.9e-04 - 3.3e-03** |

## The measurement that settled it

`neuron/probe_autocast.sh` compiles one model three ways on one host in one session.
Output at `results/featurewire-log20-prec-g5-trn1-100M/raw/probe/autocast-default.txt`:

| variant | trace error | NEFF bytes | vs FP32 weights |
|---|--:|--:|--:|
| no `--auto-cast` flag (the old code path) | 1.43e-06 | 34,284,544 | 0.937x |
| `--auto-cast=none` | 1.43e-06 | 34,284,544 | 0.937x |
| `--auto-cast=matmult --auto-cast-type=bf16` | 6.92e-04 | 14,736,384 | 0.403x |

No flag and `--auto-cast=none` produce a **byte-identical NEFF**, and both reproduce the
old round's recorded 10M b100 figures exactly. `torch_neuronx.trace` builds its own
compiler argument list and does not inherit the CLI default; the CLI reference describes
the standalone `neuronx-cc compile` entry point, not the framework tracing path.

**So `featurewire-log20-g5-trn1-10M-20M` needs no correction.** Its Trainium rows are
FP32, its `"precision": "fp32"` is right, and they are directly comparable with this
round's FP32 arm -- same wire, same `sample_rate` 0.2, same session count, same cast, at
10M and 20M instead of 100M.

## What that comparability is worth

The FP32 Trainium series now spans three model sizes on one configuration, and the
linear-in-layers scaling holds across a 6x extrapolation:

| rows | per layer | fixed | 31 layers projected | 100M measured | error |
|--:|--:|--:|--:|--:|--:|
| 100 | 0.738 ms | 0.167 ms | 23.06 ms | 21.93 ms | -4.9% |
| 500 | 3.632 ms | 0.825 ms | 113.42 ms | 114.89 ms | +1.3% |

Per-layer and fixed terms are solved from that round's 2-layer and 5-layer device spans.
A projection from 10M and 20M predicted a 100M device time to within 5% on both
packagings, which is a stronger statement about the device than any single cell.

## What IS still worth running

**A BF16 arm at 10M and 20M on Trainium.** The precision comparison exists only at 100M,
where BF16 bought 1.37x at 100 rows and 1.57x at 500. Whether that ratio holds at 2 and 5
layers is unmeasured, and it is the question that decides whether BF16 is worth taking on
small models. Four cells, one lane, ~1.5 h:

```bash
( cd neuron && python fanout.py cl1 -f launch.sh --timeout 300 \
    --env REV=$REV --env SPECS="10m:bf16 20m:bf16" --env RIG="$RN" \
    --env SERVER_FOR=trn1-1 --env LANE_TAG=bf16 \
    --env RUN=featurewire-log20-trn1bf16-10M-20M \
    --env HUSHAR_BENCH_BUCKET=$HUSHAR_BENCH_BUCKET )
```

`_LOADS` in `neuron/sweep.py` needs `("10m", "bf16")` and `("20m", "bf16")` entries. Use
the same rates as the FP32 cells so the pair is a precision comparison and nothing else;
`hosts.py` needs `PLANS["main"]` pointed at the two sizes with `precision="bf16"`.

## One carry-over from the original plan, still true

**The old NEFFs are not self-evidencing, and `prime_shape` now refuses them.** Their
metadata has no `auto_cast` key, because the field did not exist when they were built, and
this driver treats a missing `auto_cast` as "cannot support a precision claim" and exits.
That gate is correct -- it is what keeps a label off an artifact that cannot back it -- but
it means a BF16 round at 10M/20M will COMPILE its NEFFs rather than restore anything, and
that an FP32 re-run would be refused at the cache rather than quietly served. Budget ~15 s
per 10M shape and ~30 s per 20M shape; they are small models.

Do not delete `models/neuron/{10m,20m}/`. Those are the artifacts the published round ran,
and the probe above reproduces them bit for bit, which is the evidence for everything on
this page.
