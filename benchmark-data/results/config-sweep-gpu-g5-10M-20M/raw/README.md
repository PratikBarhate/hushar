# Raw evidence

`summary.md` one level up is generated from this tree by `driver/report.py`, and
`driver/verify.py` re-derives every number in it from the same files. Nothing in the
summary is typed by hand.

## What is here

| path | what |
|---|---|
| `results.json` | the report's input: one enriched record per measured cell |
| `probe-results.json` | the same for the parallelism probes |
| `run-status.json` | counts by arm and by model size, and how many cells carry each kind of data |
| `cells/` | one record per head-to-head cell, written to S3 the moment it finished |
| `probes/` | one record per parallelism probe |
| `selection/` | the parallelism chosen per model and stack, with the rule and the probes it read |
| `prime/` | the TensorRT engine-cache build, and the proof a fresh server reloaded rather than rebuilt it |
| `cloudwatch/` | the raw minute buckets behind every server-side percentile |
| `server/g5-1/` | per cell: hushar's server log, its resolved `service.json` and `model.json`, Triton's `config.pbtxt` and container log, per-second host samples (`s-`/`t-`) and per-second GPU samples (`g-`) |
| `server/*-stackinfo.txt` | the accelerator stack, captured at deploy: GPU, driver, TensorRT, CUDA, ONNX Runtime, plans |
| `server/*.trtexec.log` | the `trtexec` command and output that built each Triton plan |
| `client/g5-1/<cell>/` | the client's own summary, per-rate logs and host samples |
| `driver/` | the lane logs, and a copy of the driver that produced all of it |
| `failures/` | preserved failed attempts. **Empty: nothing failed to measure.** |

Note the difference between an empty `failures/` and the four cells marked `no` in the
summary. Those four were measured successfully; what they record is that the offered rate
was not served. A cell in `failures/` would be one the harness could not measure at all.

## Reading a cell id

```
g5-hushar-20m-r100t600-p4
│  │      │   │       └── parallelism: sessions for hushar, instances for Triton
│  │      │   └────────── 100 rows per request at 600 requests/second
│  │      └────────────── model: 10m or 20m
│  └───────────────────── arm: hushar or triton
└──────────────────────── the rig: one g5.2xlarge, one NVIDIA A10G
```

A `-probe` suffix is a short cell from the parallelism probe, not a measurement the report
draws a conclusion from.

## The two arms

- **`hushar`** — 210 named features in, vectorized on the host CPU into the 690-float
  vector, scored by TensorRT through ONNX Runtime's TensorRT execution provider.
- **`triton`** — Triton Inference Server, `tensorrt` backend, prebuilt `model.plan`,
  `KIND_GPU`, `max_batch_size: 0`. The 690-float tensor in; nothing transformed.

The wires differ, and the summary prices the difference rather than hiding it. Only the
hushar arm publishes CloudWatch percentiles, so `graph` and the decomposition exist for
nine cells and not eighteen; Triton's own `nv_inference_*` quantiles are cumulative over
warm-up and cannot be cut to the measured window, so they are kept under
`triton_server_side` in the Triton cells and are not compared.

## The two graphs

Same design at two weights, which is what makes a 10M-against-20M reading mean anything:
the same 210 features, the same 690-float input, the same thirteen `ai.onnx` opset-17
operator types, the same matmul shapes — five encoder layers instead of two.

| | parameters | layers | MACs/row | md5 |
|---|--:|--:|--:|---|
| 10M | 9,138,693 | 2 | 53,288,960 | `5539c97131c8` |
| 20M | 18,589,701 | 5 | 128,983,040 | `30f9c5330b80` |
| ratio | 2.03x | | **2.42x** | |

The two ratios differ because `to_tokens` is a `Linear(690 -> 8x512)` applied once per row
and costs the same 2,830,336 parameters at either size, so holding the input width at 690
pins it as a constant and every added parameter lands in the trunk. **Latency should be
read against 2.42x, not 2x.**

Neither graph contains an `ai.onnx.ml` node, which is what makes both expressible as a
TensorRT engine at all: TensorRT has no string type and the ONNX parser does not implement
that domain. hushar's string handling happens in Rust, before the graph.

## One TensorRT, both stacks

The single fact the comparison rests on. Triton's backend serves an engine `trtexec`
compiled; hushar's provider compiles the same ONNX bytes at first use. Those are the same
engine only if the same TensorRT built them, so the deploy **extracts CUDA, cuDNN and
TensorRT out of the Triton image** and points hushar's ONNX Runtime at those copies —
`libnvinfer.so.10 => /root/nvlibs/libnvinfer.so.10` in the deploy log. It was TensorRT
10.16.1 on CUDA 13 for every cell here; `*-stackinfo.txt` records it.

Getting there required moving the Triton image *back*, which is worth recording: Triton
switched to TensorRT 11 at release 26.06, and **no** ONNX Runtime GPU release pairs with it
— 1.28.3 through 1.31.0 all carry `NEEDED libnvinfer.so.10`. `26.05-py3` is the newest
image whose TensorRT major matches. The deploy's `ldd` gate caught this before any cell was
measured rather than after.

## Caveats carried by the data

- **FP32 only**, at TensorRT's default builder settings, on both stacks. The A10G's FP16
  and INT8 tensor cores are untouched, so every latency here is a floor.
- **GPU utilization is occupancy**, not saturation: `nvidia-smi`'s `utilization.gpu` is the
  share of each sample second with at least one resident kernel. Power is sampled beside it
  (up to 235 W of the card's 300 W budget) so the number can be read in context.
- **The 20M parallelism probe ran above the 20M ceiling.** The probe measures the heaviest
  rung of the grid, and no 20M configuration sustained 1000 req/s, so the selection fell
  back from "lowest median among those that sustained" to "highest achieved rate". Both
  stacks chose 4. At 10M, where the probe did sustain, the choice between 2 and 4 moved the
  median by under 1%, so this is unlikely to matter — but it is a fallback, not the primary
  rule, and `selection/20m.json` records which rule fired.
- **One host, both sizes, one after the other.** `g5.2xlarge` capacity allowed one lane.
  That is what the weight comparison needs — a placement offset between two instances would
  contaminate every 10M-against-20M ratio — and the summary's repeatability bullet
  quantifies the offset against the earlier run that used a different machine.
