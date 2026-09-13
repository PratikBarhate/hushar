# benchmark-data

What a benchmark reads and writes. A file a script can reproduce is not committed; a file a
human wrote is.

```
benchmark-data/
├── generated/          ← ignored, from generate_benchmark_models.py
├── dist/               ← ignored, bundles from package.sh
└── results/
    ├── index.md        ← tracked, peak row per configuration
    ├── TEMPLATE.md     ← tracked, copy this to start a report
    ├── cpu-arm64/      ← tracked, one report per run
    ├── cpu-x86/
    ├── gpu-nvidia/
    └── raw/            ← ignored, what a run left behind
```

Generate the inputs on a fresh clone:

```bash
pip install -r scripts/requirements.txt
python scripts/generate_benchmark_models.py
```

The server host needs the `.onnx` files; the client only needs the configurations, the
vocabulary and the feature specification, and never opens the graph.

Running a benchmark is [documentation/benchmarking.md](../documentation/benchmarking.md).
A report's thirteen columns come from four places — the server's banner, the client's
`summary.txt` and `meta.txt`, `scripts/benchmark/server_latency.sh` for the server-side
percentiles, and `scripts/benchmark/sample_host.sh` for CPU and memory.
`results/TEMPLATE.md` names which column comes from which.

## Naming

```
results/<class>/<instance-type>-<provider>-<YYYY-MM-DD>.md
        │        │               │          └─ the day it ran
        │        │               └─ cpu, xnnpack, openvino, tensorrt
        │        └─ dots become dashes: c9g.48xlarge → c9g-48xlarge
        └─ cpu-arm64 · cpu-x86 · gpu-nvidia
```

`results/raw/` is ignored, so push anything worth keeping to S3 and name the prefix after the
report's stem.
