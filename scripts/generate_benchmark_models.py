# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.

"""Generates the benchmark models and the configurations that serve them.

Two architectures, both fed the same way. Every model classifies into 5 classes and is
sized to about 100M parameters, so any two of them are worth putting side by side.

The architectures:

    transformer  an 8-token encoder stack -- attention and feed-forward
    dcn          DCNv2 cross layers then MaskNet blocks, the shape a ranking model
                 usually has: one flat vector and no token axis

Same parameter count, an order of magnitude apart in arithmetic per row, because the
encoder applies its weights to each of 8 tokens and the ranking model applies them once.
Which of the two a piece of hardware prefers is not something to read off a spec sheet.

Features reach both the same way: the service transforms every feature and concatenates
them into one `[rows, width]` float input. `VectorizationTime` is reported separately from
`InferenceTime`, so the cost of turning a request into a tensor stays visible against the
cost of scoring it. The schema is padded to `TARGET_FEATURES` because that per-feature
cost is invisible at a dozen features.

A roll-out loads a model **twice** rather than serving a second set of weights: each model
gets a candidate configuration naming the same file under a second `model_id`. Two arms
that differ only in their weights cost exactly what two arms sharing them cost, and cost
is all this measures.

Model and configuration are written together, from one source, because the two have
to agree on every width. Deriving them separately is how a service comes to load a
model and then refuse every request.

    python scripts/generate_benchmark_models.py

Writes the models and their configurations into `benchmark-data/generated/`.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import onnx
import torch
import torch.nn as nn

# The default when --params is not given. A size is chosen rather than derived because
# the depth is solved to hit it -- see depth_for_target.
TARGET_PARAMS = 100_000_000


def parameter_count(text):
    """Reads `10M`, `50m`, `1.5B` or a plain integer into a parameter count.

    Accepting the suffix because the sizes being compared are named that way, and a
    run that meant 10M and typed 10000000 with one zero missing is a wasted instance
    hour.
    """
    text = text.strip().lower().replace("_", "")
    scale = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}
    if text and text[-1] in scale:
        return int(float(text[:-1]) * scale[text[-1]])
    return int(text)
CLASSES = 5

# The transformer trunk.
D_MODEL = 512
N_HEADS = 8
D_FF = 2048
N_TOKENS = 8
OPSET = 17

# The ranking trunk. Width and the two structural counts are fixed and the block count is
# solved, exactly as the transformer's layer count is -- see depth_for_target.
D_DEEP = 1024
# DCNv2's explicit interaction term. Three, which is what the paper and the deployments
# after it use: the cross network is there for a bounded interaction degree, not for depth.
CROSS_LAYERS = 3
# MaskNet computes its mask through a layer wider than the vector it masks.
MASK_RATIO = 2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Everything written here is generated, so the directory is not tracked by git.
# Configurations name the model by this relative path, which resolves from the
# repository root -- the directory the service and the benchmark script run from.
OUT_REL = "benchmark-data/generated"
OUT_DIR = os.path.join(ROOT, OUT_REL)


# --------------------------------------------------------------- feature schema

# Widths are the point of this table. A one-hot feature is as wide as its category
# list and an embedding as wide as its vectors, so the model's input width is a sum
# over this rather than a count of features.
EMBEDDINGS = [("emb_review", 96, 8), ("emb_category", 64, 12),
              ("emb_sentiment", 48, 6), ("emb_region", 32, 8)]
ONE_HOTS = [("oh_color", 48), ("oh_size", 64), ("oh_channel", 32), ("oh_status", 96)]
IDENTITIES = [("id_scores", 2), ("id_vector", 8)]

# Features to land on. A real deployment counts features in the hundreds, and the service
# pays per feature: 200 of them means 200 transformations run per request. That cost is
# the thing worth measuring, and it does not show up at a dozen features.
TARGET_FEATURES = 200


def synthetic_scalers(count):
    """Pads the schema out to `TARGET_FEATURES` with 32-bit scalars.

    They alternate between the two 32-bit transformations and vary their bounds so no
    two are identical, but they carry no meaning beyond being a feature: the count is
    what is being modelled here, not the semantics.
    """
    out = []
    for i in range(count):
        if i % 2 == 0:
            out.append((f"mm32_feature_{i:03d}", {
                "type": "min_max_scaling32",
                "min": 0.0,
                "max": float(10 * (i % 50 + 1)),
            }))
        else:
            out.append((f"sd32_feature_{i:03d}", {
                "type": "standardization32",
                "mean": float(i % 100),
                "std_dev": float(1 + i % 20),
            }))
    return out


BASE_SCALERS = [
    ("mm32_temperature", {"type": "min_max_scaling32", "min": -40.0, "max": 120.0}),
    ("mm32_price", {"type": "min_max_scaling32", "min": 0.0, "max": 10000.0}),
    ("mm32_percentage", {"type": "min_max_scaling32", "min": 0.0, "max": 100.0}),
    ("sd32_age", {"type": "standardization32", "mean": 35.5, "std_dev": 12.8}),
    ("sd32_weight", {"type": "standardization32", "mean": 170.0, "std_dev": 30.0}),
    ("sd32_duration", {"type": "standardization32", "mean": 300.0, "std_dev": 150.0}),
]

# The 64-bit scalers do not count towards the target, so the padding below is unaffected
# by how many of them there are.
WIDE_SCALERS = [
    ("mm64_revenue", {"type": "min_max_scaling64", "min": 0.0, "max": 1e9}),
    ("mm64_distance", {"type": "min_max_scaling64", "min": 0.0, "max": 40000.0}),
    ("mm64_timestamp", {"type": "min_max_scaling64", "min": 0.0, "max": 4102444800.0}),
    ("sd64_income", {"type": "standardization64", "mean": 75000.0, "std_dev": 35000.0}),
    ("sd64_population", {"type": "standardization64", "mean": 5e6, "std_dev": 2.5e6}),
    ("sd64_views", {"type": "standardization64", "mean": 1e6, "std_dev": 5e5}),
]

# What the target already accounts for: the hand-written 32-bit scalers, the one-hot
# arrays, and the two features that used to reach the graph as text. The two are still
# counted so the padding, and therefore `VECTOR_WIDTH`, is what every tracked run was
# measured at -- a schema that changed width would not be comparable with them.
_ACCOUNTED = len(BASE_SCALERS) + len(ONE_HOTS) + 2
SCALERS = (BASE_SCALERS
           + synthetic_scalers(TARGET_FEATURES - _ACCOUNTED)
           + WIDE_SCALERS)

# The width the concatenated input carries, asserted rather than assumed: it is the one
# number a model, its configuration and every tracked result all have to agree on.
VECTOR_WIDTH = 690

# The suffix the second arm of a roll-out is loaded under. It names the same file as the
# control, because a candidate that differs only in its weights costs what the control
# costs -- so exporting a second set of them would buy nothing but disk. What the pair
# does measure is real: two sessions resident, two sets of metrics, traffic split between
# them.
CANDIDATE = "candidate"


def words(prefix, count):
    """Deterministic category names, so a regenerated model matches an old config."""
    return [f"{prefix}_{i:04d}" for i in range(count)]


def vectors(rng, count, width):
    """A lookup table of `count` vectors, rounded so the JSON stays readable."""
    return np.round(rng.standard_normal((count, width)), 4).tolist()


def embedding_transformation(rng, name, width, vocab):
    return {
        "type": "embedding_lookup",
        "embeddings": dict(zip(words(f"{name}_v", vocab), vectors(rng, vocab, width))),
        "default_val": [0.0] * width,
    }


def one_hot_transformation(name, width):
    return {
        "type": "one_hot_encoding",
        "categories": sorted(words(f"{name}_c", width)),
        "default_val": [0] * width,
    }


def vectorized_config(rng, model_id, model_path, provider, rows=None):
    """Every feature transformed and concatenated into one input.

    `feature_order` fixes the layout, so the model's input width is the sum of the
    widths below and the order is not a detail the caller can get wrong.

    `rows` pins the batch size. A model exported with a fixed leading dimension is only
    servable when the configuration declares the same number, so the two are written
    together here for the same reason the width is.
    """
    transformations = {}
    for name, width, vocab in EMBEDDINGS:
        transformations[name] = embedding_transformation(rng, name, width, vocab)
    for name, width in ONE_HOTS:
        transformations[name] = one_hot_transformation(name, width)
    for name, width in IDENTITIES:
        transformations[name] = {"type": "identity", "default_val": [0.0] * width}
    for name, spec in SCALERS:
        default = 0.5 if "min_max" in spec["type"] else 0.0
        transformations[name] = {**spec, "default_val": [default]}

    order = ([n for n, _, _ in EMBEDDINGS] + [n for n, _ in ONE_HOTS]
             + [n for n, _ in IDENTITIES] + [n for n, _ in SCALERS])
    return {
        "model_id": model_id,
        "model_path": model_path,
        "execution_provider": provider,
        # A pinned graph needs every batch padded up to its size, which is what
        # is_fixed declares; without it a short last batch is a shape mismatch.
        **({"mini_batch_size": rows, "is_fixed": True} if rows is not None else {}),
        "vectorization_config": {
            "data_type": "float",
            "feature_transformations": transformations,
            "feature_order": order,
        },
    }


def service_config(control, candidate=None):
    """The process settings, pointed at one model or at a roll-out's pair.

    Every process setting lives here, not on the command line. The runner rewrites this
    file per run -- see scripts/benchmark/run_server.sh -- because the provider, the
    thread count and the log directory all vary by host.

    `threading` is deliberately absent: every field in it has a default, and the runner
    fills in the ones a run is varying, so this one file suits every instance size it is
    copied to.
    """
    config = {
        "port_number": 8279,
        "model_config_path": control,
        "inference_log": {
            "uri": "/tmp/hushar-benchmark/inference-logs",
            "batch_size": 5000,
            # A fifth of requests, so the log path is genuinely exercised without the
            # feature copies dominating what is being measured. Logging every request
            # at 100 rows x 200 features would make the benchmark partly a measurement
            # of the inference log.
            "sample_rate": 0.2,
            "max_sends_in_flight": 4,
        },
        "metrics": {"batch_size": 500, "max_sends_in_flight": 4},
    }
    if candidate is not None:
        # Half each, which is what a benchmark wants: the two arms then carry the same
        # load and their latencies are comparable. A real roll-out starts far lower.
        config["candidate_model"] = {"config_path": candidate, "traffic_percent": 50}
    return config


# ------------------------------------------------------------------- the model

class Block(nn.Module):
    """One pre-norm transformer encoder layer.

    Attention is written out rather than taken from `nn.MultiheadAttention` because
    the exported operator list is what an execution provider has to accept, and a
    fused attention path exports differently between torch versions. Split, matmul,
    softmax and `LayerNormalization` are ops every provider here implements.
    """

    def __init__(self, d_model, n_heads, d_ff, tokens):
        super().__init__()
        # Held as plain integers, and never read back off the tensor. A shape taken
        # from `x.shape` at run time exports as Shape/Gather/Concat feeding the
        # reshape, and a graph-compiling execution provider that needs a constant
        # shape -- TensorRT and MIGraphX both prefer one -- then refuses the
        # reshapes in the attention block and hands the graph back in fragments.
        self.d_model = d_model
        self.tokens = tokens
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm_attention = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.project = nn.Linear(d_model, d_model)
        self.norm_feed_forward = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, d_ff)
        self.down = nn.Linear(d_ff, d_model)
        # Tanh-approximated GELU rather than the error-function form: it decomposes
        # into ops every provider implements, where `Erf` is not always accelerated.
        self.activation = nn.GELU(approximate="tanh")

    def forward(self, x):
        query, key, value = self.qkv(self.norm_attention(x)).split(self.d_model, dim=-1)
        shape = (-1, self.tokens, self.n_heads, self.head_dim)
        query = query.reshape(shape).transpose(1, 2)
        key = key.reshape(shape).transpose(1, 2)
        value = value.reshape(shape).transpose(1, 2)
        weights = torch.softmax(
            query @ key.transpose(-2, -1) * self.head_dim**-0.5, dim=-1
        )
        attended = (weights @ value).transpose(1, 2).reshape(-1, self.tokens, self.d_model)
        x = x + self.project(attended)
        return x + self.down(self.activation(self.up(self.norm_feed_forward(x))))


class Encoder(nn.Module):
    """The transformer trunk: tokens in, class probabilities out.

    Keeping it identical across the models that use it is what makes their latencies
    comparable -- any difference then comes from how features reached it.
    """

    def __init__(self, layers, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
                 tokens=N_TOKENS):
        super().__init__()
        self.positions = nn.Parameter(torch.zeros(1, tokens, d_model))
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, d_ff, tokens) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, CLASSES)

    def forward(self, tokens):
        x = tokens + self.positions
        for block in self.blocks:
            x = block(x)
        # Mean over positions rather than a class token: one fewer position to carry
        # for a model with no pretrained head to preserve.
        return torch.softmax(self.classifier(self.norm(x).mean(dim=1)), dim=-1)


class CrossLayer(nn.Module):
    """One DCNv2 cross layer: `x0 * (W x + b) + x`.

    The multiplication by the original vector is what makes the interaction explicit --
    after `l` layers the output carries terms of degree `l + 1` in the input features,
    which is the whole reason a ranking model has a cross network in front of its tower.

    Full-rank `W` rather than the paper's low-rank `U V^T`. The low-rank form exists to
    save parameters against an embedding far wider than this one, and here it would only
    turn one matmul into two.
    """

    def __init__(self, width):
        super().__init__()
        self.linear = nn.Linear(width, width)

    def forward(self, original, x):
        return original * self.linear(x) + x


class MaskBlock(nn.Module):
    """One MaskNet block: an instance-guided mask, then a projection.

    The mask is computed from the block's own input and multiplied into it elementwise,
    so the weighting is per request rather than learned once. That elementwise product is
    the point of MaskNet: it gives the network a multiplicative path, which no stack of
    additive layers can express.
    """

    def __init__(self, width, ratio):
        super().__init__()
        self.mask = nn.Sequential(
            nn.Linear(width, width * ratio),
            nn.ReLU(),
            nn.Linear(width * ratio, width),
        )
        self.norm = nn.LayerNorm(width)
        self.project = nn.Linear(width, width)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.project(self.norm(self.mask(x) * x)))


class CrossMaskNet(nn.Module):
    """The ranking trunk: one flat vector in, class probabilities out.

    DCNv2 for explicit interactions, then a stack of MaskBlocks, then the classifier --
    the arrangement a ranking model usually has. `blocks` is what the sizing solves,
    because the width and the cross depth are structural and the tower's depth is not.

    Against the encoder the difference that matters for a benchmark is not the operators
    -- both are matmuls -- but the missing token axis. This applies its weights once per
    row where the encoder applies them to each of 8 tokens, so at one parameter count it
    is an order of magnitude less arithmetic. Whether that translates into an order of
    magnitude of latency is the sort of thing only the hardware can answer.
    """

    def __init__(self, width, blocks, deep=D_DEEP, cross=CROSS_LAYERS, ratio=MASK_RATIO):
        super().__init__()
        self.project = nn.Linear(width, deep)
        self.cross = nn.ModuleList(CrossLayer(deep) for _ in range(cross))
        self.blocks = nn.ModuleList(MaskBlock(deep, ratio) for _ in range(blocks))
        self.norm = nn.LayerNorm(deep)
        self.classifier = nn.Linear(deep, CLASSES)

    def forward(self, features):
        original = self.project(features)
        x = original
        for layer in self.cross:
            x = layer(original, x)
        for block in self.blocks:
            x = block(x)
        return torch.softmax(self.classifier(self.norm(x)), dim=-1)


class VectorizedModel(nn.Module):
    """`[rows, width]` of concatenated features in, class probabilities out."""

    def __init__(self, width, layers):
        super().__init__()
        self.to_tokens = nn.Linear(width, N_TOKENS * D_MODEL)
        self.encoder = Encoder(layers)

    def forward(self, features):
        return self.encoder(self.to_tokens(features).reshape(-1, N_TOKENS, D_MODEL))


# ------------------------------------------------------------------ sizing

def parameters_of(model):
    return sum(p.numel() for p in model.parameters())


def depth_for_target(build, target, most=80):
    """Picks the depth landing closest to `target` parameters.

    Called once per architecture -- the transformer's layer count and the ranking trunk's
    block count -- and that one answer is shared by every model on it. Solving each way in
    separately would land each closest to the target but leave the trunks a layer apart,
    and one layer of 31 is 3% of the encoder's work: a fifth of the feature-path cost the
    set exists to measure, which would then be indistinguishable from an accident of
    sizing. The trunk is held still instead, and the parameter counts differ by whatever
    each way in weighs.

    Solving rather than hard-coding also keeps the size honest when a feature is added to
    the schema, which changes the input width and would otherwise leave the count stale.
    """
    best = min(range(1, most + 1), key=lambda n: abs(parameters_of(build(n)) - target))
    return best, parameters_of(build(best))


# ------------------------------------------------------------------ exporting

def export(model, example, path, input_names, rows=None, output_name="scores"):
    """Writes the ONNX file, with the row axis either dynamic or pinned to `rows`.

    A dynamic axis is what a serving path wants, since the service batches however many
    rows a request carries. A pinned one exists to be measured against it: hushar serves
    a pinned model only when the configuration declares the same row count, and refuses
    every request of another size.
    """
    dynamic = None
    if rows is None:
        dynamic = {name: {0: "rows"} for name in input_names}
        dynamic[output_name] = {0: "rows"}
    torch.onnx.export(
        model, example, path,
        export_params=True, opset_version=OPSET, do_constant_folding=True,
        input_names=input_names, output_names=[output_name], dynamic_axes=dynamic,
        dynamo=False,
    )
    onnx.checker.check_model(onnx.load(path))


def signature(path):
    model = onnx.load(path)
    types = {1: "FP32", 7: "INT64", 8: "STRING", 11: "FP64"}

    def describe(value):
        kind = types.get(value.type.tensor_type.elem_type, value.type.tensor_type.elem_type)
        dims = ",".join(
            str(d.dim_param or d.dim_value) for d in value.type.tensor_type.shape.dim
        )
        return f"{value.name} {kind}[{dims}]"

    initialised = {i.name for i in model.graph.initializer}
    return ([describe(i) for i in model.graph.input if i.name not in initialised],
            [describe(o) for o in model.graph.output])


def summarise(described, most=6):
    """A signature in full while it is short enough to read, and tallied past that.

    A 212-input model otherwise puts one line thousands of characters wide on the
    terminal, and the part worth checking there is the tally by element type -- that the
    doubles are doubles and the strings are strings. The names are in the feature
    specification, which is the authority anyway.
    """
    if len(described) <= most:
        return ", ".join(described)
    kinds = Counter(one.split(" ", 1)[1].split("[")[0] for one in described)
    tally = ", ".join(f"{count} x {kind}" for kind, count in sorted(kinds.items()))
    return f"{len(described)} [{tally}], first {most}: {', '.join(described[:most])}"


# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-batch1",
        action="store_true",
        help="also export copies with the row axis pinned to 1, and configurations "
        "declaring \"mini_batch_size\": 1 with \"is_fixed\": true so hushar will "
        "serve them. A pinned model "
        "is served by splitting a request into batches of that size and padding the "
        "last, so this exists to measure whether a fixed shape is worth that work.",
    )
    parser.add_argument(
        "--params",
        default=str(TARGET_PARAMS),
        help="parameters to size each model to, as 10M, 50M, 100M or an integer "
        "(default: %(default)s). One depth is solved per architecture and shared by "
        "every way in, so all the models on a trunk run the identical trunk -- their "
        "counts then differ by what each way in weighs, about 1.4%% at 100M.",
    )
    parser.add_argument(
        "--out-dir",
        default=OUT_REL,
        help="where to write, relative to the repository root (default: %(default)s). "
        "Give each parameter count its own directory to keep several sizes side by "
        "side, and point the runners at it with GEN_DIR.",
    )
    arguments = parser.parse_args()
    target_params = parameter_count(arguments.params)
    out_rel = arguments.out_dir.strip("/")
    out_dir = os.path.join(ROOT, out_rel)
    # The row axis is dynamic because that is what the service needs: it batches
    # however many rows a request carries. A pinned copy is opt-in and not servable.
    batches = [("", None)] + ([("_batch1", 1)] if arguments.with_batch1 else [])

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    written, described = {}, []

    # The schema settles the width, and the width settles both models' input. Checked
    # rather than assumed: a feature added to one of the lists above changes it, and a
    # run at a different width is not comparable with anything already recorded.
    template = vectorized_config(rng, "", "", "cpu")["vectorization_config"]
    width = sum(len(template["feature_transformations"][n]["default_val"])
                for n in template["feature_order"])
    if width != VECTOR_WIDTH:
        raise SystemExit(
            f"the schema now concatenates to {width} values per row, not the "
            f"{VECTOR_WIDTH} every tracked result was measured at. Update "
            f"VECTOR_WIDTH deliberately if that is intended."
        )
    print(f"schema     : {len(template['feature_order'])} features, width {width}")

    # One depth per architecture, solved to the parameter target so the two are worth
    # putting side by side. The encoder counts layers and the ranking trunk blocks.
    layers, encoder_total = depth_for_target(
        lambda n: VectorizedModel(width, n), target_params
    )
    blocks, ranking_total = depth_for_target(
        lambda n: CrossMaskNet(width, n), target_params
    )
    architectures = [
        ("bench_model", "bench-vectorized", "transformer",
         lambda: VectorizedModel(width, layers),
         f"{layers} encoder layers", encoder_total),
        ("bench_dcn_model", "bench-vectorized-dcn", "dcn",
         lambda: CrossMaskNet(width, blocks),
         f"DCNv2 {CROSS_LAYERS} cross + MaskNet {blocks} blocks at {D_DEEP} wide",
         ranking_total),
    ]

    for stem, model_id, label, build_arch, shape, total in architectures:
        # Reseeded before each, so a regenerated model matches an old one.
        torch.manual_seed(0)
        model = build_arch().eval()
        print(f"  {label:<11}: {shape}, {total:,} parameters")

        for suffix, rows in batches:
            name = f"{stem}{suffix}"
            path = os.path.join(out_dir, f"{name}.onnx")
            export(model, torch.randn(rows or 2, width), path, ["features"], rows=rows)
            dashed = suffix.replace("_", "-")
            written[f"{name}_config.json"] = vectorized_config(
                rng, f"{model_id}{dashed}", f"{out_rel}/{name}.onnx", "cpu", rows=rows
            )
            # The candidate arm: the same file under a second name, so a roll-out can be
            # served without a second set of weights to export, copy and keep straight.
            written[f"{name}_{CANDIDATE}_config.json"] = vectorized_config(
                rng, f"{model_id}{dashed}-{CANDIDATE}", f"{out_rel}/{name}.onnx", "cpu",
                rows=rows,
            )
            described.append((f"{label} rows={rows or 'dynamic'}", path))

    written["bench_service_config.json"] = service_config(
        f"{out_rel}/bench_model_config.json"
    )
    written["bench_dcn_service_config.json"] = service_config(
        f"{out_rel}/bench_dcn_model_config.json"
    )
    # The two-model configs: both loaded, traffic split between them. Separate from the
    # single-model configs above because most runs want one model, and a run that wants
    # two should have to say so -- loading the second costs its whole size in memory and
    # produces metrics nobody asked for.
    #
    # This pair is the interesting one: both architectures resident, half the traffic
    # each, so their latencies come from one session on one host with nothing else moving.
    written["bench_ab_service_config.json"] = service_config(
        f"{out_rel}/bench_model_config.json",
        f"{out_rel}/bench_dcn_model_config.json",
    )
    # And the roll-out shape: one model loaded twice, which is what two arms of the same
    # architecture cost.
    written["bench_same_ab_service_config.json"] = service_config(
        f"{out_rel}/bench_model_config.json",
        f"{out_rel}/bench_model_{CANDIDATE}_config.json",
    )
    for name, body in written.items():
        with open(os.path.join(out_dir, name), "w") as handle:
            json.dump(body, handle, indent=2)
            handle.write("\n")

    print()
    for label, model_path in described:
        megabytes = os.path.getsize(model_path) / (1024 * 1024)
        graph_inputs, outputs = signature(model_path)
        print(f"{label} -> {os.path.relpath(model_path, ROOT)} ({megabytes:.1f} MB)")
        print(f"  inputs  : {summarise(graph_inputs)}")
        print(f"  outputs : {summarise(outputs)}")
    print(f"\nwrote {len(written)} files to {os.path.relpath(out_dir, ROOT)}")


if __name__ == "__main__":
    main()
