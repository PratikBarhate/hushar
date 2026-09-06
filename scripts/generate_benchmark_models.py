# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.

"""Generates the two benchmark models and the configurations that serve them.

Both are transformer encoders classifying into 5 classes, sized to about 50M
parameters so that the two are worth comparing to each other. They differ only in
how features reach them, which is the thing being measured:

    vectorized  one `[rows, width]` float input; the service transforms every
                feature and concatenates them
    named       one input per feature, each with its own type; the two text
                features arrive as strings and the graph embeds them itself

Model and configuration are written together, from one source, because the two have
to agree on every width. Deriving them separately is how a service comes to load a
model and then refuse every request.

    python scripts/generate_benchmark_models.py

Writes `bench_model.onnx`, `bench_named_model.onnx`, their configurations and the
string vocabulary into `benchmark-data/generated/`.
"""

import argparse
import json
import os

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper

TARGET_PARAMS = 50_000_000
CLASSES = 5

D_MODEL = 512
N_HEADS = 8
D_FF = 2048
N_TOKENS = 8
OPSET = 17

# Tokens the graph's own embedding tables cover. Anything else is out of vocabulary
# and lands on the default row, which costs exactly the same to look up.
VOCAB = 2000

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
SCALERS = [
    ("mm32_temperature", {"type": "min_max_scaling32", "min": -40.0, "max": 120.0}),
    ("mm32_price", {"type": "min_max_scaling32", "min": 0.0, "max": 10000.0}),
    ("mm32_percentage", {"type": "min_max_scaling32", "min": 0.0, "max": 100.0}),
    ("mm64_revenue", {"type": "min_max_scaling64", "min": 0.0, "max": 1e9}),
    ("mm64_distance", {"type": "min_max_scaling64", "min": 0.0, "max": 40000.0}),
    ("mm64_timestamp", {"type": "min_max_scaling64", "min": 0.0, "max": 4102444800.0}),
    ("sd32_age", {"type": "standardization32", "mean": 35.5, "std_dev": 12.8}),
    ("sd32_weight", {"type": "standardization32", "mean": 170.0, "std_dev": 30.0}),
    ("sd32_duration", {"type": "standardization32", "mean": 300.0, "std_dev": 150.0}),
    ("sd64_income", {"type": "standardization64", "mean": 75000.0, "std_dev": 35000.0}),
    ("sd64_population", {"type": "standardization64", "mean": 5e6, "std_dev": 2.5e6}),
    ("sd64_views", {"type": "standardization64", "mean": 1e6, "std_dev": 5e5}),
]

# The named model's float inputs, paired with 32-bit scalers only. A 64-bit scaler
# would demand a double on the wire and then narrow it to feed an FP32 input, which
# buys nothing -- that path earns its keep when the model input is FP64 as well.
NAMED_SCALARS = [name for name, spec in SCALERS if spec["type"].endswith("32")]

# The named model's text features, embedded inside the graph rather than by a
# configured lookup table.
TEXT_INPUTS = ["text_title", "text_brand"]


def words(prefix, count):
    """Deterministic category names, so a regenerated model matches an old config."""
    return [f"{prefix}_{i:04d}" for i in range(count)]


def vectors(rng, count, width):
    """A lookup table of `count` vectors, rounded so the JSON stays readable."""
    return np.round(rng.standard_normal((count, width)), 4).tolist()


def embedding_transformation(rng, name, width, vocab):
    return {
        "type": "embedding",
        "embeddings": dict(zip(words(f"{name}_v", vocab), vectors(rng, vocab, width))),
        "default_val": [0.0] * width,
    }


def one_hot_transformation(name, width):
    return {
        "type": "one_hot_encoding",
        "categories": sorted(words(f"{name}_c", width)),
        "default_val": [0] * width,
    }


def vectorized_config(rng, model_path, provider, rows=None):
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
        "model_id": "bench-vectorized-v1",
        "model_path": model_path,
        "execution_provider": provider,
        "intra_op_threads": 1,
        **({"fixed_batch_size": rows} if rows is not None else {}),
        "vectorization_config": {
            "data_type": "float",
            "feature_transformations": transformations,
            "feature_order": order,
        },
    }


def named_config(model_path, provider, rows=None):
    """One input per feature, and the text features passed through verbatim.

    The scalars and one-hots are built by transformations because they feed float
    inputs. The text features feed string inputs, so configuring a transformation for
    them would be refused at startup -- the graph does that work instead.

    `rows` pins the batch size, as in `vectorized_config`.
    """
    transformations = {}
    inputs = {}
    for name in NAMED_SCALARS:
        spec = dict(next(s for n, s in SCALERS if n == name))
        spec["default_val"] = [0.5 if "min_max" in spec["type"] else 0.0]
        transformations[name] = spec
        inputs[name] = {"data_type": "float"}
    for name, width in ONE_HOTS:
        transformations[name] = one_hot_transformation(name, width)
        inputs[name] = {"data_type": "float_array"}
    for name in TEXT_INPUTS:
        inputs[name] = {"data_type": "string"}

    return {
        "model_id": "bench-named-v1",
        "model_path": model_path,
        "execution_provider": provider,
        "intra_op_threads": 1,
        **({"fixed_batch_size": rows} if rows is not None else {}),
        "vectorization_config": {
            "feature_transformations": transformations,
            "model_inputs": inputs,
        },
    }


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
        # reshape, and an execution provider that needs a constant shape -- CoreML
        # does -- then refuses every reshape in the attention block and hands the
        # graph back in fragments.
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
    """The part both models share: tokens in, class probabilities out.

    Keeping it identical across the two models is what makes their latencies
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


class VectorizedModel(nn.Module):
    """`[rows, width]` of concatenated features in, class probabilities out."""

    def __init__(self, width, layers):
        super().__init__()
        self.to_tokens = nn.Linear(width, N_TOKENS * D_MODEL)
        self.encoder = Encoder(layers)

    def forward(self, features):
        return self.encoder(self.to_tokens(features).reshape(-1, N_TOKENS, D_MODEL))


class NamedModel(nn.Module):
    """One input per feature in, class probabilities out.

    Text arrives already encoded to a row index, because mapping a string to an index
    is not something torch can export. Those inputs are replaced by the string
    front-end after export -- see `add_string_front_end`.
    """

    def __init__(self, scalars, arrays, texts, layers):
        super().__init__()
        self.scalars, self.arrays, self.texts = scalars, arrays, texts
        dense_tokens = N_TOKENS - len(texts)
        dense_width = len(scalars) + sum(width for _, width in arrays)
        self.to_tokens = nn.Linear(dense_width, dense_tokens * D_MODEL)
        self.tables = nn.ModuleList(nn.Embedding(VOCAB + 1, D_MODEL) for _ in texts)
        self.dense_tokens = dense_tokens
        self.encoder = Encoder(layers)

    def forward(self, *inputs):
        dense = torch.cat(inputs[: len(self.scalars) + len(self.arrays)], dim=-1)
        tokens = [self.to_tokens(dense).reshape(-1, self.dense_tokens, D_MODEL)]
        for table, ids in zip(self.tables, inputs[len(self.scalars) + len(self.arrays):]):
            tokens.append(table(ids.reshape(-1)).reshape(-1, 1, D_MODEL))
        return self.encoder(torch.cat(tokens, dim=1))


# ------------------------------------------------------------------ sizing

def parameters_of(model):
    return sum(p.numel() for p in model.parameters())


def layers_for_target(build, target=TARGET_PARAMS, most=40):
    """Picks the layer count landing closest to `target`.

    Solving for it rather than hard-coding keeps the two models the same size when a
    feature is added to either schema, which would otherwise move one and not the
    other.
    """
    best = min(range(1, most + 1), key=lambda n: abs(parameters_of(build(n)) - target))
    return best, parameters_of(build(best))


# ------------------------------------------------------------------ exporting

def export(model, example, path, input_names, rows=None, output_name="scores"):
    """Writes the ONNX file, with the row axis either dynamic or pinned to `rows`.

    A dynamic axis is what a serving path wants, since the service batches however many
    rows a request carries. A pinned one exists to be measured against it: hushar serves
    a pinned model only when the configuration declares the same row count, and refuses
    every request of another size. See benchmark-data/results/ for what it is worth.
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


def add_string_front_end(path, vocabularies, rows=None):
    """Replaces the encoded-index inputs with string inputs the graph maps itself.

    A `LabelEncoder` per text input turns the string into a row index, defaulting to
    the last row for anything out of vocabulary, and the exported `Gather` then reads
    the embedding. This is the whole reason a string reaches the model at all: the
    service passes text through untouched, so the mapping can live with the weights
    it belongs to instead of in configuration.
    """
    body = onnx.load(path)
    axis = rows if rows is not None else "rows"
    nodes, inputs = [], []
    for name, vocabulary in vocabularies.items():
        inputs.append(helper.make_tensor_value_info(
            name, TensorProto.STRING, [axis, 1]))
        nodes.append(helper.make_node(
            "LabelEncoder", [name], [f"{name}_id"], domain="ai.onnx.ml",
            keys_strings=vocabulary,
            values_int64s=list(range(len(vocabulary))),
            default_int64=len(vocabulary),
        ))
    front = helper.make_model(
        helper.make_graph(
            nodes, "string_front_end", inputs,
            [helper.make_tensor_value_info(f"{n}_id", TensorProto.INT64, [axis, 1])
             for n in vocabularies],
        ),
        opset_imports=[helper.make_opsetid("", OPSET), helper.make_opsetid("ai.onnx.ml", 2)],
    )
    front.ir_version = body.ir_version
    merged = onnx.compose.merge_models(
        front, body, io_map=[(f"{n}_id", f"{n}_id") for n in vocabularies]
    )
    onnx.checker.check_model(merged)
    onnx.save(merged, path)
    return merged


def signature(path):
    model = onnx.load(path)
    types = {1: "FP32", 7: "INT64", 8: "STRING"}

    def describe(value):
        kind = types.get(value.type.tensor_type.elem_type, value.type.tensor_type.elem_type)
        dims = [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]
        return f"{value.name} {kind}{dims}"

    initialised = {i.name for i in model.graph.initializer}
    return ([describe(i) for i in model.graph.input if i.name not in initialised],
            [describe(o) for o in model.graph.output])


# ---------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-batch1",
        action="store_true",
        help="also export copies with the row axis pinned to 1, and configurations "
        "declaring \"fixed_batch_size\": 1 so hushar will serve them. A pinned model "
        "is served by splitting a request into batches of that size and padding the "
        "last, so this exists to measure whether a fixed shape is worth that work.",
    )
    arguments = parser.parse_args()
    # The row axis is dynamic because that is what the service needs: it batches
    # however many rows a request carries. A pinned copy is opt-in and not servable.
    batches = [("", None)] + ([("_batch1", 1)] if arguments.with_batch1 else [])

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    vocabulary = {name: words(f"{name}_v", VOCAB) for name in TEXT_INPUTS}
    written, described = {}, []

    # ---- vectorized
    template = vectorized_config(rng, "", "cpu")
    vectorization = template["vectorization_config"]
    width = sum(len(vectorization["feature_transformations"][n]["default_val"])
                for n in vectorization["feature_order"])
    layers, total = layers_for_target(lambda n: VectorizedModel(width, n))
    print(f"vectorized : width {width}, {layers} layers, {total:,} parameters")
    model = VectorizedModel(width, layers).eval()

    for suffix, rows in batches:
        name = f"bench_model{suffix}"
        path = os.path.join(OUT_DIR, f"{name}.onnx")
        export(model, torch.randn(rows or 2, width), path, ["features"], rows=rows)
        config = vectorized_config(rng, f"{OUT_REL}/{name}.onnx", "cpu", rows=rows)
        config["model_id"] = f"bench-vectorized{suffix.replace('_', '-')}-v1"
        written[f"{name}_config.json"] = config
        described.append((f"vectorized rows={rows or 'dynamic'}", path))

    # ---- named
    scalars, arrays = NAMED_SCALARS, ONE_HOTS
    build = lambda n: NamedModel(scalars, arrays, TEXT_INPUTS, n)
    named_layers, named_total = layers_for_target(build)
    dense_width = len(scalars) + sum(w for _, w in arrays)
    print(f"named      : {len(scalars)} scalars + {len(arrays)} arrays "
          f"({dense_width} dense) + {len(TEXT_INPUTS)} strings, "
          f"{named_layers} layers, {named_total:,} parameters")
    named_model = build(named_layers).eval()
    input_names = scalars + [n for n, _ in arrays] + [f"{n}_id" for n in TEXT_INPUTS]

    for suffix, rows in batches:
        name = f"bench_named_model{suffix}"
        path = os.path.join(OUT_DIR, f"{name}.onnx")
        batch = rows or 2
        example = (tuple(torch.randn(batch, 1) for _ in scalars)
                   + tuple(torch.randn(batch, w) for _, w in arrays)
                   + tuple(torch.zeros(batch, 1, dtype=torch.int64) for _ in TEXT_INPUTS))
        export(named_model, example, path, input_names, rows=rows)
        add_string_front_end(path, vocabulary, rows=rows)
        config = named_config(f"{OUT_REL}/{name}.onnx", "cpu", rows=rows)
        config["model_id"] = f"bench-named{suffix.replace('_', '-')}-v1"
        written[f"{name}_config.json"] = config
        described.append((f"named rows={rows or 'dynamic'}", path))

    written["bench_service_config.json"] = {
        "connection_concurrency": 100,
        "port_number": 8279,
        "model_config_path": f"{OUT_REL}/bench_model_config.json",
    }
    written["bench_string_vocabulary.json"] = vocabulary
    for name, body in written.items():
        with open(os.path.join(OUT_DIR, name), "w") as handle:
            json.dump(body, handle, indent=2)
            handle.write("\n")

    print()
    for label, model_path in described:
        megabytes = os.path.getsize(model_path) / (1024 * 1024)
        model_inputs, outputs = signature(model_path)
        print(f"{label} -> {os.path.relpath(model_path, ROOT)} ({megabytes:.1f} MB)")
        print(f"  inputs  : {', '.join(model_inputs)}")
        print(f"  outputs : {', '.join(outputs)}")
    print(f"\nwrote {len(written)} files to {os.path.relpath(OUT_DIR, ROOT)}")


if __name__ == "__main__":
    main()
