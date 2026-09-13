# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.

"""Generates the benchmark models and the configurations that serve them.

Two architectures, and three ways for features to reach them. Every model classifies into
5 classes and is sized to about 100M parameters, so any two of them are worth putting side
by side.

The architectures:

    transformer  an 8-token encoder stack -- attention and feed-forward
    dcn          DCNv2 cross layers then MaskNet blocks, the shape a ranking model
                 usually has: one flat vector and no token axis

Same parameter count, an order of magnitude apart in arithmetic per row, because the
encoder applies its weights to each of 8 tokens and the ranking model applies them once.
Which of the two a piece of hardware prefers is not something to read off a spec sheet.

The ways in, all measured on the transformer, so that comparison holds the architecture
still:

    raw         one input per feature, every value exactly as the caller sent it; the
                graph scales, one-hot encodes and embeds all of it itself, and the
                configuration says nothing about features at all
    vectorized  one `[rows, width]` float input; the service transforms every feature
                and concatenates them
    named       one input per feature; the service transforms the numbers, and the two
                text features reach the graph as strings

`VectorizationTime` is reported separately from `InferenceTime`, because turning a request
into tensors is the cost those three exist to compare: `vectorized` puts that work in the
service, `raw` puts all of it in the graph, and `named` splits it. The schema is padded to
`TARGET_FEATURES` because a per-input path pays per input, and that cost is invisible at a
dozen features.

`raw` is the shape a trained model usually arrives in: featurization is part of the graph,
so a deployment has nothing to configure and nothing to keep in sync -- its configuration
has no `vectorization_config`, which is how hushar is told that every model input is fed by
the feature of its own name. Both architectures take it, and they share the featurization
front end weight for weight, so a run comparing them is comparing trunks.

A roll-out loads a model **twice** rather than serving a second set of weights: each raw
model gets a candidate configuration naming the same file under a second `model_id`. Two
arms that differ only in their weights cost exactly what two arms sharing them cost, and
cost is all this measures.

Model and configuration are written together, from one source, because the two have
to agree on every width. Deriving them separately is how a service comes to load a
model and then refuse every request.

    python scripts/generate_benchmark_models.py

Writes the models, their configurations, the string vocabulary and the raw models'
feature specification into `benchmark-data/generated/`.
"""

import argparse
import json
import os
from collections import Counter

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import TensorProto, helper

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

# The named model's text features, embedded inside the graph rather than by a
# configured lookup table.
TEXT_INPUTS = ["text_title", "text_brand"]

# Named inputs to land on. A real deployment counts features in the hundreds, and the
# named path pays per input: 200 features means 200 tensors built per request, against
# one concatenated vector for the vectorized path. That per-input cost is the thing
# worth measuring, and it does not show up at a dozen features.
TARGET_FEATURES = 200


def synthetic_scalers(count):
    """Pads the schema out to `TARGET_FEATURES` with 32-bit scalars.

    Only 32-bit, so every one is also a named input -- see `NAMED_SCALARS`. They
    alternate between the two 32-bit transformations and vary their bounds so no two
    are identical, but they carry no meaning beyond being a feature: the count is what
    is being modelled here, not the semantics.
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


NAMED_SCALERS_BASE = [
    ("mm32_temperature", {"type": "min_max_scaling32", "min": -40.0, "max": 120.0}),
    ("mm32_price", {"type": "min_max_scaling32", "min": 0.0, "max": 10000.0}),
    ("mm32_percentage", {"type": "min_max_scaling32", "min": 0.0, "max": 100.0}),
    ("sd32_age", {"type": "standardization32", "mean": 35.5, "std_dev": 12.8}),
    ("sd32_weight", {"type": "standardization32", "mean": 170.0, "std_dev": 30.0}),
    ("sd32_duration", {"type": "standardization32", "mean": 300.0, "std_dev": 150.0}),
]

# 64-bit scalers stay out of the named model on purpose -- see `NAMED_SCALARS` -- so
# they do not count towards the target.
WIDE_SCALERS = [
    ("mm64_revenue", {"type": "min_max_scaling64", "min": 0.0, "max": 1e9}),
    ("mm64_distance", {"type": "min_max_scaling64", "min": 0.0, "max": 40000.0}),
    ("mm64_timestamp", {"type": "min_max_scaling64", "min": 0.0, "max": 4102444800.0}),
    ("sd64_income", {"type": "standardization64", "mean": 75000.0, "std_dev": 35000.0}),
    ("sd64_population", {"type": "standardization64", "mean": 5e6, "std_dev": 2.5e6}),
    ("sd64_views", {"type": "standardization64", "mean": 1e6, "std_dev": 5e5}),
]

# Everything named already accounted for: the 32-bit scalers, the one-hot arrays and
# the two text inputs. The remainder is made up with synthetic 32-bit scalars.
_NAMED_ALREADY = len(NAMED_SCALERS_BASE) + len(ONE_HOTS) + len(TEXT_INPUTS)
SCALERS = (NAMED_SCALERS_BASE
           + synthetic_scalers(TARGET_FEATURES - _NAMED_ALREADY)
           + WIDE_SCALERS)

# The named model's float inputs, paired with 32-bit scalers only. A 64-bit scaler
# would demand a double on the wire and then narrow it to feed an FP32 input, which
# buys nothing -- that path earns its keep when the model input is FP64 as well.
NAMED_SCALARS = [name for name, spec in SCALERS if spec["type"].endswith("32")]

# The raw model's numeric inputs, split by the precision the value arrives in. It makes
# the same distinction the transformation suffixes make -- a 32-bit scaler's value feeds
# an FP32 input and a 64-bit one an FP64 input -- but honours it in the graph, taking the
# double and casting it there rather than having the service do the narrowing. So the raw
# model keeps every feature the vectorized configuration has, including the six the named
# model leaves out.
RAW_SCALARS_32 = [(name, spec) for name, spec in SCALERS if spec["type"].endswith("32")]
RAW_SCALARS_64 = [(name, spec) for name, spec in SCALERS if spec["type"].endswith("64")]

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


def affine_of(spec):
    """One scaler as `(value - offset) * scale`, for the version living in the graph.

    Min-max scaling is `(x - min) / (max - min)` and standardization is
    `(x - mean) / std_dev`, so both are an affine and 200 of them are two constant
    vectors -- one `Sub` and one `Mul` for the whole numeric block, rather than 200
    per-feature subgraphs. Which is the point of doing it in the graph at all.
    """
    if spec["type"].startswith("min_max"):
        return spec["min"], 1.0 / (spec["max"] - spec["min"])
    return spec["mean"], 1.0 / spec["std_dev"]


def plausible_range(spec):
    """The raw values a scaler expects to see, for the load generator to sample from.

    Read off the transformation's own parameters, so the numbers on the wire land where
    the scaling was defined for: a min-max scaler's own bounds, and three standard
    deviations either side of a standardization's mean. The service derives the same
    range from the same fields when it is the one transforming, so a vectorized run and
    a raw run put the same values on the wire.
    """
    if spec["type"].startswith("min_max"):
        return spec["min"], spec["max"]
    return spec["mean"] - 3.0 * spec["std_dev"], spec["mean"] + 3.0 * spec["std_dev"]


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
        # A pinned graph needs every batch padded up to its size, which is what
        # is_fixed declares; without it a short last batch is a shape mismatch.
        **({"mini_batch_size": rows, "is_fixed": True} if rows is not None else {}),
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
        # A pinned graph needs every batch padded up to its size, which is what
        # is_fixed declares; without it a short last batch is a shape mismatch.
        **({"mini_batch_size": rows, "is_fixed": True} if rows is not None else {}),
        "vectorization_config": {
            "feature_transformations": transformations,
            "model_inputs": inputs,
        },
    }


def raw_config(model_id, model_path, provider, rows=None):
    """No `vectorization_config` at all: the model's inputs *are* the features.

    Which is the whole point of this shape. Omitting the field is how hushar is told to
    feed every model input from the feature of its own name, typed from the model's own
    signature -- so there is nothing to declare and nothing that can disagree with the
    graph. The scaling, one-hot encoding and embedding lookups the other two
    configurations spell out are inside the model instead.

    `rows` pins the batch size, as in `vectorized_config`.
    """
    return {
        "model_id": model_id,
        "model_path": model_path,
        "execution_provider": provider,
        # A pinned graph needs every batch padded up to its size, which is what
        # is_fixed declares; without it a short last batch is a shape mismatch.
        **({"mini_batch_size": rows, "is_fixed": True} if rows is not None else {}),
    }


def raw_vocabularies():
    """Every string input the raw graph maps to a row index itself.

    The categories and the embedding keys are the same lists the other two
    configurations carry, so all three shapes see the same feature values -- only the
    party doing the lookup changes.
    """
    vocabulary = {name: words(f"{name}_v", vocab) for name, _, vocab in EMBEDDINGS}
    vocabulary.update({name: sorted(words(f"{name}_c", w)) for name, w in ONE_HOTS})
    vocabulary.update({name: words(f"{name}_v", VOCAB) for name in TEXT_INPUTS})
    return vocabulary


def raw_feature_spec():
    """What the load generator sends for each of the raw model's features.

    The client reads a model configuration to learn which features to send, and a raw
    model's configuration mentions none -- the graph's inputs are the features. So this
    file takes its place, and says only what a generator needs: a range, a width, or a
    word list.

    Written from the same schema the other two configurations are, so the values on the
    wire are the same in all three runs. A client inventing its own feature names would
    make every feature fall back to its default and measure a request path production
    never sees.
    """
    vocabulary = raw_vocabularies()
    spec = {}
    for name, _, _ in EMBEDDINGS:
        spec[name] = {"kind": "choice", "words": vocabulary[name]}
    for name, _ in ONE_HOTS:
        spec[name] = {"kind": "choice", "words": vocabulary[name]}
    for name, width in IDENTITIES:
        spec[name] = {"kind": "float_array", "width": width}
    for name, scaler in SCALERS:
        low, high = plausible_range(scaler)
        spec[name] = {
            # The suffix decides the wire variant, as it does everywhere else: a 64-bit
            # scaler's feature feeds an FP64 input, which takes a double and not a float.
            "kind": "double" if scaler["type"].endswith("64") else "float",
            "low": low,
            "high": high,
        }
    for name in TEXT_INPUTS:
        spec[name] = {"kind": "choice", "words": vocabulary[name]}
    return spec


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


def zeroed_last_row(table):
    """Zeros an embedding table's last row, which is where out of vocabulary lands.

    The configured `embedding` transformation answers an unknown key with `default_val`,
    all zeros. Matching it keeps the two paths equivalent, so the only difference between
    them is which side did the lookup.
    """
    with torch.no_grad():
        table.weight[-1].zero_()
    return table


class RawFeatures(nn.Module):
    """Raw features in -- unscaled numbers and category names -- one dense vector out.

    Every transformation the vectorized configuration describes happens here instead,
    inside the graph:

        scalers      concatenated, then one `(x - offset) * scale` over the block
        64-bit ones  cast to FP32 first, which is all the wider input costs
        one-hots     a lookup in a frozen identity table whose last row is zeros
        embeddings   a lookup in a learned table whose last row is zeros
        arrays       straight through, already the vector the model wants
        text         a lookup in a learned table, returned separately

    The dense vector has the same layout and width as the vectorized model's single
    input, so a raw model and the vectorized one differ in *where* featurization runs and
    in nothing else. That is what makes their latencies worth subtracting.

    Text comes back on its own rather than concatenated, because the two trunks want it
    differently: the encoder takes each vector as a token, and the ranking trunk has no
    token axis to put it on and concatenates it. Everything else about this module is
    shared between them weight for weight.

    Lookups rather than a `OneHot` operator, because a one-hot *is* a lookup in an
    identity matrix and `Gather` is the op every execution provider here implements. The
    table carries one row more than it has categories, and the string front end sends
    anything out of vocabulary to it -- so an unknown category contributes zeros, exactly
    as `default_val` does on the configured path.

    Strings arrive already encoded to a row index, because mapping a string to an index
    is not something torch can export. Those inputs are replaced by the string front end
    after export -- see `add_string_front_end`.
    """

    def __init__(self, scalars32, scalars64, arrays, one_hots, embeddings, texts):
        super().__init__()
        offsets, scales = zip(*(affine_of(spec) for _, spec in scalars32 + scalars64))
        # Buffers rather than parameters: they are the transformation, not something
        # learned, so they belong with the graph and not in what a size target counts.
        self.register_buffer("offset", torch.tensor(offsets, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scales, dtype=torch.float32))
        self.narrow, self.wide = len(scalars32), len(scalars64)
        self.array_widths = [width for _, width in arrays]
        self.one_hots = nn.ModuleList(
            nn.Embedding.from_pretrained(
                torch.cat([torch.eye(width), torch.zeros(1, width)]), freeze=True
            )
            for _, width in one_hots
        )
        self.embeddings = nn.ModuleList(
            zeroed_last_row(nn.Embedding(vocab + 1, width))
            for _, width, vocab in embeddings
        )
        self.tables = nn.ModuleList(nn.Embedding(VOCAB + 1, D_MODEL) for _ in texts)
        self.width = (
            self.narrow
            + self.wide
            + sum(self.array_widths)
            + sum(width for _, width in one_hots)
            + sum(width for _, width, _ in embeddings)
        )
        # What a trunk with no token axis has to take on as well.
        self.text_width = len(texts) * D_MODEL

    def forward(self, *inputs):
        taken = 0

        def take(count):
            """The next `count` inputs, in the order `export` names them."""
            nonlocal taken
            chunk = inputs[taken:taken + count]
            taken += count
            return chunk

        narrow = take(self.narrow)
        wide = take(self.wide)
        arrays = take(len(self.array_widths))
        one_hot_ids = take(len(self.one_hots))
        embedding_ids = take(len(self.embeddings))
        text_ids = take(len(self.tables))

        # One Sub and one Mul for every scaler in the schema, rather than a subgraph per
        # feature: min-max scaling and standardization are the same affine, so 200 of
        # them are two constant vectors -- see affine_of.
        numbers = torch.cat(narrow + tuple(w.to(torch.float32) for w in wide), dim=-1)
        # Concatenated in the vectorized model's feature order, so this is the tensor the
        # service would otherwise have handed us.
        parts = [table(ids.reshape(-1)) for table, ids in zip(self.embeddings, embedding_ids)]
        parts += [table(ids.reshape(-1)) for table, ids in zip(self.one_hots, one_hot_ids)]
        parts += list(arrays)
        parts.append((numbers - self.offset) * self.scale)

        texts = tuple(
            table(ids.reshape(-1)) for table, ids in zip(self.tables, text_ids)
        )
        return torch.cat(parts, dim=-1), texts


class RawTransformer(nn.Module):
    """Raw features into the encoder: the dense vector becomes tokens, text one each."""

    def __init__(self, features, layers):
        super().__init__()
        self.features = features
        self.dense_tokens = N_TOKENS - len(features.tables)
        self.to_tokens = nn.Linear(features.width, self.dense_tokens * D_MODEL)
        self.encoder = Encoder(layers)

    def forward(self, *inputs):
        dense, texts = self.features(*inputs)
        tokens = [self.to_tokens(dense).reshape(-1, self.dense_tokens, D_MODEL)]
        tokens += [text.reshape(-1, 1, D_MODEL) for text in texts]
        return self.encoder(torch.cat(tokens, dim=1))


class RawCrossMaskNet(nn.Module):
    """Raw features into the ranking trunk: everything in one vector, text included."""

    def __init__(self, features, blocks):
        super().__init__()
        self.features = features
        self.trunk = CrossMaskNet(features.width + features.text_width, blocks)

    def forward(self, *inputs):
        dense, texts = self.features(*inputs)
        return self.trunk(torch.cat((dense,) + texts, dim=-1))


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

    A `LabelEncoder` per string input turns the value into a row index, defaulting to
    the last row for anything out of vocabulary, and the exported `Gather` then reads
    the table. This is the whole reason a string reaches the model at all: the service
    passes text through untouched, so the mapping can live with the weights it belongs
    to instead of in configuration.

    Text, categories and embedding keys all take this route -- the difference between
    them is only what the table they index holds.
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
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    vocabulary = {name: words(f"{name}_v", VOCAB) for name in TEXT_INPUTS}
    written, described = {}, []

    # ---- vectorized
    template = vectorized_config(rng, "", "cpu")
    vectorization = template["vectorization_config"]
    width = sum(len(vectorization["feature_transformations"][n]["default_val"])
                for n in vectorization["feature_order"])
    layers, total = depth_for_target(lambda n: VectorizedModel(width, n), target_params)
    print(f"vectorized : width {width}, transformer {layers} layers, "
          f"{total:,} parameters")
    model = VectorizedModel(width, layers).eval()

    for suffix, rows in batches:
        name = f"bench_model{suffix}"
        path = os.path.join(out_dir, f"{name}.onnx")
        export(model, torch.randn(rows or 2, width), path, ["features"], rows=rows)
        config = vectorized_config(rng, f"{out_rel}/{name}.onnx", "cpu", rows=rows)
        config["model_id"] = f"bench-vectorized{suffix.replace('_', '-')}"
        written[f"{name}_config.json"] = config
        described.append((f"vectorized rows={rows or 'dynamic'}", path))

    # ---- named
    scalars, arrays = NAMED_SCALARS, ONE_HOTS
    build = lambda n: NamedModel(scalars, arrays, TEXT_INPUTS, n)
    named_total = parameters_of(build(layers))
    dense_width = len(scalars) + sum(w for _, w in arrays)
    features = len(scalars) + len(arrays) + len(TEXT_INPUTS)
    print(f"named      : {features} features = {len(scalars)} scalars + "
          f"{len(arrays)} arrays ({dense_width} dense) + {len(TEXT_INPUTS)} strings, "
          f"transformer {layers} layers, {named_total:,} parameters")
    input_names = scalars + [n for n, _ in arrays] + [f"{n}_id" for n in TEXT_INPUTS]
    named_model = build(layers).eval()

    for suffix, rows in batches:
        name = f"bench_named_model{suffix}"
        path = os.path.join(out_dir, f"{name}.onnx")
        batch = rows or 2
        example = (tuple(torch.randn(batch, 1) for _ in scalars)
                   + tuple(torch.randn(batch, w) for _, w in arrays)
                   + tuple(torch.zeros(batch, 1, dtype=torch.int64)
                           for _ in TEXT_INPUTS))
        export(named_model, example, path, input_names, rows=rows)
        add_string_front_end(path, vocabulary, rows=rows)
        config = named_config(f"{out_rel}/{name}.onnx", "cpu", rows=rows)
        config["model_id"] = f"bench-named{suffix.replace('_', '-')}"
        written[f"{name}_config.json"] = config
        described.append((f"named rows={rows or 'dynamic'}", path))

    # ---- raw, one model per architecture: featurization inside the graph
    #
    # The dense vector it builds is the vectorized model's input, so the two are only
    # comparable while the layouts agree. Checked rather than assumed: a feature added to
    # one list and not the other would quietly make them different models, and the whole
    # pair exists to hold everything but the featurization site still.
    raw_strings = [n for n, _ in ONE_HOTS] + [n for n, _, _ in EMBEDDINGS] + TEXT_INPUTS
    raw_order = ([n for n, _, _ in EMBEDDINGS] + [n for n, _ in ONE_HOTS]
                 + [n for n, _ in IDENTITIES]
                 + [n for n, _ in RAW_SCALARS_32] + [n for n, _ in RAW_SCALARS_64])
    if raw_order != vectorization["feature_order"]:
        raise SystemExit(
            "the raw model's dense layout no longer matches the vectorized model's "
            "feature_order, so the two would not be comparable"
        )

    raw_features = lambda: RawFeatures(RAW_SCALARS_32, RAW_SCALARS_64, IDENTITIES,
                                       ONE_HOTS, EMBEDDINGS, TEXT_INPUTS)
    # The ranking trunk's block count, solved the same way the encoder's layer count was.
    blocks, _ = depth_for_target(
        lambda n: RawCrossMaskNet(raw_features(), n), target_params
    )
    feature_count = len(raw_order) + len(TEXT_INPUTS)
    print(f"raw        : {feature_count} features = {len(RAW_SCALARS_32)} float + "
          f"{len(RAW_SCALARS_64)} double + {len(IDENTITIES)} array + "
          f"{len(ONE_HOTS) + len(EMBEDDINGS)} category + {len(TEXT_INPUTS)} text, "
          f"scaled and looked up in the graph into the same {width} dense")
    raw_input_names = ([n for n, _ in RAW_SCALARS_32] + [n for n, _ in RAW_SCALARS_64]
                       + [n for n, _ in IDENTITIES]
                       + [f"{n}_id" for n in raw_strings])
    raw_vocabulary = raw_vocabularies()

    # Two trunks on that one front end. Reseeded before each so they are built from the
    # same weights for the part they share, leaving the trunk as the only difference.
    architectures = [
        ("bench_raw_model", "bench-raw", "transformer",
         lambda: RawTransformer(raw_features(), layers), f"{layers} encoder layers"),
        ("bench_raw_dcn_model", "bench-raw-dcn", "dcn",
         lambda: RawCrossMaskNet(raw_features(), blocks),
         f"DCNv2 {CROSS_LAYERS} cross + MaskNet {blocks} blocks at {D_DEEP} wide"),
    ]

    for stem, model_id, label, build_arch, shape in architectures:
        torch.manual_seed(0)
        raw_model = build_arch().eval()
        print(f"  {label:<11}: {shape}, {parameters_of(raw_model):,} parameters")

        for suffix, rows in batches:
            name = f"{stem}{suffix}"
            path = os.path.join(out_dir, f"{name}.onnx")
            batch = rows or 2
            # The 64-bit scalers are exported as doubles, so a request carries the
            # double the schema asks for and the graph does the narrowing.
            example = (tuple(torch.randn(batch, 1) for _ in RAW_SCALARS_32)
                       + tuple(torch.randn(batch, 1, dtype=torch.float64)
                               for _ in RAW_SCALARS_64)
                       + tuple(torch.randn(batch, w) for _, w in IDENTITIES)
                       + tuple(torch.zeros(batch, 1, dtype=torch.int64)
                               for _ in raw_strings))
            export(raw_model, example, path, raw_input_names, rows=rows)
            add_string_front_end(path, raw_vocabulary, rows=rows)
            dashed = suffix.replace("_", "-")
            written[f"{name}_config.json"] = raw_config(
                f"{model_id}{dashed}", f"{out_rel}/{name}.onnx", "cpu", rows=rows
            )
            # The candidate arm: the same file under a second name, so a roll-out can be
            # served without a second set of weights to export, copy and keep straight.
            written[f"{name}_{CANDIDATE}_config.json"] = raw_config(
                f"{model_id}{dashed}-{CANDIDATE}", f"{out_rel}/{name}.onnx", "cpu",
                rows=rows,
            )
            described.append((f"raw {label} rows={rows or 'dynamic'}", path))

    written["bench_service_config.json"] = service_config(
        f"{out_rel}/bench_raw_model_config.json"
    )
    written["bench_dcn_service_config.json"] = service_config(
        f"{out_rel}/bench_raw_dcn_model_config.json"
    )
    # The two-model configs: both loaded, traffic split between them. Separate from the
    # single-model configs above because most runs want one model, and a run that wants
    # two should have to say so -- loading the second costs its whole size in memory and
    # produces metrics nobody asked for.
    #
    # This pair is the interesting one: both architectures resident, half the traffic
    # each, so their latencies come from one session on one host with nothing else moving.
    written["bench_ab_service_config.json"] = service_config(
        f"{out_rel}/bench_raw_model_config.json",
        f"{out_rel}/bench_raw_dcn_model_config.json",
    )
    # And the roll-out shape: one model loaded twice, which is what two arms of the same
    # architecture cost.
    written["bench_same_ab_service_config.json"] = service_config(
        f"{out_rel}/bench_raw_model_config.json",
        f"{out_rel}/bench_raw_model_{CANDIDATE}_config.json",
    )
    written["bench_string_vocabulary.json"] = vocabulary
    # A raw model's configuration says nothing about features, so the load generator is
    # told what to send by this instead -- see raw_feature_spec. One file for both
    # architectures, because they take the same features.
    written["bench_raw_feature_spec.json"] = raw_feature_spec()
    for name, body in written.items():
        with open(os.path.join(out_dir, name), "w") as handle:
            json.dump(body, handle, indent=2)
            handle.write("\n")

    print()
    for label, model_path in described:
        megabytes = os.path.getsize(model_path) / (1024 * 1024)
        model_inputs, outputs = signature(model_path)
        print(f"{label} -> {os.path.relpath(model_path, ROOT)} ({megabytes:.1f} MB)")
        print(f"  inputs  : {summarise(model_inputs)}")
        print(f"  outputs : {summarise(outputs)}")
    print(f"\nwrote {len(written)} files to {os.path.relpath(out_dir, ROOT)}")


if __name__ == "__main__":
    main()
