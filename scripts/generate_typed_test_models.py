# Copyright (c) 2025 Pratik Barhate
# Licensed under the MIT License. See the LICENSE file in the project root for more information.

"""Generates the fixtures that exercise typed, heterogeneous tensor I/O.

`generate_test_model.py` builds the single-input f32 model the original request
path used. These three cover what that one cannot, and each exists to make a
specific claim testable:

* `float_multi_io.onnx` -- one f32 input, two f32 outputs of different widths.
  This is what hushar itself serves: the service is float-in, float-out, and this
  covers multi-output scoring without needing a type it refuses.
* `f64_scale.onnx` -- DOUBLE in and out, so the FP64 path is exercised against a
  real engine rather than only asserted in a unit test.
* `per_feature_io.onnx` -- one input per feature (`age`, `city`) plus a text input
  (`tags`), so `feature_order` is unused. Covers the per-feature and vector-plus-text
  signatures.
* `numbers_and_text.onnx` -- an f32 vector plus a text input, both outputs f32. The
  vector-plus-text signature hushar serves. The label code is cast to float, because a
  response carries FP32 or FP64 scores.
* `mixed_io.onnx` -- two inputs of *different element types* (f32 and string) and
  four outputs of different types and shapes. This is the case
  `Session::run<T: Element>` cannot express at any dtype count, and the reason
  `InputValue` exists. It stays for `onnxrt-rs`, whose job is the whole ONNX type
  system. hushar cannot serve it -- its INT64 and STRING *outputs* are refused when the
  model loads -- which is what makes it the fixture for testing that refusal.
* `f16_identity.onnx` -- FLOAT16 in and out, to prove `half::f16` really is
  ONNX's 16-bit layout and survives a round trip rather than merely compiling.
* `bool_not.onnx` -- BOOL in and out, for the type that is deliberately not an
  `Element`.

Each fixture is written to the `test-data` of every crate whose tests read it, and
nowhere else. A crate's tests resolve fixtures under its own manifest directory, so
a suite depends on that crate and a `libonnxruntime` and nothing more -- `onnxrt-rs`
is publishable on its own, and reaching into `hushar/test-data` would have made its
suite unrunnable outside this workspace. `mixed_io` is the one fixture both crates
test, so it is written twice from one definition here; copying it by hand is what
would let the two drift.

Built with the ONNX helper API rather than exported from PyTorch, because torch
cannot express a string input.

    conda activate local && python scripts/generate_typed_test_models.py
"""

import pathlib

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HUSHAR = pathlib.Path("hushar/test-data")
ONNXRT = pathlib.Path("onnxrt-rs/test-data")

# Fixed so results are reproducible across runs and machines.
WEIGHTS = np.full((3, 2), 0.5, dtype=np.float32)
BIAS = np.full((2,), 0.1, dtype=np.float32)
EMBED = np.arange(3 * 4, dtype=np.float32).reshape(3, 4) / 10.0

TAG_KEYS = ["alpha", "beta", "gamma"]
TAG_VALUES = [10, 20, 30]
TAG_DEFAULT = -1


def mixed_io() -> onnx.ModelProto:
    """f32 + string in; f32, f32 and i64 out."""
    nodes = [
        # scores = sigmoid(numbers @ W + b)   -> [batch, 2]
        helper.make_node("MatMul", ["numbers", "weights"], ["projected"]),
        helper.make_node("Add", ["projected", "bias"], ["biased"]),
        helper.make_node("Sigmoid", ["biased"], ["scores"]),
        # embedding = numbers @ E             -> [batch, 4]
        helper.make_node("MatMul", ["numbers", "embed"], ["embedding"]),
        # codes = lookup(tags)                -> [batch]
        # LabelEncoder is the one ONNX operator that consumes strings and emits
        # integers, which is what makes a string input observable in the output.
        helper.make_node(
            "LabelEncoder",
            ["tags"],
            ["codes"],
            domain="ai.onnx.ml",
            keys_strings=TAG_KEYS,
            values_int64s=TAG_VALUES,
            default_int64=TAG_DEFAULT,
        ),
        # echoed = tags                       -> [batch], STRING
        # Gives the tests a string *output* to read back, which is the mirror of
        # the string input and the only way to cover `OwnedTensor::strings`.
        helper.make_node("Identity", ["tags"], ["echoed"]),
    ]

    graph = helper.make_graph(
        nodes,
        "mixed_io",
        inputs=[
            helper.make_tensor_value_info("numbers", TensorProto.FLOAT, ["batch", 3]),
            helper.make_tensor_value_info("tags", TensorProto.STRING, ["batch"]),
        ],
        outputs=[
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["batch", 2]),
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 4]),
            helper.make_tensor_value_info("codes", TensorProto.INT64, ["batch"]),
            helper.make_tensor_value_info("echoed", TensorProto.STRING, ["batch"]),
        ],
        initializer=[
            numpy_helper.from_array(WEIGHTS, "weights"),
            numpy_helper.from_array(BIAS, "bias"),
            numpy_helper.from_array(EMBED, "embed"),
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("ai.onnx.ml", 2),
        ],
    )


def float_multi_io() -> onnx.ModelProto:
    """f32 in; two f32 outputs of different widths. What hushar actually serves.

    Deliberately the same arithmetic as `mixed_io`, so the expected values are the
    same and a test can be moved between them without recomputing anything.
    """
    nodes = [
        # scores = sigmoid(numbers @ W + b)   -> [batch, 2]
        helper.make_node("MatMul", ["numbers", "weights"], ["projected"]),
        helper.make_node("Add", ["projected", "bias"], ["biased"]),
        helper.make_node("Sigmoid", ["biased"], ["scores"]),
        # embedding = numbers @ E             -> [batch, 4]
        helper.make_node("MatMul", ["numbers", "embed"], ["embedding"]),
    ]
    graph = helper.make_graph(
        nodes,
        "float_multi_io",
        inputs=[helper.make_tensor_value_info("numbers", TensorProto.FLOAT, ["batch", 3])],
        outputs=[
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["batch", 2]),
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["batch", 4]),
        ],
        initializer=[
            numpy_helper.from_array(WEIGHTS, "weights"),
            numpy_helper.from_array(BIAS, "bias"),
            numpy_helper.from_array(EMBED, "embed"),
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def numbers_and_text() -> onnx.ModelProto:
    """f32 vector + text in; f32 out. The vector-plus-text signature hushar serves.

    `code` is a `LabelEncoder` lookup on the text, cast to float. The cast matters: a
    response carries FP32 or FP64 scores, so an INT64 output is refused when the model
    loads. Casting keeps the string input's effect observable -- "beta" still becomes
    20 -- in a type the service can return.
    """
    nodes = [
        helper.make_node("MatMul", ["numbers", "weights"], ["projected"]),
        helper.make_node("Add", ["projected", "bias"], ["biased"]),
        helper.make_node("Sigmoid", ["biased"], ["scores"]),
        helper.make_node(
            "LabelEncoder",
            ["tags"],
            ["code_i64"],
            domain="ai.onnx.ml",
            keys_strings=TAG_KEYS,
            values_int64s=TAG_VALUES,
            default_int64=TAG_DEFAULT,
        ),
        helper.make_node("Cast", ["code_i64"], ["code"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        "numbers_and_text",
        inputs=[
            helper.make_tensor_value_info("numbers", TensorProto.FLOAT, ["batch", 3]),
            helper.make_tensor_value_info("tags", TensorProto.STRING, ["batch"]),
        ],
        outputs=[
            helper.make_tensor_value_info("scores", TensorProto.FLOAT, ["batch", 2]),
            helper.make_tensor_value_info("code", TensorProto.FLOAT, ["batch"]),
        ],
        initializer=[
            numpy_helper.from_array(WEIGHTS, "weights"),
            numpy_helper.from_array(BIAS, "bias"),
        ],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("ai.onnx.ml", 2),
        ],
    )


def per_feature_io() -> onnx.ModelProto:
    """One input per feature, plus a text input. Signature 2 and 3 in one model.

    `age` is a scalar feature, `city` a 5-wide one-hot, and `tags` text passed through
    untransformed. Nothing is concatenated, so `feature_order` is irrelevant -- which
    is the property the tests need to observe.
    """
    city_weights = np.full((5, 1), 0.2, dtype=np.float32)
    nodes = [
        # score = age + sum(city * 0.2)      -> [batch, 1]
        helper.make_node("MatMul", ["city", "city_w"], ["city_part"]),
        helper.make_node("Add", ["age", "city_part"], ["score"]),
        # code = lookup(tags)                -> [batch]
        helper.make_node(
            "LabelEncoder",
            ["tags"],
            ["code_i64"],
            domain="ai.onnx.ml",
            keys_strings=TAG_KEYS,
            values_int64s=TAG_VALUES,
            default_int64=TAG_DEFAULT,
        ),
        helper.make_node("Cast", ["code_i64"], ["code"], to=TensorProto.FLOAT),
    ]
    graph = helper.make_graph(
        nodes,
        "per_feature_io",
        inputs=[
            helper.make_tensor_value_info("age", TensorProto.FLOAT, ["batch", 1]),
            helper.make_tensor_value_info("city", TensorProto.FLOAT, ["batch", 5]),
            helper.make_tensor_value_info("tags", TensorProto.STRING, ["batch", 1]),
        ],
        outputs=[
            helper.make_tensor_value_info("score", TensorProto.FLOAT, ["batch", 1]),
            helper.make_tensor_value_info("code", TensorProto.FLOAT, ["batch", 1]),
        ],
        initializer=[numpy_helper.from_array(city_weights, "city_w")],
    )
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 13),
            helper.make_opsetid("ai.onnx.ml", 2),
        ],
    )


def f64_scale() -> onnx.ModelProto:
    """DOUBLE in, DOUBLE out, doubled then offset so the test can tell it ran."""
    two = numpy_helper.from_array(np.array([2.0], dtype=np.float64), "two")
    one = numpy_helper.from_array(np.array([1.0], dtype=np.float64), "one")
    graph = helper.make_graph(
        [
            helper.make_node("Mul", ["input", "two"], ["scaled"]),
            helper.make_node("Add", ["scaled", "one"], ["output"]),
        ],
        "f64_scale",
        inputs=[helper.make_tensor_value_info("input", TensorProto.DOUBLE, ["batch", 3])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.DOUBLE, ["batch", 3])],
        initializer=[two, one],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def f16_identity() -> onnx.ModelProto:
    """FLOAT16 in, FLOAT16 out, doubled so the test can tell it ran."""
    two = numpy_helper.from_array(np.array([2.0], dtype=np.float16), "two")
    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "two"], ["output"])],
        "f16_identity",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT16, ["batch", 3])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT16, ["batch", 3])],
        initializer=[two],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def bool_not() -> onnx.ModelProto:
    """BOOL in, BOOL out. Negated, so an all-false result cannot pass by accident."""
    graph = helper.make_graph(
        [helper.make_node("Not", ["input"], ["output"])],
        "bool_not",
        inputs=[helper.make_tensor_value_info("input", TensorProto.BOOL, ["batch"])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, ["batch"])],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


if __name__ == "__main__":
    # Each fixture names the crates that test it. A fixture is written to those
    # `test-data` directories and no others, so what a directory holds is exactly
    # what that crate's tests read.
    for name, build, destinations in [
        ("mixed_io", mixed_io, [HUSHAR, ONNXRT]),
        ("float_multi_io", float_multi_io, [HUSHAR]),
        ("f64_scale", f64_scale, [HUSHAR]),
        ("per_feature_io", per_feature_io, [HUSHAR]),
        ("numbers_and_text", numbers_and_text, [HUSHAR]),
        ("f16_identity", f16_identity, [ONNXRT]),
        ("bool_not", bool_not, [ONNXRT]),
    ]:
        model = build()
        onnx.checker.check_model(model)
        for out_dir in destinations:
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{name}.onnx"
            onnx.save(model, path)
            print(f"wrote {path} ({path.stat().st_size} bytes)")
