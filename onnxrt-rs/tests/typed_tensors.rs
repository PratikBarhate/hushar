// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Typed, heterogeneous tensor I/O against a real ONNX Runtime.
//!
//! [`onnxrt_rs::Session::run`] is generic over one [`onnxrt_rs::Element`], so every
//! input in such a call holds the same Rust type; `run_many` and `run_named` take
//! inputs of differing element types.
//!
//! The fixtures come from `scripts/generate_typed_test_models.py`, which writes them
//! into this crate's own `test-data`. Each test skips with a reason when no runtime can
//! be loaded.

use onnxrt_rs::{DataType, Error, InputValue, Session, SessionBuilder};

mod common;

use common::{api, fixture};

/// Builds a single-threaded CPU session from the named fixture.
fn session_for(name: &str) -> Session {
    SessionBuilder::new(&common::environment())
        .expect("session builder")
        .intra_op_threads(1)
        .expect("intra op threads")
        .build_from_memory(&fixture(name))
        .expect("session")
}

/// The mixed model's expected outputs for `numbers = [1, 2, 3]`, `tags = ["beta"]`,
/// cross-checked against Python onnxruntime when the fixture was generated.
const EXPECTED_SCORES: [f32; 2] = [0.9568927, 0.9568927];
const EXPECTED_EMBEDDING: [f32; 4] = [3.2, 3.8, 4.4, 5.0];
const EXPECTED_CODE: i64 = 20;

/// Asserts one f32 and one string input run in a single `run_many` call, and all four
/// outputs read back as expected — floats within 1e-5, and the `codes` output, which
/// only a genuinely read string input can produce.
#[test]
fn inputs_of_different_element_types_run_in_one_call() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    let mut numbers = vec![1.0f32, 2.0, 3.0];
    let outputs = session
        .run_many(&[
            InputValue::numeric(api, &mut numbers, &[1, 3]).expect("f32 input"),
            InputValue::strings(api, &["beta"], &[1]).expect("string input"),
        ])
        .expect("heterogeneous inference should succeed");

    assert_eq!(outputs.len(), 4, "model declares four outputs");

    let scores = outputs[0].as_slice::<f32>().expect("f32 scores");
    for (i, (got, want)) in scores.iter().zip(EXPECTED_SCORES.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "score {i} was {got}, expected {want}"
        );
    }

    let embedding = outputs[1].as_slice::<f32>().expect("f32 embedding");
    for (i, (got, want)) in embedding.iter().zip(EXPECTED_EMBEDDING.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "embedding {i} was {got}, expected {want}"
        );
    }

    let codes = outputs[2].as_slice::<i64>().expect("i64 codes");
    assert_eq!(codes, [EXPECTED_CODE]);

    let echoed = outputs[3].strings().expect("string output");
    assert_eq!(echoed, vec!["beta".to_owned()]);
}

/// Asserts one call returns four outputs carrying their own element types and shapes,
/// each row encoded from its own tag and in order.
///
/// The outputs in index order are logits (one row per input row), embedding, label, and
/// echoed text.
#[test]
fn outputs_carry_their_own_types_and_shapes() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    let mut numbers = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let outputs = session
        .run_many(&[
            InputValue::numeric(api, &mut numbers, &[2, 3]).expect("f32 input"),
            InputValue::strings(api, &["alpha", "gamma"], &[2]).expect("string input"),
        ])
        .expect("inference should succeed");

    let described: Vec<(DataType, Vec<i64>)> = outputs
        .iter()
        .map(|t| (t.data_type().expect("dtype"), t.shape().expect("shape")))
        .collect();

    assert_eq!(
        described,
        vec![
            (DataType::F32, vec![2, 2]),
            (DataType::F32, vec![2, 4]),
            (DataType::I64, vec![2]),
            (DataType::String, vec![2]),
        ]
    );

    assert_eq!(
        outputs[2].as_slice::<i64>().expect("codes"),
        [10, 30],
        "each row's own tag should be encoded, in order"
    );
}

/// Asserts `run_named` matches inputs by name, supplying them in the opposite order to
/// the model's declaration.
#[test]
fn run_named_matches_inputs_by_name_not_position() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    let mut numbers = vec![1.0f32, 2.0, 3.0];
    let outputs = session
        .run_named(&[
            (
                "tags",
                InputValue::strings(api, &["beta"], &[1]).expect("strings"),
            ),
            (
                "numbers",
                InputValue::numeric(api, &mut numbers, &[1, 3]).expect("f32"),
            ),
        ])
        .expect("named inference should succeed regardless of order");

    assert_eq!(
        outputs[2].as_slice::<i64>().expect("codes"),
        [EXPECTED_CODE]
    );
}

/// Asserts omitting a declared input is an `InputNotSuppliedOnce` with a count of 0.
#[test]
fn run_named_rejects_an_input_that_was_not_supplied() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    let mut numbers = vec![1.0f32, 2.0, 3.0];
    let err = session
        .run_named(&[(
            "numbers",
            InputValue::numeric(api, &mut numbers, &[1, 3]).expect("f32"),
        )])
        .expect_err("omitting a declared input must fail");

    match err {
        Error::InputNotSuppliedOnce { name, count } => {
            assert_eq!(name, "tags");
            assert_eq!(count, 0);
        }
        other => panic!("expected InputNotSuppliedOnce, got {other:?}"),
    }
}

/// Asserts an undeclared input name is an `UnknownInput` listing the real names, rather
/// than being silently dropped.
#[test]
fn run_named_rejects_an_unknown_input_name() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    let mut numbers = vec![1.0f32, 2.0, 3.0];
    let err = session
        .run_named(&[
            (
                "numbers",
                InputValue::numeric(api, &mut numbers, &[1, 3]).expect("f32"),
            ),
            (
                "tags",
                InputValue::strings(api, &["beta"], &[1]).expect("strings"),
            ),
            (
                "tag",
                InputValue::strings(api, &["beta"], &[1]).expect("strings"),
            ),
        ])
        .expect_err("an undeclared input name must fail");

    match err {
        Error::UnknownInput { name, declared } => {
            assert_eq!(name, "tag");
            assert!(
                declared.contains("tags"),
                "error should list the real names, got {declared:?}"
            );
        }
        other => panic!("expected UnknownInput, got {other:?}"),
    }
}

/// Asserts supplying one input twice is an `InputNotSuppliedOnce` reporting the count,
/// rather than resolving the ambiguity.
#[test]
fn run_named_rejects_a_duplicated_input() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    let mut a = vec![1.0f32, 2.0, 3.0];
    let mut b = vec![4.0f32, 5.0, 6.0];
    let err = session
        .run_named(&[
            (
                "numbers",
                InputValue::numeric(api, &mut a, &[1, 3]).expect("f32"),
            ),
            (
                "numbers",
                InputValue::numeric(api, &mut b, &[1, 3]).expect("f32"),
            ),
            (
                "tags",
                InputValue::strings(api, &["beta"], &[1]).expect("strings"),
            ),
        ])
        .expect_err("supplying one input twice must fail");

    match err {
        Error::InputNotSuppliedOnce { name, count } => {
            assert_eq!(name, "numbers");
            assert_eq!(count, 2, "ambiguity should be reported, not resolved");
        }
        other => panic!("expected InputNotSuppliedOnce, got {other:?}"),
    }
}

/// Asserts `half::f16` really is ONNX's FLOAT16 layout, by round-tripping through a
/// doubling model.
///
/// The comparison is exact: the inputs and their doubles are all representable in f16.
#[test]
fn f16_tensors_survive_a_round_trip() {
    let Some(api) = api() else { return };
    let session = session_for("f16_identity.onnx");

    use half::f16;
    let mut input = vec![
        f16::from_f32(1.5),
        f16::from_f32(-2.25),
        f16::from_f32(0.125),
    ];
    let outputs = session
        .run_many(&[InputValue::numeric(api, &mut input, &[1, 3]).expect("f16 input")])
        .expect("f16 inference should succeed");

    assert_eq!(outputs[0].data_type().expect("dtype"), DataType::F16);
    let got: Vec<f32> = outputs[0]
        .as_slice::<f16>()
        .expect("f16 output")
        .iter()
        .map(|h| h.to_f32())
        .collect();

    assert_eq!(got, vec![3.0, -4.5, 0.25]);
}

/// Asserts bool tensors round-trip through a `Not` model, coming back negated rather
/// than zeroed.
#[test]
fn bool_tensors_survive_a_round_trip() {
    let Some(api) = api() else { return };
    let session = session_for("bool_not.onnx");

    let input = [true, false, true, true];
    let outputs = session
        .run_many(&[InputValue::bools(api, &input, &[4]).expect("bool input")])
        .expect("bool inference should succeed");

    assert_eq!(outputs[0].data_type().expect("dtype"), DataType::Bool);
    assert_eq!(
        outputs[0].bools().expect("bool output"),
        vec![false, true, false, false]
    );
}

/// Asserts a session reports its own input and output names, element types and shapes,
/// with -1 for a dynamic batch axis.
#[test]
fn the_model_describes_its_own_inputs_and_outputs() {
    let Some(_api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    let inputs = session.input_specs().expect("input specs");
    assert_eq!(inputs.len(), 2);

    assert_eq!(inputs[0].name, "numbers");
    assert_eq!(inputs[0].data_type, DataType::F32);
    assert_eq!(inputs[0].shape, vec![-1, 3], "batch axis is dynamic");
    assert_eq!(inputs[0].row_width(), Some(3));
    assert_eq!(inputs[0].concrete_shape(8), vec![8, 3]);

    assert_eq!(inputs[1].name, "tags");
    assert_eq!(inputs[1].data_type, DataType::String);

    let outputs = session.output_specs().expect("output specs");
    let summary: Vec<(&str, DataType)> = outputs
        .iter()
        .map(|s| (s.name.as_str(), s.data_type))
        .collect();
    assert_eq!(
        summary,
        vec![
            ("scores", DataType::F32),
            ("embedding", DataType::F32),
            ("codes", DataType::I64),
            ("echoed", DataType::String),
        ]
    );
}

/// Asserts a string tensor whose value count disagrees with its shape is a
/// `ShapeMismatch` naming both counts, before the runtime is called.
#[test]
fn a_string_tensor_with_a_wrong_shape_is_rejected_before_the_runtime_sees_it() {
    let Some(api) = api() else { return };

    let err = InputValue::strings(api, &["a", "b"], &[3])
        .expect_err("two values cannot fill a three-element shape");
    assert!(
        matches!(
            err,
            Error::ShapeMismatch {
                expected: 3,
                actual: 2,
                ..
            }
        ),
        "expected a ShapeMismatch naming both counts, got {err:?}"
    );
}

/// Asserts reading a tensor as the wrong element type is an `ElementTypeMismatch`, for
/// `i64` as `f64` — the same width, a different meaning — and for a string tensor as
/// `f32`.
#[test]
fn reading_a_tensor_as_the_wrong_type_is_an_error_not_reinterpreted_bytes() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    let mut numbers = vec![1.0f32, 2.0, 3.0];
    let outputs = session
        .run_many(&[
            InputValue::numeric(api, &mut numbers, &[1, 3]).expect("f32"),
            InputValue::strings(api, &["beta"], &[1]).expect("strings"),
        ])
        .expect("inference");

    let err = outputs[2]
        .as_slice::<f64>()
        .expect_err("i64 must not be readable as f64");
    assert!(
        matches!(
            err,
            Error::ElementTypeMismatch {
                requested: "f64",
                ..
            }
        ),
        "expected ElementTypeMismatch, got {err:?}"
    );

    let err = outputs[3]
        .as_slice::<f32>()
        .expect_err("a string tensor must not be readable as f32");
    assert!(
        matches!(err, Error::ElementTypeMismatch { .. }),
        "got {err:?}"
    );
}

/// Asserts reading an f32 tensor with `strings()` is an `ElementTypeMismatch`.
#[test]
fn strings_reader_rejects_a_numeric_tensor() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    let mut numbers = vec![1.0f32, 2.0, 3.0];
    let outputs = session
        .run_many(&[
            InputValue::numeric(api, &mut numbers, &[1, 3]).expect("f32"),
            InputValue::strings(api, &["beta"], &[1]).expect("strings"),
        ])
        .expect("inference");

    let err = outputs[0]
        .strings()
        .expect_err("an f32 tensor must not be readable as strings");
    assert!(
        matches!(err, Error::ElementTypeMismatch { .. }),
        "got {err:?}"
    );
}

/// Asserts a borrowed numeric input and an owned `'static` string input can share one
/// call.
///
/// Numeric inputs borrow the caller's buffer, which must outlive the call; string inputs
/// own runtime memory and are `'static`. `InputValue`'s lifetime is therefore covariant,
/// without which the two could not sit in one slice and this would not compile.
#[test]
fn a_borrowed_tensor_and_an_owned_one_can_share_a_call() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    let mut numbers = vec![0.5f32, 0.5, 0.5];
    let borrowed = InputValue::numeric(api, &mut numbers, &[1, 3]).expect("f32");
    let owned: InputValue<'static> = InputValue::strings(api, &["alpha"], &[1]).expect("strings");

    let outputs = session.run_many(&[borrowed, owned]).expect("inference");
    assert_eq!(outputs[2].as_slice::<i64>().expect("codes"), [10]);
}
