// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Typed, heterogeneous tensor I/O against a real ONNX Runtime.
//!
//! These are the tests for the capability the crate did not have:
//! [`onnxrt_rs::Session::run`] is generic over one [`onnxrt_rs::Element`], so
//! every input in a call had to hold the same Rust type. A model taking a float
//! feature vector *and* a string vector was unrepresentable — not for want of a
//! dtype, but because of that single type parameter.
//!
//! The fixtures come from `scripts/generate_typed_test_models.py`, which writes
//! them into this crate's own `test-data`. Like the rest of the suite, each test
//! skips with a reason when no runtime can be loaded.

use onnxrt_rs::{DataType, Error, InputValue, Session, SessionBuilder};

mod common;

use common::{api, fixture};

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

#[test]
fn inputs_of_different_element_types_run_in_one_call() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    // The whole point: one f32 tensor and one string tensor, same call. This does
    // not compile against `run`, whose single `T: Element` forces both to agree.
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

    // Reached only by consuming the string input, so this is what proves the
    // string tensor was genuinely read rather than accepted and ignored.
    let codes = outputs[2].as_slice::<i64>().expect("i64 codes");
    assert_eq!(codes, [EXPECTED_CODE]);

    let echoed = outputs[3].strings().expect("string output");
    assert_eq!(echoed, vec!["beta".to_owned()]);
}

#[test]
fn outputs_carry_their_own_types_and_shapes() {
    // The claim from the design: named, typed, shaped outputs cover multi-head
    // classification, regression, embeddings and forecasts without four code
    // paths. This fixture returns four outputs of three different element types
    // and three different shapes from one call, which is that claim in miniature.
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
            (DataType::F32, vec![2, 2]), // logits, one row per input row
            (DataType::F32, vec![2, 4]), // embedding
            (DataType::I64, vec![2]),    // label
            (DataType::String, vec![2]), // echoed text
        ]
    );

    assert_eq!(
        outputs[2].as_slice::<i64>().expect("codes"),
        [10, 30],
        "each row's own tag should be encoded, in order"
    );
}

#[test]
fn run_named_matches_inputs_by_name_not_position() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    // Supplied in the opposite order to the model's declaration. If names were
    // ignored and position used, ONNX Runtime would be handed a string where it
    // wants floats and the call would fail.
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

#[test]
fn run_named_rejects_an_unknown_input_name() {
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    // A typo in a tensor name is the failure this guards: without the check the
    // extra input would be silently dropped and the model would run on defaults.
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

#[test]
fn f16_tensors_survive_a_round_trip() {
    // `half::f16` is claimed to be ONNX's FLOAT16 layout. Compiling proves
    // nothing about that; the runtime doubling the values and giving back what we
    // expect does.
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

    // Exact: all three inputs and their doubles are representable in f16.
    assert_eq!(got, vec![3.0, -4.5, 0.25]);
}

#[test]
fn bool_tensors_survive_a_round_trip() {
    let Some(api) = api() else { return };
    let session = session_for("bool_not.onnx");

    let input = [true, false, true, true];
    let outputs = session
        .run_many(&[InputValue::bools(api, &input, &[4]).expect("bool input")])
        .expect("bool inference should succeed");

    assert_eq!(outputs[0].data_type().expect("dtype"), DataType::Bool);
    // Negated, so an implementation that returned zeroes would fail here.
    assert_eq!(
        outputs[0].bools().expect("bool output"),
        vec![false, true, false, false]
    );
}

#[test]
fn the_model_describes_its_own_inputs_and_outputs() {
    // This is what replaces `input_width() -> usize`. A caller reading these can
    // build a request for any model without the service knowing what it serves.
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

    // `codes` is i64. Asking for f64 -- same width, different meaning -- is
    // exactly the mistake that would otherwise produce plausible garbage.
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

    // And a string tensor is not a numeric slice at all.
    let err = outputs[3]
        .as_slice::<f32>()
        .expect_err("a string tensor must not be readable as f32");
    assert!(
        matches!(err, Error::ElementTypeMismatch { .. }),
        "got {err:?}"
    );
}

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

#[test]
fn a_borrowed_tensor_and_an_owned_one_can_share_a_call() {
    // Lifetime check as much as a behaviour check. Numeric inputs borrow the
    // caller's buffer; string inputs own runtime memory and are `'static`. If
    // `InputValue`'s lifetime were invariant the two could not sit in one slice,
    // and this would not compile.
    let Some(api) = api() else { return };
    let session = session_for("mixed_io.onnx");

    let mut numbers = vec![0.5f32, 0.5, 0.5];
    let borrowed = InputValue::numeric(api, &mut numbers, &[1, 3]).expect("f32");
    let owned: InputValue<'static> = InputValue::strings(api, &["alpha"], &[1]).expect("strings");

    let outputs = session.run_many(&[borrowed, owned]).expect("inference");
    assert_eq!(outputs[2].as_slice::<i64>().expect("codes"), [10]);
}
