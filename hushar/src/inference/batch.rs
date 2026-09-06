// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! What the service hands an engine, and gets back.
//!
//! Every model here is tabular. A batch is `rows` rows, and each model input is one
//! named [`Feature`] of `width` values per row, laid out row-major.
//!
//! Two properties are held by the types rather than checked at each use:
//!
//! * `rows` lives on the batch, once, so two inputs cannot disagree about how many
//!   rows they carry.
//! * `width` is derived when a feature is pushed, so a shape cannot disagree with the
//!   number of values it describes.
//!
//! The two directions are separate types. An [`InputBatch`] carries features, which
//! may be any of five element types. An [`OutputBatch`] carries scores, which are
//! `FP32` or `FP64` and nothing else, so the response path has no third case to
//! handle.

use crate::inference::InferenceError;

/// Element types the service serves.
///
/// Five, each earning its place against a real model: `F32` for a vectorised feature
/// matrix, `F64` because a tree ensemble is not continuous and a narrowing cast can
/// move a value across a split, `I32` and `I64` for token ids and category codes, and
/// `Str` for a model that encodes text in the graph.
///
/// Left out are `FP16` and `BF16`, which have no protobuf field and would need a
/// parallel raw-bytes encoding, and the integers *narrower* than 32 bits, which share
/// protobuf's `int32` field and so would need a range check per element. `INT32` has a
/// field of its own, which is why it costs nothing to serve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ElementType {
    F32,
    F64,
    I32,
    I64,
    Str,
}

impl ElementType {
    /// The name used in the startup banner and in errors, following ONNX.
    pub(crate) fn name(&self) -> &'static str {
        match self {
            ElementType::F32 => "FP32",
            ElementType::F64 => "FP64",
            ElementType::I32 => "INT32",
            ElementType::I64 => "INT64",
            ElementType::Str => "STRING",
        }
    }

    /// Whether a transformation's output can feed this type.
    ///
    /// Transformations are defined over floats. Rounding one into an integer, or
    /// inventing text from one, would be silent corruption, so the other types are fed
    /// verbatim from the request instead.
    pub(crate) fn is_float(&self) -> bool {
        matches!(self, ElementType::F32 | ElementType::F64)
    }
}

impl std::fmt::Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One feature's values for a whole batch, row-major.
///
/// An enum rather than a type parameter, because a batch is heterogeneous: a float
/// matrix and a text column travel in the same call, so they have to be one Rust type.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum FeatureData {
    F32(Vec<f32>),
    F64(Vec<f64>),
    I32(Vec<i32>),
    I64(Vec<i64>),
    Str(Vec<String>),
}

impl FeatureData {
    /// Total values held, across every row.
    pub(crate) fn len(&self) -> usize {
        match self {
            FeatureData::F32(v) => v.len(),
            FeatureData::F64(v) => v.len(),
            FeatureData::I32(v) => v.len(),
            FeatureData::I64(v) => v.len(),
            FeatureData::Str(v) => v.len(),
        }
    }
}

/// One model output's values for a whole batch, row-major.
///
/// Two variants, because a response carries scores. A model declaring anything else is
/// refused when it loads, so that decision lives there rather than being re-checked on
/// every response.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ScoreData {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

impl ScoreData {
    pub(crate) fn element_type(&self) -> ElementType {
        match self {
            ScoreData::F32(_) => ElementType::F32,
            ScoreData::F64(_) => ElementType::F64,
        }
    }

    /// Total values held, across every row.
    pub(crate) fn len(&self) -> usize {
        match self {
            ScoreData::F32(v) => v.len(),
            ScoreData::F64(v) => v.len(),
        }
    }
}

/// One named model input: `rows × width`, row-major.
///
/// Fields:
/// - `name` — the model's own name for this input.
/// - `width` — values per row. Private, and set only by [`InputBatch::push`], which
///   derives it from the value count. That is what makes the shape unable to lie.
/// - `data` — the values, all of one element type.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Feature {
    pub name: String,
    width: usize,
    pub data: FeatureData,
}

impl Feature {
    /// Values per row.
    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// One row's values as `f32`.
    ///
    /// The integers widen and `f64` narrows, so a caller reading a feature back gets
    /// one stable numeric column. `None` for text, which has no float reading, and for
    /// a row past the end.
    ///
    /// This is the read half of the API a backend uses on the inputs it is handed. The
    /// ONNX backend takes ownership and lends its buffers to the engine untouched, so
    /// today only the test backends call it.
    #[allow(dead_code)]
    pub(crate) fn row_as_f32(&self, row: usize) -> Option<Vec<f32>> {
        let range = row_range(row, self.width, self.data.len())?;
        Some(match &self.data {
            FeatureData::F32(v) => v[range].to_vec(),
            FeatureData::F64(v) => v[range].iter().map(|d| *d as f32).collect(),
            FeatureData::I32(v) => v[range].iter().map(|i| *i as f32).collect(),
            FeatureData::I64(v) => v[range].iter().map(|i| *i as f32).collect(),
            FeatureData::Str(_) => return None,
        })
    }
}

/// One named model output: `rows × width`, row-major, always floating point.
///
/// Fields:
/// - `name` — the model's own name for this output.
/// - `width` — values per row. Private, and derived by [`OutputBatch::push`].
/// - `data` — the scores, either single or double precision.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Output {
    pub name: String,
    width: usize,
    pub data: ScoreData,
}

impl Output {
    /// Values per row.
    #[allow(dead_code)]
    pub(crate) fn width(&self) -> usize {
        self.width
    }

    /// One row's scores as `f32`, or `None` for a row past the end.
    ///
    /// Always available for a row that exists, because an output is float by
    /// construction. The inference log carries one stable float column whatever the
    /// model's precision, which is why the `f64` case narrows here.
    pub(crate) fn row_as_f32(&self, row: usize) -> Option<Vec<f32>> {
        let range = row_range(row, self.width, self.data.len())?;
        Some(match &self.data {
            ScoreData::F32(v) => v[range].to_vec(),
            ScoreData::F64(v) => v[range].iter().map(|d| *d as f32).collect(),
        })
    }

    /// One row's scores as `f64`, without narrowing.
    ///
    /// Separate from [`Output::row_as_f32`] so a double-precision model's scores reach
    /// the response at full precision. `None` for a single-precision output.
    pub(crate) fn row_f64(&self, row: usize) -> Option<&[f64]> {
        let range = row_range(row, self.width, self.data.len())?;
        match &self.data {
            ScoreData::F64(v) => Some(&v[range]),
            ScoreData::F32(_) => None,
        }
    }
}

/// Where one row's values sit in a flat buffer, or `None` past the end.
fn row_range(row: usize, width: usize, len: usize) -> Option<std::ops::Range<usize>> {
    let start = row.checked_mul(width)?;
    let end = start.checked_add(width)?;
    (end <= len).then_some(start..end)
}

/// Values per row, derived from the value count.
///
/// Refuses a count that is not a whole number of rows, and an empty column for a
/// non-empty batch. Either would otherwise become a wrong answer rather than an error.
/// `kind` names the side, so the message reads as "input" or "output".
fn derive_width(kind: &str, name: &str, rows: usize, len: usize) -> Result<usize, InferenceError> {
    if rows == 0 {
        if len != 0 {
            return Err(
                format!("{kind} {name:?} carries {len} values for a batch of no rows").into(),
            );
        }
        return Ok(0);
    }
    if len == 0 {
        return Err(format!("{kind} {name:?} carries no values for a batch of {rows} rows").into());
    }
    if !len.is_multiple_of(rows) {
        return Err(format!(
            "{kind} {name:?} carries {len} values, which is not a whole number of rows \
             for a batch of {rows}; every row must be the same width"
        )
        .into());
    }
    Ok(len / rows)
}

/// A batch of named model inputs, all with the same number of rows.
///
/// Fields, both private so the invariants hold:
/// - `rows` — rows every feature carries, stated once for the whole batch.
/// - `features` — the inputs, matched by name rather than position. Name matching is
///   what an engine's signature is keyed on; a positional list mislabels everything
///   the moment a model's inputs are reordered.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InputBatch {
    rows: usize,
    features: Vec<Feature>,
}

impl InputBatch {
    /// An empty batch expecting `rows` rows per feature.
    pub(crate) fn new(rows: usize) -> Self {
        Self {
            rows,
            features: Vec::new(),
        }
    }

    /// Adds one feature, deriving its width from the values it carries.
    ///
    /// Refuses a repeated name, which would leave the engine reading whichever of the
    /// two it found first, and anything [`derive_width`] refuses.
    pub(crate) fn push(
        &mut self,
        name: impl Into<String>,
        data: FeatureData,
    ) -> Result<(), InferenceError> {
        let name = name.into();
        if self.features.iter().any(|f| f.name == name) {
            return Err(format!("input {name:?} was supplied twice").into());
        }
        let width = derive_width("input", &name, self.rows, data.len())?;
        self.features.push(Feature { name, width, data });
        Ok(())
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// The feature of this name, if the batch carries one.
    ///
    /// How a backend picks out the input it wants. The ONNX backend takes ownership
    /// instead, so today only the test backends call it.
    #[allow(dead_code)]
    pub(crate) fn get(&self, name: &str) -> Option<&Feature> {
        self.features.iter().find(|f| f.name == name)
    }

    /// Takes the features, so their buffers can be lent to an engine.
    pub(crate) fn into_features(self) -> Vec<Feature> {
        self.features
    }
}

/// A batch of named model outputs, all with the same number of rows.
///
/// Fields, both private so the invariants hold:
/// - `rows` — rows every output carries.
/// - `outputs` — in the order the model declares them. The first is the one scored.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct OutputBatch {
    rows: usize,
    outputs: Vec<Output>,
}

impl OutputBatch {
    /// An empty batch expecting `rows` rows per output.
    pub(crate) fn new(rows: usize) -> Self {
        Self {
            rows,
            outputs: Vec::new(),
        }
    }

    /// Adds one output, deriving its width from the values it carries.
    ///
    /// Refuses a repeated name, and anything [`derive_width`] refuses.
    pub(crate) fn push(
        &mut self,
        name: impl Into<String>,
        data: ScoreData,
    ) -> Result<(), InferenceError> {
        let name = name.into();
        if self.outputs.iter().any(|o| o.name == name) {
            return Err(format!("output {name:?} was returned twice").into());
        }
        let width = derive_width("output", &name, self.rows, data.len())?;
        self.outputs.push(Output { name, width, data });
        Ok(())
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    /// The output of this name, if the batch carries one.
    #[allow(dead_code)]
    pub(crate) fn get(&self, name: &str) -> Option<&Output> {
        self.outputs.iter().find(|o| o.name == name)
    }

    /// The first output, which is the one scored.
    pub(crate) fn first(&self) -> Option<&Output> {
        self.outputs.first()
    }

    #[allow(dead_code)]
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, Output> {
        self.outputs.iter()
    }
}

/// What a model declares for one input or output.
///
/// Fields:
/// - `name` — the model's own name for it.
/// - `element_type` — what it takes or produces.
/// - `width` — values per row, or `None` where the model leaves the axis dynamic and
///   the configuration decides.
/// - `rank` — 1 for `[batch]`, 2 for `[batch, width]`. The only reason a rank survives
///   into the service: an engine rejects a `[rows, 1]` shape for a `[batch]` input even
///   though the values are identical. Ranks above 2 are refused when the model loads,
///   which is what lets everything else here work in rows and widths.
///
/// A configuration is checked against this at startup, which is why nothing needs to
/// be inferred at request time.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IoSpec {
    pub name: String,
    pub element_type: ElementType,
    pub width: Option<usize>,
    pub rank: usize,
}

impl IoSpec {
    /// A rank-2 `[batch, width]` input or output, which is the ordinary case.
    pub(crate) fn new(
        name: impl Into<String>,
        element_type: ElementType,
        width: Option<usize>,
    ) -> Self {
        Self {
            name: name.into(),
            element_type,
            width,
            rank: 2,
        }
    }

    /// The same spec at the rank the model declared.
    pub(crate) fn with_rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// The shape to hand the engine for a batch of `rows × width`.
    pub(crate) fn engine_shape(&self, rows: usize, width: usize) -> Vec<i64> {
        if self.rank <= 1 {
            vec![rows as i64]
        } else {
            vec![rows as i64, width as i64]
        }
    }

    /// How the signature reads in the startup banner and in errors.
    pub(crate) fn describe(&self) -> String {
        match self.width {
            Some(w) => format!("{} {}[{}]", self.name, self.element_type, w),
            None => format!("{} {}[?]", self.name, self.element_type),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-feature batch, for the tests that need only one.
    fn inputs(rows: usize, name: &str, data: FeatureData) -> InputBatch {
        let mut b = InputBatch::new(rows);
        b.push(name, data).expect("valid feature");
        b
    }

    /// The point of the type: six values over two rows is a width of three, and there
    /// is no second number that could claim otherwise.
    #[test]
    fn the_width_is_derived_rather_than_supplied() {
        let b = inputs(2, "x", FeatureData::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        assert_eq!(b.rows(), 2);
        assert_eq!(b.get("x").expect("x").width(), 3);
    }

    #[test]
    fn a_value_count_that_is_not_whole_rows_is_refused_with_both_numbers() {
        let mut b = InputBatch::new(2);
        let err = b
            .push("x", FeatureData::F32(vec![1.0, 2.0, 3.0]))
            .expect_err("3 values do not divide into 2 rows");
        let m = err.to_string();
        assert!(m.contains('3') && m.contains('2'), "got {m}");
    }

    /// Two inputs of different widths are fine; two inputs of different row counts
    /// cannot be expressed at all, which is the invariant worth having.
    #[test]
    fn every_feature_shares_the_batchs_row_count() {
        let mut b = InputBatch::new(2);
        b.push("numbers", FeatureData::F32(vec![0.0; 6]))
            .expect("3 wide");
        b.push("tags", FeatureData::Str(vec!["a".into(), "b".into()]))
            .expect("1 wide");
        assert_eq!(b.get("numbers").expect("numbers").width(), 3);
        assert_eq!(b.get("tags").expect("tags").width(), 1);
        assert_eq!(b.rows(), 2);
    }

    #[test]
    fn a_repeated_input_name_is_refused() {
        let mut b = InputBatch::new(1);
        b.push("x", FeatureData::F32(vec![1.0])).expect("first");
        let err = b
            .push("x", FeatureData::F32(vec![2.0]))
            .expect_err("a name must not repeat");
        assert!(err.to_string().contains("twice"), "got {err}");
    }

    #[test]
    fn an_empty_feature_is_refused_for_a_non_empty_batch() {
        let mut b = InputBatch::new(2);
        let err = b
            .push("x", FeatureData::F32(vec![]))
            .expect_err("no values for two rows");
        assert!(err.to_string().contains("no values"), "got {err}");
    }

    #[test]
    fn an_empty_batch_takes_empty_features() {
        let mut b = InputBatch::new(0);
        b.push("x", FeatureData::F32(vec![]))
            .expect("nothing to lay out");
        assert_eq!(b.rows(), 0);
        assert_eq!(b.get("x").expect("x").width(), 0);
    }

    #[test]
    fn features_are_found_by_name_not_position() {
        let mut b = InputBatch::new(1);
        b.push("a", FeatureData::F32(vec![1.0])).expect("a");
        b.push("b", FeatureData::I64(vec![7])).expect("b");
        assert_eq!(b.get("b").expect("b").data, FeatureData::I64(vec![7]));
        assert!(b.get("c").is_none(), "an absent name reads as absent");
    }

    #[test]
    fn a_row_reads_as_floats_whatever_the_element_type() {
        for (label, data) in [
            ("FP32", FeatureData::F32(vec![1.0, 2.0, 3.0, 4.0])),
            ("FP64", FeatureData::F64(vec![1.0, 2.0, 3.0, 4.0])),
            ("INT32", FeatureData::I32(vec![1, 2, 3, 4])),
            ("INT64", FeatureData::I64(vec![1, 2, 3, 4])),
        ] {
            let b = inputs(2, "x", data);
            assert_eq!(
                b.get("x").expect("x").row_as_f32(1),
                Some(vec![3.0, 4.0]),
                "{label} should read as floats"
            );
        }
    }

    #[test]
    fn text_has_no_float_reading() {
        let b = inputs(1, "x", FeatureData::Str(vec!["hello".into()]));
        assert!(
            b.get("x").expect("x").row_as_f32(0).is_none(),
            "text must report rather than invent a number"
        );
    }

    #[test]
    fn a_row_past_the_end_reads_as_absent() {
        let b = inputs(2, "x", FeatureData::F32(vec![1.0, 2.0]));
        let feature = b.get("x").expect("x");
        assert!(feature.row_as_f32(1).is_some(), "row 1 is the last row");
        assert!(feature.row_as_f32(2).is_none(), "row 2 is past the end");
    }

    /// Why `row_f64` exists: 0.1 has no exact `f32`, so a response that narrowed it
    /// would not be the number the model produced.
    #[test]
    fn a_double_output_row_is_available_without_narrowing() {
        let mut b = OutputBatch::new(1);
        b.push("scores", ScoreData::F64(vec![0.1, 0.2]))
            .expect("valid");
        assert_eq!(b.first().expect("scores").row_f64(0), Some(&[0.1, 0.2][..]));
    }

    #[test]
    fn a_float_output_has_no_double_reading() {
        let mut b = OutputBatch::new(1);
        b.push("scores", ScoreData::F32(vec![0.5])).expect("valid");
        assert!(
            b.first().expect("scores").row_f64(0).is_none(),
            "an f32 output must not claim double precision"
        );
    }

    /// Unlike a feature, an output is float by construction, so a row that exists
    /// always has a float reading and the response path has no unreachable branch.
    #[test]
    fn an_output_always_reads_as_floats() {
        let mut b = OutputBatch::new(2);
        b.push("scores", ScoreData::F64(vec![1.0, 2.0, 3.0, 4.0]))
            .expect("valid");
        assert_eq!(
            b.first().expect("scores").row_as_f32(1),
            Some(vec![3.0, 4.0])
        );
    }

    #[test]
    fn a_repeated_output_name_is_refused() {
        let mut b = OutputBatch::new(1);
        b.push("s", ScoreData::F32(vec![1.0])).expect("first");
        assert!(
            b.push("s", ScoreData::F32(vec![2.0])).is_err(),
            "a repeated output name would be read by whichever came first"
        );
    }

    #[test]
    fn a_spec_describes_a_dynamic_width_as_unknown() {
        assert_eq!(
            IoSpec::new("x", ElementType::F32, Some(4)).describe(),
            "x FP32[4]"
        );
        assert_eq!(
            IoSpec::new("x", ElementType::Str, None).describe(),
            "x STRING[?]"
        );
    }

    /// A model input declared `[batch]` and one declared `[batch, 1]` carry the same
    /// values, and an engine rejects the wrong one of the two.
    #[test]
    fn the_declared_rank_decides_the_shape_handed_to_the_engine() {
        let rank2 = IoSpec::new("x", ElementType::Str, Some(1));
        assert_eq!(rank2.engine_shape(4, 1), vec![4, 1]);

        let rank1 = IoSpec::new("x", ElementType::Str, Some(1)).with_rank(1);
        assert_eq!(rank1.engine_shape(4, 1), vec![4]);
    }

    #[test]
    fn only_the_float_types_can_be_built_from_transformations() {
        assert!(ElementType::F32.is_float());
        assert!(ElementType::F64.is_float());
        assert!(!ElementType::I32.is_float(), "rounding would be corruption");
        assert!(!ElementType::I64.is_float(), "rounding would be corruption");
        assert!(!ElementType::Str.is_float(), "text cannot be invented");
    }

    #[test]
    fn the_element_types_are_named_as_onnx_names_them() {
        assert_eq!(ElementType::Str.name(), "STRING");
        assert_eq!(ElementType::I32.name(), "INT32");
        assert_eq!(ElementType::F64.name(), "FP64");
    }
}
