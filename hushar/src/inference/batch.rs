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
//! The two directions are separate types. An [`InputBatch`] carries features, which may
//! be any of the five element types; an [`OutputBatch`] carries scores, which are `FP32`
//! or `FP64` and nothing else.

use crate::inference::InferenceError;

/// Element types the service serves.
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
}

impl std::fmt::Display for ElementType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// One feature's values for a whole batch, row-major.
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

    /// The element type of this column.
    pub(crate) fn element_type(&self) -> ElementType {
        match self {
            FeatureData::F32(_) => ElementType::F32,
            FeatureData::F64(_) => ElementType::F64,
            FeatureData::I32(_) => ElementType::I32,
            FeatureData::I64(_) => ElementType::I64,
            FeatureData::Str(_) => ElementType::Str,
        }
    }

    /// One row's values, borrowed from this column, or `None` for a row past the end.
    ///
    /// # Arguments
    ///
    /// * `row` - zero-based row index
    /// * `width` - values per row
    pub(crate) fn row(&self, row: usize, width: usize) -> Option<FeatureSlice<'_>> {
        let r = row_range(row, width, self.len())?;
        Some(match self {
            FeatureData::F32(v) => FeatureSlice::F32(&v[r]),
            FeatureData::F64(v) => FeatureSlice::F64(&v[r]),
            FeatureData::I32(v) => FeatureSlice::I32(&v[r]),
            FeatureData::I64(v) => FeatureSlice::I64(&v[r]),
            FeatureData::Str(v) => FeatureSlice::Str(&v[r]),
        })
    }
}

/// One row's values of one feature, borrowed from the column that holds them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum FeatureSlice<'a> {
    F32(&'a [f32]),
    F64(&'a [f64]),
    I32(&'a [i32]),
    I64(&'a [i64]),
    Str(&'a [String]),
}

impl FeatureSlice<'_> {
    /// How the value reads in an error message.
    pub(crate) fn describe(&self) -> &'static str {
        match self {
            FeatureSlice::F32(_) => "a float",
            FeatureSlice::F64(_) => "a double",
            FeatureSlice::I32(_) => "an integer",
            FeatureSlice::I64(_) => "a long",
            FeatureSlice::Str(_) => "text",
        }
    }
}

/// One model output's values for a whole batch, row-major.
///
/// Scores are single or double precision; a model declaring any other output type is
/// refused when it loads.
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

    /// Every score as `f32`, narrowing a double-precision batch.
    ///
    /// Nothing on the serving path narrows scores: the response and the log both carry
    /// the model's own precision.
    #[allow(dead_code)]
    pub(crate) fn as_f32(&self) -> Vec<f32> {
        match self {
            ScoreData::F32(v) => v.clone(),
            ScoreData::F64(v) => v.iter().map(|d| *d as f32).collect(),
        }
    }
}

/// One named model input: `rows × width`, row-major.
///
/// # Fields
///
/// * `name` - the model's own name for this input
/// * `width` - values per row; private, and set only by [`InputBatch::push`], which
///   derives it from the value count
/// * `data` - the values, all of one element type
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
    /// Integers widen and `f64` narrows, so any numeric feature reads as one stable
    /// column. `None` for text, which has no float reading, and for a row past the end.
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
/// # Fields
///
/// * `name` - the model's own name for this output
/// * `width` - values per row; private, and derived by [`OutputBatch::push`]
/// * `data` - the scores, either single or double precision
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
    /// Always available for a row that exists, since an output is float by construction.
    #[allow(dead_code)]
    pub(crate) fn row_as_f32(&self, row: usize) -> Option<Vec<f32>> {
        let range = row_range(row, self.width, self.data.len())?;
        Some(match &self.data {
            ScoreData::F32(v) => v[range].to_vec(),
            ScoreData::F64(v) => v[range].iter().map(|d| *d as f32).collect(),
        })
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
/// `kind` names the side in the error message, `"input"` or `"output"`.
///
/// # Errors
///
/// Fails when `len` is not a whole number of `rows`, when a non-empty batch carries an
/// empty column, or when a batch of no rows carries values.
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
/// # Fields
///
/// Both private, so the invariants hold.
///
/// * `rows` - rows every feature carries, stated once for the whole batch
/// * `features` - the inputs, matched by name rather than position, since an engine's
///   signature is keyed on name
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
    /// # Errors
    ///
    /// Fails on a name the batch already carries, and on anything [`derive_width`]
    /// refuses.
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
/// # Fields
///
/// Both private, so the invariants hold.
///
/// * `rows` - rows every output carries
/// * `outputs` - in the order the model declares them; the first is the one scored
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
    /// # Errors
    ///
    /// Fails on a name the batch already carries, and on anything [`derive_width`]
    /// refuses.
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
/// A configuration is checked against this at startup, so nothing is inferred at request
/// time.
///
/// # Fields
///
/// * `name` - the model's own name for it
/// * `element_type` - what it takes or produces
/// * `width` - values per row, or `None` where the model leaves the axis dynamic and
///   the configuration decides
/// * `rank` - 1 for `[batch]`, 2 for `[batch, width]`. An engine rejects a `[rows, 1]`
///   shape for a `[batch]` input even though the values are identical, so the rank is
///   carried. Ranks above 2 are refused when the model loads, which is what lets
///   everything else here work in rows and widths.
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

    /// A one-feature batch.
    fn inputs(rows: usize, name: &str, data: FeatureData) -> InputBatch {
        let mut b = InputBatch::new(rows);
        b.push(name, data).expect("valid feature");
        b
    }

    /// Six values over two rows derive a width of three.
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

    /// Two inputs of different widths share the batch's row count.
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

    /// A double-precision batch keeps a value no `f32` can hold exactly.
    #[test]
    fn a_double_batch_keeps_its_precision() {
        let doubles = ScoreData::F64(vec![0.1, 0.2]);
        match &doubles {
            ScoreData::F64(v) => assert_eq!(v, &[0.1, 0.2], "read exactly, not narrowed"),
            ScoreData::F32(_) => panic!("a double batch must not be an f32 one"),
        }
        assert_eq!(doubles.element_type(), ElementType::F64);
        assert_ne!(
            f64::from(doubles.as_f32()[0]),
            0.1,
            "0.1 does not survive an f32, which is why neither the response nor the log \
             is given one"
        );
    }

    #[test]
    fn a_float_batch_reads_as_floats_without_a_conversion() {
        let floats = ScoreData::F32(vec![0.5, 0.25]);
        assert_eq!(floats.as_f32(), vec![0.5, 0.25]);
        assert_eq!(floats.element_type(), ElementType::F32);
    }

    /// An output row that exists always has a float reading.
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

    /// The declared rank decides whether the engine gets `[rows]` or `[rows, width]`.
    #[test]
    fn the_declared_rank_decides_the_shape_handed_to_the_engine() {
        let rank2 = IoSpec::new("x", ElementType::Str, Some(1));
        assert_eq!(rank2.engine_shape(4, 1), vec![4, 1]);

        let rank1 = IoSpec::new("x", ElementType::Str, Some(1)).with_rank(1);
        assert_eq!(rank1.engine_shape(4, 1), vec![4]);
    }

    #[test]
    fn the_element_types_are_named_as_onnx_names_them() {
        assert_eq!(ElementType::Str.name(), "STRING");
        assert_eq!(ElementType::I32.name(), "INT32");
        assert_eq!(ElementType::F64.name(), "FP64");
    }
}
