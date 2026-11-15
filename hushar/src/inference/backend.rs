// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! The inference engine abstraction.
//!
//! Everything above this module works in terms of [`InferenceBackend`] and knows nothing
//! about ONNX Runtime, Neuron, or any other engine. The boundary is named, typed features
//! in and named, typed scores out; feature building, timing, logging and metrics all sit
//! outside it.

use std::fmt::Debug;

use crate::inference::InferenceError;
use crate::inference::batch::{InputBatch, IoSpec, OutputBatch};

/// How many of a request's mini batches one leased unit runs at once.
///
/// Reported per backend, so this describes one ONNX Runtime session or one Neuron
/// replica. A request leases one unit, so this is the whole bound on how far its rows can
/// be spread.
///
/// ```text
/// ONNX Runtime, host provider    the host's cores    a call runs on its calling thread
/// ONNX Runtime, device provider  1                   one stream per session
/// AWS Neuron                     execution contexts  configured per replica
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ExecutionProfile {
    concurrent_mini_batches: usize,
}

impl ExecutionProfile {
    /// A unit overlapping `concurrent` mini batches, floored at one.
    ///
    /// The only way to build a profile, so every reader can rely on the floor.
    pub(crate) fn new(concurrent: usize) -> Self {
        Self {
            concurrent_mini_batches: concurrent.max(1),
        }
    }

    /// A unit whose concurrent `run` calls serialize, such as one ONNX Runtime session on
    /// a device execution provider.
    pub(crate) fn serialized() -> Self {
        Self::new(1)
    }

    /// Mini batches of one request this unit runs at once, never fewer than one.
    ///
    /// `1` means concurrent [`InferenceBackend::run`] calls serialize, so cutting a
    /// request into mini batches would queue them rather than overlap them.
    pub(crate) fn concurrent_mini_batches(self) -> usize {
        self.concurrent_mini_batches
    }

    /// Whether mini batching can overlap anything on one leased unit.
    pub(crate) fn overlaps(self) -> bool {
        self.concurrent_mini_batches > 1
    }
}

/// An inference engine.
///
/// Implementations must be usable from many request threads at once through a shared
/// reference: the service holds one backend in an [`std::sync::Arc`] and does not
/// serialise access to it.
pub(crate) trait InferenceBackend: Debug + Send + Sync {
    /// Engine and device, for logs and metric dimensions, such as
    /// `"onnxruntime/CoreML(ALL)"`.
    fn name(&self) -> &str;

    /// How this backend overlaps concurrent [`Self::run`] calls.
    ///
    /// Read once at startup to choose the scheduling plan, so it must not vary with load.
    fn profile(&self) -> ExecutionProfile;

    /// The row count this backend demands per call, when its graph is pinned to one.
    ///
    /// `None` accepts any row count. `Some(n)` accepts only `n`, so a request of any
    /// other size must be cut into mini batches of `n` with the last padded. That makes
    /// the mini batch a requirement of the model rather than a scheduling choice, which
    /// is the difference between a split that cannot be avoided and one that buys
    /// nothing.
    fn required_rows(&self) -> Option<usize>;

    /// What the model takes, read from the model rather than configured.
    fn inputs(&self) -> &[IoSpec];

    /// What the model produces.
    fn outputs(&self) -> &[IoSpec];

    /// Runs one batch.
    ///
    /// Inputs are taken by value so an implementation may lend their buffers to the
    /// engine mutably rather than copying them.
    ///
    /// Every input the model declares must be supplied, and matching is by name, not
    /// position.
    fn run(&self, inputs: InputBatch) -> Result<OutputBatch, InferenceError>;
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;
    use crate::inference::batch::{ElementType, FeatureData, ScoreData};
    use std::sync::Mutex;
    use std::time::Duration;

    /// A generous default, so a test fanning out many mini batches still runs them at
    /// once rather than in rounds.
    pub(crate) const TEST_CONCURRENCY: usize = 64;

    /// Builds a single-output batch, for a test backend's result.
    fn one(rows: usize, name: &str, data: ScoreData) -> Result<OutputBatch, InferenceError> {
        let mut batch = OutputBatch::new(rows);
        batch.push(name, data)?;
        Ok(batch)
    }

    /// A backend that runs no model, for testing everything around one.
    ///
    /// Each output row is the sum of that input row's values, repeated to the output
    /// width.
    #[derive(Debug)]
    pub(crate) struct SummingBackend {
        inputs: Vec<IoSpec>,
        outputs: Vec<IoSpec>,
        input_width: usize,
        output_width: usize,
        profile: ExecutionProfile,
    }

    impl SummingBackend {
        /// A single `FP32` input named `features` and a single `FP32` output named
        /// `scores`.
        pub(crate) fn new(input_width: usize, output_width: usize) -> Self {
            Self {
                inputs: vec![IoSpec::new("features", ElementType::F32, Some(input_width))],
                outputs: vec![IoSpec::new("scores", ElementType::F32, Some(output_width))],
                input_width,
                output_width,
                profile: ExecutionProfile::new(TEST_CONCURRENCY),
            }
        }

        /// The same backend reporting `profile`, for testing a scheduler decision.
        pub(crate) fn with_profile(mut self, profile: ExecutionProfile) -> Self {
            self.profile = profile;
            self
        }

        /// The name this backend expects its input to be called.
        pub(crate) fn input_name(&self) -> &str {
            &self.inputs[0].name
        }
    }

    /// A [`SummingBackend`] whose engine call takes a known, non-trivial amount of time.
    ///
    /// Timings are recorded in microseconds, so the delay must be far longer than the
    /// clock's resolution for a lower-bound assertion on a measured stage to hold.
    #[derive(Debug)]
    pub(crate) struct SlowBackend {
        inner: SummingBackend,
        delay: Duration,
    }

    impl SlowBackend {
        pub(crate) fn new(input_width: usize, output_width: usize, delay: Duration) -> Self {
            Self {
                inner: SummingBackend::new(input_width, output_width),
                delay,
            }
        }
    }

    impl InferenceBackend for SlowBackend {
        fn name(&self) -> &str {
            "test/slow"
        }

        fn profile(&self) -> ExecutionProfile {
            self.inner.profile()
        }

        fn required_rows(&self) -> Option<usize> {
            self.inner.required_rows()
        }

        fn inputs(&self) -> &[IoSpec] {
            self.inner.inputs()
        }

        fn outputs(&self) -> &[IoSpec] {
            self.inner.outputs()
        }

        /// Sleeps for the configured delay, then delegates to the inner backend.
        fn run(&self, inputs: InputBatch) -> Result<OutputBatch, InferenceError> {
            std::thread::sleep(self.delay);
            self.inner.run(inputs)
        }
    }

    impl InferenceBackend for SummingBackend {
        fn name(&self) -> &str {
            "test/summing"
        }

        fn profile(&self) -> ExecutionProfile {
            self.profile
        }

        fn required_rows(&self) -> Option<usize> {
            None
        }

        fn inputs(&self) -> &[IoSpec] {
            &self.inputs
        }

        fn outputs(&self) -> &[IoSpec] {
            &self.outputs
        }

        fn run(&self, inputs: InputBatch) -> Result<OutputBatch, InferenceError> {
            let column = inputs
                .get(self.input_name())
                .ok_or_else(|| format!("missing input {:?}", self.input_name()))?;

            let rows = inputs.rows();
            if column.width() != self.input_width {
                return Err(format!(
                    "expected {} values per row for {rows} rows, got {}",
                    self.input_width,
                    column.width()
                )
                .into());
            }

            let mut out = Vec::with_capacity(rows * self.output_width);
            for i in 0..rows {
                let row = column.row_as_f32(i).ok_or("row out of range")?;
                let sum: f32 = row.iter().sum();
                out.extend(std::iter::repeat_n(sum, self.output_width));
            }

            one(rows, "scores", ScoreData::F32(out))
        }
    }

    /// A backend that records the size of every batch it is given.
    ///
    /// Scores are the row's sum, as [`SummingBackend`] computes them.
    #[derive(Debug)]
    pub(crate) struct RecordingBackend {
        inner: SummingBackend,
        seen: Mutex<Vec<usize>>,
    }

    impl RecordingBackend {
        pub(crate) fn new(input_width: usize, output_width: usize) -> Self {
            Self {
                inner: SummingBackend::new(input_width, output_width),
                seen: Mutex::new(Vec::new()),
            }
        }

        /// The row count of each batch that ran, in order.
        pub(crate) fn batches(&self) -> Vec<usize> {
            self.seen
                .lock()
                .expect("no test panics while holding this")
                .clone()
        }
    }

    impl InferenceBackend for RecordingBackend {
        fn name(&self) -> &str {
            "test/recording"
        }

        fn profile(&self) -> ExecutionProfile {
            self.inner.profile()
        }

        fn required_rows(&self) -> Option<usize> {
            self.inner.required_rows()
        }

        fn inputs(&self) -> &[IoSpec] {
            self.inner.inputs()
        }

        fn outputs(&self) -> &[IoSpec] {
            self.inner.outputs()
        }

        fn run(&self, inputs: InputBatch) -> Result<OutputBatch, InferenceError> {
            self.seen
                .lock()
                .expect("no test panics while holding this")
                .push(inputs.rows());
            self.inner.run(inputs)
        }
    }

    /// A backend with several outputs of different widths and both float types.
    ///
    /// Returns logits, an embedding and a double-precision total, each derived from the
    /// input row.
    #[derive(Debug)]
    pub(crate) struct MultiOutputBackend {
        inputs: Vec<IoSpec>,
        outputs: Vec<IoSpec>,
    }

    impl MultiOutputBackend {
        pub(crate) fn new(input_width: usize) -> Self {
            Self {
                inputs: vec![IoSpec::new("features", ElementType::F32, Some(input_width))],
                outputs: vec![
                    IoSpec::new("scores", ElementType::F32, Some(2)),
                    IoSpec::new("embedding", ElementType::F32, Some(3)),
                    IoSpec::new("total", ElementType::F64, Some(1)),
                ],
            }
        }
    }

    impl InferenceBackend for MultiOutputBackend {
        fn name(&self) -> &str {
            "test/multi-output"
        }

        fn profile(&self) -> ExecutionProfile {
            ExecutionProfile::new(TEST_CONCURRENCY)
        }

        fn required_rows(&self) -> Option<usize> {
            None
        }

        fn inputs(&self) -> &[IoSpec] {
            &self.inputs
        }

        fn outputs(&self) -> &[IoSpec] {
            &self.outputs
        }

        fn run(&self, inputs: InputBatch) -> Result<OutputBatch, InferenceError> {
            let column = inputs.get("features").ok_or("missing input \"features\"")?;
            let rows = inputs.rows();

            let mut scores = Vec::with_capacity(rows * 2);
            let mut embedding = Vec::with_capacity(rows * 3);
            let mut total = Vec::with_capacity(rows);
            for i in 0..rows {
                let row = column.row_as_f32(i).ok_or("row out of range")?;
                let sum: f32 = row.iter().sum();
                scores.extend([sum, -sum]);
                embedding.extend([sum, sum * 2.0, sum * 3.0]);
                total.push(f64::from(sum));
            }

            let mut batch = OutputBatch::new(rows);
            batch.push("scores", ScoreData::F32(scores))?;
            batch.push("embedding", ScoreData::F32(embedding))?;
            batch.push("total", ScoreData::F64(total))?;
            Ok(batch)
        }
    }

    /// A backend whose input and output are double precision.
    #[derive(Debug)]
    pub(crate) struct DoubleBackend {
        inputs: Vec<IoSpec>,
        outputs: Vec<IoSpec>,
    }

    impl DoubleBackend {
        pub(crate) fn new(width: usize) -> Self {
            Self {
                inputs: vec![IoSpec::new("features", ElementType::F64, Some(width))],
                outputs: vec![IoSpec::new("scores", ElementType::F64, Some(1))],
            }
        }
    }

    impl InferenceBackend for DoubleBackend {
        fn name(&self) -> &str {
            "test/double"
        }

        fn profile(&self) -> ExecutionProfile {
            ExecutionProfile::new(TEST_CONCURRENCY)
        }

        fn required_rows(&self) -> Option<usize> {
            None
        }

        fn inputs(&self) -> &[IoSpec] {
            &self.inputs
        }

        fn outputs(&self) -> &[IoSpec] {
            &self.outputs
        }

        fn run(&self, inputs: InputBatch) -> Result<OutputBatch, InferenceError> {
            let column = inputs.get("features").ok_or("missing input \"features\"")?;
            let FeatureData::F64(values) = &column.data else {
                return Err("input \"features\" must be double precision".into());
            };
            let rows = inputs.rows();
            let width = column.width().max(1);
            let sums: Vec<f64> = values.chunks(width).map(|r| r.iter().sum()).collect();
            one(rows, "scores", ScoreData::F64(sums))
        }
    }

    /// A backend that always fails, for testing error propagation.
    ///
    /// It declares a plausible signature rather than none, so a request reaches `run` and
    /// the failure comes from the engine.
    #[derive(Debug)]
    pub(crate) struct FailingBackend;

    /// The signature `FailingBackend` claims: one `FP32` input, one `FP32` output.
    static FAILING_IO: std::sync::LazyLock<Vec<IoSpec>> =
        std::sync::LazyLock::new(|| vec![IoSpec::new("features", ElementType::F32, Some(1))]);

    impl InferenceBackend for FailingBackend {
        fn name(&self) -> &str {
            "test/failing"
        }

        fn profile(&self) -> ExecutionProfile {
            ExecutionProfile::new(TEST_CONCURRENCY)
        }

        fn required_rows(&self) -> Option<usize> {
            None
        }

        fn inputs(&self) -> &[IoSpec] {
            &FAILING_IO
        }

        fn outputs(&self) -> &[IoSpec] {
            &FAILING_IO
        }

        fn run(&self, _inputs: InputBatch) -> Result<OutputBatch, InferenceError> {
            Err("engine exploded".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::{MultiOutputBackend, SummingBackend};
    use super::*;
    use crate::inference::batch::{ElementType, FeatureData};

    fn features(rows: usize, width: usize, value: f32) -> InputBatch {
        let mut batch = InputBatch::new(rows);
        batch
            .push("features", FeatureData::F32(vec![value; rows * width]))
            .expect("valid column");
        batch
    }

    /// Each row's scores are that row's sum: two rows of width 3 sum to 3 and 6.
    #[test]
    fn the_test_backend_maps_each_row_to_its_own_sum() {
        let backend = SummingBackend::new(3, 2);
        let mut inputs = InputBatch::new(2);
        inputs
            .push(
                "features",
                FeatureData::F32(vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
            )
            .expect("valid");

        let out = backend.run(inputs).expect("run");
        let scores = out.get("scores").expect("scores output");
        assert_eq!(scores.row_as_f32(0).expect("row 0"), vec![3.0, 3.0]);
        assert_eq!(scores.row_as_f32(1).expect("row 1"), vec![6.0, 6.0]);
    }

    /// A batch whose rows are narrower than the model's input is rejected.
    #[test]
    fn the_test_backend_rejects_a_batch_of_the_wrong_width() {
        let backend = SummingBackend::new(3, 2);
        assert!(backend.run(features(2, 1, 1.0)).is_err());
    }

    #[test]
    fn an_input_is_matched_by_name_not_position() {
        let backend = SummingBackend::new(2, 1);
        let mut wrong_name = InputBatch::new(1);
        wrong_name
            .push("wrong", FeatureData::F32(vec![1.0, 2.0]))
            .expect("valid");
        let err = backend
            .run(wrong_name)
            .expect_err("a misnamed input must fail");
        assert!(
            err.to_string().contains("features"),
            "the error should name the input it wanted: {err}"
        );
    }

    /// One call returns outputs of different element types and widths, all row-aligned.
    #[test]
    fn a_model_can_return_several_outputs_of_different_types_and_widths() {
        let backend = MultiOutputBackend::new(2);
        let out = backend.run(features(2, 2, 1.5)).expect("run");

        assert_eq!(out.iter().count(), 3);
        let described: Vec<(&str, ElementType, usize)> = out
            .iter()
            .map(|c| (c.name.as_str(), c.data.element_type(), c.width()))
            .collect();
        assert_eq!(
            described,
            vec![
                ("scores", ElementType::F32, 2),
                ("embedding", ElementType::F32, 3),
                ("total", ElementType::F64, 1),
            ]
        );
        assert_eq!(out.rows(), 2, "every output shares the batch's row count");

        assert_eq!(
            out.get("scores")
                .expect("scores")
                .row_as_f32(0)
                .expect("row"),
            vec![3.0, -3.0]
        );
        assert_eq!(
            out.get("embedding")
                .expect("embedding")
                .row_as_f32(1)
                .expect("row"),
            vec![3.0, 6.0, 9.0]
        );
        assert_eq!(
            out.get("total").expect("total").row_as_f32(0).expect("row"),
            vec![3.0]
        );
    }

    #[test]
    fn a_backend_declares_what_it_takes_and_returns() {
        let backend = SummingBackend::new(4, 2);
        assert_eq!(backend.inputs().len(), 1);
        assert_eq!(backend.inputs()[0].name, "features");
        assert_eq!(backend.inputs()[0].element_type, ElementType::F32);
        assert_eq!(backend.inputs()[0].width, Some(4));
        assert_eq!(backend.outputs()[0].width, Some(2));
    }

    /// A serializing unit is the one case that gates a scheduling decision.
    #[test]
    fn a_serialized_profile_overlaps_nothing() {
        let profile = ExecutionProfile::serialized();
        assert_eq!(profile.concurrent_mini_batches(), 1);
        assert!(
            !profile.overlaps(),
            "one mini batch at a time is what refuses a split"
        );
    }

    /// The Neuron shape: a replica's execution contexts are its mini batches in flight.
    ///
    /// `NeuronBackend::profile` is `ExecutionProfile::new(contexts.total())`, and
    /// constructing the backend itself needs a Neuron host, so the figure is asserted
    /// here with the number named.
    #[test]
    fn a_unit_overlaps_as_many_mini_batches_as_it_reports() {
        let four_contexts = ExecutionProfile::new(4);
        assert_eq!(four_contexts.concurrent_mini_batches(), 4);
        assert!(four_contexts.overlaps());
    }

    /// A count of zero would describe a unit that can serve nothing.
    #[test]
    fn a_profile_is_never_below_one() {
        assert_eq!(ExecutionProfile::new(0), ExecutionProfile::serialized());
    }

    /// The scheduler reads the profile off a trait object, so it must survive erasure.
    #[test]
    fn a_profile_is_readable_through_the_trait_object() {
        let serialized: Box<dyn InferenceBackend> =
            Box::new(SummingBackend::new(3, 1).with_profile(ExecutionProfile::serialized()));
        assert!(!serialized.profile().overlaps());

        let overlapping: Box<dyn InferenceBackend> = Box::new(SummingBackend::new(3, 1));
        assert!(
            overlapping.profile().overlaps(),
            "the test default fans out, so existing fan-out tests still run at once"
        );
    }
}
