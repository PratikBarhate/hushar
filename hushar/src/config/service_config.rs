// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Service configuration: what is true of the server rather than of the model.
//!
//! The model, its execution provider and its feature configuration live in
//! [`crate::config::model_config`], beside the model they apply to. Everything else
//! about the process is here — the listener, where observability goes, and how much it
//! is allowed to cost.
//!
//! ```json
//! {
//!   "bind_address": "0.0.0.0",
//!   "port_number": 8279,
//!   "model_config_path": "s3://configs/fraud/v3/model_config.json",
//!
//!   "inference_log": {
//!     "uri": "s3://my-bucket/inference-logs",
//!     "batch_size": 5000,
//!     "sample_rate": 1.0,
//!     "max_sends_in_flight": 4
//!   },
//!
//!   "metrics": {
//!     "cloudwatch_namespace": "HusharService",
//!     "batch_size": 500,
//!     "max_sends_in_flight": 4
//!   }
//! }
//! ```
//!
//! `threading` is absent above, which is the usual case: every one of its fields has a
//! default. See [`ThreadingConfig`].
//!
//! `--config-uri` says where *this* file is, so it cannot live in it. `--port` and
//! `--bind-address` remain as overrides for running two instances on one host.

use std::net::IpAddr;

use serde::Deserialize;

use crate::inference::InferenceError;

fn default_bind_address() -> IpAddr {
    IpAddr::from([0, 0, 0, 0])
}

fn default_port_number() -> u16 {
    8279
}

/// In-flight requests allowed per connection, when the configuration does not give one.
///
/// Defaults to 500. This is an **admission** ceiling on one HTTP/2 connection, not a
/// bound on work: `threading.inference_concurrency` decides how many inferences run at
/// once, and a request admitted past it queues rather than consuming a core. The limit
/// applies per connection, and one tonic `Channel` is one connection.
pub fn default_connection_concurrency() -> u16 {
    500
}

/// Threads ONNX Runtime may use inside one operator, the calling thread included.
///
/// One. Raise it when latency matters more than requests per second, and lower
/// `threading.inference_concurrency` to match.
fn default_intra_op_threads() -> i32 {
    1
}

fn default_sessions_per_model() -> usize {
    1
}

fn default_metrics_batch_size() -> usize {
    500
}

fn default_data_batch_size() -> usize {
    5000
}

fn default_sample_rate() -> f64 {
    1.0
}

fn default_max_sends_in_flight() -> usize {
    4
}

/// Where inference logs go, and how much they are allowed to cost.
///
/// # Fields
///
/// * `uri` - the destination. `kinesis://stream-name` selects a data stream; anything
///   else is a blob-store prefix, written beneath and partitioned by time. Same URI
///   rules as `model_config_path`
/// * `batch_size` - batches accumulated before a send. Default 5000. Kinesis caps one
///   call at 500 records, so a larger number is capped there; a blob store takes an
///   object of any size
/// * `sample_rate` - fraction of requests whose features are recorded, in `0.0..=1.0`.
///   Default 1.0. An unsampled request never copies its feature values anywhere
/// * `max_sends_in_flight` - sends that may be in the air before the oldest batch is
///   dropped and reported. Default 4. Holds at most `batch_size × (this + 1)` records
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InferenceLogConfig {
    pub uri: String,
    #[serde(default = "default_data_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: f64,
    #[serde(default = "default_max_sends_in_flight")]
    pub max_sends_in_flight: usize,
}

/// Where timings go, and how often.
///
/// # Fields
///
/// * `cloudwatch_namespace` - publish to CloudWatch under this namespace. Absent
///   summarises to stderr instead, which needs no credentials
/// * `batch_size` - scored batches accumulated before a send, and the number of batches
///   per stderr summary line. Default 500. CloudWatch takes at most 1000 data points per
///   call and each batch contributes three, so a larger number is capped at 333 there
/// * `max_sends_in_flight` - as in [`InferenceLogConfig`], but over a much smaller
///   record. Default 4
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    pub cloudwatch_namespace: Option<String>,
    #[serde(default = "default_metrics_batch_size")]
    pub batch_size: usize,
    #[serde(default = "default_max_sends_in_flight")]
    pub max_sends_in_flight: usize,
}

impl Default for MetricsConfig {
    /// Absent `metrics` is stderr at the default batch size.
    fn default() -> Self {
        Self {
            cloudwatch_namespace: None,
            batch_size: default_metrics_batch_size(),
            max_sends_in_flight: default_max_sends_in_flight(),
        }
    }
}

/// How the process spends the host's cores, and how much it lets in.
///
/// The CPU load a configuration implies is `inference_concurrency × intra_op_threads`, so
/// the two are chosen together or the process ends up with more runnable threads than
/// cores. `connection_concurrency` is the gate in front of them.
///
/// ```text
///   connection_concurrency  requests one connection may have in flight. Admission,
///                           not work -- what is admitted past the next number queues.
///   worker_threads          async only -- sockets, HTTP/2 framing, spawning sends.
///                           Near-zero CPU, because inference never runs here.
///   inference_concurrency   how many inferences run at once. Each one holds a
///                           blocking thread for the whole vectorise-plus-run.
///   intra_op_threads        threads ONNX Runtime may use inside one operator, the
///                           calling thread included.
/// ```
///
/// These choose where the parallelism goes rather than buying throughput, since the total
/// work in a request is fixed. `inference_concurrency = cores, intra_op_threads = 1` runs
/// one inference per core: the most requests per second, each as slow as one core can
/// manage. `inference_concurrency = 1, intra_op_threads = cores` puts the whole machine
/// behind one request: the same rows per second, a fraction of the latency, and almost
/// nothing in flight.
///
/// # Fields
///
/// * `connection_concurrency` - in-flight requests allowed **per connection**, both as a
///   tower concurrency limit and as HTTP/2's `MAX_CONCURRENT_STREAMS`. Default 500. It
///   bounds one connection and not the process: ten connections admit ten times this
/// * `worker_threads` - tokio async workers. Absent means one per core. This pool does no
///   inference, so two or three is usually enough
/// * `inference_concurrency` - the ceiling on concurrent inferences, which is the size of
///   tokio's blocking pool, and the queue in front of the engine. Absent means one per
///   core
/// * `intra_op_threads` - passed to ONNX Runtime. Default 1, which keeps each inference
///   on its calling thread. `0` leaves the runtime's own default, which is the physical
///   core count. The pool is **per session and shared**, so `n` concurrent callers do not
///   get `n` pools
/// * `inter_op_threads` - the alternative to the above, and **only one of the two may
///   exceed 1**. Intra-op spends threads inside a single operator; inter-op spends them on
///   operators that do not depend on each other. Setting this switches ONNX Runtime to its
///   parallel executor, which is worth doing only on a branching graph — two towers, a
///   trunk with several output heads. A chain of transformer blocks has no branches, and
///   attention heads are not branches either: they are a dimension inside one batched
///   operator, which the intra-op pool already splits. Absent means intra-op
/// * `worker_cores`, `compute_cores` - which logical CPUs each pool may run on, as a
///   `taskset`-style list such as `"0-1"` or `"2-9,12"`. Absent leaves placement to the
///   kernel. Keeping the two lists disjoint stops a burst of connection handling
///   preempting an inference mid-operator. Applied on Linux; accepted and reported as not
///   applied elsewhere, since macOS affinity is advisory
/// * `allow_spinning` - whether idle ONNX Runtime intra-op threads spin waiting for work.
///   Absent leaves the runtime's own default of `true`, which trades CPU for latency. Set
///   it `false` when the pool is larger than the cores it has, where spinning threads
///   steal time from the ones doing work
/// * `sessions_per_model` - independent ONNX Runtime sessions each model is loaded into.
///   Default 1. The intra-op pool belongs to the **session**, so with one session every
///   concurrent request to a model queues for the same pool. `n` gives `n` pools per
///   model, each pinned to its own slice of the compute cores. There are `n` slices, not
///   `models × n`: replica `i` of every model shares slice `i`. Costs one copy of the
///   weights per session, so `models × n` copies
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ThreadingConfig {
    #[serde(default = "default_connection_concurrency")]
    pub connection_concurrency: u16,
    #[serde(default)]
    pub worker_threads: Option<usize>,
    #[serde(default)]
    pub inference_concurrency: Option<usize>,
    #[serde(default = "default_intra_op_threads")]
    pub intra_op_threads: i32,
    #[serde(default)]
    pub inter_op_threads: Option<i32>,
    #[serde(default)]
    pub worker_cores: Option<String>,
    #[serde(default)]
    pub compute_cores: Option<String>,
    #[serde(default)]
    pub allow_spinning: Option<bool>,
    #[serde(default = "default_sessions_per_model")]
    pub sessions_per_model: usize,
}

impl Default for ThreadingConfig {
    /// Absent `threading` is one thread per core on both pools, no intra-op parallelism
    /// and 500 admitted per connection.
    fn default() -> Self {
        Self {
            connection_concurrency: default_connection_concurrency(),
            worker_threads: None,
            inference_concurrency: None,
            intra_op_threads: default_intra_op_threads(),
            inter_op_threads: None,
            worker_cores: None,
            compute_cores: None,
            allow_spinning: None,
            sessions_per_model: default_sessions_per_model(),
        }
    }
}

impl ThreadingConfig {
    /// Async workers to build the serving runtime with.
    pub fn worker_threads(&self, cores: usize) -> usize {
        self.worker_threads.unwrap_or(cores).max(1)
    }

    /// Concurrent inferences allowed, which is the size of tokio's blocking pool.
    pub fn inference_concurrency(&self, cores: usize) -> usize {
        self.inference_concurrency.unwrap_or(cores).max(1)
    }

    /// The logical CPUs the async pool may use, empty when unset.
    ///
    /// [`Self::validate_cores`] has already refused an unparseable list at startup, so
    /// the parse cannot fail here.
    pub fn worker_cores(&self) -> Vec<usize> {
        self.worker_cores
            .as_deref()
            .and_then(|spec| crate::affinity::parse_cores(spec).ok())
            .unwrap_or_default()
    }

    /// The logical CPUs the inference threads may use, empty when unset.
    pub fn compute_cores(&self) -> Vec<usize> {
        self.compute_cores
            .as_deref()
            .and_then(|spec| crate::affinity::parse_cores(spec).ok())
            .unwrap_or_default()
    }

    /// Refuses a core list that will not parse, naming the field.
    fn validate_cores(&self) -> Result<(), InferenceError> {
        for (field, spec) in [
            ("threading.worker_cores", &self.worker_cores),
            ("threading.compute_cores", &self.compute_cores),
        ] {
            if let Some(spec) = spec {
                crate::affinity::parse_cores(spec)
                    .map_err(|e| format!("\"{field}\": {e}; write it as \"0-1\" or \"2-9,12\""))?;
            }
        }
        Ok(())
    }

    /// Which pool carries the width, and how wide it is.
    ///
    /// Inter-op when `inter_op_threads` asks for more than one thread, intra-op otherwise.
    /// [`Self::validate_parallelism`] has already refused a configuration that asks for
    /// both, so this does not have to choose between them.
    pub fn parallelism(&self) -> Parallelism {
        match self.inter_op_threads {
            Some(n) if n > 1 => Parallelism::Inter(n),
            _ => Parallelism::Intra(self.intra_op_threads),
        }
    }

    /// Refuses a configuration that spends threads on both pools.
    ///
    /// The two are alternatives: intra-op widens one operator, inter-op overlaps several.
    /// Asking for both multiplies the pools per session, which no longer divides into the
    /// compute cores the way [`Self::slice_pressure`] reports, so the placement the banner
    /// prints would stop describing the process.
    ///
    /// Choosing inter-op requires `intra_op_threads` to be exactly 1, and that includes
    /// refusing 0. Zero is not "no intra-op pool" — it is ONNX Runtime's own default of one
    /// thread per physical core — so a session with `inter_op_threads` of 8 and
    /// `intra_op_threads` of 0 would claim the machine eight times over, quietly.
    fn validate_parallelism(&self) -> Result<(), InferenceError> {
        let inter = self.inter_op_threads.unwrap_or(1);
        if inter < 0 {
            return Err(format!(
                "\"threading.inter_op_threads\" is {inter}; use a count of 1 or more, or 0 to \
                 leave it to the runtime"
            )
            .into());
        }
        if inter > 1 && self.intra_op_threads != 1 {
            let intra = self.intra_op_threads;
            let because = if intra == 0 {
                "0 is not \"none\" -- it is one intra-op thread per physical core, which this \
                 session would hold alongside the inter-op pool"
            } else {
                "only one of the two may exceed 1, since intra-op widens a single operator \
                 and inter-op overlaps independent ones"
            };
            return Err(format!(
                "\"threading.inter_op_threads\" is {inter} and \"threading.intra_op_threads\" \
                 is {intra}; set intra-op to 1 when choosing inter-op, because {because}"
            )
            .into());
        }
        Ok(())
    }

    /// Sessions to load each model into, never zero.
    ///
    /// Per model, so the total is `models × this`. The slices are not: there are `this`
    /// many, and replica `i` of every model shares slice `i`.
    pub fn sessions_per_model(&self) -> usize {
        self.sessions_per_model.max(1)
    }

    /// Threads in the chosen pool that can be runnable on one slice, and that slice's
    /// width.
    ///
    /// Replica `i` of every model shares slice `i` and each session owns a pool, so the
    /// threads on a slice are `models × pool width` against that slice's cores. An absent
    /// `compute_cores` makes the width the host's `cores` divided by the replicas.
    pub fn slice_pressure(&self, cores: usize, models: usize) -> (usize, usize) {
        let pool = usize::try_from(self.parallelism().width())
            .unwrap_or(1)
            .max(1);
        let replicas = self.sessions_per_model();
        let width = self.compute_cores().len();
        let width = if width == 0 { cores } else { width };
        (pool.saturating_mul(models.max(1)), width / replicas.max(1))
    }

    /// Compute threads this configuration can have runnable at once.
    ///
    /// `inference_concurrency` calling threads, plus the `width - 1` extra threads in each
    /// session's pool, where the width is whichever of intra-op or inter-op was chosen.
    /// Counts threads that *exist* rather than threads running: only
    /// `inference_concurrency` inferences run at once, so with several sessions per model
    /// most pools are idle at any instant — free when `allow_spinning` is false, and not
    /// when it is true.
    ///
    /// # Arguments
    ///
    /// * `cores` - logical CPUs, used when `inference_concurrency` is absent
    /// * `sessions` - the **total** sessions across models, since the pool belongs to the
    ///   session and not to the process: a candidate model resident alongside the control
    ///   brings its own pools, and `sessions_per_model` multiplies both
    pub fn compute_threads(&self, cores: usize, sessions: usize) -> usize {
        let per_pool = usize::try_from(self.parallelism().width())
            .unwrap_or(0)
            .saturating_sub(1);
        self.inference_concurrency(cores)
            .saturating_add(per_pool.saturating_mul(sessions.max(1)))
    }
}

/// Which of ONNX Runtime's two thread pools carries a session's width.
///
/// The pools are alternatives rather than a pair to tune together: the configuration
/// permits only one of the counts to exceed 1, so this records which was chosen. That
/// keeps the thread budget a single number — see
/// [`ThreadingConfig::compute_threads`] — instead of a product of two.
///
/// The choice is not symmetric, and the asymmetry is placement. ONNX Runtime pins each
/// intra-op thread to a named core through `session.intra_op_thread_affinities`; it has no
/// equivalent for the inter-op pool. An inter-op pool is therefore only as confined as the
/// mask of the thread that created the session, which is why sessions are built on a
/// thread pinned to their own slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Parallelism {
    /// Threads split the work inside one operator. `1` is no parallelism at all, and `0`
    /// leaves ONNX Runtime to pick.
    Intra(i32),
    /// Threads run operators that do not depend on each other. Needs a branching graph to
    /// be worth anything.
    Inter(i32),
}

impl Parallelism {
    /// Threads in the chosen pool, counting the calling thread.
    pub fn width(self) -> i32 {
        match self {
            Self::Intra(n) | Self::Inter(n) => n,
        }
    }

    /// Whether ONNX Runtime should run operators concurrently.
    pub fn is_parallel(self) -> bool {
        matches!(self, Self::Inter(_))
    }

    /// The pool's name, for the banner.
    pub fn pool_name(self) -> &'static str {
        match self {
            Self::Intra(_) => "intra-op",
            Self::Inter(_) => "inter-op",
        }
    }
}

/// A second model to serve alongside the first, and how much traffic it takes.
///
/// This is a roll-out: `model_config_path` is the control, this is the candidate, and
/// `traffic_percent` is the share the candidate serves.
///
/// A caller naming a model in its request overrides this entirely — see
/// `InferenceRequest.model_id`.
///
/// # Fields
///
/// * `config_path` - the candidate's own model configuration, with its own URI scheme. It
///   carries its own `vectorization_config`, so the candidate may take its features
///   differently from the control
/// * `traffic_percent` - whole percent, `0..=100`, that the candidate serves. `0` loads
///   it and sends it nothing; `100` sends everything to it
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CandidateModelConfig {
    pub config_path: String,
    pub traffic_percent: u8,
}

/// The server process's own settings.
///
/// # Fields
///
/// * `bind_address` - address to listen on. Default `0.0.0.0`, which accepts from
///   anywhere; `127.0.0.1` restricts to the host. `--bind-address` overrides it
/// * `port_number` - the port to listen on. Default 8279, which `--port` overrides
/// * `model_config_path` - where the model's own configuration is. Carries its own URI
///   scheme, so a model configuration in S3 beside a service configuration on local disk
///   is expressible. This is the control arm when a candidate is configured
/// * `threading` - [`ThreadingConfig`]. Optional; absent is one thread per core on both
///   pools, no intra-op parallelism, and 500 requests admitted per connection
/// * `candidate_model` - [`CandidateModelConfig`]. Optional; absent serves one model
/// * `inference_log` - [`InferenceLogConfig`]. Required
/// * `metrics` - [`MetricsConfig`]. Optional; absent means stderr
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HusharServiceConfig {
    #[serde(default = "default_bind_address")]
    pub bind_address: IpAddr,
    #[serde(default = "default_port_number")]
    pub port_number: u16,
    pub model_config_path: String,
    #[serde(default)]
    pub threading: ThreadingConfig,
    #[serde(default)]
    pub candidate_model: Option<CandidateModelConfig>,
    pub inference_log: InferenceLogConfig,
    #[serde(default)]
    pub metrics: MetricsConfig,
}

impl HusharServiceConfig {
    /// Parses a service configuration from JSON and validates it.
    ///
    /// # Errors
    ///
    /// Returns an error if the JSON is missing a required field or has one of the wrong
    /// type, or if [`Self::validate`] refuses a value.
    pub fn from_json(json_str: &str) -> Result<Self, InferenceError> {
        let config: HusharServiceConfig = serde_json::from_str(json_str)?;
        config.validate()?;
        Ok(config)
    }

    /// Refuses values a sink or sampler could not honour, naming the field and the range.
    ///
    /// Values are refused rather than clamped.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero `threading.connection_concurrency`,
    /// `threading.worker_threads` or `threading.inference_concurrency`; a negative
    /// `threading.intra_op_threads`; a core list that will not parse; an empty
    /// `candidate_model.config_path` or a `traffic_percent` above 100; an
    /// `inference_log.sample_rate` outside `0.0..=1.0`; a zero `batch_size` in either
    /// group; a `max_sends_in_flight` that is zero or above `u32::MAX` in either group;
    /// or an empty `metrics.cloudwatch_namespace`.
    pub fn validate(&self) -> Result<(), InferenceError> {
        if self.threading.connection_concurrency == 0 {
            return Err(format!(
                "\"threading.connection_concurrency\" is 0, so no request would ever be \
                 admitted and the server would accept connections and answer none of \
                 them; leave the field out to admit {} per connection",
                default_connection_concurrency()
            )
            .into());
        }
        for (field, value) in [
            ("threading.worker_threads", self.threading.worker_threads),
            (
                "threading.inference_concurrency",
                self.threading.inference_concurrency,
            ),
        ] {
            if value == Some(0) {
                return Err(format!(
                    "\"{field}\" is 0, so the pool would have no thread to run on; \
                     leave the field out to get one per core"
                )
                .into());
            }
        }
        self.threading.validate_cores()?;
        self.threading.validate_parallelism()?;
        if self.threading.intra_op_threads < 0 {
            return Err(format!(
                "\"threading.intra_op_threads\" is {}, which is not a thread count; \
                 1 keeps each inference on its calling thread and 0 leaves the \
                 runtime's own default",
                self.threading.intra_op_threads
            )
            .into());
        }

        if let Some(candidate) = &self.candidate_model {
            if candidate.config_path.trim().is_empty() {
                return Err("\"candidate_model.config_path\" is empty; remove the \
                            candidate_model block to serve one model"
                    .into());
            }
            if candidate.traffic_percent > 100 {
                return Err(format!(
                    "\"candidate_model.traffic_percent\" is {}, which is not a percentage; \
                     it is the share of traffic the candidate serves, so 0 to 100",
                    candidate.traffic_percent
                )
                .into());
            }
        }

        validate_sample_rate(self.inference_log.sample_rate)?;
        validate_batch_size("inference_log.batch_size", self.inference_log.batch_size)?;
        validate_sends_in_flight(
            "inference_log.max_sends_in_flight",
            self.inference_log.max_sends_in_flight,
        )?;

        validate_batch_size("metrics.batch_size", self.metrics.batch_size)?;
        validate_sends_in_flight(
            "metrics.max_sends_in_flight",
            self.metrics.max_sends_in_flight,
        )?;

        if let Some(namespace) = &self.metrics.cloudwatch_namespace
            && namespace.trim().is_empty()
        {
            return Err(
                "\"metrics.cloudwatch_namespace\" is empty; leave the field out \
                        to summarise timings to stderr instead"
                    .into(),
            );
        }
        Ok(())
    }
}

/// Checks a batch size, refusing zero because every batch would then be sent empty.
fn validate_batch_size(field: &str, size: usize) -> Result<(), InferenceError> {
    if size == 0 {
        return Err(format!(
            "{field:?} is 0, so nothing would ever accumulate and every batch would be \
             sent empty; 1 sends each one as it is produced"
        )
        .into());
    }
    Ok(())
}

/// Checks the in-flight bound, refusing what a sink could not honour.
///
/// # Errors
///
/// Returns an error if `sends` is zero, which would leave no slot for a send to start in,
/// or above `u32::MAX`: draining at shutdown acquires every permit at once, and the
/// semaphore counts them in a `u32`.
fn validate_sends_in_flight(field: &str, sends: usize) -> Result<(), InferenceError> {
    if sends == 0 {
        return Err(format!(
            "{field:?} is 0, which would leave no slot for a send to start in; 1 sends \
             one at a time"
        )
        .into());
    }
    if sends > u32::MAX as usize {
        return Err(format!(
            "{field:?} is {sends}, more sends than can be tracked; the practical range \
             is single digits, since each one in flight holds a whole batch in memory"
        )
        .into());
    }
    Ok(())
}

/// Checks a sampling rate, refusing what a sampler could not honour.
fn validate_sample_rate(rate: f64) -> Result<(), InferenceError> {
    if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
        return Err(format!(
            "\"inference_log.sample_rate\" is {rate}, which is not a fraction between \
             0.0 and 1.0; 0.01 is one request in a hundred and 1.0 is all of them"
        )
        .into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The smallest configuration that starts.
    const MINIMAL: &str = r#"{
        "model_config_path": "/tmp/model.json",
        "inference_log": { "uri": "/tmp/logs" }
    }"#;

    /// Asserts slice geometry is fixed by the configuration and does not move with the
    /// model count.
    #[test]
    fn slices_do_not_move_with_the_model_count() {
        let threading = ThreadingConfig {
            sessions_per_model: 4,
            intra_op_threads: 22,
            compute_cores: Some("8-95".to_owned()),
            ..Default::default()
        };
        assert_eq!(
            threading.sessions_per_model(),
            4,
            "sessions per model are configured, not derived"
        );
        let (one_model, width) = threading.slice_pressure(96, 1);
        let (two_models, width_again) = threading.slice_pressure(96, 2);
        assert_eq!(
            (width, width_again),
            (22, 22),
            "22 cores per slice either way"
        );
        assert_eq!(one_model, 22, "one model fills the slice exactly");
        assert_eq!(
            two_models, 44,
            "a candidate doubles the threads on the slice"
        );
    }

    /// Asserts pressure is computed from the slice width, not the host's core count.
    #[test]
    fn slice_pressure_uses_the_slice_not_the_host() {
        let threading = ThreadingConfig {
            sessions_per_model: 8,
            intra_op_threads: 11,
            compute_cores: Some("8-95".to_owned()),
            ..Default::default()
        };
        let (per_slice, width) = threading.slice_pressure(96, 2);
        assert_eq!(width, 11, "88 compute cores over 8 slices");
        assert_eq!(
            per_slice, 22,
            "two 11-thread pools land on one 11-core slice"
        );
    }

    /// Asserts an absent `compute_cores` divides the host's cores across the slices.
    #[test]
    fn slice_pressure_falls_back_to_the_host_width() {
        let threading = ThreadingConfig {
            sessions_per_model: 4,
            intra_op_threads: 8,
            ..Default::default()
        };
        let (per_slice, width) = threading.slice_pressure(64, 1);
        assert_eq!(
            (per_slice, width),
            (8, 16),
            "64 cores over 4 slices is 16 each"
        );
    }

    #[test]
    fn a_service_config_carries_the_listener_the_model_pointer_and_both_sinks() {
        let config = HusharServiceConfig::from_json(
            r#"{
                "bind_address": "127.0.0.1",
                "port_number": 8080,
                "threading": { "connection_concurrency": 25 },
                "model_config_path": "s3://configs/embedding-v1/model_config.json",
                "inference_log": {
                    "uri": "kinesis://inference-logs",
                    "batch_size": 400,
                    "sample_rate": 0.01,
                    "max_sends_in_flight": 8
                },
                "metrics": {
                    "cloudwatch_namespace": "HusharService",
                    "batch_size": 100,
                    "max_sends_in_flight": 2
                }
            }"#,
        )
        .expect("valid");

        assert_eq!(config.bind_address, IpAddr::from([127, 0, 0, 1]));
        assert_eq!(config.port_number, 8080);
        assert_eq!(config.threading.connection_concurrency, 25);
        assert_eq!(
            config.model_config_path,
            "s3://configs/embedding-v1/model_config.json"
        );
        assert_eq!(config.inference_log.uri, "kinesis://inference-logs");
        assert_eq!(config.inference_log.batch_size, 400);
        assert_eq!(config.inference_log.sample_rate, 0.01);
        assert_eq!(config.inference_log.max_sends_in_flight, 8);
        assert_eq!(
            config.metrics.cloudwatch_namespace.as_deref(),
            Some("HusharService")
        );
        assert_eq!(config.metrics.batch_size, 100);
        assert_eq!(config.metrics.max_sends_in_flight, 2);
    }

    /// Asserts only the model path and the log destination are required, everything else
    /// taking its default.
    #[test]
    fn only_the_model_path_and_the_log_destination_are_required() {
        let config = HusharServiceConfig::from_json(MINIMAL).expect("valid");

        assert_eq!(config.bind_address, default_bind_address());
        assert_eq!(config.port_number, default_port_number());
        assert_eq!(
            config.threading.connection_concurrency,
            default_connection_concurrency(),
            "left out, admission takes the default rather than depending on the host"
        );
        assert_eq!(config.inference_log.batch_size, default_data_batch_size());
        assert_eq!(
            config.inference_log.max_sends_in_flight,
            default_max_sends_in_flight()
        );
        assert_eq!(
            config.inference_log.sample_rate, 1.0,
            "logging every request is what the service did before there was a lever"
        );
    }

    /// Asserts timings go to stderr when no CloudWatch namespace is configured.
    #[test]
    fn metrics_go_to_stderr_unless_a_namespace_is_given() {
        let config = HusharServiceConfig::from_json(MINIMAL).expect("valid");
        assert_eq!(config.metrics, MetricsConfig::default());
        assert_eq!(config.metrics.cloudwatch_namespace, None);
        assert_eq!(config.metrics.batch_size, default_metrics_batch_size());
    }

    #[test]
    fn a_missing_model_config_path_is_refused() {
        assert!(
            HusharServiceConfig::from_json(
                r#"{"port_number": 8000, "inference_log": {"uri": "/tmp/logs"}}"#
            )
            .is_err(),
            "there is nothing sensible to serve without one"
        );
    }

    /// Asserts a missing `inference_log` is refused with the field named in the error.
    #[test]
    fn a_missing_inference_log_is_refused_by_name() {
        let err = HusharServiceConfig::from_json(r#"{"model_config_path": "/tmp/m.json"}"#)
            .expect_err("the log destination is required");
        assert!(
            err.to_string().contains("inference_log"),
            "the error should name the missing field: {err}"
        );
    }

    #[test]
    fn an_inference_log_without_a_uri_is_refused() {
        assert!(
            HusharServiceConfig::from_json(
                r#"{"model_config_path": "/tmp/m.json", "inference_log": {"batch_size": 10}}"#
            )
            .is_err(),
            "a batch size with nowhere to send it is not a destination"
        );
    }

    /// Asserts a flat top-level sink field is refused rather than silently ignored.
    #[test]
    fn a_flat_field_from_the_old_command_line_is_refused_rather_than_ignored() {
        for stale in [
            r#""inference_log_uri": "/tmp/logs""#,
            r#""cloudwatch_namespace": "Hushar""#,
            r#""metrics_batch_size": 500"#,
            r#""data_batch_size": 5000"#,
            r#""data_sample_rate": 0.5"#,
            r#""max_sends_in_flight": 4"#,
        ] {
            let json = format!(
                r#"{{"model_config_path": "/tmp/m.json",
                     "inference_log": {{"uri": "/tmp/logs"}}, {stale}}}"#
            );
            assert!(
                HusharServiceConfig::from_json(&json).is_err(),
                "{stale} is grouped now and must be refused at the top level"
            );
        }
    }

    /// Asserts a model-level field name is refused in the service configuration.
    #[test]
    fn a_stale_model_field_is_refused_rather_than_ignored() {
        for stale in ["model_path", "model_id", "vectorization_instruction_path"] {
            let json = format!(
                r#"{{"model_config_path": "/tmp/m.json",
                     "inference_log": {{"uri": "/tmp/logs"}}, "{stale}": "x"}}"#
            );
            assert!(
                HusharServiceConfig::from_json(&json).is_err(),
                "{stale} moved into the model configuration and must be refused here"
            );
        }
    }

    #[test]
    fn an_invalid_type_is_refused() {
        assert!(
            HusharServiceConfig::from_json(
                r#"{"threading": {"connection_concurrency": "not-a-number"},
                    "model_config_path": "/tmp/m.json",
                    "inference_log": {"uri": "/tmp/logs"}}"#
            )
            .is_err()
        );
    }

    /// Asserts a malformed bind address is refused and an IPv6 address is accepted.
    #[test]
    fn a_bad_bind_address_is_refused_and_ipv6_is_accepted() {
        let with = |address: &str| {
            format!(
                r#"{{"bind_address": "{address}", "model_config_path": "/tmp/m.json",
                     "inference_log": {{"uri": "/tmp/logs"}}}}"#
            )
        };
        assert!(HusharServiceConfig::from_json(&with("not-an-address")).is_err());
        let config = HusharServiceConfig::from_json(&with("::1")).expect("IPv6 is an address");
        assert_eq!(config.bind_address, "::1".parse::<IpAddr>().expect("valid"));
    }

    #[test]
    fn a_sampling_rate_is_accepted_across_its_range() {
        for rate in ["0", "0.001", "0.5", "1", "1.0"] {
            let json = format!(
                r#"{{"model_config_path": "/tmp/m.json",
                     "inference_log": {{"uri": "/tmp/logs", "sample_rate": {rate}}}}}"#
            );
            assert!(
                HusharServiceConfig::from_json(&json).is_ok(),
                "{rate} should be a valid rate"
            );
        }
    }

    /// Asserts a sample rate outside `0.0..=1.0` is refused, with the field and range in
    /// the message.
    #[test]
    fn a_sampling_rate_outside_the_range_is_refused_with_an_explanation() {
        for rate in ["10", "-0.5", "1.5"] {
            let json = format!(
                r#"{{"model_config_path": "/tmp/m.json",
                     "inference_log": {{"uri": "/tmp/logs", "sample_rate": {rate}}}}}"#
            );
            let err = HusharServiceConfig::from_json(&json).expect_err("not a fraction");
            let message = err.to_string();
            assert!(
                message.contains("0.0") && message.contains("1.0"),
                "the error for {rate} should give the range: {message}"
            );
            assert!(
                message.contains("sample_rate"),
                "the error for {rate} should name the field: {message}"
            );
        }
    }

    /// Asserts admission defaults to 500 per connection, independently of the host.
    #[test]
    fn the_concurrency_limit_defaults_to_five_hundred_per_connection() {
        let config = HusharServiceConfig::from_json(MINIMAL).expect("valid");
        assert_eq!(config.threading.connection_concurrency, 500);
        assert_eq!(default_connection_concurrency(), 500);
    }

    /// Asserts a top-level `connection_concurrency` is refused by name.
    #[test]
    fn a_top_level_concurrency_limit_is_refused_now_that_it_lives_in_threading() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "connection_concurrency": 500,
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("the field moved");
        assert!(
            err.to_string().contains("connection_concurrency"),
            "the error should name the field that moved: {err}"
        );
    }

    /// Asserts a configured admission limit is used as given, not adjusted to the host.
    #[test]
    fn a_configured_concurrency_limit_is_used_as_given() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "threading": {"connection_concurrency": 5},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let config = HusharServiceConfig::from_json(json).expect("valid");
        assert_eq!(
            config.threading.connection_concurrency, 5,
            "a stated limit is not raised to fit the machine"
        );
    }

    /// Asserts a zero admission limit is refused, naming the field and the remedy.
    #[test]
    fn a_concurrency_limit_of_zero_is_refused() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "threading": {"connection_concurrency": 0},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("zero admits nothing");
        let message = err.to_string();
        assert!(
            message.contains("connection_concurrency"),
            "the error should name the field: {message}"
        );
        assert!(
            message.contains("leave the field out"),
            "the error should say what to do instead: {message}"
        );
    }

    /// Absent is the ordinary case: one model, no split.
    #[test]
    fn a_configuration_without_a_candidate_serves_one_model() {
        let config = HusharServiceConfig::from_json(MINIMAL).expect("valid");
        assert_eq!(config.candidate_model, None);
    }

    #[test]
    fn a_candidate_model_parses_with_its_share() {
        let json = r#"{"model_config_path": "/tmp/control.json",
                       "candidate_model": {"config_path": "/tmp/candidate.json",
                                           "traffic_percent": 10},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let config = HusharServiceConfig::from_json(json).expect("valid");
        let candidate = config.candidate_model.expect("a candidate was configured");
        assert_eq!(candidate.config_path, "/tmp/candidate.json");
        assert_eq!(candidate.traffic_percent, 10);
    }

    /// Asserts both ends of a roll-out, 0 and 100 percent, are accepted.
    #[test]
    fn the_ends_of_a_rollout_are_accepted() {
        for percent in [0, 100] {
            let json = format!(
                r#"{{"model_config_path": "/tmp/c.json",
                      "candidate_model": {{"config_path": "/tmp/n.json",
                                           "traffic_percent": {percent}}},
                      "inference_log": {{"uri": "/tmp/logs"}}}}"#
            );
            let config = HusharServiceConfig::from_json(&json)
                .unwrap_or_else(|e| panic!("{percent}% should be valid: {e}"));
            assert_eq!(config.candidate_model.unwrap().traffic_percent, percent);
        }
    }

    /// Asserts a `traffic_percent` above 100 is refused by name and range.
    #[test]
    fn a_share_above_a_hundred_is_refused() {
        let json = r#"{"model_config_path": "/tmp/c.json",
                       "candidate_model": {"config_path": "/tmp/n.json",
                                           "traffic_percent": 150},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("150% is not a share");
        let message = err.to_string();
        assert!(
            message.contains("traffic_percent"),
            "the error should name the field: {message}"
        );
        assert!(
            message.contains("0 to 100"),
            "the error should give the range: {message}"
        );
    }

    /// Asserts a share above `u8::MAX` is refused by serde on the field's type.
    #[test]
    fn a_share_beyond_the_field_is_refused_on_type() {
        let json = r#"{"model_config_path": "/tmp/c.json",
                       "candidate_model": {"config_path": "/tmp/n.json",
                                           "traffic_percent": 300},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        assert!(HusharServiceConfig::from_json(json).is_err());
    }

    #[test]
    fn a_candidate_without_a_path_is_refused() {
        let json = r#"{"model_config_path": "/tmp/c.json",
                       "candidate_model": {"config_path": "  ", "traffic_percent": 10},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("an empty path is not a model");
        assert!(
            err.to_string().contains("candidate_model.config_path"),
            "the error should name the field: {err}"
        );
        let json = r#"{"model_config_path": "/tmp/c.json",
                       "candidate_model": {"traffic_percent": 10},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        assert!(HusharServiceConfig::from_json(json).is_err());
    }

    /// Asserts `traffic_percent` is required rather than defaulted, and named in the
    /// error.
    #[test]
    fn a_candidate_without_a_share_is_refused() {
        let json = r#"{"model_config_path": "/tmp/c.json",
                       "candidate_model": {"config_path": "/tmp/n.json"},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("the share is required");
        assert!(
            err.to_string().contains("traffic_percent"),
            "the error should name the missing field: {err}"
        );
    }

    #[test]
    fn an_unknown_field_in_the_candidate_block_is_refused() {
        let json = r#"{"model_config_path": "/tmp/c.json",
                       "candidate_model": {"config_path": "/tmp/n.json",
                                           "traffic_percent": 10, "weight": 3},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        assert!(HusharServiceConfig::from_json(json).is_err());
    }

    /// Asserts an unparseable core list is refused at startup, naming the field and the
    /// offending fragment.
    #[test]
    fn a_core_list_that_will_not_parse_is_refused_by_field() {
        for field in ["worker_cores", "compute_cores"] {
            let json = format!(
                r#"{{"model_config_path": "/tmp/m.json",
                      "threading": {{"{field}": "0-2,nope"}},
                      "inference_log": {{"uri": "/tmp/logs"}}}}"#
            );
            let err = HusharServiceConfig::from_json(&json).expect_err("bad list");
            let message = err.to_string();
            assert!(
                message.contains(field),
                "the error should name the field: {message}"
            );
            assert!(
                message.contains("nope"),
                "the error should name the offending fragment: {message}"
            );
        }
    }

    #[test]
    fn core_lists_parse_into_ascending_cores() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "threading": {"worker_cores": "0-1", "compute_cores": "2-5,9"},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let config = HusharServiceConfig::from_json(json).expect("valid");
        assert_eq!(config.threading.worker_cores(), vec![0, 1]);
        assert_eq!(config.threading.compute_cores(), vec![2, 3, 4, 5, 9]);
    }

    /// Asserts absent core lists and `allow_spinning` pin nothing, leaving placement to
    /// the kernel.
    #[test]
    fn absent_core_lists_pin_nothing() {
        let config = HusharServiceConfig::from_json(MINIMAL).expect("valid");
        assert!(config.threading.worker_cores().is_empty());
        assert!(config.threading.compute_cores().is_empty());
        assert_eq!(config.threading.allow_spinning, None);
    }

    /// Asserts a second resident session adds a second intra-op pool to the thread count.
    #[test]
    fn a_resident_candidate_model_doubles_the_intra_op_pool() {
        let config = ThreadingConfig {
            worker_threads: Some(24),
            inference_concurrency: Some(8),
            intra_op_threads: 21,
            ..ThreadingConfig::default()
        };

        assert_eq!(
            config.compute_threads(192, 1),
            28,
            "one session: 8 calling threads + 20 pool threads"
        );
        assert_eq!(
            config.compute_threads(192, 2),
            48,
            "two sessions: 8 calling threads + two pools of 20, not one"
        );
        assert!(
            config.compute_threads(192, 2) * 2 <= 192,
            "48 of 192 cores is the undersubscription the banner has to report"
        );
    }

    /// Asserts `intra_op_threads` of 0 or 1 adds no pool threads and does not underflow.
    #[test]
    fn a_pool_of_one_or_fewer_adds_no_threads() {
        for intra_op in [0, 1] {
            let config = ThreadingConfig {
                inference_concurrency: Some(8),
                intra_op_threads: intra_op,
                ..ThreadingConfig::default()
            };
            assert_eq!(
                config.compute_threads(192, 2),
                8,
                "intra_op_threads {intra_op} contributes no pool threads"
            );
        }
    }

    /// Asserts the width is read from whichever pool was chosen, so an inter-op session
    /// budgets its threads exactly as an intra-op one of the same width does.
    #[test]
    fn the_chosen_pool_carries_the_width() {
        let intra = ThreadingConfig {
            inference_concurrency: Some(8),
            intra_op_threads: 14,
            ..ThreadingConfig::default()
        };
        let inter = ThreadingConfig {
            inference_concurrency: Some(8),
            intra_op_threads: 1,
            inter_op_threads: Some(14),
            ..ThreadingConfig::default()
        };

        assert_eq!(intra.parallelism(), Parallelism::Intra(14));
        assert_eq!(inter.parallelism(), Parallelism::Inter(14));
        assert!(!intra.parallelism().is_parallel());
        assert!(inter.parallelism().is_parallel());
        assert_eq!(
            inter.compute_threads(64, 4),
            intra.compute_threads(64, 4),
            "the same width costs the same threads whichever pool holds it"
        );
        assert_eq!(
            inter.slice_pressure(64, 1),
            intra.slice_pressure(64, 1),
            "and divides into the slices the same way"
        );
    }

    /// Asserts an inter-op count of 1 or less is not a choice, so the default stays intra-op
    /// and nothing switches ONNX Runtime's executor by accident.
    #[test]
    fn a_narrow_inter_op_pool_is_not_a_choice() {
        for inter in [None, Some(0), Some(1)] {
            let config = ThreadingConfig {
                intra_op_threads: 7,
                inter_op_threads: inter,
                ..ThreadingConfig::default()
            };
            assert_eq!(
                config.parallelism(),
                Parallelism::Intra(7),
                "inter_op_threads {inter:?} should leave the width on the intra-op pool"
            );
        }
    }

    /// Asserts both pools cannot be widened at once, and that the refusal names both counts.
    #[test]
    fn spending_threads_on_both_pools_is_refused() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "threading": {"intra_op_threads": 8, "inter_op_threads": 4},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("both pools widened");
        let message = err.to_string();
        for expected in ["intra_op_threads", "inter_op_threads", "8", "4"] {
            assert!(
                message.contains(expected),
                "the error should name {expected}: {message}"
            );
        }
    }

    /// Asserts inter-op alongside `intra_op_threads` of 0 is refused.
    ///
    /// The quiet one: 0 is not "no pool", it is one intra-op thread per physical core, so
    /// this pairing would hold the machine once for each pool and the banner's thread count
    /// would be wrong by a factor of the core count.
    #[test]
    fn inter_op_with_the_runtime_default_intra_op_is_refused() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "threading": {"intra_op_threads": 0, "inter_op_threads": 8},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("0 is one thread per core");
        let message = err.to_string();
        assert!(
            message.contains("per physical core"),
            "the error should explain what 0 means: {message}"
        );
    }

    /// Asserts choosing inter-op alone is accepted, since the intra-op default is already 1.
    #[test]
    fn inter_op_alone_needs_no_other_field() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "threading": {"inter_op_threads": 8},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let config = HusharServiceConfig::from_json(json).expect("valid");
        assert_eq!(config.threading.parallelism(), Parallelism::Inter(8));
        assert_eq!(config.threading.parallelism().pool_name(), "inter-op");
    }

    /// Asserts a negative inter-op count is refused rather than wrapping into a huge pool.
    #[test]
    fn a_negative_inter_op_count_is_refused() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "threading": {"inter_op_threads": -2},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("negative count");
        assert!(
            err.to_string().contains("inter_op_threads"),
            "the error should name the field: {err}"
        );
    }

    #[test]
    fn spinning_is_a_tri_state() {
        for (json_value, expected) in [("true", Some(true)), ("false", Some(false))] {
            let json = format!(
                r#"{{"model_config_path": "/tmp/m.json",
                      "threading": {{"allow_spinning": {json_value}}},
                      "inference_log": {{"uri": "/tmp/logs"}}}}"#
            );
            let config = HusharServiceConfig::from_json(&json).expect("valid");
            assert_eq!(config.threading.allow_spinning, expected);
        }
    }

    /// Asserts a zero send bound is refused in either group, with the reason and the
    /// group in the message.
    #[test]
    fn a_send_bound_of_zero_is_refused_with_an_explanation() {
        let cases = [
            (
                "inference_log",
                r#"{"model_config_path": "/tmp/m.json",
                    "inference_log": {"uri": "/tmp/logs", "max_sends_in_flight": 0}}"#,
            ),
            (
                "metrics",
                r#"{"model_config_path": "/tmp/m.json",
                    "inference_log": {"uri": "/tmp/logs"},
                    "metrics": {"max_sends_in_flight": 0}}"#,
            ),
        ];
        for (group, json) in cases {
            let err = HusharServiceConfig::from_json(json).expect_err("zero leaves no slot");
            let message = err.to_string();
            assert!(
                message.contains("no slot"),
                "the error should say why: {message}"
            );
            assert!(
                message.contains(group),
                "the error should name which group: {message}"
            );
        }
    }

    #[test]
    fn a_send_bound_wider_than_the_semaphore_is_refused() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "inference_log": {"uri": "/tmp/logs",
                                         "max_sends_in_flight": 99999999999}}"#;
        assert!(
            HusharServiceConfig::from_json(json).is_err(),
            "a bound wider than the semaphore can drain must be refused"
        );
    }

    #[test]
    fn a_batch_size_of_zero_is_refused_in_either_group() {
        for json in [
            r#"{"model_config_path": "/tmp/m.json",
                "inference_log": {"uri": "/tmp/logs", "batch_size": 0}}"#,
            r#"{"model_config_path": "/tmp/m.json",
                "inference_log": {"uri": "/tmp/logs"}, "metrics": {"batch_size": 0}}"#,
        ] {
            let err = HusharServiceConfig::from_json(json).expect_err("zero accumulates nothing");
            assert!(
                err.to_string().contains("batch_size"),
                "the error should name the field: {err}"
            );
        }
    }

    /// Asserts an empty CloudWatch namespace is refused at startup rather than at the
    /// first publish.
    #[test]
    fn an_empty_cloudwatch_namespace_is_refused() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "inference_log": {"uri": "/tmp/logs"},
                       "metrics": {"cloudwatch_namespace": "  "}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("empty is not a namespace");
        assert!(
            err.to_string().contains("stderr"),
            "the error should say what to do instead: {err}"
        );
    }

    /// Asserts the shipped `test-data/service_config.json` fixture parses.
    #[test]
    fn a_complete_service_configuration_parses() {
        let path = std::path::Path::new("test-data/service_config.json");
        let json = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

        let config = HusharServiceConfig::from_json(&json).expect("must parse");
        assert_eq!(
            config.threading.connection_concurrency,
            default_connection_concurrency(),
            "the shipped example omits admission, which is the shape a fleet should copy"
        );
        assert_eq!(config.threading.worker_threads, Some(2));
        assert_eq!(config.threading.inference_concurrency, Some(8));
        assert_eq!(config.threading.intra_op_threads, 1);
        assert_eq!(
            config.threading.compute_threads(16, 1),
            8,
            "8 concurrent inferences + (1 intra-op - 1): the counts add, because the \
             intra-op pool belongs to the session and is shared by every caller"
        );
        assert_eq!(config.port_number, 8279);
        assert_eq!(
            config.model_config_path, "test-data/model_config_vectorized.json",
            "it should point at the model configuration"
        );
        assert_eq!(config.inference_log.uri, "/tmp/hushar-inference-logs");
        assert_eq!(config.inference_log.sample_rate, 1.0);
        assert_eq!(
            config.metrics.cloudwatch_namespace, None,
            "the fixture must start without credentials"
        );
    }
}
