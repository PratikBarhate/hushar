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
//!     "max_sends_in_flight": 4,
//!     "prefix_shards": 16
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

use std::collections::BTreeMap;
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
/// bound on work: `threading.sessions_per_model` decides how many inferences run at
/// once, and a request admitted past it queues rather than consuming a core. The limit
/// applies per connection, and one tonic `Channel` is one connection.
pub fn default_connection_concurrency() -> u16 {
    500
}

/// Threads ONNX Runtime may use inside one operator, the calling thread included.
///
/// One. Raise it when latency matters more than requests per second, and lower
/// `threading.sessions_per_model` to match.
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

/// Spread log objects over this many key prefixes. See
/// [`partitioned_key`](crate::io::data_sink).
fn default_prefix_shards() -> u32 {
    16
}

/// Where inference logs go, and how much they are allowed to cost.
///
/// # Fields
///
/// * `uri` - a blob-store prefix, written beneath and partitioned by shard and time.
///   Same URI rules as `model_config_path`
/// * `batch_size` - batches accumulated before a write. Default 5000. Not clamped, since
///   a blob store takes an object of any size
/// * `sample_rate` - fraction of requests whose features are recorded, in `0.0..=1.0`.
///   Default 1.0. An unsampled request never copies its feature values anywhere
/// * `max_sends_in_flight` - sends that may be in the air before the oldest batch is
///   dropped and reported. Default 4. Holds at most `batch_size × (this + 1)` records
/// * `prefix_shards` - key prefixes the objects are spread over, so that a fleet writing
///   to S3 is not sharing one request-rate limit. Default 16; 1 writes them all beneath
///   one prefix
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
    #[serde(default = "default_prefix_shards")]
    pub prefix_shards: u32,
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

/// How the process admits work and how ONNX Runtime spends host cores.
///
/// `connection_concurrency`, `worker_threads`, and `sessions_per_model` apply to every
/// backend. A session is an ONNX Runtime session or an AWS Neuron model replica. The
/// remaining fields configure ONNX Runtime only; NeuronCore placement and reusable
/// execution contexts live in [`crate::config::NeuronConfig`].
///
/// For ONNX Runtime, the CPU load a configuration implies is
/// `sessions_per_model × intra_op_threads`, so the two are chosen together.
///
/// ```text
///   connection_concurrency  requests one connection may have in flight. Admission,
///                           not work -- what is admitted past the next number queues.
///   worker_threads          async only -- sockets, HTTP/2 framing, spawning sends.
///                           Near-zero CPU, because inference never runs here.
///   sessions_per_model      how many inferences run at once: one blocking thread per
///                           ONNX Runtime session or Neuron model replica.
///   intra_op_threads        ONNX Runtime threads inside one operator, caller included.
///                           Not used by Neuron.
/// ```
///
/// These choose where the parallelism goes rather than buying throughput, since the total
/// work in a request is fixed. `sessions_per_model = cores, intra_op_threads = 1` runs one
/// inference per core: the most requests per second, each as slow as one core can manage.
/// `sessions_per_model = 1, intra_op_threads = cores` puts the whole machine behind one
/// request: the same rows per second, a fraction of the latency, and one in flight.
///
/// ONE NUMBER, NOT TWO. Concurrency used to be its own setting, separate from the session
/// count, and the two could disagree in both directions: fewer concurrent inferences than
/// sessions left sessions that were never called, and more put several callers on one
/// session's intra-op pool -- the contention the pools exist to avoid. A sweep over 16, 32
/// and 64 concurrent inferences against 4 sessions moved the median by under 4 %, so the
/// knob was not buying anything to justify the ways it could be set wrong. The blocking
/// pool is now exactly as wide as there are sessions, so a request always finds an idle
/// session to lease and the thread budget is one multiplication.
///
/// # Fields
///
/// * `connection_concurrency` - in-flight requests allowed **per connection**, both as a
///   tower concurrency limit and as HTTP/2's `MAX_CONCURRENT_STREAMS`. Default 500. It
///   bounds one connection and not the process: ten connections admit ten times this
/// * `worker_threads` - tokio async workers. Absent means one per core. This pool does no
///   inference, so two or three is usually enough
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
/// * `allow_spinning` - whether idle ONNX Runtime intra-op threads spin waiting for work.
///   Absent leaves the runtime's own default of `true`, which trades CPU for latency. Set
///   it `false` when the pool is larger than the cores it has, where spinning threads
///   steal time from the ones doing work
/// * `session_config` - further ONNX Runtime session settings, as the string key/value
///   pairs the runtime documents in `onnxruntime_session_options_config_keys.h` and
///   applied to every session verbatim — `"session.intra_op.spin_duration_us": "500"`
///   bounds the spin to half a millisecond, `"session.force_spinning_stop": "1"` stops it
///   the moment a session's last `Run()` returns. Absent means none. The runtime accepts a
///   key it does not know without complaint, so the banner prints exactly what was sent
/// * `sessions_per_model` - backend replicas and therefore concurrent inferences. Default
///   1.
///
/// An ONNX Runtime replica is an independent session with its own thread pool. A Neuron
/// replica is a separate model load on consecutive NeuronCores. Either costs one model
/// copy per replica.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ThreadingConfig {
    #[serde(default = "default_connection_concurrency")]
    pub connection_concurrency: u16,
    #[serde(default)]
    pub worker_threads: Option<usize>,
    #[serde(default = "default_intra_op_threads")]
    pub intra_op_threads: i32,
    #[serde(default)]
    pub inter_op_threads: Option<i32>,
    #[serde(default)]
    pub allow_spinning: Option<bool>,
    #[serde(default)]
    pub session_config: BTreeMap<String, String>,
    #[serde(default = "default_sessions_per_model")]
    pub sessions_per_model: usize,
}

impl Default for ThreadingConfig {
    /// Absent `threading` is one async thread per core, one session, no intra-op
    /// parallelism, no extra session settings and 500 admitted per connection.
    fn default() -> Self {
        Self {
            connection_concurrency: default_connection_concurrency(),
            worker_threads: None,
            intra_op_threads: default_intra_op_threads(),
            inter_op_threads: None,
            allow_spinning: None,
            session_config: BTreeMap::new(),
            sessions_per_model: default_sessions_per_model(),
        }
    }
}

impl ThreadingConfig {
    /// Async workers to build the serving runtime with.
    pub fn worker_threads(&self, cores: usize) -> usize {
        self.worker_threads.unwrap_or(cores).max(1)
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
    /// Asking for both multiplies the pools per session, so the thread budget the banner
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

    /// Refuses a session setting ONNX Runtime could not take, or one set in two places.
    ///
    /// The runtime accepts any key it does not recognise, so spelling cannot be checked
    /// here; what can be is the shape of an entry. A key with whitespace or over 128 bytes
    /// and a value over 1024 bytes are the runtime's own refusals, made here at start-up
    /// with the field named rather than at model load with a bare status code. The two
    /// spinning keys are refused when `allow_spinning` is also set, since one setting
    /// written twice would leave the banner unable to say which won.
    fn validate_session_config(&self) -> Result<(), InferenceError> {
        const SPINNING_KEYS: [&str; 2] = [
            "session.intra_op.allow_spinning",
            "session.inter_op.allow_spinning",
        ];
        for (key, value) in &self.session_config {
            if key.is_empty() || key.chars().any(char::is_whitespace) {
                return Err(format!(
                    "\"threading.session_config\" has the key {key:?}, which is not an ONNX \
                     Runtime setting name; keys look like \"session.intra_op.spin_duration_us\""
                )
                .into());
            }
            if key.len() > 128 || value.len() > 1024 {
                return Err(format!(
                    "\"threading.session_config\" entry {key:?} is too long for ONNX Runtime, \
                     which takes keys up to 128 bytes and values up to 1024"
                )
                .into());
            }
            if self.allow_spinning.is_some() && SPINNING_KEYS.contains(&key.as_str()) {
                return Err(format!(
                    "\"threading.session_config\" sets {key:?} while \"threading.allow_spinning\" \
                     is also set; spinning is one setting, so choose one place to set it"
                )
                .into());
            }
        }
        Ok(())
    }

    /// Sessions to load each model into, never zero.
    ///
    /// Per model, so the total is `models × this`. This is also the width of tokio's
    /// blocking pool, so it is the number of inferences that can run at once: one thread
    /// per session.
    pub fn sessions_per_model(&self) -> usize {
        self.sessions_per_model.max(1)
    }

    /// Compute threads this configuration can have runnable at once.
    ///
    /// One calling thread per session, plus the `width - 1` extra threads in that session's
    /// pool, where the width is whichever of intra-op or inter-op was chosen. So
    /// `sessions × width`.
    ///
    /// Counts threads that *exist* rather than threads running. With a candidate model
    /// resident the sessions double but the blocking pool does not, because a thread serves
    /// one model at a time: half the pools are idle at any instant — free when
    /// `allow_spinning` is false, and not when it is true.
    ///
    /// # Arguments
    ///
    /// * `sessions` - the **total** sessions across models, since the pool belongs to the
    ///   session and not to the process: a candidate model resident alongside the control
    ///   brings its own pools, and `sessions_per_model` multiplies both
    pub fn compute_threads(&self, sessions: usize) -> usize {
        let per_pool = usize::try_from(self.parallelism().width())
            .unwrap_or(1)
            .max(1);
        per_pool.saturating_mul(sessions.max(1))
    }
}
/// Which of ONNX Runtime's two thread pools carries a session's width.
///
/// The pools are alternatives rather than a pair to tune together: the configuration
/// permits only one of the counts to exceed 1, so this records which was chosen. That
/// keeps the thread budget a single number — see
/// [`ThreadingConfig::compute_threads`] — instead of a product of two.
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
    /// `threading.worker_threads` or `threading.sessions_per_model`; a negative
    /// `threading.intra_op_threads`; a `threading.session_config` entry ONNX Runtime
    /// could not take, or one that sets spinning alongside `threading.allow_spinning`; an
    /// empty `candidate_model.config_path` or a `traffic_percent` above 100; an
    /// `inference_log.sample_rate` outside `0.0..=1.0`; a zero `batch_size` in either
    /// group; a `max_sends_in_flight` that is zero or above `u32::MAX` in either group;
    /// a zero `inference_log.prefix_shards`; or an empty `metrics.cloudwatch_namespace`.
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
        if self.threading.worker_threads == Some(0) {
            return Err(
                "\"threading.worker_threads\" is 0, so the pool would have no thread \
                        to run on; leave the field out to get one per core"
                    .into(),
            );
        }
        // Refused rather than clamped, now that it sizes the blocking pool as well as the
        // session count: 0 would ask for a server that can run no inference at all.
        if self.threading.sessions_per_model == 0 {
            return Err(format!(
                "\"threading.sessions_per_model\" is 0, so the model would be loaded into no \
                 session and no inference could run; leave the field out to get {}",
                default_sessions_per_model()
            )
            .into());
        }
        self.threading.validate_parallelism()?;
        self.threading.validate_session_config()?;
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
        if self.inference_log.prefix_shards == 0 {
            return Err(format!(
                "\"inference_log.prefix_shards\" is 0, so there would be no prefix to \
                 write beneath; 1 writes every object beneath one, and the default is {}",
                default_prefix_shards()
            )
            .into());
        }

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

    /// Asserts the thread budget is the sessions times the pool width, and that a
    /// candidate model resident alongside the control doubles it.
    #[test]
    fn the_thread_budget_is_the_sessions_times_the_pool_width() {
        let threading = ThreadingConfig {
            sessions_per_model: 4,
            intra_op_threads: 22,
            ..Default::default()
        };
        assert_eq!(
            threading.sessions_per_model(),
            4,
            "sessions per model are configured, not derived"
        );
        assert_eq!(
            threading.compute_threads(4),
            88,
            "four 22-thread pools for one model"
        );
        assert_eq!(
            threading.compute_threads(8),
            176,
            "a candidate brings four pools of its own"
        );
    }

    /// Asserts the width comes from whichever pool was chosen, not always intra-op.
    #[test]
    fn the_thread_budget_follows_the_chosen_pool() {
        let inter = ThreadingConfig {
            sessions_per_model: 2,
            intra_op_threads: 1,
            inter_op_threads: Some(14),
            ..Default::default()
        };
        assert_eq!(
            inter.compute_threads(2),
            28,
            "two inter-op pools of 14, and intra-op contributes nothing"
        );
    }

    /// Asserts a pool width of one costs exactly one thread per session.
    #[test]
    fn a_pool_of_one_is_one_thread_per_session() {
        let threading = ThreadingConfig {
            sessions_per_model: 8,
            ..Default::default()
        };
        assert_eq!(
            threading.compute_threads(8),
            8,
            "intra_op_threads defaults to 1, so each session is its calling thread alone"
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
                    "uri": "s3://my-bucket/inference-logs",
                    "batch_size": 400,
                    "sample_rate": 0.01,
                    "max_sends_in_flight": 8,
                    "prefix_shards": 64
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
        assert_eq!(config.inference_log.uri, "s3://my-bucket/inference-logs");
        assert_eq!(config.inference_log.batch_size, 400);
        assert_eq!(config.inference_log.sample_rate, 0.01);
        assert_eq!(config.inference_log.max_sends_in_flight, 8);
        assert_eq!(config.inference_log.prefix_shards, 64);
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
        assert_eq!(
            config.inference_log.prefix_shards,
            default_prefix_shards(),
            "logs are spread over prefixes without being asked to be"
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

    /// Asserts a zero session count is refused rather than clamped, since it now sizes the
    /// blocking pool as well as the sessions.
    #[test]
    fn a_session_count_of_zero_is_refused() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "threading": {"sessions_per_model": 0},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("zero sessions");
        let message = err.to_string();
        assert!(
            message.contains("threading.sessions_per_model"),
            "the error should name the field: {message}"
        );
    }

    /// Asserts the removed placement fields are now rejected outright rather than
    /// silently ignored, so a configuration written for the old shape fails loudly.
    #[test]
    fn the_removed_threading_fields_are_refused() {
        for field in ["worker_cores", "compute_cores", "inference_concurrency"] {
            let json = format!(
                r#"{{"model_config_path": "/tmp/m.json",
                      "threading": {{"{field}": "0-1"}},
                      "inference_log": {{"uri": "/tmp/logs"}}}}"#
            );
            let err = HusharServiceConfig::from_json(&json).expect_err("removed field");
            assert!(
                err.to_string().contains(field),
                "the error should name {field}: {err}"
            );
        }
    }

    /// Asserts `allow_spinning` is absent by default, leaving the runtime's own choice.
    #[test]
    fn absent_spinning_leaves_the_runtime_default() {
        let config = HusharServiceConfig::from_json(MINIMAL).expect("valid");
        assert_eq!(config.threading.allow_spinning, None);
    }

    /// Asserts a second resident session adds a second intra-op pool to the thread count.
    #[test]
    fn a_resident_candidate_model_doubles_the_intra_op_pool() {
        let config = ThreadingConfig {
            worker_threads: Some(24),
            intra_op_threads: 21,
            ..ThreadingConfig::default()
        };

        assert_eq!(
            config.compute_threads(1),
            21,
            "one session: its calling thread plus 20 pool threads"
        );
        assert_eq!(
            config.compute_threads(2),
            42,
            "two sessions: two pools of 21, not one"
        );
        assert!(
            config.compute_threads(2) * 2 <= 192,
            "42 of 192 cores is the undersubscription the banner has to report"
        );
    }

    /// Asserts `intra_op_threads` of 0 or 1 costs one thread per session and does not
    /// collapse to zero.
    #[test]
    fn a_pool_of_one_or_fewer_is_one_thread_per_session() {
        for intra_op in [0, 1] {
            let config = ThreadingConfig {
                intra_op_threads: intra_op,
                ..ThreadingConfig::default()
            };
            assert_eq!(
                config.compute_threads(2),
                2,
                "intra_op_threads {intra_op} is the calling thread alone, once per session"
            );
        }
    }

    /// Asserts the width is read from whichever pool was chosen, so an inter-op session
    /// budgets its threads exactly as an intra-op one of the same width does.
    #[test]
    fn the_chosen_pool_carries_the_width() {
        let intra = ThreadingConfig {
            intra_op_threads: 14,
            ..ThreadingConfig::default()
        };
        let inter = ThreadingConfig {
            intra_op_threads: 1,
            inter_op_threads: Some(14),
            ..ThreadingConfig::default()
        };

        assert_eq!(intra.parallelism(), Parallelism::Intra(14));
        assert_eq!(inter.parallelism(), Parallelism::Inter(14));
        assert!(!intra.parallelism().is_parallel());
        assert!(inter.parallelism().is_parallel());
        assert_eq!(
            inter.compute_threads(4),
            intra.compute_threads(4),
            "the same width costs the same threads whichever pool holds it"
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

    /// Asserts extra session settings are read as an ordered string map and default to none.
    #[test]
    fn session_config_is_a_string_map_that_defaults_to_empty() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "threading": {"session_config": {
                           "session.intra_op.spin_duration_us": "500",
                           "session.force_spinning_stop": "1"}},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let config = HusharServiceConfig::from_json(json).expect("valid");
        let entries: Vec<(&str, &str)> = config
            .threading
            .session_config
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(
            entries,
            vec![
                ("session.force_spinning_stop", "1"),
                ("session.intra_op.spin_duration_us", "500"),
            ],
            "entries are kept as strings, in key order"
        );
        assert!(
            HusharServiceConfig::from_json(MINIMAL)
                .expect("valid")
                .threading
                .session_config
                .is_empty(),
            "absent means no extra settings"
        );
    }

    /// Asserts an entry ONNX Runtime would refuse is refused here first, naming the field.
    #[test]
    fn a_malformed_session_config_entry_is_refused_by_name() {
        let long_value = "x".repeat(1025);
        for (entry, expected) in [
            (r#""": "1""#.to_owned(), "not an ONNX Runtime setting name"),
            (
                r#""session. intra_op": "1""#.to_owned(),
                "not an ONNX Runtime setting name",
            ),
            (format!(r#""session.k": "{long_value}""#), "too long"),
        ] {
            let json = format!(
                r#"{{"model_config_path": "/tmp/m.json",
                     "threading": {{"session_config": {{{entry}}}}},
                     "inference_log": {{"uri": "/tmp/logs"}}}}"#
            );
            let err = HusharServiceConfig::from_json(&json).expect_err("malformed entry");
            let message = err.to_string();
            assert!(
                message.contains("threading.session_config") && message.contains(expected),
                "the error should name the field and say {expected}: {message}"
            );
        }
    }

    /// Asserts spinning cannot be set both as `allow_spinning` and as a raw session entry.
    #[test]
    fn spinning_set_in_two_places_is_refused() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "threading": {"allow_spinning": false,
                                     "session_config": {"session.intra_op.allow_spinning": "1"}},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("set twice");
        let message = err.to_string();
        assert!(
            message.contains("allow_spinning") && message.contains("one place"),
            "the error should point at both settings: {message}"
        );

        // Alone, the raw entry is a legitimate way to spell the same thing.
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "threading": {"session_config": {"session.intra_op.allow_spinning": "1"}},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        HusharServiceConfig::from_json(json).expect("one place is fine");
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

    /// Asserts a zero shard count is refused, since it names no prefix to write beneath.
    #[test]
    fn a_prefix_shard_count_of_zero_is_refused() {
        let json = r#"{"model_config_path": "/tmp/m.json",
                       "inference_log": {"uri": "s3://b/logs", "prefix_shards": 0}}"#;
        let err = HusharServiceConfig::from_json(json).expect_err("zero names no prefix");
        assert!(
            err.to_string().contains("prefix_shards"),
            "the error should name the field: {err}"
        );
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
        assert_eq!(config.threading.sessions_per_model, 8);
        assert_eq!(config.threading.intra_op_threads, 1);
        assert_eq!(
            config.threading.compute_threads(8),
            8,
            "8 sessions x 1 intra-op thread: with a pool of one, a session is its calling \
             thread and nothing more"
        );
        assert_eq!(config.port_number, 8279);
        assert_eq!(
            config.model_config_path, "test-data/model_config_vectorized.json",
            "it should point at the model configuration"
        );
        assert_eq!(config.inference_log.uri, "/tmp/hushar-inference-logs");
        assert_eq!(config.inference_log.sample_rate, 1.0);
        assert_eq!(config.inference_log.prefix_shards, 16);
        assert_eq!(
            config.metrics.cloudwatch_namespace, None,
            "the fixture must start without credentials"
        );
    }
}
