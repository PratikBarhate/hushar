// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Service configuration: what is true of the server rather than of the model.
//!
//! The model, its execution provider and its feature configuration all live in
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
//! default, and the two that matter — how many inferences run at once and how many
//! requests a connection may have in flight — are the group's whole subject. See
//! [`ThreadingConfig`].
//!
//! # Why one file rather than a file and a command line
//!
//! These settings were split: the listener and the model pointer here, and the log
//! destination, the sampling rate and both sinks' batch sizes as command-line flags.
//! The split did not follow anything — `threading.connection_concurrency` is no more a
//! property of the deployment than `inference_log.uri` is — and it meant the destination
//! a deployment writes to could not be versioned alongside the deployment.
//!
//! One exception is forced. `--config-uri` says where *this* file is, so it cannot
//! live in it. `--port` and `--bind-address` remain as overrides for running two
//! instances on one host, which is a developer concern rather than a deployment one.
//!
//! # Nested rather than flat
//!
//! `batch_size` means a different thing to each sink — 500 scored batches of timings
//! against 5000 requests' worth of feature maps — and the same is true of
//! `max_sends_in_flight`, which bounds memory as `batch_size × (n + 1)` in each. A flat
//! `metrics_batch_size`/`data_batch_size` pair only reads as parallel to someone who
//! already knows that; grouping puts each number beside the thing it describes.

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
/// 500, and deliberately generous. This is an **admission** ceiling on one HTTP/2
/// connection, not a bound on work: `threading.inference_concurrency` is what decides how
/// many inferences run at once, and a request admitted past it queues rather than
/// consuming a core. Sizing admission from the core count instead — which this once did —
/// made a single-connection client the ceiling on a large host, because one tonic
/// `Channel` is one connection and the limit applies per connection.
///
/// Low enough to stay a ceiling rather than an invitation to unbounded queueing, and
/// well above what any one caller pipelines.
pub fn default_connection_concurrency() -> u16 {
    500
}

/// Threads ONNX Runtime may use inside one operator, the calling thread included.
///
/// One, because concurrent requests already keep every core busy and a second thread
/// inside an operator then only contends with them. Raise it when latency matters more
/// than requests per second, and lower `threading.inference_concurrency` to match.
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
/// Fields:
/// - `uri` — the destination. `kinesis://stream-name` selects a data stream; anything
///   else is a blob-store prefix, written beneath and partitioned by time. Same URI
///   rules as `model_config_path`.
/// - `batch_size` — batches accumulated before a send. Larger means fewer, bigger
///   writes and more memory held. Kinesis caps one call at 500 records, so a larger
///   number is capped there; a blob store takes an object of any size, which is why the
///   default is well above what a stream can use.
/// - `sample_rate` — fraction of requests whose features are recorded, in `0.0..=1.0`.
///   The cheapest lever over what the log costs, and the only one that also removes
///   work: an unsampled request never copies its feature values anywhere.
/// - `max_sends_in_flight` — sends that may be in the air before the oldest batch is
///   dropped and reported. Holds at most `batch_size × (this + 1)` records, so raising
///   it buys tolerance of a slow destination and costs memory in proportion.
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
/// Fields:
/// - `cloudwatch_namespace` — publish to CloudWatch under this namespace. Absent means
///   summarise to stderr instead, which is what a local run wants and needs no
///   credentials.
/// - `batch_size` — scored batches accumulated before a send, and the number of batches
///   per stderr summary line. CloudWatch takes at most 1000 data points per call and
///   each batch contributes three, so a larger number is capped at 333 there.
/// - `max_sends_in_flight` — as in [`InferenceLogConfig`], but over a much smaller
///   record, so the memory this implies is a fraction of the log's.
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
    /// Absent `metrics` is stderr at the default batch size — the same as a local run
    /// with nothing configured, so a minimal configuration still starts.
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
/// Two of these numbers multiply, which is the whole reason they are one group: the load a
/// configuration puts on the CPU is `inference_concurrency × intra_op_threads`, and
/// choosing either without the other is how a service ends up with more runnable threads
/// than cores. The first is the gate in front of them.
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
/// # The dial this is
///
/// The total work in a request is fixed, so these do not buy throughput — they choose
/// where the parallelism goes. `inference_concurrency = cores, intra_op_threads = 1` runs
/// one inference per core: the most requests per second, and each one as slow as a single
/// core can manage. `inference_concurrency = 1, intra_op_threads = cores` puts the whole
/// machine behind one request: the same rows per second, a fraction of the latency, and
/// almost nothing in flight.
///
/// Admission belongs here rather than beside the port because it is the front of the same
/// queue: `connection_concurrency` decides how deep the line in front of
/// `inference_concurrency` may get on one connection, and the two are read together or
/// neither makes sense.
///
/// Fields:
/// - `connection_concurrency` — in-flight requests allowed **per connection**, both as a
///   tower concurrency limit and as HTTP/2's `MAX_CONCURRENT_STREAMS`. Default 500. It
///   bounds one connection and not the process: ten connections admit ten times this.
/// - `worker_threads` — tokio async workers. Absent means one per core, which is
///   generous: this pool does no inference. Two or three is usually enough, and the
///   cores it does not take are cores the engine can have.
/// - `inference_concurrency` — the ceiling on concurrent inferences, which is the size of
///   tokio's blocking pool, and the queue in front of the engine. Absent means one per
///   core.
/// - `intra_op_threads` — passed to ONNX Runtime. `1` keeps each inference on its
///   calling thread. `0` leaves the runtime's own default, which is the physical core
///   count. Note the pool is **per session and shared**, so `n` concurrent callers do
///   not get `n` pools.
/// - `worker_cores`, `compute_cores` — which logical CPUs each pool may run on, as a
///   `taskset`-style list such as `"0-1"` or `"2-9,12"`. Absent leaves placement to the
///   kernel. Keeping the two lists disjoint is the point: a burst of connection handling
///   then cannot preempt an inference mid-operator. Applied on Linux; accepted and
///   reported as not applied elsewhere, since macOS affinity is advisory.
/// - `allow_spinning` — whether idle ONNX Runtime intra-op threads spin waiting for
///   work. `true` is the runtime's own default and trades CPU for latency. Set it
///   `false` when the pool is larger than the cores it has, where spinning threads
///   steal time from the ones doing work.
/// - `sessions_per_model` — independent ONNX Runtime sessions each model is loaded into.
///   The intra-op pool belongs to the **session**, so with one session every concurrent
///   request to a model queues for the same pool and latency climbs with load on an
///   otherwise idle machine. `n` gives `n` pools per model, each pinned to its own slice
///   of the compute cores, so a request gets a pool to itself. There are `n` slices, not
///   `models × n`: replica `i` of every model shares slice `i`. Costs one copy of the
///   weights per session, so `models × n` copies. Default 1.
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
    /// and 500 admitted per connection — the throughput end of the dial, and what the
    /// service did before these were configurable.
    fn default() -> Self {
        Self {
            connection_concurrency: default_connection_concurrency(),
            worker_threads: None,
            inference_concurrency: None,
            intra_op_threads: default_intra_op_threads(),
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
    /// Already validated, so the parse cannot fail here — [`Self::validate_cores`] runs
    /// from `validate` and reports a bad list with the field name.
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

    /// Sessions to load each model into, never zero.
    ///
    /// Per model, so the total is `models × this`. The **slices** are not: there are
    /// `this` many, and replica `i` of every model shares slice `i`, so a thread that
    /// serves the control and then the candidate stays on the same cores. That asymmetry
    /// is why a candidate adds a pool to each slice without adding a core to it.
    pub fn sessions_per_model(&self) -> usize {
        self.sessions_per_model.max(1)
    }

    /// Intra-op threads that can be runnable on one slice, and that slice's width.
    ///
    /// The comparison the banner makes, and it is per slice because that is where
    /// contention happens — the whole-host thread total hides it. Replica `i` of every
    /// model shares slice `i` and each session owns a pool, so the threads on a slice are
    /// `models × intra_op_threads` against that slice's cores.
    ///
    /// Exceeding the width is not automatically a cost. Measured at the same nominal 2.0x:
    /// 6 requests a slice cost 20 % of p50, while 1 request a slice gave the tightest tail
    /// on the host, because the second pool only sleeps. The banner says which case it is.
    pub fn slice_pressure(&self, cores: usize, models: usize) -> (usize, usize) {
        let pool = usize::try_from(self.intra_op_threads).unwrap_or(1).max(1);
        let replicas = self.sessions_per_model();
        let width = self.compute_cores().len();
        let width = if width == 0 { cores } else { width };
        (pool.saturating_mul(models.max(1)), width / replicas.max(1))
    }

    /// Compute threads this configuration can have runnable at once.
    ///
    /// `inference_concurrency` calling threads, plus the `intra_op_threads - 1` extra
    /// threads in each session's pool. `sessions` is the **total** across models, because
    /// the pool belongs to the session and not to the process: a candidate model resident
    /// alongside the control brings its own pools, and `sessions_per_model` multiplies
    /// both. So this figure moves with a deployment's model count even though the
    /// configuration has not changed — measured at 20 % of p50 when it does.
    ///
    /// Note this counts threads that *exist*. Only `inference_concurrency` inferences run
    /// at once, so with several sessions per model most pools are idle at any instant —
    /// which is free when `allow_spinning` is false and very much not when it is true.
    ///
    /// Compared against the core count in the banner, because either side of that
    /// comparison being a surprise is expensive. Too many threads oversubscribes; too few
    /// leaves an expensive machine idle, and this figure is what says which.
    pub fn compute_threads(&self, cores: usize, sessions: usize) -> usize {
        let per_pool = usize::try_from(self.intra_op_threads)
            .unwrap_or(0)
            .saturating_sub(1);
        self.inference_concurrency(cores)
            .saturating_add(per_pool.saturating_mul(sessions.max(1)))
    }
}

/// A second model to serve alongside the first, and how much traffic it takes.
///
/// This is a roll-out: `model_config_path` is the control, this is the candidate, and
/// `traffic_percent` is the share the candidate serves. Stating only the candidate's
/// share is how a roll-out is actually described — "ten percent on the new model" — and
/// it makes the two numbers impossible to contradict.
///
/// Fields:
/// - `config_path` — the candidate's own model configuration, with its own URI scheme.
///   It carries its own `vectorization_config`, so the candidate may take its features
///   differently from the control: a request is named values, not a tensor, so one
///   request can feed a vectorized model and a named one.
/// - `traffic_percent` — whole percent, `0..=100`, that the candidate serves. `0` loads
///   it and sends it nothing, which is how you check it loads before it serves anyone.
///   `100` sends everything to it, which is the end of a roll-out.
///
/// A caller naming a model in its request overrides this entirely — see
/// `InferenceRequest.model_id`.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CandidateModelConfig {
    pub config_path: String,
    pub traffic_percent: u8,
}

/// The server process's own settings.
///
/// Fields:
/// - `bind_address` — address to listen on. `0.0.0.0` accepts from anywhere, which is
///   what a container wants; `127.0.0.1` restricts to the host. `--bind-address`
///   overrides it.
/// - `port_number` — the port to listen on, which `--port` overrides.
/// - `model_config_path` — where the model's own configuration is. Carries its own URI
///   scheme, so a model configuration in S3 beside a service configuration on local
///   disk is expressible. This is the control arm when a candidate is configured.
/// - `threading` — [`ThreadingConfig`]. Optional; absent is one thread per core on both
///   pools, no intra-op parallelism, and 500 requests admitted per connection.
/// - `candidate_model` — [`CandidateModelConfig`]. Optional; absent serves one model.
/// - `inference_log` — [`InferenceLogConfig`]. Required, because a service that records
///   nothing is a decision rather than a default.
/// - `metrics` — [`MetricsConfig`]. Optional; absent means stderr.
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
    /// Parses and then checks. Both failures are startup failures.
    ///
    /// `serde` establishes that the fields are present and of the right type;
    /// [`Self::validate`] establishes that the numbers are ones the process can honour.
    /// Keeping the second step separate means a configuration built in a test can be
    /// checked the same way as one read from disk.
    pub fn from_json(json_str: &str) -> Result<Self, InferenceError> {
        let config: HusharServiceConfig = serde_json::from_str(json_str)?;
        config.validate()?;
        Ok(config)
    }

    /// Refuses values a sink or sampler could not honour, naming the field and the range.
    ///
    /// Refused rather than clamped. `"sample_rate": 10` is someone who meant 10%, and
    /// reading it as "log everything" is the reading least likely to be what they
    /// wanted; `"max_sends_in_flight": 0` would leave no slot for a send to start in.
    /// The startup banner prints these numbers, so a silently corrected one would
    /// misreport the memory bound in force — which is the kind of thing discovered while
    /// wondering why a limit is not what was configured.
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

/// A batch of zero would send an empty payload on every request.
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
fn validate_sends_in_flight(field: &str, sends: usize) -> Result<(), InferenceError> {
    if sends == 0 {
        return Err(format!(
            "{field:?} is 0, which would leave no slot for a send to start in; 1 sends \
             one at a time"
        )
        .into());
    }
    // Draining at shutdown acquires every permit at once, which the semaphore counts in
    // a `u32`, so a wider bound could not be waited on.
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

    /// The smallest configuration that starts, for tests about something else.
    const MINIMAL: &str = r#"{
        "model_config_path": "/tmp/model.json",
        "inference_log": { "uri": "/tmp/logs" }
    }"#;

    /// Slots are absolute: the same configuration keeps its geometry when a candidate
    /// arrives or leaves, and only the session count moves. This is the property that
    /// lets one deployment configuration serve a roll-out and a single model.
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
        // Threads on a slice scale with the models resident; the slice itself does not.
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

    /// The oversubscription the banner warns about is a per-slice property, so it must be
    /// computed from the slice width and not from the host's core count.
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

    /// An absent `compute_cores` means the whole host, so the width still has to divide.
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

    /// Everything but the model and the log destination has a default, so the shortest
    /// useful configuration is two fields.
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

    /// The default has to need no credentials, or a local run cannot start.
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

    /// Recording nothing is a decision, so it has to be stated rather than defaulted
    /// into. A missing destination is the one field a deployment cannot forget.
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

    /// These were command-line flags. Rejecting the old names means an operator
    /// upgrading gets an error naming the field rather than a service that starts and
    /// silently logs everywhere it used to be told not to.
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

    /// These moved into the model configuration.
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

    /// Typed, so this cannot reach the runtime and fail there.
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

    /// `"sample_rate": 10` is someone who meant 10%, and reading it as "log everything"
    /// is the reading least likely to be what they wanted. Refusing it with the range in
    /// the message is the only outcome that tells them.
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

    /// Admission is a fixed number rather than one derived from the host: sizing it from
    /// the cores made a single-connection client the ceiling on a large instance, since
    /// the limit is per connection and one tonic `Channel` is one connection.
    #[test]
    fn the_concurrency_limit_defaults_to_five_hundred_per_connection() {
        let config = HusharServiceConfig::from_json(MINIMAL).expect("valid");
        assert_eq!(config.threading.connection_concurrency, 500);
        assert_eq!(default_connection_concurrency(), 500);
    }

    /// It moved into `threading`, so the old spelling has to fail loudly rather than be
    /// accepted and ignored — a silently dropped admission limit is invisible until a
    /// benchmark disagrees with its own report.
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

    /// Configured wins, and is not second-guessed against the core count: a benchmark
    /// pinning every host to the same number is the reason the field exists.
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

    /// Zero would let connections in and serve none of them -- a hang rather than an
    /// error, which is the worst way for a configuration mistake to present.
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

    /// Both ends of a roll-out are legitimate configurations: 0 loads the candidate
    /// without serving it, 100 has finished moving over.
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

    /// A share above 100 is not a share. Refused by name and by range, like the rest.
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

    /// Above `u8::MAX` serde refuses on the type, which is the earlier and clearer
    /// failure -- worth pinning so the field's type is not widened without thought.
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
        // And the field is required within the block, so omitting it is also refused.
        let json = r#"{"model_config_path": "/tmp/c.json",
                       "candidate_model": {"traffic_percent": 10},
                       "inference_log": {"uri": "/tmp/logs"}}"#;
        assert!(HusharServiceConfig::from_json(json).is_err());
    }

    /// The share is required rather than defaulted. A candidate loaded with an implied
    /// share would be a guess about how much production traffic to move, which is not
    /// a guess this service gets to make.
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

    /// Parsed at startup rather than at first pin, so a list copied from a larger
    /// instance fails while someone is watching.
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

    /// Absent means "leave placement to the kernel", which has to be distinguishable
    /// from an empty list rather than collapsing into it.
    #[test]
    fn absent_core_lists_pin_nothing() {
        let config = HusharServiceConfig::from_json(MINIMAL).expect("valid");
        assert!(config.threading.worker_cores().is_empty());
        assert!(config.threading.compute_cores().is_empty());
        assert_eq!(config.threading.allow_spinning, None);
    }

    /// The intra-op pool belongs to the session, so a resident candidate model brings a
    /// second one. Getting this wrong under-reports the thread budget by almost half on an
    /// A/B roll-out, and the banner is where an operator learns how much of the host the
    /// configuration can actually use.
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
        // The point of the count: on a 192-core host this configuration can keep a quarter
        // of it busy, so no request rate will ever saturate the machine.
        assert!(
            config.compute_threads(192, 2) * 2 <= 192,
            "48 of 192 cores is the undersubscription the banner has to report"
        );
    }

    /// `intra_op_threads` of 0 means "the runtime's own default", and 1 means "no pool at
    /// all". Neither may underflow the subtraction into a huge pool count.
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

    /// Refused rather than clamped, because the banner prints this number and a
    /// silently corrected one would misreport the memory bound. Checked in both groups,
    /// since each has its own bound now.
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

    /// An empty namespace would be sent to CloudWatch and rejected there, on the first
    /// publish, long after startup.
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

    /// A complete configuration must parse, read from a file.
    ///
    /// This is not only about the field names: `port_number` is a `u16`, and a
    /// configuration once carried 827942, which cannot be one. The fixture belongs to
    /// this test rather than being borrowed from `benchmark-data/generated`, whose files
    /// are generated and change shape when the benchmark does.
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
        // Present for the opposite reason to `connection_concurrency` above: that one is
        // left out to show the defaulted shape, whereas these are spelled out so the file
        // proves the `threading` section parses and matches the worked example in
        // documentation/configuration.md. A fleet is free to omit the two counts and take
        // one per core; `intra_op_threads` is the one that is a real decision, since its
        // default of 1 gives a single request no parallelism at all.
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
