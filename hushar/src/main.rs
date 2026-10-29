// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

use clap::Parser;
use hushar::hushar_proto::{
    InferenceLogBatch, InferenceRequest, InferenceResponse,
    hushar_server::{Hushar, HusharServer},
};
use inference::InferenceBackend;
use inference::input_builder::RequestColumns;
use inference::scoring::{FeatureLogging, to_score_type};
use io::{BlobStores, DataSink, Location, MetricsSink, Sampler};
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::{Request, Response, Status, transport::Server};

pub(crate) mod cli;
pub(crate) mod config;
pub(crate) mod inference;
pub(crate) mod io;

/// One servable model: everything a request needs once an arm has been chosen.
///
/// The three parts must never be crossed between models: one model's inputs fed through
/// another's builder is a shape error at best.
///
/// # Fields
///
/// * `model_id` - from the model's own configuration; recorded on every inference log
///   batch and every metric, and returned in the response
/// * `sessions` - interchangeable backend replicas. For ONNX Runtime each owns its own
///   thread pool; for Neuron each is a separate model load on its configured core range.
///   A request leases one for as long as it scores
/// * `input_builder` - how a feature payload becomes this model's inputs
/// * `mini_batch` - how a request's rows are cut, when the configuration says to; the
///   batches are scored at the same time. `None` runs each request as a single batch
/// * `backend` - backend-specific settings used to describe the loaded execution plan
#[derive(Debug)]
pub(crate) struct ServableModel {
    model_id: String,
    sessions: inference::SessionPool<Arc<dyn InferenceBackend>>,
    input_builder: Arc<inference::input_builder::InputBuilder>,
    mini_batch: Option<inference::scoring::MiniBatch>,
    backend: config::BackendConfig,
}

/// Everything a request handler needs, built once at startup.
///
/// # Fields
///
/// * `control` - the model from `model_config_path`, and the only one when no candidate
///   is configured
/// * `candidate` - the model from `candidate_model.config_path`, when there is one
/// * `split` - which arm an unnamed request goes to
/// * `metrics_sink`, `data_sink` - where timings and inference logs accumulate; a
///   handler's whole interaction with either is one non-blocking call
/// * `sampler` - what fraction of requests is logged, decided before scoring so an
///   unsampled request never copies its features
#[derive(Debug)]
pub(crate) struct HusharService {
    control: Arc<ServableModel>,
    candidate: Option<Arc<ServableModel>>,
    split: inference::traffic_split::TrafficSplit,
    metrics_sink: Arc<dyn MetricsSink>,
    data_sink: Arc<dyn DataSink>,
    sampler: Sampler,
}

impl HusharService {
    /// The model that serves this request.
    ///
    /// A named model wins over the configured split; an empty name takes the arm the
    /// split chooses. The split only returns the candidate arm when a candidate is
    /// loaded.
    ///
    /// # Errors
    ///
    /// `invalid_argument` when `requested` names a model this server has not loaded. It
    /// is refused rather than served by the other arm, which would attribute the request
    /// to the wrong model; the message names every loaded model.
    fn choose(&self, requested: &str) -> Result<&Arc<ServableModel>, Status> {
        use inference::traffic_split::Arm;

        if requested.is_empty() {
            return Ok(match self.split.choose() {
                Arm::Control => &self.control,
                Arm::Candidate => self.candidate.as_ref().unwrap_or(&self.control),
            });
        }
        if requested == self.control.model_id {
            return Ok(&self.control);
        }
        if let Some(candidate) = &self.candidate
            && requested == candidate.model_id
        {
            return Ok(candidate);
        }
        let loaded = match &self.candidate {
            Some(candidate) => format!("{:?} and {:?}", self.control.model_id, candidate.model_id),
            None => format!("{:?}", self.control.model_id),
        };
        Err(Status::invalid_argument(format!(
            "no model {requested:?} is loaded; this server serves {loaded}"
        )))
    }
}

#[tonic::async_trait]
impl Hushar for HusharService {
    /// Scores one batch of feature rows.
    ///
    /// Inference is CPU-bound and can run for milliseconds, so it runs on the blocking
    /// pool rather than on the async worker that accepted the request. There it leases one
    /// of the model's sessions for exactly as long as it scores: the blocking pool is as
    /// wide as there are sessions, so the lease never waits, and a session is never shared
    /// between two requests however the pool's threads come and go.
    ///
    /// Neither sink call waits on anything: each appends to a buffer and returns, and the
    /// request that fills a buffer hands the whole batch to a spawned task, so no request
    /// waits on S3, CloudWatch or the local filesystem. See
    /// [`io::accumulator`](crate::io::accumulator). The logged scores are the same
    /// values, in the same precision, that the response carries.
    ///
    /// Sampling is decided before scoring, so an unsampled request drops its features
    /// instead of retaining them for the log.
    ///
    /// # Errors
    ///
    /// `invalid_argument` when the request names a model this server has not loaded, or
    /// when its feature map does not decode into columns of the stated row count.
    /// `internal` when scoring fails or the blocking task panics.
    async fn inference_service(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>, Status> {
        let inference_req = request.into_inner();
        let request_id = inference_req.request_id;

        let model = Arc::clone(self.choose(&inference_req.model_id)?);
        let model_id = model.model_id.clone();

        let columns = RequestColumns::from_request(inference_req.row_ids, inference_req.features)
            .map_err(|e| Status::invalid_argument(e.to_string()))?;

        let input_builder = Arc::clone(&model.input_builder);
        let mini_batch = model.mini_batch;
        let logging = if self.sampler.sample() {
            FeatureLogging::Record
        } else {
            FeatureLogging::Skip
        };

        let scored = tokio::task::spawn_blocking(move || {
            let session = model.sessions.acquire();
            let scored = inference::scoring::score_features(
                session.as_ref(),
                &columns,
                &input_builder,
                mini_batch,
                logging,
            );
            (scored, columns)
        })
        .await
        .map_err(|e| Status::internal(format!("inference task failed to complete: {e}")))?;
        let (scored, columns) = scored;

        match scored {
            Ok(batch) => {
                self.metrics_sink.record(&model_id, &batch.timings);

                let scores = to_score_type(batch.scores);
                let logged_scores = (logging == FeatureLogging::Record).then(|| scores.clone());

                let inference_response = InferenceResponse {
                    request_id: request_id.clone(),
                    row_ids: columns.row_ids().to_vec(),
                    scores: Some(scores),
                    model_id: model_id.clone(),
                };
                if let Some(scores) = logged_scores {
                    let (row_ids, features) = columns.into_wire();
                    self.data_sink.record(InferenceLogBatch {
                        request_id,
                        model_id,
                        row_ids,
                        features,
                        scores: Some(scores),
                    });
                }

                Ok(Response::new(inference_response))
            }
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }
}

/// Describes a model's inputs or outputs for the banner.
///
/// Up to 8 entries are listed in full. Beyond that the result is the total count followed
/// by one entry per distinct type and shape, prefixed by `{n} x` when more than one entry
/// shares it.
fn describe_specs(specs: &[crate::inference::batch::IoSpec]) -> String {
    let described: Vec<String> = specs
        .iter()
        .map(crate::inference::batch::IoSpec::describe)
        .collect();
    if described.len() <= 8 {
        return described.join(", ");
    }
    let mut counts: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    for entry in &described {
        let shape = entry
            .split_once(' ')
            .map_or(entry.as_str(), |(_, rest)| rest);
        *counts.entry(shape).or_default() += 1;
    }
    let grouped: Vec<String> = counts
        .iter()
        .map(|(shape, n)| {
            if *n == 1 {
                (*shape).to_owned()
            } else {
                format!("{n} x {shape}")
            }
        })
        .collect();
    format!("{} total -- {}", described.len(), grouped.join(", "))
}

/// ONNX Runtime's string-keyed session settings, from the threading configuration.
///
/// An empty list leaves every ONNX Runtime default in place. The spinning keys take `"1"`
/// or `"0"` rather than a boolean, and both the intra-op and inter-op pools are set
/// together. Whatever `session_config` names follows, verbatim and in key order; the
/// configuration has already refused an entry that would set spinning a second time.
fn session_options(threading: &config::ThreadingConfig) -> Vec<(String, String)> {
    let mut options = Vec::new();
    if let Some(spinning) = threading.allow_spinning {
        let value = if spinning { "1" } else { "0" };
        options.push((
            "session.intra_op.allow_spinning".to_owned(),
            value.to_owned(),
        ));
        options.push((
            "session.inter_op.allow_spinning".to_owned(),
            value.to_owned(),
        ));
    }
    options.extend(
        threading
            .session_config
            .iter()
            .map(|(key, value)| (key.clone(), value.clone())),
    );
    options
}

/// ONNX Runtime's string-keyed session settings, one set per session.
type ReplicaOptions = Vec<Vec<(String, String)>>;

/// One set of session options per session a model is loaded into.
///
/// Every set is identical: sessions differ in that each owns its own intra-op pool, not in
/// how they are configured. The list still has one entry per session because
/// [`io::load_onnx_sessions`] builds one session per entry, so its length is how the
/// session count reaches the loader.
fn replica_options(threading: &config::ThreadingConfig) -> ReplicaOptions {
    vec![session_options(threading); threading.sessions_per_model()]
}

/// Reads one model's configuration, loads it, and checks the two agree.
///
/// The vectorization configuration is checked against the model's own signature — a
/// feature bound to an input the model does not have, a type that disagrees, a
/// transformation wider than the input it feeds — so a deployment mistake stops the
/// server starting rather than failing every request. Prints the model's signature and
/// feature bindings to stdout, and its mini-batch geometry when the model splits
/// requests.
///
/// # Arguments
///
/// * `stores` - blob stores used to resolve the configuration and model URIs
/// * `config_path` - URI of the model configuration
/// * `role` - names this arm in failures and in the banner, for example "model" or
///   "candidate model"
/// * `threading` - common replica count plus ONNX Runtime's pool settings
/// * `neuron_runtime` - the process-wide runtime, initialized on the first Neuron model
///   and reused by every later one
///
/// # Panics
///
/// Panics if the configuration cannot be reached, read or parsed; if the model cannot be
/// reached, read or loaded on the configured execution provider; or if the vectorization
/// configuration does not fit the model.
async fn load_model(
    stores: &BlobStores,
    config_path: &str,
    role: &str,
    threading: &config::ThreadingConfig,
    neuron_runtime: &mut Option<Arc<libnrt_rs::Runtime>>,
) -> ServableModel {
    let (config_location, config_store) = stores
        .resolve(config_path)
        .await
        .unwrap_or_else(|e| panic!("cannot reach the {role} configuration {config_path}: {e}"));
    let model_config = config::ModelConfig::from_json(
        &config_store
            .get_string(&config_location)
            .await
            .unwrap_or_else(|e| panic!("cannot read the {role} configuration {config_path}: {e}")),
    )
    .unwrap_or_else(|e| panic!("{config_path} is not a valid {role} configuration: {e}"));

    let model_id = model_config.model_id.clone();
    let mini_batch = model_config
        .mini_batch_size
        .map(|size| inference::scoring::MiniBatch {
            size,
            is_fixed: model_config.is_fixed,
        });

    let (model_location, model_store) = stores
        .resolve(&model_config.model_path)
        .await
        .unwrap_or_else(|e| panic!("cannot reach {role} {}: {e}", model_config.model_path));
    let model_bytes = model_store
        .get(&model_location)
        .await
        .unwrap_or_else(|e| panic!("cannot read {role} {}: {e}", model_config.model_path));
    let sessions = match &model_config.backend {
        config::BackendConfig::OnnxRuntime(backend) => io::load_onnx_sessions(
            &model_bytes,
            &backend.execution_provider,
            threading.parallelism(),
            &replica_options(threading),
            mini_batch,
        ),
        config::BackendConfig::Neuron(backend) => io::load_neuron_replicas(
            &model_bytes,
            backend,
            threading.sessions_per_model(),
            mini_batch,
            neuron_runtime,
        ),
    }
    .unwrap_or_else(|e| {
        panic!(
            "could not load {role} {} with {}: {e}",
            model_config.model_path,
            model_config.backend.name()
        )
    });
    let backend = Arc::clone(&sessions[0]);

    let input_builder = inference::input_builder::InputBuilder::resolve(
        model_config.vectorization_config,
        backend.inputs(),
    )
    .unwrap_or_else(|e| panic!("the {role} configuration does not fit the {role}: {e}"));

    let bindings = input_builder.describe();
    println!(
        "hushar: {role} {} loaded on {}\n  inputs   : {}\n  outputs  : {}\n  \
         features : {} binding(s)\n{}",
        model_id,
        backend.name(),
        describe_specs(backend.inputs()),
        describe_specs(backend.outputs()),
        bindings.lines().count(),
        bindings
            .lines()
            .map(|line| format!("             {}", line.trim_start()))
            .collect::<Vec<_>>()
            .join("\n"),
    );
    if let Some(m) = mini_batch {
        let tail = if m.is_fixed {
            "the last is padded up to it"
        } else {
            "the last is however many remain"
        };
        println!(
            "  batch  : rows split into mini batches of {}, scored together, {tail}",
            m.size
        );
    }

    ServableModel {
        model_id,
        sessions: inference::SessionPool::new(sessions),
        input_builder: Arc::new(input_builder),
        mini_batch,
        backend: model_config.backend,
    }
}

/// Chooses where metrics go, from the service configuration.
///
/// A CloudWatch namespace selects CloudWatch; without one, timings are summarised to
/// stderr, which needs no credentials.
///
/// Infallible: the CloudWatch client is built from the environment and contacts nothing,
/// so a missing permission surfaces on the first publish rather than here.
async fn build_metrics_sink(config: &config::MetricsConfig) -> Arc<dyn MetricsSink> {
    match config.cloudwatch_namespace.as_deref() {
        Some(namespace) => {
            let aws = aws_config::load_from_env().await;
            Arc::new(io::metrics::CloudWatchMetrics::new(
                Arc::new(aws_sdk_cloudwatch::Client::new(&aws)),
                namespace,
                config.batch_size,
                config.max_sends_in_flight,
            ))
        }
        None => Arc::new(io::metrics::StderrMetrics::new(config.batch_size)),
    }
}

/// Builds the inference-log sink for a blob-store prefix.
///
/// # Panics
///
/// Panics if the destination cannot be reached.
async fn build_data_sink(
    config: &config::InferenceLogConfig,
    location: &Location,
    stores: &BlobStores,
) -> Arc<dyn DataSink> {
    let store = stores
        .open(location)
        .await
        .unwrap_or_else(|e| panic!("cannot reach {}: {e}", location.uri()));
    Arc::new(io::data_sink::BlobDataSink::new(
        store,
        location.clone(),
        config.prefix_shards,
        config.batch_size,
        config.max_sends_in_flight,
    ))
}

/// How long shutdown waits for both sinks to drain.
///
/// Sits inside a typical supervisor grace period — 30 seconds for Kubernetes by default —
/// so giving up is ours to report rather than arriving as `SIGKILL` mid-write.
const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// Resolves when the process is asked to stop.
///
/// Ctrl-C, and on Unix `SIGTERM`. A signal that cannot be registered is reported to
/// stderr and then never resolves, rather than being treated as an immediate shutdown.
async fn shutdown_signal() {
    let interrupt = async {
        if let Err(e) = tokio::signal::ctrl_c().await {
            eprintln!("hushar: cannot listen for ctrl-c, so it will not drain: {e}");
            std::future::pending::<()>().await;
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(e) => {
                eprintln!("hushar: cannot listen for SIGTERM, so it will not drain: {e}");
                std::future::pending::<()>().await;
            }
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = interrupt => println!("hushar: interrupted, draining what is buffered"),
        () = terminate => println!("hushar: SIGTERM, draining what is buffered"),
    }
}

/// Starts the service.
///
/// The order is chosen so that every deployment mistake fails at startup rather than on
/// the first request:
///
/// 1. Parse the command line and count the cores.
/// 2. Parse the configuration URI, before any runtime exists.
/// 3. On a single-threaded runtime, read the configuration, parse the inference log
///    destination out of it, load the models, check each vectorization configuration
///    against its model's signature, and build the two sinks. All of this can await;
///    none of it needs workers.
/// 4. Build the serving runtime, and the mini-batch pool when a model splits requests.
/// 5. Print the banner, serve, then flush both sinks concurrently under
///    [`DRAIN_TIMEOUT`] before leaving.
///
/// One multi-threaded runtime serves everything. Its workers only poll; every inference
/// runs on the blocking pool, which is capped at the core count rather than tokio's
/// default ceiling of 512, so roughly one inference is in flight per core and further
/// work queues instead of oversubscribing. The spawned sink sends are async tasks and use
/// the workers, not the blocking pool.
///
/// `--port` and `--bind-address` override the configured listener. The HTTP/2 stream
/// limit and the per-connection concurrency limit are set from one configured value and
/// must stay equal, or a connection can stall.
///
/// # Errors
///
/// Returns an error if the core count cannot be read, if the configuration URI does not
/// parse, if a runtime or the mini-batch pool cannot be built, or if the server could not
/// bind and serve. A serve failure is returned after draining, so the sinks are flushed
/// either way.
///
/// # Panics
///
/// Panics on a configuration this host cannot serve: a service configuration that cannot
/// be reached, read or parsed, a malformed `inference_log.uri`, a model or candidate
/// model that cannot be loaded, or a candidate whose `model_id` matches the control's.
pub fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = cli::Cli::parse();
    let num_cpus = std::thread::available_parallelism()?.get();

    let config_location = io::Location::parse(&args.config_uri)?;
    let stores = Arc::new(BlobStores::new());

    let config_runtime = tokio::runtime::Builder::new_current_thread()
        .thread_name("hushar-config-loader")
        .enable_all()
        .build()?;

    let (service_config, control, candidate, split, metrics_sink, data_sink, sampler) =
        config_runtime.block_on(async {
            let config_store = stores
                .open(&config_location)
                .await
                .unwrap_or_else(|e| panic!("cannot reach {}: {e}", config_location.uri()));
            let config_json = config_store
                .get_string(&config_location)
                .await
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", config_location.uri()));
            let service_config = config::HusharServiceConfig::from_json(&config_json)
                .unwrap_or_else(|e| {
                    panic!(
                        "{} is not a valid service configuration: {e}",
                        config_location.uri()
                    )
                });

            let log_location = Location::parse(&service_config.inference_log.uri)
                .unwrap_or_else(|e| {
                    panic!(
                        "\"inference_log.uri\" {:?} is not a destination: {e}",
                        service_config.inference_log.uri
                    )
                });

            let mut neuron_runtime = None;
            let control = load_model(
                &stores,
                &service_config.model_config_path,
                "model",
                &service_config.threading,
                &mut neuron_runtime,
            )
            .await;
            let candidate = match &service_config.candidate_model {
                Some(spec) => {
                    let model = load_model(
                        &stores,
                        &spec.config_path,
                        "candidate model",
                        &service_config.threading,
                        &mut neuron_runtime,
                    )
                    .await;
                    assert_ne!(
                        model.model_id, control.model_id,
                        "the candidate and the control both call themselves {:?}; \
                     give the candidate its own model_id so metrics, logs and \
                     responses can tell them apart",
                        control.model_id
                    );
                    Some(Arc::new(model))
                }
                None => None,
            };

            let split = match &service_config.candidate_model {
                Some(spec) => inference::traffic_split::TrafficSplit::new(spec.traffic_percent),
                None => inference::traffic_split::TrafficSplit::control_only(),
            };
            if let Some(candidate) = &candidate {
                println!(
                    "hushar: A/B roll-out -> control {} keeps the rest, candidate {} takes {}",
                    control.model_id,
                    candidate.model_id,
                    split.describe(),
                );
            }

            let metrics_sink = build_metrics_sink(&service_config.metrics).await;
            let data_sink =
                build_data_sink(&service_config.inference_log, &log_location, &stores).await;
            let sampler = Sampler::new(service_config.inference_log.sample_rate);

            println!(
                "hushar: metrics -> {} [every {} batches, {} in flight]",
                metrics_sink.backend(),
                metrics_sink.batch_size(),
                service_config.metrics.max_sends_in_flight,
            );
            println!(
                "hushar: inference logs -> {} [{}, {} prefix shard(s), every {} batches, {} in flight, logging {}]",
                log_location.uri(),
                data_sink.backend(),
                service_config.inference_log.prefix_shards,
                data_sink.batch_size(),
                service_config.inference_log.max_sends_in_flight,
                sampler.describe(),
            );

            (
                service_config,
                Arc::new(control),
                candidate,
                split,
                metrics_sink,
                data_sink,
                sampler,
            )
        });

    config_runtime.shutdown_background();

    let worker_threads = service_config.threading.worker_threads(num_cpus);
    // The blocking pool is the backend replica count, so a request that reaches it always
    // finds an idle ORT session or Neuron model replica to lease.
    //
    // NOTE that tokio uses this pool for its own blocking work as well -- notably DNS
    // resolution behind the metric and log sinks -- so with one replica a lookup queues
    // behind an inference. The sinks retry and are off the request path, so a delayed
    // lookup costs a late batch rather than a failed request.
    let blocking_threads = service_config.threading.sessions_per_model();
    let server_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .max_blocking_threads(blocking_threads)
        .thread_name("hushar-server-worker")
        .enable_all()
        .build()?;

    let splits_requests = control.mini_batch.is_some()
        || candidate
            .as_ref()
            .is_some_and(|model| model.mini_batch.is_some());
    if splits_requests {
        let fan_out_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_cpus)
            .thread_name(|i| format!("hushar-fanout-{i}"))
            .build()?;
        if inference::scoring::init_fan_out(fan_out_pool).is_err() {
            return Err("the mini-batch pool was installed twice".into());
        }
    }

    let models = 1 + usize::from(candidate.is_some());
    let replicas = control.sessions.total();
    let loaded_models: Vec<&ServableModel> = std::iter::once(control.as_ref())
        .chain(candidate.iter().map(AsRef::as_ref))
        .collect();
    let ort_models = loaded_models
        .iter()
        .filter(|model| matches!(model.backend, config::BackendConfig::OnnxRuntime(_)))
        .count();
    let neuron_models = models - ort_models;
    let parallelism = service_config.threading.parallelism();

    if ort_models > 0 {
        let sessions = replicas * ort_models;
        let compute_threads = service_config.threading.compute_threads(sessions);
        println!(
            "hushar: threads -> {worker_threads} async, {replicas} concurrent inference(s) x \
             {} {} x {sessions} ONNX Runtime session(s) = {compute_threads} compute on {num_cpus} core(s)",
            parallelism.width(),
            parallelism.pool_name(),
        );
        if parallelism.is_parallel() {
            println!(
                "  exec   : operators run concurrently, so this pool needs a graph with \
                 independent branches to be worth anything -- a chain of blocks has none, and \
                 attention heads are a dimension inside one operator, not branches"
            );
            println!(
                "  exec   : memory pattern optimisation is off, which ONNX Runtime only offers \
                 under sequential execution"
            );
        }
        if compute_threads > num_cpus {
            println!(
                "  note   : {compute_threads} compute threads on {num_cpus} core(s) is \
                 oversubscribed; lower \"threading.sessions_per_model\" or \
                 \"threading.intra_op_threads\" -- their product is the budget"
            );
        } else if compute_threads * 2 <= num_cpus {
            println!(
                "  note   : {compute_threads} compute threads leaves {} of {num_cpus} core(s) \
                 unusable; raise \"threading.sessions_per_model\" for throughput, or \
                 \"threading.intra_op_threads\" to spend the cores on one request's latency",
                num_cpus - compute_threads,
            );
        }
        if let Some(spinning) = service_config.threading.allow_spinning {
            println!(
                "  spin   : idle intra-op and inter-op threads {}",
                if spinning { "spin" } else { "sleep" },
            );
        }
        if !service_config.threading.session_config.is_empty() {
            let sent: Vec<String> = service_config
                .threading
                .session_config
                .iter()
                .map(|(key, value)| format!("{key}={value}"))
                .collect();
            println!("  config : {}", sent.join(", "));
        }
        let stops_when_idle = service_config
            .threading
            .session_config
            .get("session.force_spinning_stop")
            .is_some_and(|value| value == "1");
        if replicas > 1 && service_config.threading.allow_spinning == Some(true) && !stops_when_idle
        {
            println!(
                "  note   : {sessions} spinning pools on {num_cpus} core(s) means an idle pool \
                 keeps burning time the busy ones need; add \"session.force_spinning_stop\": \
                 \"1\" to \"threading.session_config\" so a pool stops spinning when its Run() \
                 returns, or set \"allow_spinning\": false"
            );
        }
    }

    if neuron_models > 0 {
        if ort_models == 0 {
            println!(
                "hushar: threads -> {worker_threads} async, {replicas} concurrent inference(s), \
                 Neuron execution is device-managed on {num_cpus} host core(s)"
            );
        }
        for model in &loaded_models {
            let config::BackendConfig::Neuron(backend) = &model.backend else {
                continue;
            };
            let per_core = backend.replicas_per_neuron_core.max(1);
            let ranges = replicas.div_ceil(per_core);
            let occupied = ranges * backend.neuron_cores_per_replica as usize;
            let last = backend.start_neuron_core as usize + occupied - 1;
            println!(
                "hushar: Neuron -> model {}, {replicas} replica(s) on NeuronCore {}-{last}, \
                 {} core(s) and {} execution context(s) per replica",
                model.model_id,
                backend.start_neuron_core,
                backend.neuron_cores_per_replica,
                backend.execution_contexts_per_replica,
            );
            if backend.neuron_cores_per_replica > 1 {
                println!(
                    "  cores  : each logical replica loads one complete NEFF copy per core; \
                     its contexts spread across those copies (data parallel, not model sharding)"
                );
            }
            if per_core > 1 {
                println!(
                    "  share  : {per_core} replicas per core, so a core serves up to {per_core} \
                     requests at once; it still executes one at a time, and each replica \
                     holds its own copy of the weights"
                );
            }
            if let Some(batch) = model.mini_batch {
                let contexts = backend.execution_contexts_per_replica;
                println!(
                    "  ctx    : a request of N rows is ceil(N / {size}) execution(s) of {size} \
                     rows on the one replica it leases; {contexts} can be in flight at once and \
                     any further one waits for a context",
                    size = batch.size,
                );
                println!(
                    "  note   : a replica serves one request at a time, so \
                     \"backend.execution_contexts_per_replica\" is worth exactly the executions \
                     one request produces -- fewer serialize the request, more stay idle"
                );
            }
        }
        println!(
            "  config : intra_op_threads, inter_op_threads, allow_spinning and session_config \
             apply only to ONNX Runtime"
        );
    }

    let connection_concurrency = service_config.threading.connection_concurrency;
    println!("  admit  : {connection_concurrency} in flight per connection");
    if neuron_models == 0 {
        println!(
            "  pool   : {replicas} blocking thread(s) for {replicas} session(s) per model, each \
             session with its own {} pool; a request leases an idle session while it scores{}",
            parallelism.pool_name(),
            if models > 1 {
                format!(" -- from whichever of the {models} models it was routed to")
            } else {
                String::new()
            },
        );
    } else {
        println!(
            "  pool   : {replicas} blocking thread(s) per request lane; a request leases one \
             backend replica while it scores{}",
            if models > 1 {
                format!(" -- from whichever of the {models} models it was routed to")
            } else {
                String::new()
            },
        );
    }
    if replicas == 1 {
        println!(
            "  note   : one replica means one request at a time; raise \
             \"threading.sessions_per_model\" for concurrency, at one model load each"
        );
    }

    let port = args.port.unwrap_or(service_config.port_number);
    let bind_address = args.bind_address.unwrap_or(service_config.bind_address);
    let server_addr = SocketAddr::new(bind_address, port);

    println!("hushar: listening on {server_addr}");
    let stream_limit = u32::from(connection_concurrency);
    let connection_concurrency = usize::from(connection_concurrency);
    let hushar_service = HusharService {
        control,
        candidate,
        split,
        metrics_sink: Arc::clone(&metrics_sink),
        data_sink: Arc::clone(&data_sink),
        sampler,
    };
    let served = server_runtime.block_on(async {
        Server::builder()
            .max_concurrent_streams(Some(stream_limit))
            .concurrency_limit_per_connection(connection_concurrency)
            .tcp_keepalive(Some(std::time::Duration::from_secs(30)))
            .initial_stream_window_size(Some(2 << 20))
            .initial_connection_window_size(Some(4 << 20))
            .tcp_nodelay(true)
            .add_service(HusharServer::new(hushar_service))
            .serve_with_shutdown(server_addr, shutdown_signal())
            .await
    });

    server_runtime.block_on(async {
        let drain = async {
            tokio::join!(metrics_sink.flush(), data_sink.flush());
        };
        if tokio::time::timeout(DRAIN_TIMEOUT, drain).await.is_err() {
            eprintln!(
                "hushar: draining did not finish within {DRAIN_TIMEOUT:?}; some \
                 buffered metrics or log rows were not written"
            );
        }
    });
    println!("hushar: drained, exiting");
    server_runtime.shutdown_background();

    served.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
        format!("could not serve on {server_addr}: {e}").into()
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Asserts one session per model yields one set of options, and that a thread always
    /// takes index 0 because there is nothing to spread over.
    #[test]
    fn a_single_session_needs_no_choice() {
        let threading = config::ThreadingConfig {
            intra_op_threads: 8,
            ..config::ThreadingConfig::default()
        };
        let options = replica_options(&threading);
        assert_eq!(options.len(), 1, "one session, one set of options");
        assert!(
            options[0].is_empty(),
            "nothing to say: intra_op_threads is passed separately, not as an option"
        );
        let pool = inference::SessionPool::new(vec![7u8]);
        for _ in 0..4 {
            assert_eq!(
                *pool.acquire(),
                7,
                "the one session serves every request in turn"
            );
        }
    }

    /// Asserts the session count decides how many option sets the loader is given, since
    /// that length is how the count reaches it.
    #[test]
    fn the_session_count_is_the_number_of_option_sets() {
        for sessions in [1usize, 2, 4, 8] {
            let threading = config::ThreadingConfig {
                sessions_per_model: sessions,
                ..config::ThreadingConfig::default()
            };
            assert_eq!(
                replica_options(&threading).len(),
                sessions,
                "{sessions} session(s) means {sessions} set(s) of options"
            );
        }
    }

    /// Asserts every session is configured identically: they differ by owning separate
    /// pools, not by their settings.
    #[test]
    fn every_session_is_configured_the_same_way() {
        let threading = config::ThreadingConfig {
            intra_op_threads: 4,
            sessions_per_model: 3,
            allow_spinning: Some(false),
            ..config::ThreadingConfig::default()
        };
        let options = replica_options(&threading);
        assert_eq!(options.len(), 3);
        for one in &options {
            assert_eq!(one, &options[0], "no session is a special case");
            assert!(
                one.contains(&("session.intra_op.allow_spinning".to_owned(), "0".to_owned())),
                "every session is told, got {one:?}"
            );
        }
    }

    /// Asserts concurrent requests on a model each lease a different session, and that a
    /// request arriving after any number of threads have come and gone still gets the idle
    /// one -- the case the per-thread counter this replaced got wrong.
    #[test]
    fn concurrent_requests_lease_distinct_sessions_whatever_the_thread_history() {
        let pool = Arc::new(inference::SessionPool::new(vec![0usize, 1, 2, 3]));
        for _ in 0..8 {
            let pool = Arc::clone(&pool);
            std::thread::spawn(move || drop(pool.acquire()))
                .join()
                .unwrap();
        }
        let held: Vec<_> = (0..3).map(|_| pool.acquire()).collect();
        let busy: Vec<usize> = held.iter().map(|lease| **lease).collect();
        let late = {
            let pool = Arc::clone(&pool);
            std::thread::spawn(move || *pool.acquire()).join().unwrap()
        };
        assert!(
            !busy.contains(&late),
            "a late request must take the idle session, not one in use: got {late} with \
             {busy:?} busy"
        );
        drop(held);
        assert_eq!(pool.idle(), 4, "every lease was returned");
    }

    /// Asserts a default threading configuration emits no session options at all.
    #[test]
    fn a_default_threading_configuration_sets_no_session_options() {
        let threading = config::ThreadingConfig::default();
        assert!(
            session_options(&threading).is_empty(),
            "the default must leave every ONNX Runtime default in place"
        );
    }

    /// Asserts a wide intra-op pool still emits no session options: the width is passed
    /// to ONNX Runtime directly, and there is no longer a placement string to derive.
    #[test]
    fn a_wide_pool_needs_no_session_option() {
        let json = r#"{"model_config_path": "/m.json",
                       "threading": {"intra_op_threads": 8},
                       "inference_log": {"uri": "/logs"}}"#;
        let config = config::HusharServiceConfig::from_json(json).expect("valid");
        assert!(
            session_options(&config.threading).is_empty(),
            "only spinning is expressed as an option now"
        );
    }

    /// Asserts both the intra-op and inter-op spinning keys are set, spelled `"1"`/`"0"`.
    #[test]
    fn spinning_sets_both_pools_with_the_runtimes_spelling() {
        for (configured, expected) in [(true, "1"), (false, "0")] {
            let json = format!(
                r#"{{"model_config_path": "/m.json",
                      "threading": {{"allow_spinning": {configured}}},
                      "inference_log": {{"uri": "/logs"}}}}"#
            );
            let config = config::HusharServiceConfig::from_json(&json).expect("valid");
            let options = session_options(&config.threading);
            let keys: Vec<&str> = options.iter().map(|(k, _)| k.as_str()).collect();
            assert_eq!(
                keys,
                vec![
                    "session.intra_op.allow_spinning",
                    "session.inter_op.allow_spinning"
                ]
            );
            assert!(options.iter().all(|(_, v)| v == expected));
        }
    }

    /// Asserts extra session settings follow the spinning keys, verbatim and in key order,
    /// and reach every session's option set.
    #[test]
    fn session_config_entries_follow_spinning_in_key_order_on_every_session() {
        let json = r#"{"model_config_path": "/m.json",
                       "threading": {"allow_spinning": true, "sessions_per_model": 3,
                                     "session_config": {
                                         "session.intra_op.spin_duration_us": "500",
                                         "session.force_spinning_stop": "1"}},
                       "inference_log": {"uri": "/logs"}}"#;
        let config = config::HusharServiceConfig::from_json(json).expect("valid");
        let options = session_options(&config.threading);
        let as_pairs: Vec<(&str, &str)> = options
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();
        assert_eq!(
            as_pairs,
            vec![
                ("session.intra_op.allow_spinning", "1"),
                ("session.inter_op.allow_spinning", "1"),
                ("session.force_spinning_stop", "1"),
                ("session.intra_op.spin_duration_us", "500"),
            ]
        );
        let replicas = replica_options(&config.threading);
        assert_eq!(replicas.len(), 3, "one option set per session");
        assert!(
            replicas.iter().all(|set| *set == options),
            "every session is configured identically"
        );
    }
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Asserts a scoring result and its error cross a `spawn_blocking` boundary, which is
    /// a compile-time check that both are `Send + Sync`.
    #[test]
    fn inference_results_can_return_from_a_blocking_task() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(1)
            .enable_all()
            .build()
            .expect("runtime");

        runtime.block_on(async {
            let outcome: Result<(), inference::InferenceError> =
                tokio::task::spawn_blocking(|| Err("inference failed".into()))
                    .await
                    .expect("join");
            assert!(outcome.is_err(), "the error should survive the trip back");

            let ok: Result<inference::scoring::ScoredBatch, inference::InferenceError> =
                tokio::task::spawn_blocking(|| {
                    Ok(inference::scoring::ScoredBatch {
                        scores: inference::batch::ScoreData::F32(Vec::new()),
                        timings: inference::InferenceMicros {
                            vec_time: 1,
                            tensor_time: 2,
                            inference_time: 3,
                        },
                    })
                })
                .await
                .expect("join");
            assert_eq!(ok.expect("ok result").timings.inference_time, 3);
        });
    }

    /// Asserts CPU-bound work handed to `spawn_blocking` from a spawned task leaves the
    /// single async worker free to keep making progress.
    #[test]
    fn blocking_work_does_not_starve_the_async_workers() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(2)
            .enable_all()
            .build()
            .expect("runtime");

        runtime.block_on(async {
            let ticks = Arc::new(AtomicUsize::new(0));
            let ticker = {
                let ticks = Arc::clone(&ticks);
                tokio::spawn(async move {
                    loop {
                        ticks.fetch_add(1, Ordering::Relaxed);
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                })
            };

            tokio::time::sleep(Duration::from_millis(10)).await;
            let before = ticks.load(Ordering::Relaxed);

            tokio::spawn(async {
                tokio::task::spawn_blocking(|| std::thread::sleep(Duration::from_millis(60)))
                    .await
                    .expect("blocking task");
            })
            .await
            .expect("handler task");

            let advanced = ticks.load(Ordering::Relaxed) - before;
            ticker.abort();
            assert!(
                advanced > 5,
                "the async worker advanced only {advanced} ticks during 60ms of \
                 CPU-bound work, so it was starved rather than freed"
            );
        });
    }

    /// Asserts the sampler alternates at one in two, so a second call for the same
    /// request would contradict the first.
    #[test]
    fn the_sampling_decision_is_taken_once_per_request() {
        let sampler = Sampler::new(0.5);

        let first = sampler.sample();
        let second = sampler.sample();
        assert_ne!(
            first, second,
            "consecutive calls alternate at one in two, so a second call for the \
             same request would contradict the first"
        );
    }
}
