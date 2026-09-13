// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

use clap::Parser;
use hushar::hushar_proto::{
    InferenceLogBatch, InferenceRequest, InferenceResponse,
    hushar_server::{Hushar, HusharServer},
};
use inference::InferenceBackend;
use inference::scoring::FeatureLogging;
use io::{BlobStores, DataDestination, DataSink, MetricsSink, Sampler};
use std::net::SocketAddr;
use std::sync::Arc;
use tonic::{Request, Response, Status, transport::Server};

pub(crate) mod affinity;
pub(crate) mod cli;
pub(crate) mod config;
pub(crate) mod inference;
pub(crate) mod io;

/// One servable model: everything a request needs once an arm has been chosen.
///
/// Grouped into a struct rather than left as parallel fields because the three travel
/// together and must never be crossed — feeding one model's inputs through another's
/// builder would be a shape error at best and a wrong answer at worst.
///
/// Fields:
/// - `model_id` — from the model's own configuration. Recorded on every inference log
///   batch and every metric, and returned in the response, so an outcome can be
///   attributed to the model that produced it.
/// - `replicas` — the engine, once per session. One entry is the ordinary case; several
///   give concurrent requests independent intra-op pools. Every entry is the same graph.
/// - `input_builder` — how a feature payload becomes this model's inputs. Each arm has
///   its own, which is what lets a candidate take its features differently from the
///   control.
/// - `mini_batch` — how a request's rows are cut, when the configuration says to. The
///   batches are scored at the same time, which is what makes a split lower latency
///   rather than raise it. `None` runs each request as a single batch.
#[derive(Debug)]
pub(crate) struct ServableModel {
    model_id: String,
    replicas: Vec<Arc<dyn InferenceBackend>>,
    input_builder: Arc<inference::input_builder::InputBuilder>,
    mini_batch: Option<inference::scoring::MiniBatch>,
}

impl ServableModel {
    /// The session this thread uses, and the cores it belongs on.
    ///
    /// Chosen by **thread** rather than per request, and both halves of that matter. A
    /// thread keeps its replica for life, so it is pinned once rather than on every
    /// request; and because slice `i` is shared by replica `i` of every model, a thread
    /// that serves the control and then the candidate stays on the same cores. It runs
    /// only one of them at a time, so they cannot contend.
    fn replica_for_this_thread(
        &self,
        slices: &[Arc<Vec<usize>>],
    ) -> (Arc<dyn InferenceBackend>, Arc<Vec<usize>>) {
        let (index, cores) = replica_slot(self.replicas.len(), slices);
        (Arc::clone(&self.replicas[index]), cores)
    }
}

/// The replica index this thread owns, and the cores that replica's pool is pinned to.
///
/// The two are returned together because they must agree: the session's pool is pinned to
/// slice `i`, so a caller that runs replica `i` has to be pinned to slice `i` too. Pairing
/// them here is what stops the two being chosen from different places.
fn replica_slot(replicas: usize, slices: &[Arc<Vec<usize>>]) -> (usize, Arc<Vec<usize>>) {
    let index = if replicas <= 1 {
        0
    } else {
        this_thread_replica(replicas)
    };
    let cores = slices
        .get(index)
        .cloned()
        .unwrap_or_else(|| Arc::new(Vec::new()));
    (index, cores)
}

/// A stable replica index for the calling thread, assigned on first use.
///
/// Handed out round-robin as threads arrive, so tokio's blocking pool spreads itself over
/// the replicas. Sized to the caller's replica count, which is the same for every model in
/// a deployment, so one index serves them all.
fn this_thread_replica(replicas: usize) -> usize {
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    thread_local! {
        static MINE: Cell<Option<usize>> = const { Cell::new(None) };
    }
    match MINE.get() {
        Some(index) => index % replicas.max(1),
        None => {
            let index = NEXT.fetch_add(1, Ordering::Relaxed) % replicas.max(1);
            MINE.set(Some(index));
            index
        }
    }
}

/// Everything a request handler needs, built once at startup.
///
/// Fields:
/// - `control` — the model from `model_config_path`, and the only one when no candidate
///   is configured.
/// - `candidate` — the model from `candidate_model.config_path`, when there is one.
/// - `split` — which arm an unnamed request goes to.
/// - `metrics_sink`, `data_sink` — where timings and inference logs accumulate. Held
///   directly rather than reached through a channel: there are no sidecars, and a
///   handler's whole interaction with either is one non-blocking call.
/// - `sampler` — what fraction of requests is logged, decided before scoring so an
///   unsampled request never copies its features.
#[derive(Debug)]
pub(crate) struct HusharService {
    control: Arc<ServableModel>,
    candidate: Option<Arc<ServableModel>>,
    split: inference::traffic_split::TrafficSplit,
    metrics_sink: Arc<dyn MetricsSink>,
    data_sink: Arc<dyn DataSink>,
    sampler: Sampler,
    /// Logical CPUs each replica's session may run on, one entry per replica. A single
    /// entry is the ordinary case; empty inside an entry leaves placement to the kernel.
    core_slices: Arc<CoreSlices>,
}

impl HusharService {
    /// The model that serves this request.
    ///
    /// A named model wins over the configured split, and a name that is not loaded is
    /// refused rather than falling back. Falling back would be the worse failure: the
    /// request would be answered, so nothing would look wrong, while the experiment it
    /// belongs to quietly recorded the wrong arm. The error names both loaded models so
    /// the caller can see what it should have asked for.
    fn choose(&self, requested: &str) -> Result<&Arc<ServableModel>, Status> {
        use inference::traffic_split::Arm;

        if requested.is_empty() {
            return Ok(match self.split.choose() {
                Arm::Control => &self.control,
                // Only reachable with a candidate loaded: without one the split is
                // built as control-only, which never returns this arm.
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
    /// # Why inference does not run on this thread
    ///
    /// Inference is CPU-bound and can run for milliseconds. Running it inline would
    /// hold this worker for that whole time, and a tokio worker that is not polling
    /// is a worker not accepting connections or driving other requests -- with `N`
    /// workers, `N` concurrent inferences stall the server. `spawn_blocking` moves it
    /// to the blocking pool, which is bounded where the runtime is built.
    ///
    /// # Why recording is safe to do here
    ///
    /// Neither sink call waits on anything. Each appends to a buffer and returns; the
    /// request that fills a buffer hands the whole batch to a spawned task, so no
    /// request ever waits on S3, CloudWatch or Kinesis. That is what makes the
    /// sidecars unnecessary rather than merely unfashionable -- they existed to keep
    /// this latency off the request path, and an accumulator keeps it off without
    /// reserving a thread or shedding into a full channel. See
    /// [`io::accumulator`](crate::io::accumulator).
    ///
    /// # Why sampling is decided before scoring
    ///
    /// The features are the largest thing a request carries, and logging them means
    /// keeping them after the response is built. Deciding first lets an unsampled
    /// request drop them where they lie, so a low sample rate reduces the work done
    /// and not just the bytes retained.
    ///
    /// A join error means the task panicked: a bug, not a bad request.
    ///
    /// # Why the row count is checked here
    ///
    /// A deployment serving a pinned-shape model can only run batches of one size, and
    /// a request of any other size would fail inside the engine on a shape mismatch --
    /// after the features were built and a blocking thread was taken. Checking it on
    /// this thread makes it the caller's error, with the status code that says so, and
    /// costs nothing when nothing is pinned. An empty request stays a no-op, because it
    /// never reaches the engine to mismatch anything.
    async fn inference_service(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>, Status> {
        let inference_req = request.into_inner();
        let request_id = inference_req.request_id;

        let rows = inference_req.inputs;

        // Before anything is built, so a request naming a model this server does not
        // serve costs one string comparison rather than a vectorization pass.
        let model = Arc::clone(self.choose(&inference_req.model_id)?);
        let model_id = model.model_id.clone();

        let input_builder = Arc::clone(&model.input_builder);
        let mini_batch = model.mini_batch;
        let logging = if self.sampler.sample() {
            FeatureLogging::Record
        } else {
            FeatureLogging::Skip
        };

        let core_slices = Arc::clone(&self.core_slices);
        let scored = tokio::task::spawn_blocking(move || {
            // The replica is chosen *here*, on the thread that will run it, and not on the
            // async worker that accepted the request. The session's pool is pinned to one
            // slice of the cores, so the thread calling into it has to be on that same
            // slice -- picking on another thread would pin this one to a slice whose pool
            // it then does not use, which is the worst of both.
            let (backend, compute_cores) = model.replica_for_this_thread(&core_slices);
            // Before any weights are touched, so the pages it warms are the ones on the
            // cores it will stay on.
            pin_compute_thread(&compute_cores);
            inference::scoring::score_features(
                backend.as_ref(),
                rows,
                &input_builder,
                mini_batch,
                logging,
            )
        })
        .await
        .map_err(|e| Status::internal(format!("inference task failed to complete: {e}")))?;

        match scored {
            Ok(batch) => {
                self.metrics_sink.record(&model_id, &batch.timings);

                let inference_response = InferenceResponse {
                    request_id: request_id.clone(),
                    outputs: batch.outputs,
                    model_id: model_id.clone(),
                };
                if logging == FeatureLogging::Record {
                    self.data_sink.record(InferenceLogBatch {
                        request_id,
                        model_id,
                        inference_log_rows: batch.logs,
                    });
                }

                Ok(Response::new(inference_response))
            }
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }
}

/// One entry per distinct type and shape when there are many, the full list when few.
///
/// A 200-feature model rendered as a single 5,000-character line, which is not a contract
/// anyone reads. Grouping loses nothing that matters here: every distinct type and shape
/// still appears with a count, and the names are on the `features` lines below, one per
/// line, where they can be grepped.
fn describe_specs(specs: &[crate::inference::batch::IoSpec]) -> String {
    let described: Vec<String> = specs
        .iter()
        .map(crate::inference::batch::IoSpec::describe)
        .collect();
    if described.len() <= 8 {
        return described.join(", ");
    }
    // Keyed on everything after the name, which is the type and shape.
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
/// Only what was asked for: an empty list leaves every default in place, so a
/// configuration that says nothing about placement produces exactly the session it did
/// before these settings existed.
///
/// Affinity is only emitted when there is an extra intra-op thread to place — at the
/// default of one thread the calling thread is the whole pool, and it is pinned by
/// [`pin_compute_thread`] rather than by the runtime.
fn session_options(threading: &config::ThreadingConfig) -> Vec<(String, String)> {
    session_options_for(threading, &threading.compute_cores())
}

/// The same, for one replica's slice of the cores.
fn session_options_for(
    threading: &config::ThreadingConfig,
    cores: &[usize],
) -> Vec<(String, String)> {
    let mut options = Vec::new();
    if let Some(value) = affinity::ort_affinity(cores, threading.intra_op_threads) {
        options.push(("session.intra_op_thread_affinities".to_owned(), value));
    }
    if let Some(spinning) = threading.allow_spinning {
        // Upstream's key takes "1" or "0" rather than a boolean.
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
    options
}

/// The logical CPUs each replica's pool may run on, one entry per replica.
type CoreSlices = Vec<Arc<Vec<usize>>>;

/// ONNX Runtime's string-keyed session settings, one set per replica.
type ReplicaOptions = Vec<Vec<(String, String)>>;

/// How the cores are divided between replicas, and the session options that follow.
///
/// One slice per replica, so each session's pool has cores of its own and two replicas
/// cannot contend. Slice `i` is shared by replica `i` of *every* model, because a thread
/// serves one model at a time — which is what stops a candidate model doubling the
/// footprint instead of only the memory.
///
/// With one replica this is exactly what the service did before: a single session over
/// whatever `compute_cores` named, empty included.
fn replica_placement(
    threading: &config::ThreadingConfig,
    cores: usize,
) -> (CoreSlices, ReplicaOptions) {
    let replicas = threading.sessions_per_model();
    if replicas == 1 {
        return (
            vec![Arc::new(threading.compute_cores())],
            vec![session_options(threading)],
        );
    }
    // Asking for several pools is asking for them to be placed; without a list to divide
    // there is nothing to keep them apart, so the whole machine is the list.
    let configured = threading.compute_cores();
    let pool = if configured.is_empty() {
        (0..cores).collect()
    } else {
        configured
    };
    let slices = affinity::partition(&pool, replicas);
    let options = slices
        .iter()
        .map(|slice| session_options_for(threading, slice))
        .collect();
    (slices.into_iter().map(Arc::new).collect(), options)
}

/// Pins the thread running an inference to the compute cores, once per thread.
///
/// Called from inside the blocking task rather than from a runtime hook, because tokio's
/// `on_thread_start` fires for both pools and cannot tell them apart. Doing it here
/// places exactly the threads that run inference, and lets the hook place everything
/// else — a blocking thread pinned to the async cores at startup is re-pinned here
/// before it does any work.
///
/// The thread-local guard keeps this to one syscall per thread rather than one per
/// request. A failure is reported once and then tolerated: a service that refuses to
/// answer because it could not place a thread is worse than one that answers from the
/// wrong core.
fn pin_compute_thread(cores: &[usize]) {
    use std::cell::Cell;
    thread_local! {
        static PINNED: Cell<bool> = const { Cell::new(false) };
    }
    if cores.is_empty() || PINNED.get() {
        return;
    }
    PINNED.set(true);
    if let Err(e) = affinity::pin_current_thread(cores) {
        eprintln!("hushar: could not pin an inference thread, continuing unpinned: {e}");
    }
}

/// Reads one model's configuration, loads it, and checks the two agree.
///
/// `intra_op_threads` comes from the service configuration rather than the model's,
/// because it describes how this host spends its cores rather than anything about the
/// model -- and both arms of a roll-out must use the same value, or the comparison is
/// measuring thread counts instead of models.
///
/// `role` names the arm in failures and in the banner, so a roll-out's two loads are
/// distinguishable — "the candidate model could not load" is a different operational
/// problem from the control failing, and only one of them means the service is down.
///
/// Everything checkable is checked here, against the model's own signature: a feature
/// bound to an input the model does not have, a type that disagrees, a transformation
/// wider than the input it feeds. A deployment mistake stops the server starting rather
/// than producing a service that starts cleanly and rejects every request.
async fn load_model(
    stores: &BlobStores,
    config_path: &str,
    role: &str,
    intra_op_threads: i32,
    session_options: &[Vec<(String, String)>],
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
    let replicas = io::load_onnx_sessions(
        &model_store
            .get(&model_location)
            .await
            .unwrap_or_else(|e| panic!("cannot read {role} {}: {e}", model_config.model_path)),
        &model_config.execution_provider,
        intra_op_threads,
        session_options,
        mini_batch,
    )
    .unwrap_or_else(|e| {
        panic!(
            "could not load {role} {} on execution provider {:?}: {e}",
            model_config.model_path, model_config.execution_provider
        )
    });
    // Every replica is the same graph, so one of them describes the set.
    let backend = Arc::clone(&replicas[0]);

    let input_builder = inference::input_builder::InputBuilder::resolve(
        model_config.vectorization_config,
        backend.inputs(),
    )
    .unwrap_or_else(|e| panic!("the {role} configuration does not fit the {role}: {e}"));

    // The bindings are one per line and can run to hundreds, so the count goes first:
    // an operator checking that 200 features were understood should not have to count
    // 200 lines to find out.
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
    // Printed only when it changes how a request is executed, so a normal deployment's
    // banner is unchanged and a pinned one cannot be mistaken for it.
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
        replicas,
        input_builder: Arc::new(input_builder),
        mini_batch,
    }
}

/// Chooses where metrics go, from the service configuration.
///
/// A CloudWatch namespace selects CloudWatch. Without one, timings are summarised
/// to stderr, which is what a local run or a test wants and needs no credentials.
///
/// Infallible: the CloudWatch client is built from the environment and does not
/// contact anything, so there is nothing here to fail. A missing permission
/// surfaces later, on the first publish, as a reported error rather than a refusal
/// to serve traffic -- which is the right trade for observability.
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

/// Chooses where inference logs go, from the destination URI.
///
/// `kinesis://stream` selects the stream; anything else is a blob-store prefix. The
/// same shape as `build_metrics_sink`, and for the same reason: the decision is made
/// once, here, and the request path never learns the answer.
async fn build_data_sink(
    config: &config::InferenceLogConfig,
    destination: &DataDestination,
    stores: &BlobStores,
) -> Arc<dyn DataSink> {
    match destination {
        DataDestination::Blob(location) => {
            let store = stores
                .open(location)
                .await
                .unwrap_or_else(|e| panic!("cannot reach {}: {e}", location.uri()));
            Arc::new(io::data_sink::BlobDataSink::new(
                store,
                location.clone(),
                config.batch_size,
                config.max_sends_in_flight,
            ))
        }
        DataDestination::Kinesis { stream } => {
            let aws = aws_config::load_from_env().await;
            Arc::new(io::data_sink::KinesisDataSink::new(
                Arc::new(aws_sdk_kinesis::Client::new(&aws)),
                stream,
                config.batch_size,
                config.max_sends_in_flight,
            ))
        }
    }
}

/// How long shutdown waits for both sinks to drain.
///
/// A supervisor that sends `SIGTERM` follows it with `SIGKILL` after a grace period —
/// 30 seconds for Kubernetes by default — so waiting without a deadline does not buy
/// more time, it just risks being killed mid-write with no message explaining it. This
/// sits inside a typical grace period so the give-up is ours and is reported.
const DRAIN_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(20);

/// Resolves when the process is asked to stop.
///
/// Ctrl-C, and on Unix `SIGTERM` — which is what a container runtime and every process
/// supervisor send. Without it `serve` runs until the process is killed, so the flush
/// after it never happens and everything accumulated goes with the process.
///
/// That was survivable when a buffer held a few hundred batches. It is not the moment
/// the buffer holds thousands, which is why raising the batch size and handling the
/// signal are the same change: the larger the batch, the more a missing flush costs.
///
/// A signal this cannot register is reported and then ignored rather than treated as a
/// shutdown, since resolving immediately would stop the server before it served
/// anything.
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
/// The order matters, and it is chosen so that every deployment mistake fails at
/// startup rather than on the first request:
///
/// 1. Parse the command line and count the cores.
/// 2. Parse the configuration URI, before any runtime exists.
/// 3. On a single-threaded runtime, read the configuration, parse the inference log
///    destination out of it, load the model, check the vectorisation configuration
///    against the model's signature, and build the two sinks. All of this can await;
///    none of it needs workers.
/// 4. Build the serving runtime.
/// 5. Serve, then flush both sinks before leaving.
///
/// # One runtime, every core
///
/// There were three: a server runtime sized to whatever the two sidecar runtimes left,
/// and one runtime each for metrics and logs. On a ten-core host with the default
/// thread counts that meant eight serving workers and, because the blocking pool was
/// sized to the workers, eight concurrent inferences -- two cores reserved for work
/// that takes microseconds per request, against inference that takes milliseconds.
/// Removing the sidecars returns both to serving, so the blocking pool grows with them.
///
/// The blocking pool is still capped at the core count rather than tokio's default
/// ceiling of 512: for CPU-bound work that ceiling would mean hundreds of threads
/// fighting over the same cores, each slower and none finishing sooner. Sizing it to
/// the cores keeps roughly one inference in flight per core, and further work queues
/// instead of oversubscribing, which is the backpressure this path wants. The spawned
/// sends are async tasks and use the worker threads, not this pool.
///
/// The error type carries `Send + Sync` so that failures from the storage layer,
/// which carry it, convert with `?` rather than needing to be re-wrapped.
///
/// Opening the stores is deferred to the config runtime, because building an AWS client
/// is async. Each path carries its own URI, so a model in S3 beside a configuration on
/// local disk is expressible. A wrong execution provider is a deploy-time mistake and
/// not recoverable, so fail loudly rather than serving errors. The declared
/// configuration is checked against the model's signature here, so every mistake that
/// does not need a request is a boot failure. A service that starts cleanly and then
/// rejects every request is worse than one that refuses to start. Printed so an
/// operator can confirm the signature was read as intended and that each input is fed
/// from where they meant. `--port` and `--bind-address` override the configured
/// listener, which lets two instances share a host.
pub fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = cli::Cli::parse();
    let num_cpus = std::thread::available_parallelism()?.get();

    let config_location = io::Location::parse(&args.config_uri)?;
    let stores = Arc::new(BlobStores::new());

    let config_runtime = tokio::runtime::Builder::new_current_thread()
        .thread_name("hushar-config-loader")
        .enable_all()
        .build()?;

    let (service_config, control, candidate, split, metrics_sink, data_sink, sampler, core_slices) =
        config_runtime.block_on(async {
            let config_store = stores
                .open(&config_location)
                .await
                .unwrap_or_else(|e| panic!("cannot reach {}: {e}", config_location.uri()));
            let config_json = config_store
                .get_string(&config_location)
                .await
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", config_location.uri()));
            // Reported rather than unwrapped, because `validate` writes its messages for an
            // operator -- naming the field and the range -- and `unwrap` would show them
            // escaped inside a `Result::unwrap()` panic.
            let service_config = config::HusharServiceConfig::from_json(&config_json)
                .unwrap_or_else(|e| {
                    panic!(
                        "{} is not a valid service configuration: {e}",
                        config_location.uri()
                    )
                });

            // Parsed here rather than beside the config URI, because it is a field of the
            // configuration now. Still before the model is read: a malformed destination is
            // a deployment mistake, and finding it after a 190 MB load wastes the load.
            let log_destination = DataDestination::parse(&service_config.inference_log.uri)
                .unwrap_or_else(|e| {
                    panic!(
                        "\"inference_log.uri\" {:?} is not a destination: {e}",
                        service_config.inference_log.uri
                    )
                });

            let intra_op_threads = service_config.threading.intra_op_threads;
            // Built once and shared by both arms, so a roll-out cannot accidentally
            // place its two models differently and then compare them.
            let (core_slices, session_options) =
                replica_placement(&service_config.threading, num_cpus);
            let control = load_model(
                &stores,
                &service_config.model_config_path,
                "model",
                intra_op_threads,
                &session_options,
            )
            .await;
            let candidate = match &service_config.candidate_model {
                Some(spec) => {
                    let model = load_model(
                        &stores,
                        &spec.config_path,
                        "candidate model",
                        intra_op_threads,
                        &session_options,
                    )
                    .await;
                    // Both arms are named on every metric, every log row and every
                    // response, so two arms sharing a name would make all three
                    // unreadable -- and a request naming that id would be ambiguous.
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
                build_data_sink(&service_config.inference_log, &log_destination, &stores).await;
            let sampler = Sampler::new(service_config.inference_log.sample_rate);

            // The effective batch sizes rather than the configured ones: CloudWatch and
            // Kinesis each cap what one call may carry, and a number that was reduced is
            // otherwise invisible until someone wonders why the writes are so frequent.
            println!(
                "hushar: metrics -> {} [every {} batches, {} in flight]",
                metrics_sink.backend(),
                metrics_sink.batch_size(),
                service_config.metrics.max_sends_in_flight,
            );
            println!(
                "hushar: inference logs -> {} [{}, every {} batches, {} in flight, logging {}]",
                log_destination.uri(),
                data_sink.backend(),
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
                core_slices,
            )
        });

    config_runtime.shutdown_background();

    // Both sized from the configuration, defaulting to one per core. The two pools do
    // different work: workers only ever poll, and every inference happens on a blocking
    // thread, so trimming the workers gives the engine cores rather than taking them
    // from anything.
    let worker_threads = service_config.threading.worker_threads(num_cpus);
    let inference_concurrency = service_config.threading.inference_concurrency(num_cpus);
    // Fires for both pools, which is why the blocking threads re-pin themselves in
    // `pin_compute_thread` before doing any work. The effect is that everything starts on
    // the async cores and only the threads that run inference move off them.
    let worker_cores = service_config.threading.worker_cores();
    let hook_cores = worker_cores.clone();
    let server_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(worker_threads)
        .max_blocking_threads(inference_concurrency)
        .thread_name("hushar-server-worker")
        .on_thread_start(move || {
            if !hook_cores.is_empty()
                && let Err(e) = affinity::pin_current_thread(&hook_cores)
            {
                eprintln!("hushar: could not pin a runtime thread: {e}");
            }
        })
        .enable_all()
        .build()?;

    // Printed because these decide the shape of every request's latency, and because the
    // product is easy to get wrong in both directions: more compute threads than cores
    // oversubscribes, and far fewer leaves most of an expensive machine idle. A resident
    // candidate model counts, because the intra-op pool belongs to the session -- two
    // models is two pools, and nearly twice the thread budget.
    let models = 1 + usize::from(candidate.is_some());
    let replicas = service_config.threading.sessions_per_model();
    let sessions = models * replicas;
    let compute_threads = service_config.threading.compute_threads(num_cpus, sessions);
    println!(
        "hushar: threads -> {worker_threads} async, {inference_concurrency} inference x \
         {} intra-op x {sessions} session(s) = {compute_threads} compute on {num_cpus} core(s)",
        service_config.threading.intra_op_threads,
    );
    // Printed with the threads rather than with the listener because it is the front of
    // the same queue: this is how deep the line in front of `inference_concurrency` may
    // get on one connection, and a limit nobody can see is one nobody can account for
    // when latency climbs. It bounds a connection and not the process, which is why the
    // line says so -- ten connections admit ten times this.
    let connection_concurrency = service_config.threading.connection_concurrency;
    println!("  admit  : {connection_concurrency} in flight per connection");
    // Only when there is more than one, so an ordinary deployment's banner is unchanged
    // and a pooled one cannot be mistaken for it. The slices are the whole point of asking
    // for replicas, so they are named rather than implied.
    if replicas > 1 {
        let placed = core_slices
            .iter()
            .map(|slice| affinity::describe(slice))
            .collect::<Vec<_>>()
            .join(" | ");
        // Sessions are per model but slices are not, so with a candidate resident the two
        // counts differ and the line has to say which is which -- naming only one of them
        // is how the geometry gets misread.
        let shared = if models > 1 {
            format!(" over {replicas} slice(s) shared by all {models} model(s)")
        } else {
            String::new()
        };
        println!(
            "  pool   : {replicas} session(s) per model{shared}, one intra-op pool each on \
             core(s) {placed}"
        );
        // The comparison that predicts latency, and it cannot be read off the thread total
        // because that is a whole-host figure while contention is per slice. Every resident
        // model has a pool on the slice, so a candidate doubles the threads there without
        // adding cores.
        //
        // Whether that costs anything depends on how many requests reach one slice at a
        // time, so the line says which case this is rather than warning either way. Both
        // were measured at the same nominal 2.0x: `inf 24` over 4 replicas is 6 requests a
        // slice and cost 20 % of p50, while `inf 8` over 8 replicas is 1 request a slice
        // and gave the tightest tail on the host -- one pool works while the other sleeps.
        let (per_slice, width) = service_config.threading.slice_pressure(num_cpus, models);
        if width > 0 && per_slice > width {
            let in_flight = inference_concurrency.div_ceil(replicas.max(1));
            let verdict = if in_flight > 1 {
                format!(
                    "{in_flight} request(s) per slice, so the pools contend -- costs latency at saturation"
                )
            } else {
                "1 request per slice, so only one pool is busy at a time -- nominal, not paid"
                    .to_owned()
            };
            println!(
                "  note   : {per_slice} intra-op thread(s) per slice on {width} core(s), \
                 {:.1}x. {verdict}",
                per_slice as f64 / width as f64,
            );
        }
        if service_config.threading.allow_spinning == Some(true) {
            // Worth its own line: only `inference_concurrency` inferences run at once, so
            // most pools are idle at any instant, and spinning makes an idle pool cost
            // exactly as much as a busy one.
            println!(
                "  note   : {sessions} spinning pools with {inference_concurrency} \
                 inference(s) in flight means most of them burn cores waiting; \
                 \"allow_spinning\": false is usually right with several sessions"
            );
        }
    }
    if compute_threads > num_cpus {
        println!(
            "  note   : {compute_threads} compute threads on {num_cpus} core(s) is \
             oversubscribed; lower \"threading.inference_concurrency\" if latency matters \
             more than requests per second"
        );
    } else if compute_threads * 2 <= num_cpus {
        // The other half of the same warning, and the one that is easy to miss: an
        // oversubscribed service shows up as latency, while an undersubscribed one just
        // quietly costs money. Peak throughput is bounded by these threads and not by the
        // cores, so a host this far from its budget cannot be saturated by any request rate
        // -- which is indistinguishable from a slow model unless the banner says so.
        println!(
            "  note   : {compute_threads} compute threads leaves {} of {num_cpus} core(s) \
             unusable; raise \"threading.inference_concurrency\" for throughput, or \
             \"threading.intra_op_threads\" to spend the cores on one request's latency",
            num_cpus - compute_threads,
        );
    }
    // Printed only when configured, for the same reason as affinity below: a run that
    // said nothing about spinning took the runtime's default, and claiming a value it
    // did not set would make the banner disagree with the configuration. It matters
    // enough to record because an execution provider with a thread-pool of its own --
    // XNNPACK, OpenVINO -- is measured quite differently with idle ORT threads spinning
    // against it.
    if let Some(spinning) = service_config.threading.allow_spinning {
        println!(
            "  spin   : idle intra-op and inter-op threads {}",
            if spinning { "spin" } else { "sleep" },
        );
    }
    // Placement is printed only when it was asked for, and says plainly when it was not
    // applied -- an affinity that silently did nothing looks like an affinity that did
    // not help, which is a much harder thing to notice.
    let worker_placement = service_config.threading.worker_cores();
    let compute_placement = service_config.threading.compute_cores();
    if !worker_placement.is_empty() || !compute_placement.is_empty() {
        let show = |cores: &[usize]| {
            if cores.is_empty() {
                "anywhere".to_owned()
            } else {
                format!("core(s) {}", affinity::describe(cores))
            }
        };
        println!(
            "  cores  : async on {}, compute on {}{}",
            show(&worker_placement),
            show(&compute_placement),
            if affinity::SUPPORTED {
                ""
            } else {
                " -- NOT APPLIED, this target has no thread affinity"
            }
        );
        let overlap: Vec<usize> = compute_placement
            .iter()
            .copied()
            .filter(|core| worker_placement.contains(core))
            .collect();
        if !overlap.is_empty() {
            println!(
                "  note   : the two pools share core(s) {}, so connection handling can \
                 preempt an inference; make the lists disjoint to prevent that",
                affinity::describe(&overlap)
            );
        }
    }

    // Both overrides fall through to the configured value when absent, which is what
    // lets a deployment state the listener once and a developer move it for one run.
    let port = args.port.unwrap_or(service_config.port_number);
    let bind_address = args.bind_address.unwrap_or(service_config.bind_address);
    let server_addr = SocketAddr::new(bind_address, port);

    // The admission limit is reported with the threading group above, where it is
    // configured, so this line is only about the socket.
    println!("hushar: listening on {server_addr}");
    // Kept as u16 as well: the HTTP/2 setting wants u32 and the tower layer wants usize,
    // and they must be the same number or the mismatch is the deadlock described below.
    let stream_limit = u32::from(connection_concurrency);
    let connection_concurrency = usize::from(connection_concurrency);
    let core_slices = Arc::new(core_slices);
    let hushar_service = HusharService {
        control,
        candidate,
        split,
        metrics_sink: Arc::clone(&metrics_sink),
        data_sink: Arc::clone(&data_sink),
        sampler,
        core_slices: Arc::clone(&core_slices),
    };
    let served = server_runtime.block_on(async {
        Server::builder()
            // Both, and both are load-bearing. `concurrency_limit_per_connection` is a
            // tower layer: it bounds how many requests reach the service, but HTTP/2 knows
            // nothing about it, so a client is free to open more streams than the layer
            // will dispatch. Those streams are accepted and then never served, and while
            // the connection task waits for the layer to be ready it stops driving the
            // connection at all -- so the responses already computed cannot be written
            // either, and the permits they hold are never released. The connection wedges
            // for good, both ends idle, with no error to see.
            //
            // `max_concurrent_streams` closes that by putting the same number in the HTTP/2
            // handshake, where it belongs. The client's own h2 layer then holds excess
            // requests instead of opening streams that cannot be served, which is ordinary
            // backpressure rather than a trap.
            .max_concurrent_streams(Some(stream_limit))
            .concurrency_limit_per_connection(connection_concurrency)
            .tcp_keepalive(Some(std::time::Duration::from_secs(30)))
            .tcp_nodelay(true)
            .add_service(HusharServer::new(hushar_service))
            .serve_with_shutdown(server_addr, shutdown_signal())
            .await
    });

    // What the sidecars used to do when their channels closed. The buffers hold the
    // most recent window of traffic and the spawned sends are detached tasks, so
    // shutting the runtime down without this would discard both -- and that window is
    // exactly the one worth having when a service has just stopped.
    //
    // Concurrently and with a deadline, because a supervisor that sent SIGTERM will
    // send SIGKILL after its own grace period: draining the two sinks in sequence
    // would let an unreachable CloudWatch spend that whole budget and take the
    // inference logs down with it. Losing one sink's buffer is better than losing
    // both.
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

    // Reported after draining rather than with `?` above, so a bind failure still
    // shuts down cleanly -- and reported at all, because exiting 0 on a port already
    // in use tells a supervisor the service is healthy.
    served.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
        format!("could not serve on {server_addr}: {e}").into()
    })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The replica and the cores must be the *same* index. Choosing them separately -- or on
    /// different threads -- pins a caller to one slice while it runs a pool pinned to
    /// another, which is slower than not pinning at all.
    #[test]
    fn the_replica_and_its_cores_come_from_one_index() {
        let slices: Vec<Arc<Vec<usize>>> = (0..4)
            .map(|i| Arc::new((i * 8..i * 8 + 8).collect::<Vec<usize>>()))
            .collect();

        let (index, cores) = replica_slot(4, &slices);
        assert_eq!(*cores, *slices[index], "cores belong to the chosen replica");

        // Stable for this thread, so pinning once is correct.
        for _ in 0..4 {
            let (again, same) = replica_slot(4, &slices);
            assert_eq!(again, index);
            assert_eq!(*same, *cores);
        }
    }

    /// One replica needs no bookkeeping and must not consume a round-robin slot, or a
    /// single-session deployment would drift across slices for no reason.
    #[test]
    fn a_single_replica_always_takes_the_first_slice() {
        let slices = vec![Arc::new(vec![0, 1, 2, 3])];
        for _ in 0..4 {
            let (index, cores) = replica_slot(1, &slices);
            assert_eq!(index, 0);
            assert_eq!(*cores, vec![0, 1, 2, 3]);
        }
    }

    /// Fewer slices than replicas must not panic: an empty list means "anywhere", which is
    /// what an unplaced deployment already does.
    #[test]
    fn a_missing_slice_places_the_replica_anywhere() {
        let (_, cores) = replica_slot(8, &[]);
        assert!(cores.is_empty());
    }

    /// One session is the old behaviour exactly: one slice, whatever `compute_cores` said,
    /// and no placement invented on its behalf.
    #[test]
    fn a_single_session_is_placed_exactly_as_before() {
        let threading = config::ThreadingConfig {
            intra_op_threads: 8,
            ..config::ThreadingConfig::default()
        };
        let (slices, options) = replica_placement(&threading, 64);
        assert_eq!(slices.len(), 1);
        assert!(
            slices[0].is_empty(),
            "no cores were configured, so none are chosen"
        );
        assert_eq!(options.len(), 1);
        assert!(
            options[0].is_empty(),
            "nothing to say about placement or spinning"
        );
    }

    /// Asking for several pools is asking for them to be placed, so with no `compute_cores`
    /// the whole machine is divided rather than leaving every pool unpinned and contending.
    #[test]
    fn several_sessions_divide_the_whole_machine_by_default() {
        let threading = config::ThreadingConfig {
            intra_op_threads: 4,
            sessions_per_model: 4,
            ..config::ThreadingConfig::default()
        };
        let (slices, options) = replica_placement(&threading, 16);
        assert_eq!(slices.len(), 4);
        assert_eq!(*slices[0], vec![0, 1, 2, 3]);
        assert_eq!(*slices[3], vec![12, 13, 14, 15]);

        // Each session is told about its own cores and no others, which is what keeps two
        // pools off the same core.
        let affinity = |o: &Vec<(String, String)>| {
            o.iter()
                .find(|(k, _)| k == "session.intra_op_thread_affinities")
                .map(|(_, v)| v.clone())
                .expect("an affinity entry")
        };
        // 1-based, as ONNX Runtime requires: slice 0 is cores 0-3, so its pool threads
        // land on cores 1,2,3 and are written 2,3,4.
        assert_eq!(affinity(&options[0]), "2;3;4");
        assert_eq!(affinity(&options[3]), "14;15;16");
    }

    /// A configured `compute_cores` is divided rather than ignored, so a host shared with
    /// something else still only uses the cores it was given.
    #[test]
    fn several_sessions_divide_the_configured_cores() {
        let threading = config::ThreadingConfig {
            intra_op_threads: 2,
            sessions_per_model: 2,
            compute_cores: Some("8-11".to_owned()),
            ..config::ThreadingConfig::default()
        };
        let (slices, _) = replica_placement(&threading, 64);
        assert_eq!(*slices[0], vec![8, 9]);
        assert_eq!(*slices[1], vec![10, 11]);
    }

    /// Spinning is a session setting, so every replica has to agree -- one pool spinning
    /// while its neighbours sleep would make the arms incomparable.
    #[test]
    fn spinning_reaches_every_session() {
        let threading = config::ThreadingConfig {
            intra_op_threads: 4,
            sessions_per_model: 3,
            allow_spinning: Some(false),
            ..config::ThreadingConfig::default()
        };
        let (_, options) = replica_placement(&threading, 24);
        assert_eq!(options.len(), 3);
        for one in &options {
            assert!(
                one.contains(&("session.intra_op.allow_spinning".to_owned(), "0".to_owned())),
                "every session is told, got {one:?}"
            );
        }
    }

    /// Threads take a replica each in turn and keep it, which is what lets a thread be
    /// pinned once rather than on every request.
    #[test]
    fn a_thread_keeps_the_replica_it_was_given() {
        let mine = this_thread_replica(4);
        for _ in 0..5 {
            assert_eq!(this_thread_replica(4), mine, "stable within a thread");
        }

        // Fresh threads keep taking the next one, so the blocking pool spreads itself.
        let seen: Vec<usize> = (0..8)
            .map(|_| {
                std::thread::spawn(|| this_thread_replica(4))
                    .join()
                    .unwrap()
            })
            .collect();
        let mut distinct = seen.clone();
        distinct.sort_unstable();
        distinct.dedup();
        assert_eq!(distinct.len(), 4, "all four replicas used, got {seen:?}");
    }

    /// A configuration that says nothing about placement must produce the session it
    /// produced before these settings existed — no keys, no behaviour change.
    #[test]
    fn a_default_threading_configuration_sets_no_session_options() {
        let threading = config::ThreadingConfig::default();
        assert!(
            session_options(&threading).is_empty(),
            "the default must leave every ONNX Runtime default in place"
        );
    }

    /// Cores without an extra thread to place is not an affinity: at `intra_op_threads`
    /// of 1 the calling thread is the whole pool, and it is pinned by us instead.
    #[test]
    fn cores_alone_do_not_produce_an_affinity_entry() {
        let json = r#"{"model_config_path": "/m.json",
                       "threading": {"intra_op_threads": 1, "compute_cores": "2-9"},
                       "inference_log": {"uri": "/logs"}}"#;
        let config = config::HusharServiceConfig::from_json(json).expect("valid");
        assert!(session_options(&config.threading).is_empty());
    }

    /// One entry per intra-op thread other than the caller, which is what ONNX Runtime
    /// validates the string against when the session is built.
    #[test]
    fn an_affinity_entry_is_emitted_for_the_extra_threads() {
        let json = r#"{"model_config_path": "/m.json",
                       "threading": {"intra_op_threads": 8, "compute_cores": "2-9"},
                       "inference_log": {"uri": "/logs"}}"#;
        let config = config::HusharServiceConfig::from_json(json).expect("valid");
        let options = session_options(&config.threading);
        assert_eq!(options.len(), 1, "affinity only: {options:?}");
        assert_eq!(options[0].0, "session.intra_op_thread_affinities");
        assert_eq!(
            options[0].1.split(';').count(),
            7,
            "8 threads leave 7 to place: {:?}",
            options[0].1
        );
    }

    /// Spinning is a tri-state, and the key takes "1"/"0" rather than a boolean. Both
    /// pools are set, since the inter-op pool spins on the same default.
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// The inference result must be able to cross a thread boundary.
    ///
    /// This is the property that lets the handler use `spawn_blocking`, and it is
    /// a compile-time one: before the inference path moved to an error type that
    /// is `Send + Sync`, this would not build. Keeping it as a test means reverting
    /// that error type breaks the build here, with an explanation attached, rather
    /// than at the call site with a bare trait-bound error.
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
                        outputs: Vec::new(),
                        logs: Vec::new(),
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

    /// CPU-bound work must not stop the async workers from making progress.
    ///
    /// The setup matters, and a first attempt at this test had none of it: the
    /// work under test has to run inside a **spawned task**, because that is where
    /// a request handler runs. Work done directly in `block_on` executes on the
    /// calling thread, so blocking there never touches a worker and the test
    /// passes either way — proving nothing.
    ///
    /// With one worker thread, a ticker task and the handler-shaped task compete
    /// for it. If the handler blocks that worker inline, the ticker cannot advance;
    /// handing the work to `spawn_blocking` frees the worker and it does. Verified
    /// to fail when the `spawn_blocking` is removed.
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

    /// Sampling has to be decided once per request and used for both halves of it.
    ///
    /// The bug this guards against is subtle and would be silent: asking the sampler
    /// again when the response is ready would score with the log switched on and then
    /// decide not to record it — or the reverse, recording a batch whose rows were
    /// never collected, producing an empty log entry per request. One call, one
    /// decision.
    #[test]
    fn the_sampling_decision_is_taken_once_per_request() {
        let sampler = Sampler::new(0.5);

        // Two calls per "request" is what the bug looks like: the same request
        // disagrees with itself.
        let first = sampler.sample();
        let second = sampler.sample();
        assert_ne!(
            first, second,
            "consecutive calls alternate at one in two, so a second call for the \
             same request would contradict the first"
        );
    }
}
