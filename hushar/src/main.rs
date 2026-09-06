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

pub(crate) mod cli;
pub(crate) mod config;
pub(crate) mod inference;
pub(crate) mod io;

/// Everything a request handler needs, built once at startup.
///
/// Fields:
/// - `backend` — the engine, shared across request threads.
/// - `metrics_sink`, `data_sink` — where timings and inference logs accumulate. Held
///   directly rather than reached through a channel: there are no sidecars, and a
///   handler's whole interaction with either is one non-blocking call.
/// - `sampler` — what fraction of requests is logged, decided before scoring so an
///   unsampled request never copies its features.
/// - `model_id` — recorded on every inference log batch, so logs join to a model
///   version.
/// - `input_builder` — how a feature payload becomes the model's inputs.
/// - `fixed_batch_size` — the batch size the model's graph is pinned to, when the
///   model configuration declares one. Requests of any size are still served: rows
///   are cut into batches of this many and the last one is padded. `None` runs each
///   request as a single batch.
#[derive(Debug)]
pub(crate) struct HusharService {
    backend: Arc<dyn InferenceBackend>,
    metrics_sink: Arc<dyn MetricsSink>,
    data_sink: Arc<dyn DataSink>,
    sampler: Sampler,
    model_id: String,
    input_builder: Arc<inference::input_builder::InputBuilder>,
    fixed_batch_size: Option<usize>,
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

        let backend = Arc::clone(&self.backend);
        let input_builder = Arc::clone(&self.input_builder);
        let fixed_batch_size = self.fixed_batch_size;
        let logging = if self.sampler.sample() {
            FeatureLogging::Record
        } else {
            FeatureLogging::Skip
        };

        let scored = tokio::task::spawn_blocking(move || {
            inference::scoring::score_features(
                backend.as_ref(),
                rows,
                &input_builder,
                fixed_batch_size,
                logging,
            )
        })
        .await
        .map_err(|e| Status::internal(format!("inference task failed to complete: {e}")))?;

        match scored {
            Ok(batch) => {
                self.metrics_sink.record(&batch.timings);

                let inference_response = InferenceResponse {
                    request_id: request_id.clone(),
                    outputs: batch.outputs,
                };
                if logging == FeatureLogging::Record {
                    self.data_sink.record(InferenceLogBatch {
                        request_id,
                        model_id: self.model_id.clone(),
                        inference_log_rows: batch.logs,
                    });
                }

                Ok(Response::new(inference_response))
            }
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }
}

/// Chooses where metrics go, from the command line.
///
/// A CloudWatch namespace selects CloudWatch. Without one, timings are summarised
/// to stderr, which is what a local run or a test wants and needs no credentials.
///
/// Infallible: the CloudWatch client is built from the environment and does not
/// contact anything, so there is nothing here to fail. A missing permission
/// surfaces later, on the first publish, as a reported error rather than a refusal
/// to serve traffic -- which is the right trade for observability.
async fn build_metrics_sink(args: &cli::Cli) -> Arc<dyn MetricsSink> {
    match args.cloudwatch_namespace.as_deref() {
        Some(namespace) => {
            let config = aws_config::load_from_env().await;
            Arc::new(io::metrics::CloudWatchMetrics::new(
                Arc::new(aws_sdk_cloudwatch::Client::new(&config)),
                namespace,
                args.metrics_batch_size,
                args.max_sends_in_flight,
            ))
        }
        None => Arc::new(io::metrics::StderrMetrics::new(args.metrics_batch_size)),
    }
}

/// Chooses where inference logs go, from the destination URI.
///
/// `kinesis://stream` selects the stream; anything else is a blob-store prefix. The
/// same shape as `build_metrics_sink`, and for the same reason: the decision is made
/// once, here, and the request path never learns the answer.
async fn build_data_sink(
    args: &cli::Cli,
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
                args.data_batch_size,
                args.max_sends_in_flight,
            ))
        }
        DataDestination::Kinesis { stream } => {
            let config = aws_config::load_from_env().await;
            Arc::new(io::data_sink::KinesisDataSink::new(
                Arc::new(aws_sdk_kinesis::Client::new(&config)),
                stream,
                args.data_batch_size,
                args.max_sends_in_flight,
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
/// 2. Parse both URIs, before any runtime exists.
/// 3. On a single-threaded runtime, read the configuration, load the model, check the
///    vectorisation configuration against the model's signature, and choose the
///    metrics and log destinations. All of this can await; none of it needs workers.
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
/// from where they meant. `--port` overrides the configured port, which lets two
/// instances share a host.
pub fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = cli::Cli::parse();
    let num_cpus = std::thread::available_parallelism()?.get();

    let config_location = io::Location::parse(&args.config_uri)?;
    let log_destination = DataDestination::parse(&args.inference_log_uri)?;
    let stores = Arc::new(BlobStores::new());

    let config_runtime = tokio::runtime::Builder::new_current_thread()
        .thread_name("hushar-config-loader")
        .enable_all()
        .build()?;

    let (
        service_config,
        model_id,
        backend,
        metrics_sink,
        data_sink,
        sampler,
        input_builder,
        fixed_batch_size,
    ) = config_runtime.block_on(async {
        let config_store = stores
            .open(&config_location)
            .await
            .unwrap_or_else(|e| panic!("cannot reach {}: {e}", config_location.uri()));
        let service_config = config::HusharServiceConfig::from_json(
            &config_store.get_string(&config_location).await.unwrap(),
        )
        .unwrap();

        let (model_config_location, model_config_store) = stores
            .resolve(&service_config.model_config_path)
            .await
            .unwrap();
        let model_config = config::ModelConfig::from_json(
            &model_config_store
                .get_string(&model_config_location)
                .await
                .unwrap(),
        )
        .unwrap();
        let model_id = model_config.model_id.clone();
        let fixed_batch_size = model_config.fixed_batch_size;

        let (model_location, model_store) = stores.resolve(&model_config.model_path).await.unwrap();
        let backend = io::load_onnx_model(
            &model_store.get(&model_location).await.unwrap(),
            &model_config.execution_provider,
            model_config.intra_op_threads,
            fixed_batch_size,
        )
        .unwrap_or_else(|e| {
            panic!(
                "could not load model {} on execution provider {:?}: {e}",
                model_config.model_path, model_config.execution_provider
            )
        });

        let input_builder = inference::input_builder::InputBuilder::resolve(
            model_config.vectorization_config,
            backend.inputs(),
        )
        .unwrap_or_else(|e| panic!("the model configuration does not fit the model: {e}"));

        let describe = |specs: &[crate::inference::batch::IoSpec]| {
            specs
                .iter()
                .map(crate::inference::batch::IoSpec::describe)
                .collect::<Vec<_>>()
                .join(", ")
        };
        println!(
            "hushar: model {} loaded on {}\n  inputs   : {}\n  outputs  : {}\n  features : {}",
            model_id,
            backend.name(),
            describe(backend.inputs()),
            describe(backend.outputs()),
            input_builder.describe(),
        );
        // Printed only when it changes how a request is executed, so a normal
        // deployment's banner is unchanged and a pinned one cannot be mistaken for it.
        if let Some(rows) = fixed_batch_size {
            println!(
                "  batch    : model pinned to {rows} row(s); requests are split into \
                 batches of {rows} and the last is padded"
            );
        }

        let metrics_sink = build_metrics_sink(&args).await;
        let data_sink = build_data_sink(&args, &log_destination, &stores).await;
        let sampler = Sampler::new(args.data_sample_rate);

        // The effective batch sizes rather than the configured ones: CloudWatch and
        // Kinesis each cap what one call may carry, and a number that was reduced is
        // otherwise invisible until someone wonders why the writes are so frequent.
        println!(
            "hushar: metrics -> {} [every {} batches]",
            metrics_sink.backend(),
            metrics_sink.batch_size()
        );
        println!(
            "hushar: inference logs -> {} [{}, every {} batches, {} in flight, logging {}]",
            log_destination.uri(),
            data_sink.backend(),
            data_sink.batch_size(),
            args.max_sends_in_flight,
            sampler.describe(),
        );

        (
            service_config,
            model_id,
            backend,
            metrics_sink,
            data_sink,
            sampler,
            input_builder,
            fixed_batch_size,
        )
    });

    config_runtime.shutdown_background();

    let server_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(num_cpus)
        .max_blocking_threads(num_cpus)
        .thread_name("hushar-server-worker")
        .enable_all()
        .build()?;

    let port = args.port.unwrap_or(service_config.port_number);
    let server_addr = SocketAddr::new(args.bind_address, port);
    println!("hushar: listening on {server_addr}");
    let connection_concurrency = service_config.connection_concurrency as usize;
    let hushar_service = HusharService {
        backend,
        metrics_sink: Arc::clone(&metrics_sink),
        data_sink: Arc::clone(&data_sink),
        sampler,
        model_id,
        input_builder: Arc::new(input_builder),
        fixed_batch_size,
    };
    let served = server_runtime.block_on(async {
        Server::builder()
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
