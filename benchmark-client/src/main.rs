// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Offers a fixed request rate at a running server and reports what came back.
//!
//! Percentiles go to stdout and need no credentials. CloudWatch is opt-in with
//! `--cloudwatch-namespace`.
//!
//! Features are generated from the model configuration the server was started with, so
//! the values sent match the feature names and transformations the server expects.
//!
//! A configuration that declares named `model_inputs` feeds them verbatim and so names
//! no transformation to generate from. `--feature-spec` supplies the values for those:
//! a range, a width, or a word list per feature.
//!
//! ```bash
//! cargo run --release -p benchmark-client -- \
//!   --server http://127.0.0.1:8279 --tps 500 --duration-secs 30 \
//!   --model-config benchmark-data/generated/bench_model_config.json
//! ```

use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use benchmark_client::hushar_proto::Feature;
use benchmark_client::hushar_proto::{InferenceRequest, hushar_client::HusharClient};
use benchmark_client::proto_columns;
use clap::Parser;
use rand::RngExt;
use tokio::sync::Semaphore;

/// How much load to offer, for how long, and against which configuration.
#[derive(Debug, Parser)]
#[command(about = "Offers a fixed request rate at a hushar server")]
struct Args {
    /// Address of the running server.
    #[arg(long, env = "BENCH_SERVER", default_value = "http://127.0.0.1:8279")]
    server: String,

    /// Requests offered per second.
    ///
    /// The rate the client *attempts*, not the rate it achieves. A shortfall is
    /// reported as shed rather than hidden.
    #[arg(long, env = "BENCH_TPS", default_value_t = 500)]
    tps: u32,

    /// How long to measure for, after the warmup.
    #[arg(long, env = "BENCH_DURATION_SECS", default_value_t = 30)]
    duration_secs: u64,

    /// Time to run before measuring, discarding the results.
    ///
    /// The first requests to a freshly loaded model are not representative: an
    /// execution provider may compile the graph on first use, and nothing is in cache
    /// yet.
    #[arg(long, env = "BENCH_WARMUP_SECS", default_value_t = 10)]
    warmup_secs: u64,

    /// Rows in one request.
    #[arg(long, env = "BENCH_ROWS", default_value_t = 100)]
    rows: usize,

    /// Requests allowed in flight at once. Defaults to four seconds' worth.
    ///
    /// In flight has to cover `tps × latency`, so a number equal to `tps` permits only
    /// a one-second response. Left unset it is `tps × 4`, with a floor of 8. Set it to
    /// make the cap deliberate.
    #[arg(long, env = "BENCH_CONCURRENCY")]
    concurrency: Option<usize>,

    /// Ask the server for one model by name, instead of letting it choose.
    ///
    /// Unset leaves the choice to the server, which splits unnamed requests between a
    /// roll-out's two arms by the percentage in its configuration.
    #[arg(long, env = "BENCH_MODEL_ID")]
    model_id: Option<String>,

    /// The model configuration the server was started with.
    #[arg(long, env = "BENCH_MODEL_CONFIG")]
    model_config: PathBuf,

    /// What to send for each feature, for a model that does its own featurization.
    ///
    /// Required exactly when the model configuration declares named `model_inputs`,
    /// which are fed verbatim and so name no transformation. Ignored otherwise, where
    /// the configuration is the better authority.
    #[arg(long, env = "BENCH_FEATURE_SPEC")]
    feature_spec: Option<PathBuf>,

    /// Send results to CloudWatch under this namespace as well as stdout.
    #[arg(long, env = "BENCH_CLOUDWATCH_NAMESPACE")]
    cloudwatch_namespace: Option<String>,

    /// Printed with the results, to tell one run from another.
    #[arg(long, env = "BENCH_LABEL")]
    label: Option<String>,
}

/// How to produce a plausible value for one feature.
///
/// Variants:
/// - `Choice` — one of a fixed set of words, for a category or a vocabulary entry.
/// - `Float`, `Double` — a number in range, at the precision the transformation wants.
///   The two are separate because a 64-bit scaler refuses a float and vice versa.
/// - `FloatVector` — that many numbers, for a feature passed through as an array.
#[derive(Debug)]
enum Generator {
    Choice(Vec<String>),
    Float(f32, f32),
    Double(f64, f64),
    FloatVector(usize),
}

impl Generator {
    /// One whole feature: this feature's values for every row, back to back, with the
    /// width that says where each row ends.
    ///
    /// A feature `width` wide over `rows` rows carries `rows × width` values. The name is
    /// not here: it is the key this feature is filed under.
    fn feature(&self, rows: usize, rng: &mut impl RngExt) -> Feature {
        match self {
            Self::Choice(words) => proto_columns::strings(
                1,
                (0..rows)
                    .map(|_| words[rng.random_range(0..words.len())].clone())
                    .collect(),
            ),
            Self::Float(low, high) => proto_columns::floats(
                1,
                (0..rows).map(|_| rng.random_range(*low..*high)).collect(),
            ),
            Self::Double(low, high) => proto_columns::doubles(
                1,
                (0..rows).map(|_| rng.random_range(*low..*high)).collect(),
            ),
            Self::FloatVector(width) => proto_columns::floats(
                *width,
                (0..rows * width)
                    .map(|_| rng.random_range(-3.0..3.0))
                    .collect(),
            ),
        }
    }
}

/// Reads the model configuration into one generator per feature the server expects.
///
/// The transformation's own type decides the value: a one-hot wants a category, a
/// 64-bit scaler wants a double, an identity wants as many numbers as its default is
/// wide, and a standardization spans three deviations either side of its mean so the
/// scaled value lands in the range the model was trained to see.
///
/// # Arguments
///
/// * `config` - the model configuration the server was started with
/// * `feature_spec` - what to send per feature, read only when the configuration
///   declares named `model_inputs` and so transforms nothing
///
/// # Errors
///
/// Returns an error when the configuration has no `vectorization_config`, when a named
/// feature has no transformation or an unknown one, or when the configuration declares
/// `model_inputs` and no `feature_spec` was given.
fn generators(
    config: &serde_json::Value,
    feature_spec: Option<&serde_json::Value>,
) -> Result<Vec<(String, Generator)>, String> {
    let Some(vectorization) = config.get("vectorization_config") else {
        return Err(
            "the model configuration has no \"vectorization_config\", which hushar \
             requires -- it would refuse to start on this file. Declare \"model_inputs\" \
             to feed one input per feature, or \"feature_order\" to concatenate them."
                .to_owned(),
        );
    };
    let transformations = vectorization
        .get("feature_transformations")
        .and_then(|t| t.as_object())
        .cloned()
        .unwrap_or_default();

    let number = |spec: &serde_json::Value, key: &str, fallback: f64| {
        spec.get(key).and_then(|v| v.as_f64()).unwrap_or(fallback)
    };

    let from_transformation = |name: &str| -> Result<Generator, String> {
        let spec = transformations
            .get(name)
            .ok_or_else(|| format!("feature {name:?} has no transformation"))?;
        let kind = spec
            .get("type")
            .and_then(|t| t.as_str())
            .ok_or_else(|| format!("transformation for {name:?} has no type"))?;
        Ok(match kind {
            "embedding" => Generator::Choice(
                spec.get("embeddings")
                    .and_then(|e| e.as_object())
                    .map(|e| e.keys().cloned().collect())
                    .ok_or_else(|| format!("embedding {name:?} has no table"))?,
            ),
            "one_hot_encoding" => Generator::Choice(
                spec.get("categories")
                    .and_then(|c| c.as_array())
                    .map(|c| {
                        c.iter()
                            .filter_map(|v| v.as_str().map(str::to_owned))
                            .collect()
                    })
                    .ok_or_else(|| format!("one-hot {name:?} has no categories"))?,
            ),
            "identity" => {
                let width = spec
                    .get("default_val")
                    .and_then(|d| d.as_array())
                    .map_or(1, |d| d.len());
                if width == 1 {
                    Generator::Float(-3.0, 3.0)
                } else {
                    Generator::FloatVector(width)
                }
            }
            "min_max_scaling32" => Generator::Float(
                number(spec, "min", 0.0) as f32,
                number(spec, "max", 1.0) as f32,
            ),
            "min_max_scaling64" => {
                Generator::Double(number(spec, "min", 0.0), number(spec, "max", 1.0))
            }
            "standardization32" => {
                let (mean, deviation) = (number(spec, "mean", 0.0), number(spec, "std_dev", 1.0));
                Generator::Float(
                    (mean - 3.0 * deviation) as f32,
                    (mean + 3.0 * deviation) as f32,
                )
            }
            "standardization64" => {
                let (mean, deviation) = (number(spec, "mean", 0.0), number(spec, "std_dev", 1.0));
                Generator::Double(mean - 3.0 * deviation, mean + 3.0 * deviation)
            }
            other => return Err(format!("unknown transformation {other:?} for {name:?}")),
        })
    };

    if let Some(order) = vectorization
        .get("feature_order")
        .and_then(|o| o.as_array())
    {
        return order
            .iter()
            .filter_map(|v| v.as_str())
            .map(|name| from_transformation(name).map(|g| (name.to_owned(), g)))
            .collect();
    }

    let inputs = vectorization
        .get("model_inputs")
        .and_then(|i| i.as_object())
        .ok_or("the configuration declares neither feature_order nor model_inputs")?;
    match feature_spec {
        Some(spec) => from_feature_spec(spec),
        None => Err(format!(
            "the model configuration declares {} named inputs, which are fed verbatim, \
             so it says nothing about what values to send. Pass --feature-spec with the \
             file written beside it, such as \
             benchmark-data/generated/bench_named_feature_spec.json",
            inputs.len()
        )),
    }
}

/// Reads a feature specification into one generator per feature.
///
/// One entry per feature, and each says only what a generator needs:
///
/// ```json
/// {
///   "mm32_price": { "kind": "float",  "low": 0.0, "high": 10000.0 },
///   "sd64_income": { "kind": "double", "low": -30000.0, "high": 180000.0 },
///   "id_vector":  { "kind": "float_array", "width": 8 },
///   "oh_color":   { "kind": "choice", "words": ["oh_color_c_0000"] }
/// }
/// ```
///
/// `float` against `double` is not a precision preference: it is which protobuf field
/// the value travels in, and an FP64 model input takes a double and refuses a float.
///
/// # Errors
///
/// Returns an error when the specification is not a JSON object, or when an entry has
/// no `kind`, an unknown `kind`, or is missing a bound, a `width`, or a non-empty
/// `words` list.
fn from_feature_spec(spec: &serde_json::Value) -> Result<Vec<(String, Generator)>, String> {
    let features = spec
        .as_object()
        .ok_or("the feature specification is not a JSON object of feature -> spec")?;

    features
        .iter()
        .map(|(name, entry)| {
            let kind = entry
                .get("kind")
                .and_then(|k| k.as_str())
                .ok_or_else(|| format!("feature {name:?} has no kind"))?;
            let bound = |key: &str| {
                entry
                    .get(key)
                    .and_then(|v| v.as_f64())
                    .ok_or_else(|| format!("feature {name:?} is {kind} but has no {key}"))
            };
            let generator = match kind {
                "float" => Generator::Float(bound("low")? as f32, bound("high")? as f32),
                "double" => Generator::Double(bound("low")?, bound("high")?),
                "float_array" => Generator::FloatVector(
                    entry
                        .get("width")
                        .and_then(serde_json::Value::as_u64)
                        .ok_or_else(|| format!("feature {name:?} is an array with no width"))?
                        as usize,
                ),
                "choice" => Generator::Choice(
                    entry
                        .get("words")
                        .and_then(|w| w.as_array())
                        .map(|w| {
                            w.iter()
                                .filter_map(|v| v.as_str().map(str::to_owned))
                                .collect::<Vec<_>>()
                        })
                        .filter(|w| !w.is_empty())
                        .ok_or_else(|| format!("feature {name:?} is a choice with no words"))?,
                ),
                other => return Err(format!("unknown kind {other:?} for feature {name:?}")),
            };
            Ok((name.clone(), generator))
        })
        .collect()
}

/// One request, with fresh values for every feature of every row.
///
/// `model_id` empty leaves the choice to the server's configured split.
///
/// # Panics
///
/// Panics if the generated features do not cover the same rows, which would mean two
/// generators disagreed about the row count.
fn request(
    id: u64,
    rows: usize,
    generators: &[(String, Generator)],
    model_id: &str,
) -> InferenceRequest {
    let mut rng = rand::rng();
    let features: Vec<(String, Feature)> = generators
        .iter()
        .map(|(name, generator)| (name.clone(), generator.feature(rows, &mut rng)))
        .collect();
    InferenceRequest {
        request_id: format!("bench-{id}"),
        model_id: model_id.to_owned(),
        row_ids: (0..rows).map(|row| format!("row-{row}")).collect(),
        features: proto_columns::encode(features).expect("every generator covers the same rows"),
    }
}

/// Latency percentiles over the measured window.
#[derive(Debug)]
struct Percentiles {
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
    mean: f64,
}

impl Percentiles {
    /// Sorts `latencies` in place and reads the percentile positions off.
    ///
    /// # Returns
    ///
    /// `None` when nothing was measured, which is distinct from zero latency.
    fn of(latencies: &mut [f64]) -> Option<Self> {
        if latencies.is_empty() {
            return None;
        }
        latencies.sort_by(f64::total_cmp);
        let at =
            |q: f64| latencies[((latencies.len() as f64 * q) as usize).min(latencies.len() - 1)];
        Some(Self {
            p50: at(0.50),
            p95: at(0.95),
            p99: at(0.99),
            max: latencies[latencies.len() - 1],
            mean: latencies.iter().sum::<f64>() / latencies.len() as f64,
        })
    }
}

/// Offers the configured rate for the warmup and then the measured window, and reports.
///
/// The generator is open loop: a request that finds no permit free is counted as shed
/// rather than queued. A response whose row count does not match the request counts as
/// a failure. Only requests begun after the warmup are measured.
///
/// # Errors
///
/// Returns an error when the configuration files cannot be read or parsed, when the
/// server cannot be reached, or when the run produced no measurements.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&args.model_config)?)
            .map_err(|e| format!("{}: {e}", args.model_config.display()))?;
    let feature_spec: Option<serde_json::Value> = match &args.feature_spec {
        Some(path) => Some(
            serde_json::from_str(&std::fs::read_to_string(path)?)
                .map_err(|e| format!("{}: {e}", path.display()))?,
        ),
        None => None,
    };
    let generators =
        Arc::new(generators(&config, feature_spec.as_ref()).map_err(|e| e.to_string())?);

    let model_id = config
        .get("model_id")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    let provider = config
        .get("execution_provider")
        .and_then(|v| v.as_str())
        .unwrap_or("?");
    println!(
        "hushar benchmark{}",
        args.label
            .as_deref()
            .map_or(String::new(), |l| format!(" [{l}]"))
    );
    println!("  server    : {}", args.server);
    println!("  model     : {model_id} on {provider}");
    println!("  features  : {} per row", generators.len());
    println!(
        "  asking    : {}",
        args.model_id
            .as_deref()
            .map_or("whichever the server's split chooses", |m| m)
    );
    let concurrency = args
        .concurrency
        .unwrap_or_else(|| ((args.tps as usize).saturating_mul(4)).max(8));
    let sizing = if args.concurrency.is_some() {
        "pinned"
    } else {
        "4s of latency"
    };
    println!(
        "  offering  : {} req/s x {} row(s), {} in flight ({}), {}s warmup + {}s measured",
        args.tps, args.rows, concurrency, sizing, args.warmup_secs, args.duration_secs
    );

    let client = HusharClient::connect(args.server.clone()).await?;
    let requested_model: Arc<str> = Arc::from(args.model_id.clone().unwrap_or_default());
    let permits = Arc::new(Semaphore::new(concurrency));
    let latencies = Arc::new(std::sync::Mutex::new(Vec::<(String, f64)>::new()));
    let (sent, failed, shed, wrong_rows) = (
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
    );

    let mut ticker = tokio::time::interval(Duration::from_millis(1));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Burst);
    let per_tick = f64::from(args.tps) / 1000.0;
    let warmup = Duration::from_secs(args.warmup_secs);
    let total = warmup + Duration::from_secs(args.duration_secs);

    let start = Instant::now();
    let mut credit = 0.0;
    let mut id = 0u64;
    let mut announced_warmup = args.warmup_secs == 0;
    while start.elapsed() < total {
        ticker.tick().await;
        if !announced_warmup && start.elapsed() >= warmup {
            println!("  warmup done, measuring");
            latencies.lock().expect("latencies").clear();
            announced_warmup = true;
        }
        credit += per_tick;
        while credit >= 1.0 {
            credit -= 1.0;
            id += 1;
            let Ok(permit) = Arc::clone(&permits).try_acquire_owned() else {
                shed.fetch_add(1, Ordering::Relaxed);
                continue;
            };
            let (mut client, generators) = (client.clone(), Arc::clone(&generators));
            let model_for_task = Arc::clone(&requested_model);
            let (latencies, sent, failed, wrong_rows) = (
                Arc::clone(&latencies),
                Arc::clone(&sent),
                Arc::clone(&failed),
                Arc::clone(&wrong_rows),
            );
            let rows_per_request = args.rows;
            let measuring_from = start + warmup;
            tokio::spawn(async move {
                let payload = request(id, args.rows, &generators, &model_for_task);
                let began = Instant::now();
                let outcome = client.inference_service(payload).await;
                let elapsed = began.elapsed().as_secs_f64() * 1000.0;
                match outcome {
                    Err(_) => {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                    Ok(response) => {
                        let response = response.into_inner();
                        if response.row_ids.len() != rows_per_request {
                            failed.fetch_add(1, Ordering::Relaxed);
                            wrong_rows.fetch_add(1, Ordering::Relaxed);
                            drop(permit);
                            return;
                        }
                        sent.fetch_add(1, Ordering::Relaxed);
                        if began >= measuring_from {
                            let served_by = response.model_id;
                            latencies
                                .lock()
                                .expect("latencies")
                                .push((served_by, elapsed));
                        }
                    }
                }
                drop(permit);
            });
        }
    }

    let _ = permits.acquire_many(concurrency as u32).await;

    let served = std::mem::take(&mut *latencies.lock().expect("latencies"));
    let mut per_arm: std::collections::BTreeMap<String, Vec<f64>> = Default::default();
    for (arm, elapsed) in &served {
        per_arm.entry(arm.clone()).or_default().push(*elapsed);
    }
    let mut measured: Vec<f64> = served.iter().map(|(_, elapsed)| *elapsed).collect();
    let counts = (
        sent.load(Ordering::Relaxed),
        failed.load(Ordering::Relaxed),
        shed.load(Ordering::Relaxed),
    );
    report(
        &args,
        &mut measured,
        &mut per_arm,
        counts,
        model_id,
        provider,
        concurrency,
    )
    .await
}

/// Prints the result, and sends it to CloudWatch when a namespace was given.
///
/// The per-model table appears only when more than one model answered; its share is the
/// split as the caller observed it, which should agree with the configured percentage.
///
/// # Errors
///
/// Returns an error when nothing was measured, or when the CloudWatch call fails.
async fn report(
    args: &Args,
    measured: &mut [f64],
    per_arm: &mut std::collections::BTreeMap<String, Vec<f64>>,
    counts: (u64, u64, u64),
    model_id: &str,
    provider: &str,
    concurrency: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let (succeeded, failed, shed) = counts;
    let offered = u64::from(args.tps) * (args.warmup_secs + args.duration_secs);
    let Some(p) = Percentiles::of(measured) else {
        println!("\nno successful requests: {failed} failed, {shed} shed");
        return Err("benchmark produced no measurements".into());
    };
    let achieved = measured.len() as f64 / args.duration_secs as f64;

    println!("\n  {model_id} on {provider}");
    println!("  ---------------------------------------------");
    println!(
        "  achieved      : {achieved:>9.1} req/s   ({:.1} rows/s)",
        achieved * args.rows as f64
    );
    println!("  offered       : {:>9} req/s", args.tps);
    println!(
        "  measured      : {:>9} requests  (of {succeeded} that succeeded overall)",
        measured.len()
    );
    println!("  latency  p50  : {:>9.2} ms", p.p50);
    println!("           p95  : {:>9.2} ms", p.p95);
    println!("           p99  : {:>9.2} ms", p.p99);
    println!("           max  : {:>9.2} ms", p.max);
    println!("           mean : {:>9.2} ms", p.mean);
    if per_arm.len() > 1 {
        println!();
        println!(
            "  by model      : {:<22}{:>6}{:>9}{:>9}{:>9}{:>9}{:>11}",
            "", "share", "req/s", "p50", "p95", "p99", "requests"
        );
        let total = measured.len() as f64;
        for (arm, samples) in per_arm.iter_mut() {
            let Some(q) = Percentiles::of(samples) else {
                continue;
            };
            println!(
                "    {arm:<24}{:>6.1}% {:>8.1} {:>8.2} {:>8.2} {:>8.2} {:>10}",
                100.0 * samples.len() as f64 / total,
                samples.len() as f64 / args.duration_secs as f64,
                q.p50,
                q.p95,
                q.p99,
                samples.len(),
            );
        }
    }
    println!("  failed        : {failed:>9}");
    println!(
        "  shed          : {shed:>9}  (offered but never sent: {} in flight was the cap)",
        concurrency
    );
    if shed > offered / 100 {
        println!(
            "\n  NOTE: {:.0}% of the offered rate was shed, so {} req/s is above what this\n  \
             configuration sustains. The latencies above are for the requests that ran.",
            100.0 * shed as f64 / offered as f64,
            args.tps
        );
    }

    if let Some(namespace) = &args.cloudwatch_namespace {
        emit_cloudwatch(namespace, &p, achieved).await?;
        println!("\n  also sent to CloudWatch namespace {namespace}");
    }
    Ok(())
}

/// Sends the percentiles and the achieved rate to CloudWatch under `namespace`.
///
/// Credentials and region come from the environment.
///
/// # Errors
///
/// Returns an error when the `PutMetricData` call fails.
async fn emit_cloudwatch(
    namespace: &str,
    percentiles: &Percentiles,
    achieved: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    use aws_sdk_cloudwatch::{Client, types::*};

    let client = Client::new(&aws_config::load_from_env().await);
    let datum = |name: &str, value: f64, unit: StandardUnit| {
        MetricDatum::builder()
            .metric_name(name)
            .value(value)
            .unit(unit)
            .build()
    };
    client
        .put_metric_data()
        .namespace(namespace)
        .metric_data(datum(
            "Latency_P50",
            percentiles.p50,
            StandardUnit::Milliseconds,
        ))
        .metric_data(datum(
            "Latency_P95",
            percentiles.p95,
            StandardUnit::Milliseconds,
        ))
        .metric_data(datum(
            "Latency_P99",
            percentiles.p99,
            StandardUnit::Milliseconds,
        ))
        .metric_data(datum(
            "Latency_Avg",
            percentiles.mean,
            StandardUnit::Milliseconds,
        ))
        .metric_data(datum("Throughput", achieved, StandardUnit::CountSecond))
        .send()
        .await?;
    Ok(())
}
