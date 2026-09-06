// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Offers a fixed request rate at a running server and reports what came back.
//!
//! Local by default -- percentiles on stdout, no credentials needed. CloudWatch is
//! opt-in with `--cloudwatch-namespace`, the same way the server treats its own
//! metrics, so a laptop run needs nothing set up.
//!
//! Features are generated from the model configuration the server was started with,
//! rather than hard-coded here. A client that invents its own feature names sends
//! values that match nothing, every feature falls back to its default, and the
//! benchmark measures a request path it will never see in production.
//!
//! ```bash
//! cargo run --release -p benchmark-client -- \
//!   --server http://127.0.0.1:8279 --tps 500 --duration-secs 30 \
//!   --model-config benchmark-data/generated/bench_model_config.json
//! ```

use std::{
    collections::HashMap,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant},
};

use benchmark_client::hushar_proto::{
    DataType, FloatArray, InferenceRequest, InputRow, data_type::DataType as Value,
    hushar_client::HusharClient,
};
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
    /// This is the rate the client *attempts*, not the rate it achieves. When the
    /// server cannot keep up the shortfall is reported rather than hidden, because a
    /// benchmark that quietly slows its own generator measures the generator.
    #[arg(long, env = "BENCH_TPS", default_value_t = 500)]
    tps: u32,

    /// How long to measure for, after the warmup.
    #[arg(long, env = "BENCH_DURATION_SECS", default_value_t = 30)]
    duration_secs: u64,

    /// Time to run before measuring, discarding the results.
    ///
    /// The first requests to a freshly loaded model are not representative: an
    /// execution provider may compile the graph on first use, and nothing is in cache
    /// yet. CoreML in particular pays for its first inference.
    #[arg(long, env = "BENCH_WARMUP_SECS", default_value_t = 10)]
    warmup_secs: u64,

    /// Rows in one request.
    #[arg(long, env = "BENCH_ROWS", default_value_t = 1)]
    rows: usize,

    /// Requests allowed in flight at once.
    #[arg(long, env = "BENCH_CONCURRENCY", default_value_t = 64)]
    concurrency: usize,

    /// The model configuration the server was started with.
    #[arg(long, env = "BENCH_MODEL_CONFIG")]
    model_config: PathBuf,

    /// Vocabulary for text inputs the model embeds itself.
    ///
    /// Only needed for a configuration with string inputs. Without it those features
    /// are sent as words the model has never seen, which costs the same to look up but
    /// exercises only the out-of-vocabulary row.
    #[arg(long, env = "BENCH_VOCABULARY")]
    vocabulary: Option<PathBuf>,

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
    fn sample(&self, rng: &mut impl RngExt) -> DataType {
        let value = match self {
            Self::Choice(words) => {
                Value::StringValue(words[rng.random_range(0..words.len())].clone())
            }
            Self::Float(low, high) => Value::FloatValue(rng.random_range(*low..*high)),
            Self::Double(low, high) => Value::DoubleValue(rng.random_range(*low..*high)),
            Self::FloatVector(width) => Value::FloatArray(FloatArray {
                values: (0..*width).map(|_| rng.random_range(-3.0..3.0)).collect(),
            }),
        };
        DataType {
            data_type: Some(value),
        }
    }
}

/// Reads the model configuration into one generator per feature the server expects.
///
/// The transformation's own type decides the value: a one-hot wants a category, a
/// 64-bit scaler wants a double, an identity wants as many numbers as its default is
/// wide. A named string input has no transformation at all -- it reaches the model
/// verbatim -- so its words come from the vocabulary file instead.
fn generators(
    config: &serde_json::Value,
    vocabulary: &HashMap<String, Vec<String>>,
) -> Result<Vec<(String, Generator)>, String> {
    let vectorization = config
        .get("vectorization_config")
        .ok_or("the model configuration declares no vectorization_config")?;
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
            // Three standard deviations either side, so the scaled value lands in the
            // range the model was trained to see rather than always near zero.
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

    // Which features to send follows the shape of the configuration, exactly as it
    // does for the server: an order for the vectorised shape, a set of names for the
    // named one.
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
    inputs
        .iter()
        .map(|(name, spec)| {
            let data_type = spec
                .get("data_type")
                .and_then(|t| t.as_str())
                .unwrap_or("float");
            let generator = if data_type.starts_with("string") {
                Generator::Choice(vocabulary.get(name).cloned().unwrap_or_else(|| {
                    vec![format!("{name}-unseen-0"), format!("{name}-unseen-1")]
                }))
            } else {
                from_transformation(name)?
            };
            Ok((name.clone(), generator))
        })
        .collect()
}

/// One request, with fresh values for every feature of every row.
fn request(id: u64, rows: usize, generators: &[(String, Generator)]) -> InferenceRequest {
    let mut rng = rand::rng();
    InferenceRequest {
        request_id: format!("bench-{id}"),
        inputs: (0..rows)
            .map(|row| InputRow {
                row_id: format!("row-{row}"),
                features: generators
                    .iter()
                    .map(|(name, generator)| (name.clone(), generator.sample(&mut rng)))
                    .collect(),
            })
            .collect(),
    }
}

/// Latency percentiles over the measured window.
#[derive(Debug)]
struct Percentiles {
    p50: f64,
    p90: f64,
    p99: f64,
    max: f64,
    mean: f64,
}

impl Percentiles {
    /// Sorts in place and reads the positions off. `None` when nothing was measured,
    /// which is a result worth distinguishing from zero latency.
    fn of(latencies: &mut [f64]) -> Option<Self> {
        if latencies.is_empty() {
            return None;
        }
        latencies.sort_by(f64::total_cmp);
        let at =
            |q: f64| latencies[((latencies.len() as f64 * q) as usize).min(latencies.len() - 1)];
        Some(Self {
            p50: at(0.50),
            p90: at(0.90),
            p99: at(0.99),
            max: latencies[latencies.len() - 1],
            mean: latencies.iter().sum::<f64>() / latencies.len() as f64,
        })
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&args.model_config)?)
            .map_err(|e| format!("{}: {e}", args.model_config.display()))?;
    let vocabulary: HashMap<String, Vec<String>> = match &args.vocabulary {
        Some(path) => serde_json::from_str(&std::fs::read_to_string(path)?)?,
        None => HashMap::new(),
    };
    let generators = Arc::new(generators(&config, &vocabulary).map_err(|e| e.to_string())?);

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
        "  offering  : {} req/s x {} row(s), {} in flight, {}s warmup + {}s measured",
        args.tps, args.rows, args.concurrency, args.warmup_secs, args.duration_secs
    );

    let client = HusharClient::connect(args.server.clone()).await?;
    let permits = Arc::new(Semaphore::new(args.concurrency));
    let latencies = Arc::new(std::sync::Mutex::new(Vec::<f64>::new()));
    let (sent, failed, shed) = (
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
        Arc::new(AtomicU64::new(0)),
    );

    // A 1 ms tick with a fractional budget rather than one timer per request: at any
    // useful rate the per-request interval is below the timer's resolution, so ticking
    // per request would silently offer less load than asked for.
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
            // An open-loop generator: when nothing is free the request is counted as
            // shed rather than queued, so the offered rate stays honest and the
            // shortfall shows up in the report.
            let Ok(permit) = Arc::clone(&permits).try_acquire_owned() else {
                shed.fetch_add(1, Ordering::Relaxed);
                continue;
            };
            let (mut client, generators) = (client.clone(), Arc::clone(&generators));
            let (latencies, sent, failed) = (
                Arc::clone(&latencies),
                Arc::clone(&sent),
                Arc::clone(&failed),
            );
            let measuring_from = start + warmup;
            tokio::spawn(async move {
                let payload = request(id, args.rows, &generators);
                let began = Instant::now();
                let outcome = client.inference_service(payload).await;
                let elapsed = began.elapsed().as_secs_f64() * 1000.0;
                if outcome.is_err() {
                    failed.fetch_add(1, Ordering::Relaxed);
                } else {
                    sent.fetch_add(1, Ordering::Relaxed);
                    if began >= measuring_from {
                        latencies.lock().expect("latencies").push(elapsed);
                    }
                }
                drop(permit);
            });
        }
    }

    // Let what is in flight finish, so the last requests are not counted as failures.
    let _ = permits.acquire_many(args.concurrency as u32).await;

    let mut measured = std::mem::take(&mut *latencies.lock().expect("latencies"));
    let counts = (
        sent.load(Ordering::Relaxed),
        failed.load(Ordering::Relaxed),
        shed.load(Ordering::Relaxed),
    );
    report(&args, &mut measured, counts, model_id, provider).await
}

/// Prints the result, and sends it to CloudWatch when a namespace was given.
async fn report(
    args: &Args,
    measured: &mut [f64],
    counts: (u64, u64, u64),
    model_id: &str,
    provider: &str,
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
    println!("           p90  : {:>9.2} ms", p.p90);
    println!("           p99  : {:>9.2} ms", p.p99);
    println!("           max  : {:>9.2} ms", p.max);
    println!("           mean : {:>9.2} ms", p.mean);
    println!("  failed        : {failed:>9}");
    println!(
        "  shed          : {shed:>9}  (offered but never sent: {} in flight was the cap)",
        args.concurrency
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

/// Sends the same numbers to CloudWatch, for a run that is not on a laptop.
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
            "Latency_P90",
            percentiles.p90,
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
