// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Command-line configuration.
//!
//! This replaces five positional arguments:
//!
//! ```text
//! ./hushar 0.0.0.0 s3://bucket/config.json s3://bucket/logs 1 1
//! ```
//!
//! Positional arguments are a poor interface for a deployment: nothing names them,
//! nothing validates them, and transposing two of them produces a service that
//! starts and then behaves oddly rather than one that refuses. Named arguments with
//! types make both mistakes impossible.
//!
//! Every option also reads from an environment variable, so a container is
//! configured from the environment and a developer from flags without the service
//! needing to know which.
//!
//! # The two thread counts are gone
//!
//! `--metrics-threads` and `--log-threads` sized the sidecar runtimes, and the server
//! got whatever cores they left. There are no sidecars now, so every core serves
//! requests and there is nothing to divide. What replaced them are the levers that
//! actually matter for observability under load — how much is accumulated before a
//! send, how many sends may be in the air, and what fraction of requests is logged at
//! all.

use std::net::IpAddr;

use clap::Parser;

/// A gRPC server for machine-learning model inference.
#[derive(Debug, Clone, Parser)]
#[command(name = "hushar", version, about, long_about = None)]
pub(crate) struct Cli {
    /// Address to bind the gRPC server to.
    ///
    /// Accepts IPv4 or IPv6. `0.0.0.0` accepts from anywhere, which is what a
    /// container wants; `127.0.0.1` restricts to the host.
    #[arg(long, env = "HUSHAR_BIND_ADDRESS", default_value = "0.0.0.0")]
    pub(crate) bind_address: IpAddr,

    /// Override the port from the service configuration.
    ///
    /// The port normally lives in the service configuration alongside the rest of
    /// the deployment's settings. This exists for running two instances on one
    /// machine, which is a developer concern rather than a deployment one.
    #[arg(long, env = "HUSHAR_PORT")]
    pub(crate) port: Option<u16>,

    /// Where the service configuration lives.
    ///
    /// A URI, whose scheme selects the storage: `s3://bucket/key`,
    /// `file:///path`, or a bare filesystem path.
    #[arg(long, env = "HUSHAR_CONFIG_URI")]
    pub(crate) config_uri: String,

    /// Prefix to write inference logs under.
    ///
    /// Same URI rules as `--config-uri`, plus `kinesis://stream-name` for a data
    /// stream. Objects are written beneath a prefix, partitioned by time.
    #[arg(long, env = "HUSHAR_INFERENCE_LOG_URI")]
    pub(crate) inference_log_uri: String,

    /// CloudWatch namespace to publish metrics to.
    ///
    /// Without it, timings are summarised to stderr, which is what a local run
    /// wants and needs no credentials.
    #[arg(long, env = "HUSHAR_CLOUDWATCH_NAMESPACE")]
    pub(crate) cloudwatch_namespace: Option<String>,

    /// Scored batches accumulated before metrics are sent.
    ///
    /// Also the number of batches per stderr summary line. CloudWatch takes at most
    /// 1000 data points per call and each batch contributes three, so a larger number
    /// is capped at 333 there; the startup banner prints what was settled on.
    #[arg(long, env = "HUSHAR_METRICS_BATCH_SIZE", default_value_t = 500)]
    pub(crate) metrics_batch_size: usize,

    /// Inference log batches accumulated before they are sent.
    ///
    /// Larger means fewer, bigger writes and more memory held: the buffer costs
    /// roughly this many requests' worth of features, and a request that is not
    /// sampled costs nothing at all. Kinesis takes at most 500 records per call, so a
    /// larger number is capped there; a blob store takes an object of any size, which
    /// is why the default is well above what a stream can use.
    #[arg(long, env = "HUSHAR_DATA_BATCH_SIZE", default_value_t = 5000)]
    pub(crate) data_batch_size: usize,

    /// Fraction of requests whose features reach the inference log, in `0.0..=1.0`.
    ///
    /// The cheapest lever over what the log costs, and the only one that also removes
    /// work: an unsampled request never copies its feature values anywhere. Realised
    /// as one request in `n`, so a rate that is not a unit fraction becomes the
    /// nearest one and the banner reports which.
    #[arg(
        long,
        env = "HUSHAR_DATA_SAMPLE_RATE",
        default_value_t = 1.0,
        value_parser = parse_sample_rate
    )]
    pub(crate) data_sample_rate: f64,

    /// Sends a sink may have in flight at once, before it starts shedding.
    ///
    /// This is the bound that replaced the sidecar's bounded channel. A sink holds at
    /// most `batch size × (this + 1)` records, so raising it buys tolerance of a slow
    /// destination and costs memory in the same proportion — at the defaults, 5000 × 5
    /// = 25,000 inference log batches, which is a few tens of megabytes for a typical
    /// row and is why sampling is the lever to reach for first. When every slot is
    /// taken the oldest batch is dropped and reported, which keeps a slow log
    /// destination from becoming slow inference.
    #[arg(
        long,
        env = "HUSHAR_MAX_SENDS_IN_FLIGHT",
        default_value_t = 4,
        value_parser = parse_sends_in_flight
    )]
    pub(crate) max_sends_in_flight: usize,
}

/// Parses the in-flight bound, refusing what a sink could not honour.
///
/// Validated here rather than clamped inside the sink so that the startup banner
/// prints the number in force. A silently corrected value is the kind of thing that is
/// discovered while wondering why the memory bound is not what was configured.
fn parse_sends_in_flight(value: &str) -> Result<usize, String> {
    let sends: usize = value
        .parse()
        .map_err(|_| format!("{value:?} is not a whole number"))?;
    if sends == 0 {
        return Err(
            "0 would leave no slot for a send to start in; 1 sends one at a \
                    time"
                .to_owned(),
        );
    }
    // Draining at shutdown acquires every permit at once, which the semaphore counts
    // in a `u32`, so a wider bound could not be waited on.
    if sends > u32::MAX as usize {
        return Err(format!(
            "{sends} is more sends than can be tracked; the practical range is single \
             digits, since each one in flight holds a whole batch in memory"
        ));
    }
    Ok(sends)
}

/// Parses a sampling rate, refusing what a sampler could not honour.
///
/// Rejected at parse time rather than clamped, because a rate of 50 is someone who
/// meant a percentage, and silently reading it as "log everything" is the reading
/// least likely to be what they wanted.
fn parse_sample_rate(value: &str) -> Result<f64, String> {
    let rate: f64 = value
        .parse()
        .map_err(|_| format!("{value:?} is not a number"))?;
    if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
        return Err(format!(
            "{rate} is not a fraction between 0.0 and 1.0; 0.01 is one request in a \
             hundred and 1.0 is all of them"
        ));
    }
    Ok(rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// Parses arguments as if from a command line, with the binary name prepended.
    fn parse(args: &[&str]) -> Result<Cli, clap::Error> {
        let mut argv = vec!["hushar"];
        argv.extend_from_slice(args);
        Cli::try_parse_from(argv)
    }

    /// The two required options, for tests that care about something else.
    const REQUIRED: [&str; 4] = [
        "--config-uri",
        "s3://bucket/config.json",
        "--inference-log-uri",
        "s3://bucket/logs",
    ];
    /// The default has to need no credentials, or a local run cannot start.
    #[test]
    fn metrics_go_to_stderr_unless_a_namespace_is_given() {
        assert_eq!(parse(&REQUIRED).expect("valid").cloudwatch_namespace, None);

        let mut args = REQUIRED.to_vec();
        args.extend(["--cloudwatch-namespace", "HusharService"]);
        assert_eq!(
            parse(&args).expect("valid").cloudwatch_namespace.as_deref(),
            Some("HusharService")
        );
    }

    #[test]
    fn the_defaults_are_enough_to_start() {
        let cli = parse(&REQUIRED).expect("the two URIs should be the only requirements");
        assert_eq!(cli.bind_address, IpAddr::from([0, 0, 0, 0]));
        assert_eq!(cli.metrics_batch_size, 500);
        assert_eq!(cli.data_batch_size, 5000);
        assert_eq!(cli.max_sends_in_flight, 4);
        assert_eq!(
            cli.data_sample_rate, 1.0,
            "logging every request is what the service did before there was a lever"
        );
        assert_eq!(
            cli.port, None,
            "the port comes from the config unless overridden"
        );
    }
    /// The improvement over positional arguments: the error says which one.
    #[test]
    fn a_missing_required_option_is_refused_by_name() {
        let err = parse(&["--config-uri", "file:///c.json"]).expect_err("the log URI is required");
        let message = err.to_string();
        assert!(
            message.contains("inference-log-uri"),
            "the error should name the missing option: {message}"
        );
    }
    /// Typed, so this cannot reach the runtime and fail there.
    #[test]
    fn a_bad_address_is_rejected_at_parse_time() {
        assert!(parse(&["--bind-address", "not-an-address"]).is_err());
        assert!(
            parse(&[
                REQUIRED[0],
                REQUIRED[1],
                REQUIRED[2],
                REQUIRED[3],
                "--bind-address",
                "::1"
            ])
            .is_ok(),
            "IPv6 should be accepted"
        );
    }

    #[test]
    fn a_non_numeric_batch_size_is_rejected() {
        let mut args = REQUIRED.to_vec();
        args.extend(["--data-batch-size", "lots"]);
        assert!(parse(&args).is_err());
    }

    #[test]
    fn the_batch_sizes_and_the_send_bound_are_settable() {
        let mut args = REQUIRED.to_vec();
        args.extend([
            "--metrics-batch-size",
            "50",
            "--data-batch-size",
            "20000",
            "--max-sends-in-flight",
            "16",
        ]);
        let cli = parse(&args).expect("valid");
        assert_eq!(cli.metrics_batch_size, 50);
        assert_eq!(cli.data_batch_size, 20_000);
        assert_eq!(cli.max_sends_in_flight, 16);
    }

    /// Refused rather than clamped, because the banner prints this number and a
    /// silently corrected one would misreport the memory bound.
    #[test]
    fn a_send_bound_of_zero_is_refused_with_an_explanation() {
        let mut args = REQUIRED.to_vec();
        args.extend(["--max-sends-in-flight", "0"]);
        let err = parse(&args).expect_err("zero leaves no slot");
        assert!(
            err.to_string().contains("no slot"),
            "the error should say why: {err}"
        );

        let mut args = REQUIRED.to_vec();
        args.extend(["--max-sends-in-flight", "99999999999999"]);
        assert!(
            parse(&args).is_err(),
            "a bound wider than the semaphore can drain must be refused"
        );
    }

    #[test]
    fn a_sampling_rate_is_accepted_across_its_range() {
        for rate in ["0", "0.001", "0.5", "1", "1.0"] {
            let mut args = REQUIRED.to_vec();
            args.extend(["--data-sample-rate", rate]);
            assert!(parse(&args).is_ok(), "{rate} should be a valid rate");
        }
    }
    /// `--data-sample-rate 10` is someone who meant 10%, and reading it as "log
    /// everything" is the reading least likely to be what they wanted. Refusing it
    /// with the range in the message is the only outcome that tells them.
    ///
    /// Written as `--flag=value` because a bare `-0.5` is a flag as far as a command
    /// line is concerned, and clap says so before the parser ever sees it.
    #[test]
    fn a_sampling_rate_outside_the_range_is_refused_with_an_explanation() {
        for rate in ["10", "-0.5", "1.5", "nan", "inf"] {
            let argument = format!("--data-sample-rate={rate}");
            let mut args = REQUIRED.to_vec();
            args.push(&argument);
            let err = parse(&args).expect_err("not a fraction");
            let message = err.to_string();
            assert!(
                message.contains("0.0") && message.contains("1.0"),
                "the error for {rate} should give the range: {message}"
            );
        }
    }
    /// A container is configured from the environment, so the flags have to be
    /// reachable that way -- including the two added here.
    #[test]
    fn every_option_is_also_an_environment_variable() {
        let command = Cli::command();
        let named: Vec<(&str, bool)> = command
            .get_arguments()
            .filter(|arg| !arg.is_global_set() && arg.get_long().is_some())
            .filter(|arg| !matches!(arg.get_id().as_str(), "help" | "version"))
            .map(|arg| (arg.get_id().as_str(), arg.get_env().is_some()))
            .collect();
        for (name, has_env) in &named {
            assert!(*has_env, "--{name} has no environment variable");
        }
        assert!(
            named.iter().any(|(name, _)| *name == "data_sample_rate"),
            "the sampling lever should be listed: {named:?}"
        );
    }
}
