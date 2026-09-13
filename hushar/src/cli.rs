// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Command-line configuration: where the service configuration is, and two overrides.
//!
//! Everything else lives in the service configuration — see
//! [`crate::config::service_config`] for the file and why it is one file. What is left
//! here is what cannot go in it:
//!
//! ```text
//! ./hushar --config-uri s3://bucket/service.json
//! ```
//!
//! - `--config-uri` says where that file is, so it cannot be a field inside it.
//! - `--port` and `--bind-address` override the configured listener, for running two
//!   instances on one host. That is a developer concern rather than a deployment one,
//!   which is why they are flags and not the source of truth.
//!
//! Both options also read from an environment variable, so a container is configured
//! from the environment and a developer from flags without the service needing to know
//! which.
//!
//! # What used to be here
//!
//! Six flags: the inference log URI, the CloudWatch namespace, both sinks' batch sizes,
//! the sampling rate and the in-flight bound. They were properties of a deployment
//! sitting outside the file that describes the deployment, which meant the destination a
//! service writes to could not be versioned alongside the service. They are now
//! `inference_log` and `metrics` in the service configuration, and their validation
//! moved with them into `HusharServiceConfig::validate`.

use std::net::IpAddr;

use clap::Parser;

/// A gRPC server for machine-learning model inference.
#[derive(Debug, Clone, Parser)]
#[command(name = "hushar", version, about, long_about = None)]
pub(crate) struct Cli {
    /// Where the service configuration lives.
    ///
    /// A URI, whose scheme selects the storage: `s3://bucket/key`, `file:///path`, or a
    /// bare filesystem path. Everything else the process needs is in that file.
    #[arg(long, env = "HUSHAR_CONFIG_URI")]
    pub(crate) config_uri: String,

    /// Override `port_number` from the service configuration.
    ///
    /// For running two instances on one machine. A deployment sets the port in the
    /// configuration alongside the rest of its settings.
    #[arg(long, env = "HUSHAR_PORT")]
    pub(crate) port: Option<u16>,

    /// Override `bind_address` from the service configuration.
    ///
    /// Accepts IPv4 or IPv6. `0.0.0.0` accepts from anywhere, which is what a container
    /// wants and what the configuration defaults to; `127.0.0.1` restricts to the host.
    #[arg(long, env = "HUSHAR_BIND_ADDRESS")]
    pub(crate) bind_address: Option<IpAddr>,
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

    const REQUIRED: [&str; 2] = ["--config-uri", "s3://bucket/service.json"];

    /// One required option, and the two overrides absent unless asked for. Absent
    /// matters: `None` is what lets the configured value through in `main`.
    #[test]
    fn the_config_uri_is_the_only_requirement() {
        let cli = parse(&REQUIRED).expect("the config URI should be the only requirement");
        assert_eq!(cli.config_uri, "s3://bucket/service.json");
        assert_eq!(
            cli.port, None,
            "the port comes from the configuration unless overridden"
        );
        assert_eq!(
            cli.bind_address, None,
            "the address comes from the configuration unless overridden"
        );
    }

    #[test]
    fn a_missing_config_uri_is_refused_by_name() {
        let err = parse(&[]).expect_err("the config URI is required");
        let message = err.to_string();
        assert!(
            message.contains("config-uri"),
            "the error should name the missing option: {message}"
        );
    }

    #[test]
    fn both_overrides_are_settable() {
        let mut args = REQUIRED.to_vec();
        args.extend(["--port", "50051", "--bind-address", "127.0.0.1"]);
        let cli = parse(&args).expect("valid");
        assert_eq!(cli.port, Some(50051));
        assert_eq!(cli.bind_address, Some(IpAddr::from([127, 0, 0, 1])));
    }

    /// Typed, so this cannot reach the runtime and fail there.
    #[test]
    fn a_bad_address_is_rejected_at_parse_time() {
        let mut args = REQUIRED.to_vec();
        args.extend(["--bind-address", "not-an-address"]);
        assert!(parse(&args).is_err());

        let mut args = REQUIRED.to_vec();
        args.extend(["--bind-address", "::1"]);
        assert!(parse(&args).is_ok(), "IPv6 should be accepted");
    }

    /// A flag that moved into the configuration must be refused, not ignored. clap
    /// rejects an unknown argument by default, so this is a guard against one being
    /// reintroduced here rather than in the file.
    #[test]
    fn a_flag_that_moved_into_the_configuration_is_refused() {
        for stale in [
            "--inference-log-uri",
            "--cloudwatch-namespace",
            "--metrics-batch-size",
            "--data-batch-size",
            "--data-sample-rate",
            "--max-sends-in-flight",
        ] {
            let mut args = REQUIRED.to_vec();
            args.extend([stale, "1"]);
            assert!(
                parse(&args).is_err(),
                "{stale} is in the service configuration now and must be refused here"
            );
        }
    }

    /// A container is configured from the environment, so every flag has to be
    /// reachable that way.
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
        assert_eq!(
            named.len(),
            3,
            "three options are left; anything new belongs in the service \
             configuration unless it cannot go there: {named:?}"
        );
    }
}
