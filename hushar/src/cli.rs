// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Command-line configuration: where the service configuration is, and two overrides.
//!
//! Everything else lives in the service configuration — see
//! [`crate::config::service_config`]. What is left here is what cannot go in it:
//!
//! ```text
//! ./hushar --config-uri s3://bucket/service.json
//! ```
//!
//! - `--config-uri` says where that file is, so it cannot be a field inside it.
//! - `--port` and `--bind-address` override the configured listener, for running two
//!   instances on one host.
//!
//! Every option also reads from an environment variable, so a container is configured
//! from the environment and a developer from flags.

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
    /// For running two instances on one machine.
    #[arg(long, env = "HUSHAR_PORT")]
    pub(crate) port: Option<u16>,

    /// Override `bind_address` from the service configuration.
    ///
    /// Accepts IPv4 or IPv6. The configuration defaults to `0.0.0.0`, which accepts from
    /// anywhere; `127.0.0.1` restricts to the host.
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

    /// The config URI is the only required option, and both overrides parse as `None`.
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

    /// A malformed bind address fails at parse time, and IPv6 is accepted.
    #[test]
    fn a_bad_address_is_rejected_at_parse_time() {
        let mut args = REQUIRED.to_vec();
        args.extend(["--bind-address", "not-an-address"]);
        assert!(parse(&args).is_err());

        let mut args = REQUIRED.to_vec();
        args.extend(["--bind-address", "::1"]);
        assert!(parse(&args).is_ok(), "IPv6 should be accepted");
    }

    /// A flag that belongs in the service configuration is refused, not ignored.
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

    /// Every non-global option has an environment variable, and there are three of them.
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
