// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Service configuration: what is true of the server rather than of the model.
//!
//! The model, its execution provider and its feature configuration all live in
//! [`crate::config::model_config`], beside the model they apply to. This holds only
//! what is true of the server process: the listener.
//!
//! ```json
//! {
//!   "connection_concurrency": 50,
//!   "port_number": 8279,
//!   "model_config_path": "s3://configs/fraud/v3/model_config.json"
//! }
//! ```

use serde::Deserialize;

fn default_port_number() -> u16 {
    8279
}

fn default_connection_concurrency() -> u16 {
    50
}

/// The server process's own settings.
///
/// Fields:
/// - `connection_concurrency` — in-flight requests allowed per connection.
/// - `model_config_path` — where the model's own configuration is. Carries its own URI
///   scheme, so a model configuration in S3 beside a service configuration on local
///   disk is expressible.
/// - `port_number` — the port to listen on, which `--port` overrides.
#[derive(Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HusharServiceConfig {
    #[serde(default = "default_connection_concurrency")]
    pub connection_concurrency: u16,
    pub model_config_path: String,
    #[serde(default = "default_port_number")]
    pub port_number: u16,
}

impl HusharServiceConfig {
    pub fn from_json(json_str: &str) -> Result<Self, crate::inference::InferenceError> {
        let config: HusharServiceConfig = serde_json::from_str(json_str)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_service_config_carries_the_listener_and_the_model_pointer() {
        let config = HusharServiceConfig::from_json(
            r#"{
                "connection_concurrency": 25,
                "model_config_path": "s3://configs/embedding-v1/model_config.json",
                "port_number": 8080
            }"#,
        )
        .expect("valid");

        assert_eq!(config.connection_concurrency, 25);
        assert_eq!(
            config.model_config_path,
            "s3://configs/embedding-v1/model_config.json"
        );
        assert_eq!(config.port_number, 8080);
    }

    #[test]
    fn only_the_model_config_path_is_required() {
        let config = HusharServiceConfig::from_json(r#"{"model_config_path": "/tmp/model.json"}"#)
            .expect("valid");

        assert_eq!(
            config.connection_concurrency,
            default_connection_concurrency()
        );
        assert_eq!(config.port_number, default_port_number());
    }

    #[test]
    fn a_missing_model_config_path_is_refused() {
        assert!(
            HusharServiceConfig::from_json(r#"{"port_number": 8000}"#).is_err(),
            "there is nothing sensible to serve without one"
        );
    }

    /// These moved into the model configuration. Rejecting them means an operator
    /// upgrading gets an error naming the field, not a service that starts and then
    /// cannot find a model.
    #[test]
    fn a_stale_field_is_refused_rather_than_ignored() {
        for stale in ["model_path", "model_id", "vectorization_instruction_path"] {
            let json = format!(r#"{{"model_config_path": "/tmp/m.json", "{stale}": "x"}}"#);
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
                r#"{"connection_concurrency": "not-a-number", "model_config_path": "/tmp/m.json"}"#
            )
            .is_err()
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
        assert_eq!(config.connection_concurrency, 50);
        assert_eq!(config.port_number, 8279);
        assert_eq!(
            config.model_config_path, "test-data/model_config_vectorized.json",
            "it should point at the model configuration"
        );
    }
}
