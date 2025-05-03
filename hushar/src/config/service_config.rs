// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Service configuration module to read the configurations from JSON string.

use serde::Deserialize;
use std::error::Error;

fn default_port_number() -> u32 {
    8279
}

fn default_connection_concurrency() -> u16 {
    10
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct HusharServiceConfig {
    #[serde(default = "default_connection_concurrency")]
    pub connection_concurrency: u16,
    pub model_path: String,
    pub model_id: String,
    #[serde(default = "default_port_number")]
    pub port_number: u32,
    pub vectorization_instruction_path: String,
}

impl HusharServiceConfig {
    pub fn from_json(json_str: &str) -> Result<Self, Box<dyn Error>> {
        let config: HusharServiceConfig = serde_json::from_str(json_str)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_json() {
        let json = r#"{
            "connection_concurrency": 50,
            "model_path": "s3://hushar-models/embedding-model-v1",
            "model_id": "embedding-model-v1.0",
            "port_number": 8080,
            "vectorization_instruction_path": "/hushar-instruction/embedding-model-v1"
        }"#;

        let config = HusharServiceConfig::from_json(json).unwrap();

        assert_eq!(config.connection_concurrency, 50);
        assert_eq!(config.model_path, "s3://hushar-models/embedding-model-v1");
        assert_eq!(config.model_id, "embedding-model-v1.0");
        assert_eq!(config.port_number, 8080);
        assert_eq!(
            config.vectorization_instruction_path,
            "/hushar-instruction/embedding-model-v1"
        );
    }

    #[test]
    fn test_default_values() {
        let json = r#"{
            "model_path": "s3://hushar-models/embedding-model-v1",
            "model_id": "embedding-model-v1.0",
            "vectorization_instruction_path": "/hushar-instruction/embedding-model-v1"
        }"#;

        let config = HusharServiceConfig::from_json(json).unwrap();

        assert_eq!(
            config.connection_concurrency,
            default_connection_concurrency()
        );
        assert_eq!(config.port_number, default_port_number());

        assert_eq!(config.model_path, "s3://hushar-models/embedding-model-v1");
        assert_eq!(config.model_id, "embedding-model-v1.0");
        assert_eq!(
            config.vectorization_instruction_path,
            "/hushar-instruction/embedding-model-v1"
        );
    }

    #[test]
    fn test_missing_required_fields() {
        let invalid_json = r#"{
            "connection_concurrency": 25,
            "port_number": 8000
        }"#;

        let result = HusharServiceConfig::from_json(invalid_json);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_types() {
        let invalid_json = r#"{
            "connection_concurrency": "not-a-number",
            "model_path": "s3://hushar-models/embedding-model-v1",
            "model_id": "embedding-model-v1.0",
            "port_number": 8080,
            "vectorization_instruction_path": "/hushar-instruction/embedding-model-v1"
        }"#;

        let result = HusharServiceConfig::from_json(invalid_json);
        assert!(result.is_err());
    }
}
