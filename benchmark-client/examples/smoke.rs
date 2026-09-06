// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Sends one request to a running server and prints the scores.
//!
//! The benchmark client measures throughput and needs CloudWatch to report it. This
//! needs nothing but a server, which makes it the thing to reach for when checking a
//! deployment by hand.
//!
//! The features to send are given on the command line, because a client already knows
//! them: they are in the vectorization configuration it was deployed alongside, and the
//! server prints the same mapping at startup.
//!
//! ```bash
//! # name=value for a number, name:value for a string.
//! SMOKE_FEATURES='f1=1.0,f2=2.0,f3=3.0' cargo run -p benchmark-client --example smoke
//! SMOKE_FEATURES='age=30,city:london,tags:beta' \
//!   SERVER_ADDR=http://127.0.0.1:50251 cargo run -p benchmark-client --example smoke
//! ```

use std::collections::HashMap;

use benchmark_client::hushar_proto::{
    DataType, InferenceRequest, InputRow, OutputRow, data_type::DataType as Value,
    hushar_client::HusharClient, score_type::ScoreType as Scores,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = std::env::var("SERVER_ADDR").unwrap_or_else(|_| "http://127.0.0.1:50051".into());
    let spec = std::env::var("SMOKE_FEATURES").unwrap_or_else(|_| "f1=1.0,f2=2.0,f3=3.0".into());

    let features = parse_features(&spec)?;
    println!("connecting to {addr}");
    println!("features: {} supplied", features.len());

    let mut client = HusharClient::connect(addr).await?;
    let response = client
        .inference_service(InferenceRequest {
            request_id: "smoke".into(),
            inputs: vec![InputRow {
                row_id: "row-0".into(),
                features,
            }],
        })
        .await?
        .into_inner();

    println!("\nscored:");
    for row in &response.outputs {
        println!("  {}", describe(row));
    }
    println!("\nok");
    Ok(())
}

/// Reads `name=number,name:text` into feature values.
///
/// Two separators rather than guessing, because `city=1` and `city:1` mean different
/// things to a one-hot transformation, and a client that meant text should not have it
/// silently parsed into a float.
fn parse_features(spec: &str) -> Result<HashMap<String, DataType>, String> {
    let mut features = HashMap::new();
    for entry in spec.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        let value = if let Some((name, text)) = entry.split_once(':') {
            features.insert(
                name.trim().to_owned(),
                DataType {
                    data_type: Some(Value::StringValue(text.trim().to_owned())),
                },
            );
            continue;
        } else if let Some((name, number)) = entry.split_once('=') {
            let parsed: f32 = number.trim().parse().map_err(|e| {
                format!("feature {name:?} has value {number:?}, which is not a number: {e}")
            })?;
            (name.trim().to_owned(), parsed)
        } else {
            return Err(format!(
                "{entry:?} is neither name=number nor name:text; those are the two forms"
            ));
        };
        features.insert(
            value.0,
            DataType {
                data_type: Some(Value::FloatValue(value.1)),
            },
        );
    }
    if features.is_empty() {
        return Err("no features supplied; set SMOKE_FEATURES".into());
    }
    Ok(features)
}

/// Renders one scored row.
///
/// The populated `oneof` variant *is* the precision, so there is no separate label that
/// could disagree with the values.
fn describe(row: &OutputRow) -> String {
    let (kind, body) = match row.scores.as_ref().and_then(|s| s.score_type.as_ref()) {
        Some(Scores::FloatScores(a)) => ("FP32", format!("{:?}", a.values)),
        Some(Scores::DoubleScores(a)) => ("FP64", format!("{:?}", a.values)),
        None => ("none", "[]".to_owned()),
    };
    format!("{} {kind} = {body}", row.row_id)
}
