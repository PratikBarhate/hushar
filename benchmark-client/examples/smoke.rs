// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

//! Sends one single-row request to a running server and prints the scores.
//!
//! The features to send are given by environment variable. Valid names are the ones in
//! the server's vectorization configuration, which it prints at startup.
//!
//! * `SERVER_ADDR` - the server to call. Defaults to `http://127.0.0.1:50051`.
//! * `SMOKE_FEATURES` - the features, as `name=number,name:text` pairs separated by
//!   commas. Defaults to `f1=1.0,f2=2.0,f3=3.0`.
//! * `SMOKE_MODEL` - ask for one model by name. Empty, which is the default, leaves the
//!   choice to the server: with a roll-out configured it applies its split, and the
//!   response says which arm answered.
//!
//! ```bash
//! # name=value for a number, name:value for a string.
//! SMOKE_FEATURES='f1=1.0,f2=2.0,f3=3.0' cargo run -p benchmark-client --example smoke
//! SMOKE_FEATURES='age=30,city:london,tags:beta' \
//!   SERVER_ADDR=http://127.0.0.1:50251 cargo run -p benchmark-client --example smoke
//! ```

use benchmark_client::hushar_proto::Feature;
use benchmark_client::hushar_proto::{
    InferenceRequest, InferenceResponse, hushar_client::HusharClient,
    score_type::ScoreType as Scores,
};
use benchmark_client::proto_columns;

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
            model_id: std::env::var("SMOKE_MODEL").unwrap_or_default(),
            row_ids: vec!["row-0".into()],
            features: proto_columns::encode(features)?,
        })
        .await?
        .into_inner();

    println!("\nserved by: {}", response.model_id);
    println!("scored:");
    for line in describe(&response) {
        println!("  {line}");
    }
    println!("\nok");
    Ok(())
}

/// Reads `name=number,name:text` into one single-value column per feature.
///
/// The separator picks the type: `=` is a float, `:` is a string. `city=1` and `city:1`
/// mean different things to a one-hot transformation, so neither is guessed.
///
/// # Errors
///
/// Returns an error when an entry is neither form, when a `=` value does not parse as a
/// number, or when no features were supplied at all.
fn parse_features(spec: &str) -> Result<Vec<(String, Feature)>, String> {
    let mut features = Vec::new();
    for entry in spec.split(',').map(str::trim).filter(|e| !e.is_empty()) {
        if let Some((name, text)) = entry.split_once(':') {
            features.push((
                name.trim().to_owned(),
                proto_columns::strings(1, vec![text.trim().to_owned()]),
            ));
        } else if let Some((name, number)) = entry.split_once('=') {
            let parsed: f32 = number.trim().parse().map_err(|e| {
                format!("feature {name:?} has value {number:?}, which is not a number: {e}")
            })?;
            features.push((
                name.trim().to_owned(),
                proto_columns::floats(1, vec![parsed]),
            ));
        } else {
            return Err(format!(
                "{entry:?} is neither name=number nor name:text; those are the two forms"
            ));
        }
    }
    if features.is_empty() {
        return Err("no features supplied; set SMOKE_FEATURES".into());
    }
    Ok(features)
}

/// Renders each scored row as one line.
///
/// Scores arrive as one buffer for the whole batch, so each row's slice is
/// `values / rows` wide. The populated `oneof` variant is the precision, FP32 or FP64.
///
/// # Returns
///
/// One line per row, or a single line describing the mismatch when the value count is
/// not a whole number of rows.
fn describe(response: &InferenceResponse) -> Vec<String> {
    let rows = response.row_ids.len();
    let (kind, values): (&str, Vec<String>) =
        match response.scores.as_ref().and_then(|s| s.score_type.as_ref()) {
            Some(Scores::FloatScores(a)) => {
                ("FP32", a.values.iter().map(|v| format!("{v:?}")).collect())
            }
            Some(Scores::DoubleScores(a)) => {
                ("FP64", a.values.iter().map(|v| format!("{v:?}")).collect())
            }
            None => ("none", Vec::new()),
        };
    if rows == 0 || values.len() % rows != 0 {
        return vec![format!(
            "{kind} = {} value(s) for {rows} row(s), which is not a whole number of rows",
            values.len()
        )];
    }
    let width = values.len() / rows;
    response
        .row_ids
        .iter()
        .enumerate()
        .map(|(r, id)| {
            format!(
                "{id} {kind} = [{}]",
                values[r * width..(r + 1) * width].join(", ")
            )
        })
        .collect()
}
