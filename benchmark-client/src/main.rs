// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

use aws_sdk_cloudwatch::{types::*, Client as CloudWatchClient};
use benchmark_client::hushar_proto::{hushar_client::HusharClient, *};
use rand::Rng;
use std::{collections::HashMap, time::Instant};
use tokio::time::{interval, Duration};

const TARGET_TPS: u64 = 4000;
const INTERVAL_MS: u64 = 1000;
const BATCH_SIZE: usize = 100;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let server_addr =
        std::env::var("SERVER_ADDR").unwrap_or_else(|_| "http://127.0.0.1:50051".to_string());
    let namespace =
        std::env::var("CW_NAMESPACE").unwrap_or_else(|_| "Hushar/Benchmark".to_string());

    let config = aws_config::load_from_env().await;
    let cw_client = CloudWatchClient::new(&config);
    let client = HusharClient::connect(server_addr).await?;

    let mut ticker = interval(Duration::from_millis(INTERVAL_MS));
    let requests_per_tick = TARGET_TPS as usize;

    loop {
        ticker.tick().await;
        let mut latencies = Vec::with_capacity(requests_per_tick);

        for chunk in (0..requests_per_tick)
            .collect::<Vec<_>>()
            .chunks(BATCH_SIZE)
        {
            let tasks: Vec<_> = chunk
                .iter()
                .map(|i| {
                    let mut client = client.clone();
                    let req = create_request(&format!("req_{}", i));
                    async move {
                        let start = Instant::now();
                        let _ = client.inference_service(req).await;
                        start.elapsed().as_micros() as f64 / 1000.0
                    }
                })
                .collect();

            let results = futures::future::join_all(tasks).await;
            latencies.extend(results);
        }

        emit_metrics(&cw_client, &namespace, &latencies).await?;
    }
}

fn create_request(id: &str) -> InferenceRequest {
    let mut rng = rand::rng();
    let mut features = HashMap::new();
    features.insert(
        "feature1".to_string(),
        DataType {
            data_type: Some(data_type::DataType::FloatValue(rng.random())),
        },
    );

    InferenceRequest {
        request_id: id.to_string(),
        inputs: vec![InputRow {
            row_id: "row1".to_string(),
            features,
        }],
    }
}

async fn emit_metrics(
    client: &CloudWatchClient,
    namespace: &str,
    latencies: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    if latencies.is_empty() {
        return Ok(());
    }

    let avg = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50 = sorted[sorted.len() / 2];
    let p99 = sorted[(sorted.len() as f64 * 0.99) as usize];

    client
        .put_metric_data()
        .namespace(namespace)
        .metric_data(
            MetricDatum::builder()
                .metric_name("Latency_Avg")
                .value(avg)
                .unit(StandardUnit::Milliseconds)
                .build(),
        )
        .metric_data(
            MetricDatum::builder()
                .metric_name("Latency_P50")
                .value(p50)
                .unit(StandardUnit::Milliseconds)
                .build(),
        )
        .metric_data(
            MetricDatum::builder()
                .metric_name("Latency_P99")
                .value(p99)
                .unit(StandardUnit::Milliseconds)
                .build(),
        )
        .send()
        .await?;

    Ok(())
}
