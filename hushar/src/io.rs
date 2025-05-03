// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

use hushar_proto::hushar::InferenceLogBatch;
use prost::Message;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};

pub(crate) type FileReaderResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;
pub(crate) mod file_reader;
pub(crate) mod model_loader;

pub(crate) use file_reader::*;
pub(crate) use model_loader::*;

use crate::inference;

pub(crate) async fn inference_metrics_sidecar(
    buffer_capacity: usize,
    cloudwatch_client: Arc<aws_sdk_cloudwatch::Client>,
    namespace: &'static str,
    receiver: Arc<Mutex<mpsc::Receiver<inference::InferenceMicros>>>,
    worker_id: usize,
) {
    println!("Metrics :: Starting worker {}", worker_id);
    let mut metric_buffer = Vec::new();
    loop {
        let mut receiver_lock = receiver.lock().await;
        match receiver_lock.recv().await {
            Some(inference_micros) => {
                let metric_data = vec![
                    aws_sdk_cloudwatch::types::MetricDatum::builder()
                        .metric_name("InferenceTime")
                        .value(inference_micros.inference_time as f64)
                        .unit(aws_sdk_cloudwatch::types::StandardUnit::Microseconds)
                        .build(),
                    aws_sdk_cloudwatch::types::MetricDatum::builder()
                        .metric_name("ToTensorTime")
                        .value(inference_micros.tensor_time as f64)
                        .unit(aws_sdk_cloudwatch::types::StandardUnit::Microseconds)
                        .build(),
                    aws_sdk_cloudwatch::types::MetricDatum::builder()
                        .metric_name("VectorizationTime")
                        .value(inference_micros.vec_time as f64)
                        .unit(aws_sdk_cloudwatch::types::StandardUnit::Microseconds)
                        .build(),
                ];
                metric_buffer.extend(metric_data);
                if metric_buffer.len() >= buffer_capacity {
                    let metrics_to_send = std::mem::take(&mut metric_buffer);
                    let _ = cloudwatch_client
                        .put_metric_data()
                        .namespace(namespace)
                        .set_metric_data(Some(metrics_to_send))
                        .send()
                        .await;
                }
            }
            None => {
                println!(
                    "Metrics :: Worker {} channel closed, shutting down",
                    worker_id
                );
                break;
            }
        }
    }
}

pub(crate) async fn feature_logger(
    kinesis_client: Arc<aws_sdk_kinesis::Client>,
    receiver: Arc<Mutex<mpsc::Receiver<InferenceLogBatch>>>,
    stream_name: &'static str,
    worker_id: usize,
) {
    println!("Logger :: Starting worker {}", worker_id);
    loop {
        let mut receiver_lock = receiver.lock().await;
        match receiver_lock.recv().await {
            Some(batch) => {
                drop(receiver_lock);
                let data = batch.encode_to_vec();
                let partition_key = format!("{}-{}", batch.request_id, batch.model_id);

                let _ = kinesis_client
                    .put_record()
                    .stream_name(stream_name)
                    .partition_key(partition_key)
                    .data(aws_sdk_kinesis::primitives::Blob::new(data))
                    .send()
                    .await;
            }
            None => {
                println!(
                    "Logger :: Worker {} channel closed, shutting down",
                    worker_id
                );
                break;
            }
        }
    }
}
