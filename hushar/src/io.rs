// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

use chrono::{Datelike, Timelike};
use hushar_proto::hushar::{InferenceLogBatch, InferenceLogs};
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
) -> () {
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
            }
        }
    }
}

pub(crate) async fn feature_logger(
    buffer_capacity: usize,
    s3_client: Arc<aws_sdk_s3::Client>,
    receiver: Arc<Mutex<mpsc::Receiver<InferenceLogBatch>>>,
    s3_bucket: Arc<String>,
    s3_prefix: Arc<String>,
    worker_id: usize,
) -> () {
    println!("Logger :: Starting worker {}", worker_id);
    loop {
        let mut receiver_lock = receiver.lock().await;
        let mut inference_logs_buffer = Vec::new();
        match receiver_lock.recv().await {
            Some(batch) => {
                drop(receiver_lock);
                inference_logs_buffer.push(batch);

                if inference_logs_buffer.len() > buffer_capacity {
                    let inference_logs = InferenceLogs {
                        inference_log_batches: std::mem::take(&mut inference_logs_buffer),
                    };
                    let current_time = chrono::Utc::now();
                    let date_time_str = format!(
                        "year={}/month={:02}/day={:02}/hour={:02}/mi={}",
                        current_time.year(),
                        current_time.month(),
                        current_time.day(),
                        current_time.hour(),
                        current_time.minute()
                    );
                    let file_key = format!(
                        "{}/{}/{}_{}.pb",
                        s3_prefix.clone().to_string(),
                        date_time_str,
                        uuid::Uuid::new_v4().to_string(),
                        current_time.timestamp_micros()
                    );
                    let _ = s3_client
                        .put_object()
                        .bucket(s3_bucket.clone().to_string())
                        .key(file_key)
                        .body(aws_sdk_s3::primitives::ByteStream::from(
                            inference_logs.encode_to_vec(),
                        ))
                        .send()
                        .await;
                }
            }
            None => {
                println!(
                    "Logger :: Worker {} channel closed, shutting down",
                    worker_id
                );
            }
        }
    }
}
