// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

use config::HusharServiceConfig;
use futures::Stream;
use futures::StreamExt;
use hushar_proto::hushar::{
    hushar_server::{Hushar, HusharServer},
    InferenceLogBatch, InferenceRequest, InferenceResponse,
};
use inference::TractRunnableModel;
use io::FileReader;
use std::env;
use std::net::{Ipv4Addr, SocketAddr};
use std::pin::Pin;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status};

pub(crate) mod config;
pub(crate) mod inference;
pub(crate) mod io;

#[derive(Debug)]
pub(crate) struct HusharService {
    feat_len: usize,
    log_sender: tokio::sync::mpsc::Sender<InferenceLogBatch>,
    model: Arc<TractRunnableModel>,
    metrics_sender: tokio::sync::mpsc::Sender<inference::InferenceMicros>,
    service_config: HusharServiceConfig,
    vec_config: Arc<config::VectorizationConfig>,
}

#[tonic::async_trait]
impl Hushar for HusharService {
    type InferenceServiceStream =
        Pin<Box<dyn Stream<Item = Result<InferenceResponse, Status>> + Send + 'static>>;

    async fn inference_service(
        &self,
        request: Request<tonic::Streaming<InferenceRequest>>,
    ) -> Result<Response<Self::InferenceServiceStream>, Status> {
        let mut request_stream = request.into_inner();
        let metrics_sender = self.metrics_sender.clone();
        let log_sender = self.log_sender.clone();
        let feat_len = self.feat_len.clone();
        let model = self.model.clone();
        let vec_config = self.vec_config.clone();
        let model_id = self.service_config.model_id.clone();

        let (tx, rx) = tokio::sync::mpsc::channel(200);
        let output_stream = ReceiverStream::new(rx);

        tokio::task::spawn(async move {
            while let Some(result) = request_stream.next().await {
                let send_result = match result {
                    Ok(request) => {
                        let request_id = request.request_id.to_string();
                        match inference::scoring::batch_inference(
                            &feat_len,
                            &model,
                            request.inputs,
                            &vec_config,
                        ) {
                            Ok((outputs, inference_log_rows, inference_micros)) => {
                                let inference_response = InferenceResponse {
                                    request_id: request_id.clone(),
                                    outputs,
                                };
                                let inference_log_batch = InferenceLogBatch {
                                    request_id,
                                    model_id: model_id.clone(),
                                    inference_log_rows,
                                };
                                let _ = metrics_sender.send(inference_micros);
                                let _ = log_sender.send(inference_log_batch);
                                Ok(inference_response)
                            }
                            Err(e) => Err(Status::internal(e.to_string())),
                        }
                    }
                    Err(e) => Err(Status::internal(e.to_string())),
                };
                if tx.send(send_result).await.is_err() {
                    println!("Receiver dropped, client disconnected");
                    break;
                }
            }
        });
        Ok(Response::new(Box::pin(output_stream)))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let ip_address: Ipv4Addr = args
        .get(1)
        .expect("CLI argument (1) IP address is missing")
        .parse()
        .unwrap();
    let service_config_path = args
        .get(2)
        .expect("CLI argument (2) service configuration path is missing");
    let aws_config = aws_config::load_from_env().await;
    let s3_client = aws_sdk_s3::Client::new(&aws_config);
    let cloudwatch_client = Arc::new(aws_sdk_cloudwatch::Client::new(&aws_config));
    let kinesis_client = Arc::new(aws_sdk_kinesis::Client::new(&aws_config));

    let s3_reader = io::S3Reader::new(s3_client);
    let service_config = config::HusharServiceConfig::from_json(
        &s3_reader.read_string(&service_config_path).await.unwrap(),
    )
    .unwrap();
    let vec_config = config::VectorizationConfig::from_json(
        &s3_reader
            .read_string(&service_config.vectorization_instruction_path)
            .await
            .unwrap(),
    )
    .unwrap();
    let (tract_model, feat_len) = io::load_onnx_model(
        &s3_reader
            .read_bytes(&service_config.vectorization_instruction_path)
            .await
            .unwrap(),
    )
    .unwrap();

    let (metrics_sender, metrics_receiver) =
        tokio::sync::mpsc::channel::<inference::InferenceMicros>(200);
    let shared_metrics_receiver = Arc::new(tokio::sync::Mutex::new(metrics_receiver));
    let (log_sender, log_receiver) = tokio::sync::mpsc::channel::<InferenceLogBatch>(200);
    let shared_log_receiver = Arc::new(tokio::sync::Mutex::new(log_receiver));
    for worker_id in 0..8 {
        let worker_client = Arc::clone(&kinesis_client);
        let worker_receiver = Arc::clone(&shared_log_receiver);
        tokio::spawn(io::feature_logger(
            worker_client,
            worker_receiver,
            "HusharLogStream",
            worker_id,
        ));
    }
    for worker_id in 0..4 {
        let worker_client = Arc::clone(&cloudwatch_client);
        let worker_receiver = Arc::clone(&shared_metrics_receiver);
        tokio::spawn(io::inference_metrics_sidecar(
            500,
            worker_client,
            "Hushar",
            worker_receiver,
            worker_id,
        ));
    }

    let server_addr = SocketAddr::from((ip_address, service_config.port_number.clone() as u16));
    let hushar_service = HusharService {
        feat_len,
        log_sender,
        model: Arc::new(tract_model),
        metrics_sender,
        service_config,
        vec_config: Arc::new(vec_config),
    };

    Server::builder()
        .concurrency_limit_per_connection(4)
        .tcp_keepalive(Some(std::time::Duration::from_secs(30)))
        .tcp_nodelay(true)
        .add_service(HusharServer::new(hushar_service))
        .serve(server_addr)
        .await?;

    Ok(())
}
