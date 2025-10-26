// Copyright (c) 2025 Pratik Barhate
// Licensed under the MIT License. See the LICENSE file in the project root for more information.

use hushar::hushar_proto::{
    hushar_server::{Hushar, HusharServer},
    InferenceLogBatch, InferenceRequest, InferenceResponse,
};
use inference::TractRunnableModel;
use io::FileReader;
use std::env;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use tonic::{transport::Server, Request, Response, Status};

pub(crate) mod config;
pub(crate) mod inference;
pub(crate) mod io;

#[derive(Debug)]
pub(crate) struct HusharService {
    feat_len: usize,
    model: Arc<TractRunnableModel>,
    log_sender: async_channel::Sender<InferenceLogBatch>,
    metrics_sender: async_channel::Sender<inference::InferenceMicros>,
    service_config: config::HusharServiceConfig,
    vec_config: Arc<config::VectorizationConfig>,
}

#[tonic::async_trait]
impl Hushar for HusharService {
    async fn inference_service(
        &self,
        request: Request<InferenceRequest>,
    ) -> Result<Response<InferenceResponse>, Status> {
        let inference_req = request.into_inner();
        let request_id = inference_req.request_id;
        match inference::scoring::batch_inference(
            &self.feat_len,
            &self.model,
            inference_req.inputs,
            &self.vec_config,
        ) {
            Ok((outputs, inference_log_rows, inference_micros)) => {
                let inference_response = InferenceResponse {
                    request_id: request_id.clone(),
                    outputs,
                };
                let inference_log_batch = InferenceLogBatch {
                    request_id,
                    model_id: self.service_config.model_id.clone(),
                    inference_log_rows,
                };
                let _ = &self.metrics_sender.send(inference_micros);
                let _ = &self.log_sender.send(inference_log_batch);
                Ok(Response::new(inference_response))
            }
            Err(e) => Err(Status::internal(e.to_string())),
        }
    }
}

pub fn main() -> Result<(), Box<dyn std::error::Error>> {
    let num_cpus = std::thread::available_parallelism().unwrap().get();
    let args: Vec<String> = env::args().collect();
    let ip_address: Ipv4Addr = args
        .get(1)
        .expect("CLI argument (1) IP address is missing")
        .parse()
        .unwrap();
    let service_config_path = args
        .get(2)
        .expect("CLI argument (2) service configuration path is missing");
    let inference_log_path = args
        .get(3)
        .expect("CLI argument (3) feature log path")
        .to_string();
    let met_thread_cnt = args
        .get(4)
        .expect("CLI argument (4) metrics thread count is missing")
        .parse::<usize>()
        .unwrap();
    let log_thread_cnt = args
        .get(5)
        .expect("CLI argument (5) data logger thread count is missing")
        .parse::<usize>()
        .unwrap();

    let log_s3_uri = inference_log_path
        .strip_prefix("s3://")
        .ok_or("Invalid S3 URI format")?;
    let mut log_s3_uri_parts = log_s3_uri.splitn(2, '/');
    let inference_log_bucket = Arc::new(
        log_s3_uri_parts
            .next()
            .ok_or("Missing bucket name in S3 path")?
            .to_string(),
    );
    let inference_log_prefix = Arc::new(
        log_s3_uri_parts
            .next()
            .ok_or("Missing key in S3 path")?
            .to_string(),
    );

    let server_thread_cnt = num_cpus - (met_thread_cnt + log_thread_cnt);

    let config_runtime = tokio::runtime::Builder::new_current_thread()
        .thread_name("hushar-config-loader")
        .enable_all()
        .build()?;

    let (
        service_config,
        vec_config,
        tract_model,
        feat_len,
        metrics_sender,
        metrics_receiver,
        log_sender,
        log_receiver,
        cloudwatch_client,
        s3_client,
    ) = config_runtime.block_on(async {
        let aws_config = aws_config::load_from_env().await;
        let s3_client = Arc::new(aws_sdk_s3::Client::new(&aws_config));
        let cloudwatch_client = Arc::new(aws_sdk_cloudwatch::Client::new(&aws_config));

        let s3_reader = io::S3Reader::new(Arc::clone(&s3_client));
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
                .read_bytes(&service_config.model_path)
                .await
                .unwrap(),
        )
        .unwrap();

        let (metrics_sender, metrics_receiver) = async_channel::bounded(200);
        let (log_sender, log_receiver) = async_channel::bounded(200);

        (
            service_config,
            vec_config,
            tract_model,
            feat_len,
            metrics_sender,
            metrics_receiver,
            log_sender,
            log_receiver,
            cloudwatch_client,
            s3_client,
        )
    });

    config_runtime.shutdown_background();

    let server_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(server_thread_cnt)
        .thread_name("hushar-server-worker")
        .enable_all()
        .build()?;

    let metrics_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(met_thread_cnt)
        .thread_name("hushar-metrics-worker")
        .enable_all()
        .build()?;

    let logging_runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(log_thread_cnt)
        .thread_name("hushar-logging-worker")
        .enable_all()
        .build()?;

    for worker_id in 0..log_thread_cnt {
        let worker_client = Arc::clone(&s3_client);
        logging_runtime.spawn(io::side_car::feature_logger(
            500,
            worker_client,
            log_receiver.clone(),
            inference_log_bucket.clone(),
            inference_log_prefix.clone(),
            worker_id,
        ));
    }
    for worker_id in 0..met_thread_cnt {
        let worker_client = Arc::clone(&cloudwatch_client);
        metrics_runtime.spawn(io::side_car::metrics_emitter(
            500,
            worker_client,
            "HusharService",
            metrics_receiver.clone(),
            worker_id,
        ));
    }

    let server_addr = SocketAddr::from((ip_address, service_config.port_number.clone()));
    let connection_concurrency = service_config.connection_concurrency.clone() as usize;
    let hushar_service = HusharService {
        feat_len,
        model: Arc::new(tract_model),
        log_sender,
        metrics_sender,
        service_config,
        vec_config: Arc::new(vec_config),
    };
    server_runtime.block_on(async {
        let _ = Server::builder()
            .concurrency_limit_per_connection(connection_concurrency)
            .tcp_keepalive(Some(std::time::Duration::from_secs(30)))
            .tcp_nodelay(true)
            .add_service(HusharServer::new(hushar_service))
            .serve(server_addr)
            .await;
    });

    metrics_runtime.shutdown_background();
    logging_runtime.shutdown_background();
    server_runtime.shutdown_background();

    Ok(())
}
