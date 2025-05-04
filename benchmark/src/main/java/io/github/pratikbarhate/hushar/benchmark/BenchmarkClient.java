package io.github.pratikbarhate.hushar.benchmark;

import com.google.common.util.concurrent.RateLimiter;
import hushar.HusharGrpc;
import hushar.Structs.InferenceRequest;
import hushar.Structs.InferenceResponse;
import hushar.Structs;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.stub.StreamObserver;
import software.amazon.awssdk.services.cloudwatch.CloudWatchClient;
import software.amazon.awssdk.services.cloudwatch.model.*;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;

public class BenchmarkClient {
    private final ManagedChannel channel;
    private final HusharGrpc.HusharStub asyncStub;
    private final CloudWatchClient cloudWatch;
    private final RateLimiter rateLimiter;
    private final int numThreads;
    private final long durationSeconds;
    
    // Metrics tracking
    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong successfulRequests = new AtomicLong(0);
    private final AtomicLong failedRequests = new AtomicLong(0);
    private final ConcurrentLinkedQueue<Long> latencies = new ConcurrentLinkedQueue<>();
    
    public BenchmarkClient(String host, int port, double requestsPerSecond, 
                          int numThreads, int durationMinutes) {
        this.channel = ManagedChannelBuilder.forAddress(host, port)
                .usePlaintext()
                .build();
        this.asyncStub = HusharGrpc.newStub(channel);
        this.cloudWatch = CloudWatchClient.create();
        this.rateLimiter = RateLimiter.create(requestsPerSecond);
        this.numThreads = numThreads;
        this.durationSeconds = TimeUnit.MINUTES.toSeconds(durationMinutes);
    }
    
    public void runBenchmark() throws InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(numThreads);
        List<Future<?>> futures = new ArrayList<>();
        
        long startTime = System.currentTimeMillis();
        long endTime = startTime + TimeUnit.SECONDS.toMillis(durationSeconds);
        
        // Start all worker threads
        for (int i = 0; i < numThreads; i++) {
            futures.add(executor.submit(() -> workerTask(endTime)));
        }
        
        // Wait for all threads to complete
        for (Future<?> future : futures) {
            future.get();
        }
        
        executor.shutdown();
        executor.awaitTermination(10, TimeUnit.SECONDS);
        
        // Send metrics to CloudWatch
        sendMetricsToCloudWatch();
        shutdown();
    }
    
    private void workerTask(long endTime) {
        while (System.currentTimeMillis() < endTime) {
            rateLimiter.acquire();
            
            String requestId = UUID.randomUUID().toString();
            long startNanos = System.nanoTime();
            
            InferenceRequest request = createInferenceRequest(requestId);
            CountDownLatch responseLatch = new CountDownLatch(1);
            
            StreamObserver<InferenceResponse> responseObserver = new StreamObserver<InferenceResponse>() {
                @Override
                public void onNext(InferenceResponse response) {
                    long latencyMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - startNanos);
                    latencies.add(latencyMs);
                    successfulRequests.incrementAndGet();
                    responseLatch.countDown();
                }
                
                @Override
                public void onError(Throwable t) {
                    failedRequests.incrementAndGet();
                    responseLatch.countDown();
                }
                
                @Override
                public void onCompleted() {}
            };
            
            StreamObserver<InferenceRequest> requestObserver = 
                asyncStub.inferenceService(responseObserver);
            
            try {
                requestObserver.onNext(request);
                requestObserver.onCompleted();
                responseLatch.await(5, TimeUnit.SECONDS);
            } catch (Exception e) {
                failedRequests.incrementAndGet();
            }
            
            totalRequests.incrementAndGet();
        }
    }
    
    private InferenceRequest createInferenceRequest(String requestId) {
        Random random = new Random();
        int numRows = 80 + random.nextInt(41); // 80-120 rows
        
        List<Structs.InputRow> inputs = new ArrayList<>();
        
        for (int i = 0; i < numRows; i++) {
            Map<String, Structs.DataType> features = new HashMap<>();
            
            // Add test features
            features.put("double_feature", Structs.DataType.newBuilder()
                .setDoubleValue(random.nextDouble() * 100)
                .build());
            
            features.put("float_feature", Structs.DataType.newBuilder()
                .setFloatValue(random.nextFloat() * 100)
                .build());
            
            features.put("integer_feature", Structs.DataType.newBuilder()
                .setIntegerValue(random.nextInt(1000))
                .build());
            
            features.put("string_feature", Structs.DataType.newBuilder()
                .setStringValue("test-" + random.nextInt(100))
                .build());
            
            Structs.InputRow row = Structs.InputRow.newBuilder()
                .setRowId("row-" + i)
                .putAllFeatures(features)
                .build();
            
            inputs.add(row);
        }
        
        return InferenceRequest.newBuilder()
            .setRequestId(requestId)
            .addAllInputs(inputs)
            .build();
    }
    
    private void sendMetricsToCloudWatch() {
        long total = totalRequests.get();
        long successful = successfulRequests.get();
        long failed = failedRequests.get();
        double throughput = successful / (double) durationSeconds;
        
        List<MetricDatum> metrics = new ArrayList<>();
        Instant timestamp = Instant.now();
        
        // Summary metrics
        metrics.add(MetricDatum.builder()
            .metricName("TotalRequests")
            .unit(StandardUnit.COUNT)
            .value((double) total)
            .timestamp(timestamp)
            .build());
        
        metrics.add(MetricDatum.builder()
            .metricName("SuccessfulRequests")
            .unit(StandardUnit.COUNT)
            .value((double) successful)
            .timestamp(timestamp)
            .build());
        
        metrics.add(MetricDatum.builder()
            .metricName("FailedRequests")
            .unit(StandardUnit.COUNT)
            .value((double) failed)
            .timestamp(timestamp)
            .build());
        
        metrics.add(MetricDatum.builder()
            .metricName("ThroughputRPS")
            .unit(StandardUnit.COUNT_SECOND)
            .value(throughput)
            .timestamp(timestamp)
            .build());
        
        // Send individual latency measurements
        for (Long latency : latencies) {
            metrics.add(MetricDatum.builder()
                .metricName("RequestLatency")
                .unit(StandardUnit.MILLISECONDS)
                .value(latency.doubleValue())
                .timestamp(timestamp)
                .build());
            
            // CloudWatch has a limit of 20 metrics per request
            if (metrics.size() >= 20) {
                sendMetricBatch(metrics);
                metrics.clear();
            }
        }
        
        // Send any remaining metrics
        if (!metrics.isEmpty()) {
            sendMetricBatch(metrics);
        }
    }
    
    private void sendMetricBatch(List<MetricDatum> metrics) {
        PutMetricDataRequest request = PutMetricDataRequest.builder()
            .namespace("HusharLatencyBenchmark")
            .metricData(metrics)
            .build();
        
        try {
            cloudWatch.putMetricData(request);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
    
    public void shutdown() {
        try {
            channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        cloudWatch.close();
    }
}