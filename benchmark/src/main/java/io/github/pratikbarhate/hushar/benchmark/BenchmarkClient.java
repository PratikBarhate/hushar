package io.github.pratikbarhate.hushar.benchmark;

import com.google.common.util.concurrent.RateLimiter;
import io.github.pratikbarhate.hushar.HusharGrpc;
import io.github.pratikbarhate.hushar.Structs;
import io.github.pratikbarhate.hushar.Structs.DataType;
import io.github.pratikbarhate.hushar.Structs.InferenceRequest;
import io.github.pratikbarhate.hushar.Structs.InferenceResponse;
import io.github.pratikbarhate.hushar.Structs.InputRow;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import software.amazon.awssdk.auth.credentials.DefaultCredentialsProvider;
import software.amazon.awssdk.regions.Region;
import software.amazon.awssdk.services.cloudwatch.CloudWatchClient;
import software.amazon.awssdk.services.cloudwatch.model.MetricDatum;
import software.amazon.awssdk.services.cloudwatch.model.PutMetricDataRequest;
import software.amazon.awssdk.services.cloudwatch.model.StandardUnit;

import java.time.Instant;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;

public class BenchmarkClient {
    private static final Logger logger = Logger.getLogger(BenchmarkClient.class.getName());
    
    private final String serverHost;
    private final int serverPort;
    private final int numThreads;
    private final double requestRate;
    private final int durationMinutes;
    private final String awsRegion;
    private final String cloudWatchNamespace;
    private final List<InputRow> staticInputs;
    private final CloudWatchClient cloudWatchClient;
    private final ExecutorService executorService;
    private final AtomicLong totalRequests = new AtomicLong(0);
    private final AtomicLong successfulRequests = new AtomicLong(0);
    
    public BenchmarkClient(String serverHost, int serverPort, int numThreads, double requestRate, 
                          int durationMinutes, String awsRegion, String cloudWatchNamespace) {
        this.serverHost = serverHost;
        this.serverPort = serverPort;
        this.numThreads = numThreads;
        this.requestRate = requestRate;
        this.durationMinutes = durationMinutes;
        this.awsRegion = awsRegion;
        this.cloudWatchNamespace = cloudWatchNamespace;
        this.staticInputs = createStaticInputs();
        this.cloudWatchClient = createCloudWatchClient();
        this.executorService = Executors.newFixedThreadPool(numThreads);
    }
    
    private CloudWatchClient createCloudWatchClient() {
        return CloudWatchClient.builder()
                .region(Region.of(awsRegion))
                .credentialsProvider(DefaultCredentialsProvider.create())
                .build();
    }
    
    private List<InputRow> createStaticInputs() {
        List<InputRow> inputs = new ArrayList<>();
        Random random = new Random(42); // Fixed seed for reproducible testing
        
        Map<String, DataType> features_1 = new HashMap<>();
        features_1.put("embedding_text_1", createStringValue("excellent"));
        features_1.put("embedding_category_2", createStringValue("electronics"));
        features_1.put("embedding_sentiment_3", createStringValue("positive"));
        features_1.put("embedding_location_4", createStringValue("Asia"));
        features_1.put("onehot_color_1", createStringValue("maroon"));
        features_1.put("onehot_size_2", createStringValue("16x32"));
        features_1.put("onehot_category_3", createStringValue("bikes"));
        features_1.put("onehot_status_4", createStringValue("balanced"));
        features_1.put("identity_scores_1", createDoubleArray(generateArray(random, 2)));
        features_1.put("identity_arr", createDoubleArray(generateArray(random, 4)));
        features_1.put("minmax32_temperature_1", createFloatValue(random.nextFloat() * 100));
        features_1.put("minmax32_price_2", createFloatValue(random.nextFloat() * 100));
        features_1.put("minmax32_percentage_3", createFloatValue(random.nextFloat() * 100));
        features_1.put("minmax64_revenue_1", createDoubleValue(random.nextDouble() * 100));
        features_1.put("minmax64_distance_2", createDoubleValue(random.nextDouble() * 100));
        features_1.put("minmax64_timestamp_3", createDoubleValue(random.nextDouble() * 100));
        features_1.put("standard32_age_1", createFloatValue(random.nextFloat() * 100));
        features_1.put("standard32_weight_2", createFloatValue(random.nextFloat() * 100));
        features_1.put("standard32_duration_3", createFloatValue(random.nextFloat() * 100));
        features_1.put("standard64_income_1", createDoubleValue(random.nextDouble() * 100));
        features_1.put("standard64_population_2", createDoubleValue(random.nextDouble() * 100));
        features_1.put("standard64_views_3", createDoubleValue(random.nextDouble() * 100));
        InputRow row_1 = Structs.InputRow.newBuilder()
                .setRowId("row_1")
                .putAllFeatures(features_1)
                .build();
        inputs.add(row_1);

        Map<String, DataType> features_2 = new HashMap<>();
        features_2.put("embedding_text_1", createStringValue("good"));
        features_2.put("embedding_category_2", createStringValue("clothing"));
        features_2.put("embedding_sentiment_3", createStringValue("negative"));
        features_2.put("embedding_location_4", createStringValue("Europe"));
        features_2.put("onehot_color_1", createStringValue("coral"));
        features_2.put("onehot_size_2", createStringValue("12x12"));
        features_2.put("onehot_category_3", createStringValue("bikes"));
        features_2.put("onehot_status_4", createStringValue("balanced"));
        features_2.put("identity_scores_1", createDoubleArray(generateArray(random, 2)));
        features_2.put("identity_arr", createDoubleArray(generateArray(random, 4)));
        features_2.put("minmax32_temperature_1", createFloatValue(random.nextFloat() * 100));
        features_2.put("minmax32_price_2", createFloatValue(random.nextFloat() * 100));
        features_2.put("minmax32_percentage_3", createFloatValue(random.nextFloat() * 100));
        features_2.put("minmax64_revenue_1", createDoubleValue(random.nextDouble() * 100));
        features_2.put("minmax64_distance_2", createDoubleValue(random.nextDouble() * 100));
        features_2.put("minmax64_timestamp_3", createDoubleValue(random.nextDouble() * 100));
        features_2.put("standard32_age_1", createFloatValue(random.nextFloat() * 100));
        features_2.put("standard32_weight_2", createFloatValue(random.nextFloat() * 100));
        features_2.put("standard32_duration_3", createFloatValue(random.nextFloat() * 100));
        features_2.put("standard64_income_1", createDoubleValue(random.nextDouble() * 100));
        features_2.put("standard64_population_2", createDoubleValue(random.nextDouble() * 100));
        features_2.put("standard64_views_3", createDoubleValue(random.nextDouble() * 100));
        InputRow row_2 = Structs.InputRow.newBuilder()
                .setRowId("row_2")
                .putAllFeatures(features_2)
                .build();
        inputs.add(row_2);

        Map<String, DataType> features_3 = new HashMap<>();
        features_3.put("embedding_text_1", createStringValue("poor"));
        features_3.put("embedding_category_2", createStringValue("food"));
        features_3.put("embedding_sentiment_3", createStringValue("neutral"));
        features_3.put("embedding_location_4", createStringValue("USA"));
        features_3.put("onehot_color_1", createStringValue("lavender"));
        features_3.put("onehot_size_2", createStringValue("20x20"));
        features_3.put("onehot_category_3", createStringValue("cookware"));
        features_3.put("onehot_status_4", createStringValue("budgeted"));
        features_3.put("identity_scores_1", createDoubleArray(generateArray(random, 2)));
        features_3.put("identity_arr", createDoubleArray(generateArray(random, 4)));
        features_3.put("minmax32_temperature_1", createFloatValue(random.nextFloat() * 100));
        features_3.put("minmax32_price_2", createFloatValue(random.nextFloat() * 100));
        features_3.put("minmax32_percentage_3", createFloatValue(random.nextFloat() * 100));
        features_3.put("minmax64_revenue_1", createDoubleValue(random.nextDouble() * 100));
        features_3.put("minmax64_distance_2", createDoubleValue(random.nextDouble() * 100));
        features_3.put("minmax64_timestamp_3", createDoubleValue(random.nextDouble() * 100));
        features_3.put("standard32_age_1", createFloatValue(random.nextFloat() * 100));
        features_3.put("standard32_weight_2", createFloatValue(random.nextFloat() * 100));
        features_3.put("standard32_duration_3", createFloatValue(random.nextFloat() * 100));
        features_3.put("standard64_income_1", createDoubleValue(random.nextDouble() * 100));
        features_3.put("standard64_population_2", createDoubleValue(random.nextDouble() * 100));
        features_3.put("standard64_views_3", createDoubleValue(random.nextDouble() * 100));
        InputRow row_3 = Structs.InputRow.newBuilder()
                .setRowId("row_3")
                .putAllFeatures(features_3)
                .build();
        inputs.add(row_3);

        Map<String, DataType> features_4 = new HashMap<>();
        features_4.put("embedding_text_1", createStringValue("average"));
        features_4.put("embedding_category_2", createStringValue("food"));
        features_4.put("embedding_sentiment_3", createStringValue("negative"));
        features_4.put("embedding_location_4", createStringValue("Africa"));
        features_4.put("onehot_color_1", createStringValue("aqua"));
        features_4.put("onehot_size_2", createStringValue("11x14"));
        features_4.put("onehot_category_3", createStringValue("appliances"));
        features_4.put("onehot_status_4", createStringValue("adjusted"));
        features_4.put("identity_scores_1", createDoubleArray(generateArray(random, 2)));
        features_4.put("identity_arr", createDoubleArray(generateArray(random, 4)));
        features_4.put("minmax32_temperature_1", createFloatValue(random.nextFloat() * 100));
        features_4.put("minmax32_price_2", createFloatValue(random.nextFloat() * 100));
        features_4.put("minmax32_percentage_3", createFloatValue(random.nextFloat() * 100));
        features_4.put("minmax64_revenue_1", createDoubleValue(random.nextDouble() * 100));
        features_4.put("minmax64_distance_2", createDoubleValue(random.nextDouble() * 100));
        features_4.put("minmax64_timestamp_3", createDoubleValue(random.nextDouble() * 100));
        features_4.put("standard32_age_1", createFloatValue(random.nextFloat() * 100));
        features_4.put("standard32_weight_2", createFloatValue(random.nextFloat() * 100));
        features_4.put("standard32_duration_3", createFloatValue(random.nextFloat() * 100));
        features_4.put("standard64_income_1", createDoubleValue(random.nextDouble() * 100));
        features_4.put("standard64_population_2", createDoubleValue(random.nextDouble() * 100));
        features_4.put("standard64_views_3", createDoubleValue(random.nextDouble() * 100));
        InputRow row_4 = Structs.InputRow.newBuilder()
                .setRowId("row_4")
                .putAllFeatures(features_4)
                .build();
        inputs.add(row_4);

        return inputs;
    }
    
    private DataType createDoubleValue(double value) {
        return DataType.newBuilder().setDoubleValue(value).build();
    }
    
    private DataType createFloatValue(float value) {
        return DataType.newBuilder().setFloatValue(value).build();
    }
    
    private DataType createStringValue(String value) {
        return DataType.newBuilder().setStringValue(value).build();
    }
    
    private DataType createDoubleArray(double[] values) {
        Structs.DoubleArray array = Structs.DoubleArray.newBuilder()
                .addAllValues(Arrays.stream(values).boxed().toList())
                .build();
        return DataType.newBuilder().setDoubleArray(array).build();
    }
    
    private double[] generateArray(Random random, int size) {
        double[] array = new double[size];
        for (int i = 0; i < size; i++) {
            array[i] = random.nextDouble() * 1000;
        }
        return array;
    }
    
    public void runBenchmark() {
        logger.info("Starting benchmark with " + numThreads + " threads, rate: " + requestRate + " req/sec");
        
        ExecutorService workers = Executors.newFixedThreadPool(numThreads);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch endLatch = new CountDownLatch(numThreads);
        
        for (int i = 0; i < numThreads; i++) {
            workers.submit(() -> {
                try {
                    startLatch.await();
                    runWorkerThread();
                } catch (Exception e) {
                    logger.severe("Worker thread error: " + e.getMessage());
                } finally {
                    endLatch.countDown();
                }
            });
        }
        
        // Start all threads
        startLatch.countDown();
        
        // Wait for specified duration
        try {
            TimeUnit.MINUTES.sleep(durationMinutes);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.severe("Benchmark interrupted");
        }
        
        // Shutdown
        workers.shutdownNow();
        try {
            endLatch.await(10, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        cloudWatchClient.close();
        executorService.shutdown();
        
        logger.info("Benchmark completed. Total requests: " + totalRequests.get() + 
                   ", Successful: " + successfulRequests.get());
    }
    
    private void runWorkerThread() {
        ManagedChannel channel = ManagedChannelBuilder.forAddress(serverHost, serverPort)
                .usePlaintext()
                .build();
        
        HusharGrpc.HusharStub stub = HusharGrpc.newStub(channel);
        RateLimiter rateLimiter = RateLimiter.create(requestRate / numThreads);
        Random random = new Random(42);
        
        try {
            while (!Thread.currentThread().isInterrupted()) {
                rateLimiter.acquire();
                
                // Create request with 80-120 input rows
                int numRows = 80 + random.nextInt(42);
                List<InputRow> requestInputs = new ArrayList<>();
                
                for (int i = 0; i < numRows; i++) {
                    InputRow template = staticInputs.get(random.nextInt(staticInputs.size()));
                    InputRow row = Structs.InputRow.newBuilder()
                            .setRowId(template.getRowId() + "_" + System.nanoTime())
                            .putAllFeatures(template.getFeaturesMap())
                            .build();
                    requestInputs.add(row);
                }
                
                InferenceRequest request = InferenceRequest.newBuilder()
                        .setRequestId(UUID.randomUUID().toString())
                        .addAllInputs(requestInputs)
                        .build();
                
                long startTime = System.nanoTime();
                makeRequest(stub, request, startTime);
            }
        } catch (Exception e) {
            if (!Thread.currentThread().isInterrupted()) {
                logger.severe("Worker thread error: " + e.getMessage());
            }
        } finally {
            channel.shutdown();
            try {
                channel.awaitTermination(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    private void makeRequest(HusharGrpc.HusharStub stub, InferenceRequest request, long startTime) {
        totalRequests.incrementAndGet();
        
        stub.inferenceService(request, new io.grpc.stub.StreamObserver<InferenceResponse>() {
            @Override
            public void onNext(InferenceResponse response) {
                long latencyNanos = System.nanoTime() - startTime;
                double latencyMs = latencyNanos / 1_000_000.0;
                
                successfulRequests.incrementAndGet();
                logToCloudWatch(latencyMs, "Success");
                
                if (totalRequests.get() % 100 == 0) {
                    logger.info("Processed " + totalRequests.get() + " requests");
                }
            }
            
            @Override
            public void onError(Throwable t) {
                long latencyNanos = System.nanoTime() - startTime;
                double latencyMs = latencyNanos / 1_000_000.0;
                
                logger.warning("Request failed: " + t.getMessage());
                logToCloudWatch(latencyMs, "Error");
            }
            
            @Override
            public void onCompleted() {
                // Not used in unary RPC
            }
        });
    }
    
    private void logToCloudWatch(double latencyMs, String status) {
        executorService.submit(() -> {
            try {
                MetricDatum datum = MetricDatum.builder()
                        .metricName("InferenceLatency")
                        .value(latencyMs)
                        .unit(StandardUnit.MILLISECONDS)
                        .timestamp(Instant.now())
                        .dimensions(software.amazon.awssdk.services.cloudwatch.model.Dimension.builder()
                                .name("Status")
                                .value(status)
                                .build())
                        .build();
                
                PutMetricDataRequest putRequest = PutMetricDataRequest.builder()
                        .namespace(cloudWatchNamespace)
                        .metricData(datum)
                        .build();
                
                cloudWatchClient.putMetricData(putRequest);
            } catch (Exception e) {
                logger.warning("Failed to log to CloudWatch: " + e.getMessage());
            }
        });
    }
}