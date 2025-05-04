package io.github.pratikbarhate.hushar.benchmark;

import java.util.logging.Logger;

public class Main {
    private static final Logger logger = Logger.getLogger(Main.class.getName());
    
    public static void main(String[] args) {
        if (args.length < 6) {
            System.err.println("Usage: java -jar benchmark.jar <host> <port> <requestsPerSecond> <numThreads> <durationMinutes> <awsRegion>");
            System.err.println("Example: java -jar benchmark.jar localhost 8080 100.0 10 5 us-west-2");
            System.exit(1);
        }
        
        try {
            String host = args[0];
            int port = Integer.parseInt(args[1]);
            double requestsPerSecond = Double.parseDouble(args[2]);
            int numThreads = Integer.parseInt(args[3]);
            int durationMinutes = Integer.parseInt(args[4]);
            String awsRegion = args[5];
            String cloudwatchNamespace = "HusharBenchmark";
            
            logger.info("Starting Hushar benchmark client with the following configuration:");
            logger.info("Server: " + host + ":" + port);
            logger.info("Threads: " + numThreads);
            logger.info("Request rate: " + requestsPerSecond + " req/sec");
            logger.info("Duration: " + durationMinutes + " minutes");
            logger.info("AWS Region: " + awsRegion);
            logger.info("CloudWatch Namespace: " + cloudwatchNamespace);
            
            BenchmarkClient client = new BenchmarkClient(
                host, 
                port, 
                numThreads, 
                requestsPerSecond, 
                durationMinutes,
                awsRegion,
                cloudwatchNamespace
            );
            
            client.runBenchmark();
            
        } catch (NumberFormatException e) {
            System.err.println("Error: Invalid number format");
            System.err.println("Usage: java -jar benchmark.jar <host> <port> <requestsPerSecond> <numThreads> <durationMinutes> <awsRegion> <cloudwatchNamespace>");
            System.exit(1);
        } catch (Exception e) {
            logger.severe("Benchmark failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
        
        logger.info("Benchmark client shutting down");
    }
}