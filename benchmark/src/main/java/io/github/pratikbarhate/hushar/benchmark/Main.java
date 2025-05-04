package io.github.pratikbarhate.hushar.benchmark;

public class Main {
    public static void main(String[] args) {
        if (args.length < 5) {
            System.err.println("Usage: java -jar benchmark.jar <host> <port> <requestsPerSecond> <numThreads> <durationMinutes>");
            System.exit(1);
        }
        
        String host = args[0];
        int port = Integer.parseInt(args[1]);
        double requestsPerSecond = Double.parseDouble(args[2]);
        int numThreads = Integer.parseInt(args[3]);
        int durationMinutes = Integer.parseInt(args[4]);
        
        BenchmarkClient client = new BenchmarkClient(host, port, requestsPerSecond, 
                                                   numThreads, durationMinutes);
        
        try {
            client.runBenchmark();
        } catch (InterruptedException e) {
            e.printStackTrace();
            Thread.currentThread().interrupt();
        }
    }
}
