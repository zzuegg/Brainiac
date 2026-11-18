package io.github.zzuegg.brainiac.experiments.serving;

import io.github.zzuegg.brainiac.experimental.llm.ChunkedPrefillScheduler;
import io.github.zzuegg.brainiac.experimental.nn.LoraAdapterDense;
import io.github.zzuegg.brainiac.experimental.serving.FairScheduler;
import io.github.zzuegg.brainiac.experimental.serving.MultiLoraBatcher;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Serving shootout, throughput and scheduling rows (8.4.7 and 8.4.8).
 *
 * <p>{@link #multiLora}: requests with Zipf-distributed adapter popularity arrive with Poisson
 * gaps (in decode iterations, as {@code language.ServingThroughput} draws them) and each decodes
 * 16 tokens through a 256-wide layer with rank-8 {@link LoraAdapterDense} adapters. Policies:
 * {@link MultiLoraBatcher#forward SGMV} batching of every running request (up to 32);
 * per-adapter batching with the adapter merged into the weight (each iteration serves the
 * adapter whose oldest request waited longest; merging on a switch); and per-adapter batching
 * unmerged. Reported per adapter count: tokens per iteration (the throughput of a memory-bound
 * decode step, whose cost hardly depends on the batch), p95 latency in iterations, and tokens per
 * second of measured CPU compute (where a matmul costs in proportion to its rows, so small
 * per-adapter batches are not penalised as they are on an accelerator).
 *
 * <p>{@link #fairness}: two light clients and one heavy client sending ten times their load
 * (the light clients are active only in the middle half of the trace), served by
 * {@link FairScheduler}: FCFS, VTC and per-client request-rate limiting. Reported: the largest
 * service gap between the heavy and a light client over 40-iteration windows in which both are
 * backlogged, light-client p95 latency and tokens per iteration.
 */
public final class ThroughputShootout {

    private ThroughputShootout() {
    }

    /**
     * One multi-adapter policy.
     *
     * @param tokensPerSecond decode tokens per second of measured compute
     * @param tokensPerIteration decode tokens per iteration
     * @param p95Latency 95th percentile of iterations from arrival to the last token
     */
    public record LoraRow(double tokensPerSecond, double tokensPerIteration, double p95Latency) {
    }

    private record Req(int adapter, int arrival) {
    }

    /**
     * Runs the multi-adapter rows.
     *
     * @param adapterCounts the adapter counts, such as 4, 16, 64
     * @param requests requests per run
     * @param rate mean arrivals per iteration
     * @param seed the seed
     * @return adapter count to policy to row
     */
    public static Map<Integer, Map<String, LoraRow>> multiLora(int[] adapterCounts, int requests, double rate,
            long seed) {
        int width = 256;
        int batch = 32;
        int decode = 16;
        Map<Integer, Map<String, LoraRow>> out = new LinkedHashMap<>();
        SplittableRandom init = new SplittableRandom(seed);
        Dense base = new Dense(width, width, Activation.IDENTITY, Tensor.normal(Shape.of(width, width), 0,
                0.05f, init), Tensor.zeros(Shape.of(width)));
        for (int count : adapterCounts) {
            List<LoraAdapterDense> adapters = new ArrayList<>();
            for (int i = 0; i < count; i++) {
                LoraAdapterDense ad = LoraAdapterDense.lora(base, 8, 16, init);
                ad.b().assign(Tensor.normal(Shape.of(8, width), 0, 0.05f, init));
                adapters.add(ad);
            }
            MultiLoraBatcher batcher = MultiLoraBatcher.of(base, adapters);
            // Zipf(1) popularity and Poisson arrivals
            double[] cdf = new double[count];
            double z = 0;
            for (int i = 0; i < count; i++) {
                z += 1.0 / (i + 1);
                cdf[i] = z;
            }
            SplittableRandom r = new SplittableRandom(seed + count);
            List<Req> trace = new ArrayList<>();
            double t = 0;
            for (int i = 0; i < requests; i++) {
                t += -Math.log(1 - r.nextDouble()) / rate;
                double u = r.nextDouble() * z;
                int a = 0;
                while (cdf[a] < u) {
                    a++;
                }
                trace.add(new Req(a, (int) Math.floor(t)));
            }
            Map<String, LoraRow> rows = new LinkedHashMap<>();
            if (out.isEmpty()) {   // warm up the JIT before anything is timed
                for (String policy : List.of("SGMV", "merged", "unmerged")) {
                    simulate(policy, batcher, trace, batch, decode, width, seed);
                }
            }
            for (String policy : List.of("SGMV (S-LoRA/Punica)", "merged, per-adapter batches",
                    "unmerged, per-adapter batches")) {
                rows.put(policy, simulate(policy, batcher, trace, batch, decode, width, seed));
            }
            out.put(count, rows);
        }
        return out;
    }

    private static LoraRow simulate(String policy, MultiLoraBatcher batcher, List<Req> trace, int batch, int decode,
            int width, long seed) {
        int n = trace.size();
        int[] left = new int[n];
        Arrays.fill(left, decode);
        double[] latency = new double[n];
        int next = 0;
        int done = 0;
        int iteration = 0;
        long nanos = 0;
        long tokens = 0;
        List<Integer> active = new ArrayList<>();   // arrived, unfinished, in arrival order
        SplittableRandom r = new SplittableRandom(seed);
        while (done < n) {
            while (next < n && trace.get(next).arrival() <= iteration) {
                active.add(next++);
            }
            if (active.isEmpty()) {
                iteration = trace.get(next).arrival();
                continue;
            }
            List<Integer> serve = new ArrayList<>();
            int adapter = -1;
            if (policy.startsWith("SGMV")) {
                for (int i = 0; i < active.size() && serve.size() < batch; i++) {
                    serve.add(active.get(i));
                }
            } else {
                adapter = trace.get(active.getFirst()).adapter();
                for (int i : active) {
                    if (trace.get(i).adapter() == adapter && serve.size() < batch) {
                        serve.add(i);
                    }
                }
            }
            float[] x = new float[serve.size() * width];
            for (int i = 0; i < x.length; i++) {
                x[i] = (float) r.nextGaussian();
            }
            Tensor in = Tensor.of(Shape.of(serve.size(), width), x);
            long t0 = System.nanoTime();
            if (policy.startsWith("SGMV")) {
                int[] ids = new int[serve.size()];
                for (int i = 0; i < ids.length; i++) {
                    ids[i] = trace.get(serve.get(i)).adapter();
                }
                batcher.forward(in, ids).toArray();
            } else if (policy.startsWith("merged")) {
                batcher.forwardMerged(in, adapter).toArray();
            } else {
                batcher.forwardUnmerged(in, adapter).toArray();
            }
            nanos += System.nanoTime() - t0;
            tokens += serve.size();
            for (int i : serve) {
                if (--left[i] == 0) {
                    latency[i] = iteration + 1 - trace.get(i).arrival();
                    done++;
                }
            }
            active.removeIf(i -> left[i] == 0);
            iteration++;
        }
        double[] sorted = latency.clone();
        Arrays.sort(sorted);
        return new LoraRow(tokens / (nanos / 1e9), tokens / (double) iteration,
                ChunkedPrefillScheduler.Report.quantile(sorted, 0.95));
    }

    /**
     * One scheduling policy.
     *
     * @param serviceGap the largest heavy-versus-light service gap over 40-iteration windows
     *     with both backlogged
     * @param lightP95 the light clients' p95 latency in iterations
     * @param heavyP95 the heavy client's p95 latency in iterations
     * @param throughput tokens per iteration
     */
    public record FairRow(double serviceGap, double lightP95, double heavyP95, double throughput) {
    }

    /**
     * Runs the scheduling rows.
     *
     * @param iterations the trace length in iterations
     * @param seeds the seeds
     * @return policy to row, averaged over seeds
     */
    public static Map<String, FairRow> fairness(int iterations, long[] seeds) {
        Map<String, FairScheduler> policies = new LinkedHashMap<>();
        policies.put("FCFS", FairScheduler.fcfs());
        policies.put("VTC", FairScheduler.vtc(1, 2));
        policies.put("request-rate limit", FairScheduler.requestRateLimit(2, 10));
        Map<String, double[]> sums = new LinkedHashMap<>();
        for (long seed : seeds) {
            SplittableRandom r = new SplittableRandom(seed);
            List<FairScheduler.Request> trace = new ArrayList<>();
            double[] rates = {1.0, 0.1, 0.1};   // requests per iteration: one heavy, two light clients
            for (int c = 0; c < 3; c++) {
                double t = c == 0 ? 0 : iterations / 4.0;
                double end = c == 0 ? iterations : 3 * iterations / 4.0;
                while (true) {
                    t += -Math.log(1 - r.nextDouble()) / rates[c];
                    if (t >= end) {
                        break;
                    }
                    trace.add(new FairScheduler.Request(c, 8 + r.nextInt(24), 8 + r.nextInt(24), (int) t));
                }
            }
            for (Map.Entry<String, FairScheduler> e : policies.entrySet()) {
                FairScheduler.Report rep = e.getValue().run(trace, 3, 12);
                List<Double> light = new ArrayList<>();
                List<Double> heavy = new ArrayList<>();
                for (int i = 0; i < trace.size(); i++) {
                    (trace.get(i).client() == 0 ? heavy : light).add(rep.latency()[i]);
                }
                double[] s = sums.computeIfAbsent(e.getKey(), k -> new double[4]);
                s[0] += Math.max(rep.serviceGap(0, 1, 40), rep.serviceGap(0, 2, 40)) / seeds.length;
                s[1] += p95(light) / seeds.length;
                s[2] += p95(heavy) / seeds.length;
                s[3] += rep.throughput() / seeds.length;
            }
        }
        Map<String, FairRow> out = new LinkedHashMap<>();
        sums.forEach((k, s) -> out.put(k, new FairRow(s[0], s[1], s[2], s[3])));
        return out;
    }

    private static double p95(List<Double> xs) {
        double[] a = xs.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        return ChunkedPrefillScheduler.Report.quantile(a, 0.95);
    }
}
