package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.experimental.llm.DisaggregatedScheduler;
import io.github.zzuegg.brainiac.experimental.llm.SharedPrefixAttention;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Two serving techniques next to {@link ServingThroughput}'s:
 * <ul>
 * <li>{@link #hydragen}: decode attention for many samples of one long prompt, with
 *     {@link SharedPrefixAttention} (the prefix attended once for all samples as one batched
 *     product, merged by log-sum-exp) against per-sequence attention over the shared prefix
 *     plus each suffix, as attention over copy-on-write paged blocks does;</li>
 * <li>{@link #disaggregation}: {@link DisaggregatedScheduler} with one prefill and one decode
 *     worker against two colocated workers running chunked prefill, on the same request
 *     trace and the same simulated clock, judged by goodput under TTFT and TPOT goals.</li>
 * </ul>
 */
public final class SharedPrefixServing {

    private SharedPrefixServing() {
    }

    /**
     * The shared-prefix attention comparison.
     *
     * @param perSequenceMillis per-sequence attention's time per decode step
     * @param hydragenMillis Hydragen's time per decode step
     * @param maxDifference the largest output difference between the two
     * @param keyReadsShare the keys read by Hydragen as a share of per-sequence attention's
     */
    public record HydragenResult(double perSequenceMillis, double hydragenMillis, double maxDifference,
            double keyReadsShare) {

        /** @return decode attention tokens per second, per-sequence over Hydragen's (higher is better) */
        public double speedup() {
            return perSequenceMillis / hydragenMillis;
        }
    }

    /**
     * Decode attention of {@code batch} sequences over a shared prompt.
     *
     * @param prefix the shared prompt's length
     * @param suffix each sequence's own tokens so far
     * @param batch the sequences
     * @param repeats timed decode steps (after a warm-up)
     * @param seed the seed
     * @return the timings and the difference
     */
    public static HydragenResult hydragen(int prefix, int suffix, int batch, int repeats, long seed) {
        int heads = 8;
        int kvHeads = 2;
        int dim = 32;
        SplittableRandom r = new SplittableRandom(seed);
        Tensor pk = Tensor.normal(Shape.of(prefix, kvHeads, dim), 0, 1, r);
        Tensor pv = Tensor.normal(Shape.of(prefix, kvHeads, dim), 0, 1, r);
        List<Tensor> sk = new ArrayList<>();
        List<Tensor> sv = new ArrayList<>();
        for (int b = 0; b < batch; b++) {
            sk.add(Tensor.normal(Shape.of(suffix, kvHeads, dim), 0, 1, r));
            sv.add(Tensor.normal(Shape.of(suffix, kvHeads, dim), 0, 1, r));
        }
        Tensor q = Tensor.normal(Shape.of(batch, heads, dim), 0, 1, r);
        float scale = (float) (1 / Math.sqrt(dim));
        float[] a = SharedPrefixAttention.perSequence(q, pk, pv, sk, sv, scale).toArray();
        float[] b = SharedPrefixAttention.attend(q, pk, pv, sk, sv, scale).toArray();
        double diff = 0;
        for (int i = 0; i < a.length; i++) {
            diff = Math.max(diff, Math.abs(a[i] - b[i]));
        }
        double[] best = {Double.MAX_VALUE, Double.MAX_VALUE};
        for (int rep = 0; rep < repeats; rep++) {
            long t0 = System.nanoTime();
            SharedPrefixAttention.perSequence(q, pk, pv, sk, sv, scale).toArray();
            long t1 = System.nanoTime();
            SharedPrefixAttention.attend(q, pk, pv, sk, sv, scale).toArray();
            long t2 = System.nanoTime();
            best[0] = Math.min(best[0], (t1 - t0) / 1e6);
            best[1] = Math.min(best[1], (t2 - t1) / 1e6);
        }
        double share = (prefix + (double) batch * suffix) / ((double) batch * (prefix + suffix));
        return new HydragenResult(best[0], best[1], diff, share);
    }

    /**
     * Both serving layouts.
     *
     * @param disaggregated one prefill and one decode worker
     * @param colocated two workers with chunked prefill
     * @param sameTokens whether both produced the same tokens for every request
     */
    public record DisaggregationResult(DisaggregatedScheduler.Report disaggregated,
            DisaggregatedScheduler.Report colocated, boolean sameTokens) {
    }

    /**
     * A Poisson trace of requests with long prompts and moderate outputs, served both ways.
     *
     * @param requests the requests
     * @param meanGapMillis the mean gap between arrivals
     * @param seed the seed
     * @return both reports
     */
    public static DisaggregationResult disaggregation(int requests, double meanGapMillis, long seed) {
        Llama core = ServingThroughput.model(64, 64, seed);
        SplittableRandom r = new SplittableRandom(seed);
        List<DisaggregatedScheduler.Request> trace = new ArrayList<>();
        double t = 0;
        for (int i = 0; i < requests; i++) {
            t += -Math.log(1 - r.nextDouble()) * meanGapMillis;
            trace.add(new DisaggregatedScheduler.Request(i, ServingThroughput.randomTokens(r, 96 + r.nextInt(64), 64),
                    16, t));
        }
        DisaggregatedScheduler.Cost cost = new DisaggregatedScheduler.Cost(2, 0.05, 0.002);
        DisaggregatedScheduler s = new DisaggregatedScheduler(core, cost, 16, 512);
        DisaggregatedScheduler.Report split = s.disaggregated(trace, 512);
        DisaggregatedScheduler.Report coloc = s.colocated(trace, 2, 72, 64);
        boolean same = true;
        for (int i = 0; i < requests; i++) {
            same &= java.util.Arrays.equals(split.completions().get(i).tokens(), coloc.completions().get(i).tokens());
        }
        return new DisaggregationResult(split, coloc, same);
    }
}
