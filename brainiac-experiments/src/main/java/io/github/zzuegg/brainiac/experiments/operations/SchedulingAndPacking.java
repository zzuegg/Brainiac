package io.github.zzuegg.brainiac.experiments.operations;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims;
import io.github.zzuegg.brainiac.experimental.ops.BinPacking;
import io.github.zzuegg.brainiac.experimental.ops.JobShop;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Wave 12 items 12.6.8-12.6.9: job-shop scheduling and online bin packing.
 *
 * <ul>
 *   <li>{@link #jobShop} (12.6.8): the makespan gap of the priority dispatching rules, random
 *       dispatch and the learned (L2D-style, PPO-trained) dispatcher to the branch-and-bound reference
 *       (proved optimal on small instances, else the best found within the node budget);</li>
 *   <li>{@link #binPacking} (12.6.9): bins used relative to the lower bound for Next/First/Best Fit,
 *       Harmonic, profile packing and the hybrid with predicted frequencies of a given error, and the
 *       offline First Fit Decreasing reference.</li>
 * </ul>
 */
public final class SchedulingAndPacking {

    private SchedulingAndPacking() {
    }

    /**
     * 12.6.8.
     *
     * @param trainSize jobs and machines of the training instances
     * @param testJobs jobs of the test instances
     * @param testMachines machines of the test instances
     * @param testInstances test instances
     * @param trainIterations PPO iterations ({@code 0} to skip the learned dispatcher)
     * @param nodeLimit branch-and-bound nodes per test instance
     * @param seed the seed
     * @return method to the mean relative gap to the reference; the row {@code "proved optimal"} holds the
     *     share of test instances whose reference is proved optimal
     */
    public static Map<String, Double> jobShop(int trainSize, int testJobs, int testMachines, int testInstances,
            int trainIterations, long nodeLimit, long seed) {
        JobShop.LearnedDispatcher l2d = null;
        if (trainIterations > 0) {
            l2d = JobShop.LearnedDispatcher.of(32, seed)
                    .train(s -> OperationsSims.jobShop(trainSize, trainSize, 20, 10_000 + s), trainIterations, 4, 4, 3);
        }
        Map<String, Double> out = new LinkedHashMap<>();
        double proved = 0;
        for (int i = 0; i < testInstances; i++) {
            OperationsSims.JobShopInstance inst = OperationsSims.jobShop(testJobs, testMachines, 20, seed + i);
            long[] bb = JobShop.branchAndBound(inst, nodeLimit);
            double ref = bb[0];
            proved += bb[1];
            for (JobShop.Rule rule : JobShop.Rule.values()) {
                out.merge(rule.name(), (JobShop.dispatch(inst, rule, i) - ref) / ref / testInstances, Double::sum);
            }
            if (l2d != null) {
                out.merge("L2D (learned)", (l2d.solve(inst) - ref) / ref / testInstances, Double::sum);
            }
        }
        out.put("proved optimal", proved / testInstances);
        return out;
    }

    /**
     * 12.6.9.
     *
     * @param distribution the item distribution
     * @param items items per stream
     * @param predictionError the error of the predicted frequencies (profile methods)
     * @param streams streams
     * @param lambda the profile share of the hybrid
     * @return method to the mean bins used relative to the lower bound
     */
    public static Map<String, Double> binPacking(OperationsSims.ItemDistribution distribution, int items,
            double predictionError, int streams, double lambda) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (int s = 0; s < streams; s++) {
            int[] stream = OperationsSims.binItems(distribution, items, 100, s);
            double lb = BinPacking.lowerBound(stream, 100);
            double[] predicted = OperationsSims.predictedFrequencies(predictionError, 500 + s);
            for (BinPacking.Method m : BinPacking.Method.values()) {
                out.merge(m.name(), BinPacking.pack(m, stream, 100, predicted, lambda) / lb / streams, Double::sum);
            }
            out.merge("FFD (offline)", BinPacking.firstFitDecreasing(stream, 100) / lb / streams, Double::sum);
        }
        return out;
    }

    /**
     * 12.6.9: the RL packer trained on one distribution and tested on it and on another.
     *
     * @param train the training distribution
     * @param shift the shifted test distribution
     * @param items items per stream
     * @param steps DQN steps
     * @param seed the seed
     * @return row to {@code {bins / lower bound on the training distribution, on the shifted one}}
     */
    public static Map<String, double[]> rlPacker(OperationsSims.ItemDistribution train,
            OperationsSims.ItemDistribution shift, int items, long steps, long seed) {
        List<int[]> streams = new ArrayList<>();
        for (int s = 0; s < 20; s++) {
            streams.add(OperationsSims.binItems(train, items, 100, 1000 + s));
        }
        BinPacking.RlPacker packer = BinPacking.RlPacker.of(100, seed).train(streams, steps);
        int[] a = OperationsSims.binItems(train, items, 100, seed);
        int[] b = OperationsSims.binItems(shift, items, 100, seed);
        double la = BinPacking.lowerBound(a, 100);
        double lb = BinPacking.lowerBound(b, 100);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("RL packer (DQN)", new double[] {packer.pack(a) / la, packer.pack(b) / lb});
        for (BinPacking.Method m : new BinPacking.Method[] {BinPacking.Method.FIRST_FIT, BinPacking.Method.BEST_FIT}) {
            out.put(m.name(), new double[] {BinPacking.pack(m, a, 100, null, 0) / la,
                BinPacking.pack(m, b, 100, null, 0) / lb});
        }
        return out;
    }
}
