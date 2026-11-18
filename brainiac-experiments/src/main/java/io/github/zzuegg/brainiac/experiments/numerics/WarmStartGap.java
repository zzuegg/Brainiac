package io.github.zzuegg.brainiac.experiments.numerics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.ShrinkPerturb;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * The warm-start gap (wave 7, item 7.3.17, after Ash and Adams 2020): digit images arrive in
 * chunks; after each chunk the network trains on all data so far. Compared are a plain warm
 * start (keep training the same weights), training from scratch at every stage, and
 * {@link ShrinkPerturb} before each stage. Reports the final test accuracy, averaged over seeds,
 * once all data has arrived.
 */
public final class WarmStartGap {

    private WarmStartGap() {
    }

    /**
     * Runs the three strategies.
     *
     * @param chunks the number of data chunks
     * @param stepsPerStage training steps after each chunk
     * @param shrink the shrink factor
     * @param perturb the perturbation factor
     * @param seeds the seeds
     * @return the mean final test accuracy per strategy
     */
    public static Map<String, Double> run(int chunks, int stepsPerStage, float shrink, float perturb, long... seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (String method : List.of("warm start", "from scratch", "shrink and perturb")) {
            double acc = 0;
            for (long seed : seeds) {
                acc += runOne(method, chunks, stepsPerStage, shrink, perturb, seed);
            }
            out.put(method, acc / seeds.length);
        }
        return out;
    }

    private static double runOne(String method, int chunks, int stepsPerStage, float shrink, float perturb,
            long seed) {
        ToyTasks.Data all = ToyTasks.digits(10, 40, 8, 0.12, seed);
        ToyTasks.Data test = ToyTasks.digits(10, 30, 8, 0.12, seed, 77);
        int[] order = shuffle(all.size(), seed);
        float[][] xs = new float[all.size()][];
        int[] ys = new int[all.size()];
        for (int i = 0; i < order.length; i++) {
            xs[i] = all.x()[order[i]];
            ys[i] = all.y()[order[i]];
        }
        ToyTasks.Data data = new ToyTasks.Data(xs, ys, all.classes());
        int fresh = 0;
        SimpleMlp net = SimpleMlp.relu(seed * 100 + fresh++, 64, 64, 10);
        Optimizer opt = Optimizer.adam(0.01f);
        ShrinkPerturb sp = new ShrinkPerturb(shrink, perturb);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int c = 1; c <= chunks; c++) {
            ToyTasks.Data seen = data.slice(0, data.size() * c / chunks);
            if (c > 1 && method.equals("from scratch")) {
                net = SimpleMlp.relu(seed * 100 + fresh++, 64, 64, 10);
                opt = Optimizer.adam(0.01f);
            } else if (c > 1 && method.equals("shrink and perturb")) {
                List<Tensor> init = new ArrayList<>();
                for (Var p : SimpleMlp.relu(seed * 100 + fresh++, 64, 64, 10).parameters()) {
                    init.add(p.value());
                }
                sp.apply(net.parameters(), init);
            }
            for (int t = 0; t < stepsPerStage; t++) {
                Tensor[] b = NumericsTasks.batch(seen, 32, r);
                Functional.crossEntropy(net.forward(Var.of(b[0])), b[1]).backward();
                opt.step(net.parameters());
            }
        }
        return NumericsTasks.accuracy(net, test);
    }

    private static int[] shuffle(int n, long seed) {
        int[] o = new int[n];
        for (int i = 0; i < n; i++) {
            o[i] = i;
        }
        SplittableRandom r = new SplittableRandom(seed + 5);
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = o[i];
            o[i] = o[j];
            o[j] = t;
        }
        return o;
    }
}
