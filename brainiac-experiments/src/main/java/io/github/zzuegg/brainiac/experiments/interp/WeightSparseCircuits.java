package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.train.MachineUnlearning;
import io.github.zzuegg.brainiac.experimental.train.TopKWeightSparsity;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Circuit size of weight-sparse against dense training on bracket matching. An MLP reads a
 * string of eight brackets and says whether it is balanced (every prefix has at least as
 * many openers as closers and the counts end equal); the negatives have equal counts, so
 * the model must track the running depth. One copy trains densely, one with
 * {@link TopKWeightSparsity} (15% of each matrix kept). Each is then pruned by global
 * weight magnitude to the fewest weights that still reach the same accuracy target; that
 * count is the circuit size. Weight-sparse training should leave a much smaller circuit
 * than pruning a dense model after the fact.
 *
 * <p>Gao et al. 2025, "Weight-sparse transformers have interpretable circuits",
 * <a href="https://arxiv.org/abs/2511.13653">arXiv:2511.13653</a>.
 */
public final class WeightSparseCircuits {

    /**
     * The outcome.
     *
     * @param target the accuracy both pruned circuits must reach
     * @param denseAccuracy the dense model's test accuracy
     * @param sparseAccuracy the weight-sparse model's test accuracy
     * @param totalWeights the weights in each model
     * @param denseCircuit weights left after pruning the dense model
     * @param sparseCircuit weights left after pruning the weight-sparse model
     */
    public record Outcome(double target, double denseAccuracy, double sparseAccuracy, int totalWeights,
            int denseCircuit, int sparseCircuit) {
    }

    private static final int LENGTH = 8;

    private WeightSparseCircuits() {
    }

    private static int[] balanced(SplittableRandom r) {
        int[] s = new int[LENGTH];
        int open = 0;
        int depth = 0;
        for (int i = 0; i < LENGTH; i++) {
            int remaining = LENGTH - i;
            boolean canOpen = open < LENGTH / 2;
            boolean canClose = depth > 0;
            boolean mustClose = depth == remaining;
            boolean opener = !mustClose && canOpen && (!canClose || r.nextBoolean());
            s[i] = opener ? 0 : 1;
            open += opener ? 1 : 0;
            depth += opener ? 1 : -1;
        }
        return s;
    }

    private static boolean isBalanced(int[] s) {
        int depth = 0;
        for (int c : s) {
            depth += c == 0 ? 1 : -1;
            if (depth < 0) {
                return false;
            }
        }
        return depth == 0;
    }

    private static Dataset data(int n, SplittableRandom r) {
        float[][] x = new float[n][2 * LENGTH];
        float[][] y = new float[n][2];
        for (int i = 0; i < n; i++) {
            int[] s = balanced(r);
            if (i % 2 == 1) {
                do {
                    for (int j = LENGTH - 1; j > 0; j--) {
                        int k = r.nextInt(j + 1);
                        int t = s[j];
                        s[j] = s[k];
                        s[k] = t;
                    }
                } while (isBalanced(s));
            }
            for (int t = 0; t < LENGTH; t++) {
                x[i][2 * t + s[t]] = 1;
            }
            y[i][isBalanced(s) ? 1 : 0] = 1;
        }
        return Dataset.of(x, y);
    }

    private static List<Var> weights(Model m) {
        List<Var> w = new ArrayList<>();
        for (Layer l : m.layers()) {
            if (l instanceof Dense d) {
                w.add(d.weight());
            }
        }
        return w;
    }

    /**
     * Runs the experiment.
     *
     * @param epochs training epochs
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int epochs, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Dataset train = data(2000, r);
        Dataset test = data(600, r);
        Model dense = model(seed);
        Trainer.of(dense).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.005f)).batchSize(32).epochs(epochs)
                .seed(seed).fit(train);
        Model sparse = model(seed);
        long steps = (long) epochs * (train.size() / 32);
        Trainer.of(sparse).loss(Loss.CROSS_ENTROPY)
                .optimizer(new TopKWeightSparsity(Optimizer.adam(0.005f), weights(sparse), 0.15, steps / 2))
                .batchSize(32).epochs(epochs).seed(seed).fit(train);
        double da = MachineUnlearning.accuracy(dense, test);
        double sa = MachineUnlearning.accuracy(sparse, test);
        double target = Math.min(da, sa) - 0.02;
        int total = weights(dense).stream().mapToInt(v -> Math.toIntExact(v.shape().size())).sum();
        return new Outcome(target, da, sa, total, circuit(dense, test, target), circuit(sparse, test, target));
    }

    private static Model model(long seed) {
        return Model.sequential().input(2 * LENGTH).dense(32, Activation.RELU).dense(32, Activation.RELU)
                .dense(2, Activation.SOFTMAX).seed(seed).build();
    }

    /** The fewest largest-magnitude weights that keep the accuracy at or above the target. */
    private static int circuit(Model m, Dataset test, double target) {
        List<Var> w = weights(m);
        List<float[]> original = new ArrayList<>();
        List<Float> all = new ArrayList<>();
        for (Var v : w) {
            float[] a = v.value().to(DType.FLOAT32).toArray();
            original.add(a);
            for (float x : a) {
                all.add(Math.abs(x));
            }
        }
        float[] mags = new float[all.size()];
        for (int i = 0; i < mags.length; i++) {
            mags[i] = all.get(i);
        }
        float[] sorted = mags.clone();
        Arrays.sort(sorted);
        int lo = 1;
        int hi = mags.length;
        while (lo < hi) {
            int k = (lo + hi) / 2;
            float cut = sorted[mags.length - k];
            for (int i = 0; i < w.size(); i++) {
                float[] a = original.get(i).clone();
                for (int j = 0; j < a.length; j++) {
                    if (Math.abs(a[j]) < cut) {
                        a[j] = 0;
                    }
                }
                w.get(i).assign(Tensor.of(w.get(i).shape(), a));
            }
            if (MachineUnlearning.accuracy(m, test) >= target) {
                hi = k;
            } else {
                lo = k + 1;
            }
        }
        for (int i = 0; i < w.size(); i++) {
            w.get(i).assign(Tensor.of(w.get(i).shape(), original.get(i)));
        }
        return lo;
    }
}
