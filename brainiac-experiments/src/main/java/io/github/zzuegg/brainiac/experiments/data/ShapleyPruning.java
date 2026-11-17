package io.github.zzuegg.brainiac.experiments.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DataPruner;
import io.github.zzuegg.brainiac.experimental.data.InRunShapley;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Pruning harmful data with In-Run Data Shapley ({@link InRunShapley}), on the toy digits of
 * {@link ToyTasks#digits} with a share of the labels flipped. One SGD scoring run records
 * every example's contribution to the loss of a small clean validation set. The examples
 * with the lowest value are removed, and a fresh classifier is trained on the rest. The
 * baselines remove the same number of examples by EL2N (the hardest ones, averaged over two
 * short runs) or at random.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class ShapleyPruning {

    /** How the removed examples are chosen. */
    public enum Method {
        /** A random subset. */
        RANDOM,
        /** The highest EL2N scores. */
        EL2N,
        /** The lowest first-order In-Run Shapley values. */
        IN_RUN_SHAPLEY,
        /** The lowest In-Run Shapley values with the second-order interaction term. */
        IN_RUN_SHAPLEY_2
    }

    /**
     * The outcome of {@link #run}.
     *
     * @param flippedRemoved the share of the removed examples whose label was flipped
     * @param accuracy test accuracy of the classifier trained on the kept examples
     */
    public record Result(double flippedRemoved, double accuracy) {
    }

    private record Labelled(Tensor x, Tensor y, boolean[] flipped) {
    }

    private ShapleyPruning() {
    }

    private static Labelled tensors(ToyTasks.Data d, double flip, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = d.size();
        int f = d.features();
        float[] x = new float[n * f];
        float[] y = new float[n];
        boolean[] flipped = new boolean[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(d.x()[i], 0, x, i * f, f);
            int label = d.y()[i];
            if (r.nextDouble() < flip) {
                label = (label + 1 + r.nextInt(d.classes() - 1)) % d.classes();
                flipped[i] = true;
            }
            y[i] = label;
        }
        return new Labelled(Tensor.of(Shape.of(n, f), x), Tensor.of(Shape.of(n), y), flipped);
    }

    private static SimpleMlp train(Labelled data, Optimizer opt, int epochs, long seed, DataPruner pruner,
            InRunShapley shapley, Labelled valid, boolean secondOrder) {
        SimpleMlp net = SimpleMlp.relu(seed, data.x().shape().dim(1), 64, 10);
        int n = data.x().shape().dim(0);
        int batch = 32;
        SplittableRandom r = new SplittableRandom(seed);
        for (int e = 0; e < epochs; e++) {
            int[] order = DataPruner.random(n, 1, r.nextLong());
            for (int i = n - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            for (int s = 0; s + batch <= n; s += batch) {
                int[] rows = Arrays.copyOfRange(order, s, s + batch);
                Tensor x = DataPruner.select(data.x(), rows);
                Tensor y = DataPruner.select(data.y(), rows);
                if (shapley != null) {
                    shapley.record(net, x, y, rows, valid.x(), valid.y(), opt.learningRate(), secondOrder);
                }
                Functional.crossEntropy(net.forward(Var.of(x)), y).backward();
                opt.step(net.parameters());
            }
            if (pruner != null) {
                pruner.record(net.predict(data.x()), data.y());
            }
        }
        return net;
    }

    /**
     * Scores the training set, removes {@code remove} of it and trains on the rest.
     *
     * @param method how the removed examples are chosen
     * @param remove the share removed
     * @param flip the share of training labels flipped
     * @param seed the seed
     * @return the noise found and the test accuracy
     */
    public static Result run(Method method, double remove, double flip, long seed) {
        Labelled train = tensors(ToyTasks.digits(10, 60, 8, 0.08, seed, seed + 100), flip, seed + 300);
        Labelled valid = tensors(ToyTasks.digits(10, 20, 8, 0.08, seed, seed + 400), 0, seed);
        Labelled test = tensors(ToyTasks.digits(10, 40, 8, 0.08, seed, seed + 200), 0, seed);
        int n = train.x().shape().dim(0);
        int[] kept;
        switch (method) {
            case RANDOM -> kept = DataPruner.random(n, 1 - remove, seed);
            case EL2N -> {
                DataPruner pruner = new DataPruner(n);
                for (int k = 0; k < 2; k++) {
                    train(train, Optimizer.adam(3e-3f), 4, seed + 1000 + k, pruner, null, null, false);
                }
                kept = DataPruner.keep(pruner.el2n(), 1 - remove, remove);
            }
            default -> {
                InRunShapley shapley = new InRunShapley(n);
                train(train, Optimizer.sgd(0.2f), 8, seed + 1000, null, shapley, valid,
                        method == Method.IN_RUN_SHAPLEY_2);
                double[] value = shapley.scores();
                int[] counts = shapley.counts();
                double[] harm = new double[n];
                for (int i = 0; i < n; i++) {   // the mean value per step taken part in
                    harm[i] = -value[i] / Math.max(1, counts[i]);
                }
                kept = DataPruner.keep(harm, 1 - remove, remove);
            }
        }
        boolean[] isKept = new boolean[n];
        for (int i : kept) {
            isKept[i] = true;
        }
        int removed = 0;
        int flippedRemoved = 0;
        for (int i = 0; i < n; i++) {
            if (!isKept[i]) {
                removed++;
                flippedRemoved += train.flipped()[i] ? 1 : 0;
            }
        }
        Labelled subset = new Labelled(DataPruner.select(train.x(), kept), DataPruner.select(train.y(), kept), null);
        SimpleMlp net = train(subset, Optimizer.adam(3e-3f), 15, seed, null, null, null, false);
        double accuracy = net.predict(test.x()).argmax(1).equalTo(test.y()).mean().item();
        return new Result(removed == 0 ? 0 : flippedRemoved / (double) removed, accuracy);
    }
}
