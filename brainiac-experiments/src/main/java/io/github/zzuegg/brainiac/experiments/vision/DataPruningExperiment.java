package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DataPruner;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Data pruning on the MNIST-like digits of {@link ToyTasks#digits} (8x8, ten classes) with
 * a share of the training labels flipped: score every training example early in training
 * with {@link DataPruner} (EL2N averaged over a few short runs, or forgetting events),
 * keep a fraction, train a fresh classifier on what is kept, and compare its test accuracy
 * with keeping a random subset of the same size. Pruning by score drops the easiest
 * (redundant) examples and the hardest few percent, where the flipped labels end up.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class DataPruningExperiment {

    /** How the kept subset is chosen. */
    public enum Method {
        /** A random subset, the baseline. */
        RANDOM,
        /** By EL2N score. */
        EL2N,
        /** By forgetting events. */
        FORGETTING
    }

    private DataPruningExperiment() {
    }

    /** A labelled dataset as tensors, with which labels were flipped. */
    private record Set(Tensor x, Tensor y, boolean[] flipped) {
    }

    private static Set tensors(ToyTasks.Data d, double flip, int classes, long seed) {
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
                label = (label + 1 + r.nextInt(classes - 1)) % classes;
                flipped[i] = true;
            }
            y[i] = label;
        }
        return new Set(Tensor.of(Shape.of(n, f), x), Tensor.of(Shape.of(n), y), flipped);
    }

    private static SimpleMlp train(Set data, int epochs, long seed, DataPruner recorder) {
        SimpleMlp net = SimpleMlp.relu(seed, data.x().shape().dim(1), 64, 10);
        Optimizer opt = Optimizer.adam(3e-3f);
        int n = data.x().shape().dim(0);
        int batch = 32;
        SplittableRandom r = new SplittableRandom(seed);
        for (int e = 0; e < epochs; e++) {
            int[] order = DataPruner.random(n, 1, r.nextLong());
            shuffle(order, r);
            for (int s = 0; s + batch <= n; s += batch) {
                int[] rows = java.util.Arrays.copyOfRange(order, s, s + batch);
                Functional.crossEntropy(net.forward(Var.of(DataPruner.select(data.x(), rows))),
                        DataPruner.select(data.y(), rows)).backward();
                opt.step(net.parameters());
            }
            if (recorder != null) {
                recorder.record(net.predict(data.x()), data.y());
            }
        }
        return net;
    }

    private static double[] scores(Method method, Set train, long seed) {
        DataPruner pruner = new DataPruner(train.x().shape().dim(0));
        for (int k = 0; k < 2; k++) {
            train(train, 4, seed + 1000 + k, pruner);   // short scoring runs
        }
        return method == Method.EL2N ? pruner.el2n() : pruner.forgettingScores();
    }

    /**
     * How well a score finds the flipped labels: the share of flipped examples among the
     * {@code flip * n} highest-scoring ones (random scoring gives about {@code flip}).
     *
     * @param method {@link Method#EL2N} or {@link Method#FORGETTING}
     * @param flip the share of training labels flipped
     * @param seed the seed
     * @return the precision of the top-scored examples as a noise detector
     */
    public static double noisePrecision(Method method, double flip, long seed) {
        Set train = tensors(ToyTasks.digits(10, 60, 8, 0.08, seed, seed + 100), flip, 10, seed + 300);
        double[] scores = scores(method, train, seed);
        int n = scores.length;
        int[] hardest = DataPruner.keep(scores, flip, 0);
        int hits = 0;
        for (int i : hardest) {
            hits += train.flipped()[i] ? 1 : 0;
        }
        return hardest.length == 0 ? 0 : (double) hits / Math.min(hardest.length, n);
    }

    private static void shuffle(int[] a, SplittableRandom r) {
        for (int i = a.length - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
    }

    /**
     * Test accuracy after training on the kept fraction of a noisy training set.
     *
     * @param method how examples are kept
     * @param keep the fraction kept
     * @param dropHardest the share of the dataset dropped from the hardest end (ignored by
     *     {@link Method#RANDOM}); with {@code dropHardest = 1 - keep} the easiest examples are kept
     * @param flip the share of training labels flipped
     * @param seed the seed
     * @return the test accuracy
     */
    public static double accuracy(Method method, double keep, double dropHardest, double flip, long seed) {
        ToyTasks.Data trainData = ToyTasks.digits(10, 60, 8, 0.08, seed, seed + 100);
        ToyTasks.Data testData = ToyTasks.digits(10, 40, 8, 0.08, seed, seed + 200);
        Set train = tensors(trainData, flip, 10, seed + 300);
        Set test = tensors(testData, 0, 10, seed);
        int n = train.x().shape().dim(0);
        int[] kept = method == Method.RANDOM ? DataPruner.random(n, keep, seed)
                : DataPruner.keep(scores(method, train, seed), keep, dropHardest);
        Set subset = new Set(DataPruner.select(train.x(), kept), DataPruner.select(train.y(), kept), null);
        int epochs = (int) Math.round(12 / keep);   // the same number of optimizer steps for every subset
        SimpleMlp net = train(subset, epochs, seed, null);
        return net.predict(test.x()).argmax(1).equalTo(test.y()).mean().item();
    }
}
