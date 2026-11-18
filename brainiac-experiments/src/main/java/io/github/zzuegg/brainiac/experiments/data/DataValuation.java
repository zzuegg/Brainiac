package io.github.zzuegg.brainiac.experiments.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DataPruner;
import io.github.zzuegg.brainiac.experimental.data.PrototypePruning;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Wave 8 items 8.6.14 and 8.6.15: cheap data valuation and self-supervised prototype pruning
 * with the blocks {@link io.github.zzuegg.brainiac.experimental.data.DataValuation} and
 * {@link PrototypePruning}.
 *
 * <p><b>Flipped-label detection</b> ({@link #detection}): the toy digits of
 * {@link ShapleyPruning} with the same data, seeds and 20% flipped labels. Each valuation flags
 * its lowest-valued 20%; the share of flipped labels among them, the test accuracy of an MLP
 * trained without them, and the valuation's time are compared with
 * {@link ShapleyPruning#run} (In-Run Shapley, EL2N, random) and
 * {@link TrainingAttributionExperiment#detection} (TracIn, influence), which flag the same
 * share.
 *
 * <p><b>Diverse-subset selection</b> ({@link #selection}): from the clean training set, pick
 * a small labelled subset by greedy leverage, top KNN-Shapley value or at random, and train on
 * it.
 *
 * <p><b>Prototype pruning</b> ({@link #pruning}): digits at 500, 2000 and 8000 examples, an
 * MLP encoder trained label-free with SimCLR's NT-Xent loss, k-means prototypes in its
 * embedding space, then keep the easiest (prototype-near) or hardest (prototype-far) share, or
 * a random one, or the hardest by EL2N, and train an MLP for a fixed number of steps on it.
 * (The rows live here rather than in {@code vision.DataPruningExperiment}, which stays
 * unchanged.)
 */
@SuppressWarnings("exports")
public final class DataValuation {

    /** A valuation for flipped-label detection. */
    public enum Valuation {
        /** random values */
        RANDOM,
        /** KNN-Shapley on the pixels */
        KNN_SHAPLEY,
        /** KNN-Shapley on self-supervised embeddings */
        KNN_SHAPLEY_EMBEDDING,
        /** Data-OOB with bagged trees */
        DATA_OOB,
        /** ridge leverage on the pixels (label-free) */
        LEVERAGE
    }

    /**
     * The outcome of {@link #detection}.
     *
     * @param flippedShare share of flipped labels among the lowest-valued 20%
     * @param accuracy test accuracy after removing them
     * @param millis time to compute the values
     */
    public record Detection(double flippedShare, double accuracy, long millis) {
    }

    /** How a subset is selected. */
    public enum Selection {
        /** uniformly at random */
        RANDOM,
        /** greedy ridge leverage on the pixels */
        LEVERAGE,
        /** greedy ridge leverage on self-supervised embeddings */
        LEVERAGE_EMBEDDING,
        /** the highest KNN-Shapley values */
        KNN_SHAPLEY
    }

    /** Which share pruning keeps. */
    public enum Keep {
        /** a random share */
        RANDOM,
        /** the examples closest to their prototype */
        EASY,
        /** the examples farthest from their prototype */
        HARD,
        /** the highest EL2N scores (supervised) */
        EL2N_HARD
    }

    private record Split(float[][] x, int[] y, boolean[] flipped) {
    }

    private DataValuation() {
    }

    /** The flipped training set of {@link ShapleyPruning} (same seeds, same flip procedure). */
    private static Split flipped(ToyTasks.Data d, double flip, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = d.size();
        int[] y = new int[n];
        boolean[] f = new boolean[n];
        for (int i = 0; i < n; i++) {
            int label = d.y()[i];
            if (r.nextDouble() < flip) {
                label = (label + 1 + r.nextInt(d.classes() - 1)) % d.classes();
                f[i] = true;
            }
            y[i] = label;
        }
        return new Split(d.x(), y, f);
    }

    /**
     * Values the flipped-label training set and removes the lowest-valued share.
     *
     * @param valuation the valuation
     * @param flip the share of labels flipped (and removed)
     * @param seed the seed
     * @return detection precision, accuracy after removal and time
     */
    public static Detection detection(Valuation valuation, double flip, long seed) {
        Split train = flipped(ToyTasks.digits(10, 60, 8, 0.08, seed, seed + 100), flip, seed + 300);
        ToyTasks.Data valid = ToyTasks.digits(10, 20, 8, 0.08, seed, seed + 400);
        ToyTasks.Data test = ToyTasks.digits(10, 40, 8, 0.08, seed, seed + 200);
        long start = System.nanoTime();
        double[] value = switch (valuation) {
            case RANDOM -> new SplittableRandom(seed).doubles(train.x().length).toArray();
            case KNN_SHAPLEY -> io.github.zzuegg.brainiac.experimental.data.DataValuation.knnShapley(
                    doubles(train.x()), train.y(), doubles(valid.x()), valid.y(), 5);
            case KNN_SHAPLEY_EMBEDDING -> {
                SimpleMlp enc = encoder(train.x(), 150, seed);
                yield io.github.zzuegg.brainiac.experimental.data.DataValuation.knnShapley(
                        embed(enc, train.x()), train.y(), embed(enc, valid.x()), valid.y(), 5);
            }
            case DATA_OOB -> io.github.zzuegg.brainiac.experimental.data.DataValuation.dataOob(
                    doubles(train.x()), train.y(), 10, 60, 8, seed);
            case LEVERAGE -> io.github.zzuegg.brainiac.experimental.data.DataValuation.leverageScores(
                    doubles(train.x()), 1.0);
        };
        long millis = (System.nanoTime() - start) / 1_000_000;
        int[] low = io.github.zzuegg.brainiac.experimental.data.DataValuation.lowest(value, flip);
        boolean[] removed = new boolean[value.length];
        int hits = 0;
        for (int i : low) {
            removed[i] = true;
            hits += train.flipped()[i] ? 1 : 0;
        }
        int[] kept = new int[value.length - low.length];
        for (int i = 0, k = 0; i < value.length; i++) {
            if (!removed[i]) {
                kept[k++] = i;
            }
        }
        SimpleMlp net = fit(rows(train.x(), kept), rows(train.y(), kept), 15 * kept.length / 32, seed);
        return new Detection(hits / (double) low.length, accuracy(net, test), millis);
    }

    /**
     * Point addition: trains on a small selected subset of a clean training pool.
     *
     * @param selection how the subset is chosen
     * @param size the subset size
     * @param seed the seed
     * @return test accuracy
     */
    public static double selection(Selection selection, int size, long seed) {
        ToyTasks.Data pool = ToyTasks.digits(10, 60, 8, 0.15, seed, seed + 100);
        ToyTasks.Data valid = ToyTasks.digits(10, 20, 8, 0.15, seed, seed + 400);
        ToyTasks.Data test = ToyTasks.digits(10, 40, 8, 0.15, seed, seed + 200);
        int n = pool.size();
        int[] picked = switch (selection) {
            case RANDOM -> java.util.Arrays.copyOf(DataPruner.random(n, size / (double) n, seed), size);
            case LEVERAGE, LEVERAGE_EMBEDDING -> {
                double[][] e = selection == Selection.LEVERAGE ? doubles(pool.x())
                        : embed(encoder(pool.x(), 80, seed), pool.x());
                int[] all = new int[n];
                for (int i = 0; i < n; i++) {
                    all[i] = i;
                }
                yield io.github.zzuegg.brainiac.experimental.data.DataValuation.leverageSelect(e, new int[0], all,
                        size, 1.0);
            }
            case KNN_SHAPLEY -> {
                double[] v = io.github.zzuegg.brainiac.experimental.data.DataValuation.knnShapley(
                        doubles(pool.x()), pool.y(), doubles(valid.x()), valid.y(), 5);
                double[] negative = new double[n];
                for (int i = 0; i < n; i++) {
                    negative[i] = -v[i];
                }
                yield io.github.zzuegg.brainiac.experimental.data.DataValuation.lowest(negative, size / (double) n);
            }
        };
        SimpleMlp net = fit(rows(pool.x(), picked), rows(pool.y(), picked), 200, seed);
        return accuracy(net, test);
    }

    /**
     * Prototype pruning at one dataset size and kept share.
     *
     * @param size the dataset size
     * @param keep the share kept
     * @param how which share
     * @param steps training steps on the kept share (batch 64)
     * @param seed the seed
     * @return test accuracy
     */
    public static double pruning(int size, double keep, Keep how, int steps, long seed) {
        Prepared p = prepare(size, seed);
        return pruning(p, keep, how, steps, seed);
    }

    /**
     * A dataset with its prototype and EL2N difficulties, reusable over kept shares.
     *
     * @param x inputs
     * @param y labels
     * @param test test set
     * @param prototype the prototype-pruning scores
     * @param el2n EL2N scores
     */
    public record Prepared(float[][] x, int[] y, ToyTasks.Data test, PrototypePruning prototype, double[] el2n) {
    }

    /**
     * Builds a dataset and scores it: an NT-Xent encoder, k-means prototypes and EL2N.
     *
     * @param size the dataset size
     * @param seed the seed
     * @return the scored dataset
     */
    public static Prepared prepare(int size, long seed) {
        ToyTasks.Data d = ToyTasks.digits(10, size / 10, 8, 0.15, seed, seed + 100);
        ToyTasks.Data test = ToyTasks.digits(10, 50, 8, 0.15, seed, seed + 200);
        SimpleMlp enc = encoder(d.x(), 120, seed);
        PrototypePruning proto = PrototypePruning.fit(embed(enc, d.x()), 10, seed);
        DataPruner el2n = new DataPruner(d.size());
        for (int run = 0; run < 2; run++) {
            SimpleMlp early = fit(d.x(), d.y(), 30, seed + 7 + run);
            el2n.record(early.predict(Tensor.of(d.x())), d.ids());
        }
        return new Prepared(d.x(), d.y(), test, proto, el2n.el2n());
    }

    /**
     * Prunes a prepared dataset and trains on the rest.
     *
     * @param p the scored dataset
     * @param keep the share kept
     * @param how which share
     * @param steps training steps (batch 64)
     * @param seed the seed
     * @return test accuracy
     */
    public static double pruning(Prepared p, double keep, Keep how, int steps, long seed) {
        int n = p.x().length;
        int[] kept = switch (how) {
            case RANDOM -> DataPruner.random(n, keep, seed + 11);
            case EASY -> p.prototype().keep(keep, false);
            case HARD -> p.prototype().keep(keep, true);
            case EL2N_HARD -> DataPruner.keep(p.el2n(), keep, 0);
        };
        SimpleMlp net = fit(rows(p.x(), kept), rows(p.y(), kept), steps, seed);
        return accuracy(net, p.test());
    }

    /** A label-free encoder: an MLP trained with NT-Xent on shifted, noisy, dropped-out views. */
    static SimpleMlp encoder(float[][] x, int steps, long seed) {
        SimpleMlp net = SimpleMlp.relu(seed + 50, x[0].length, 64, 32);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 51);
        int batch = Math.min(128, x.length);
        for (int s = 0; s < steps; s++) {
            float[][] a = new float[batch][];
            float[][] b = new float[batch][];
            for (int i = 0; i < batch; i++) {
                float[] img = x[r.nextInt(x.length)];
                a[i] = view(img, r);
                b[i] = view(img, r);
            }
            SelfSupervisedLosses.ntXent(net.forward(Var.of(Tensor.of(a))), net.forward(Var.of(Tensor.of(b))), 0.2f)
                    .backward();
            opt.step(net.parameters());
        }
        return net;
    }

    private static float[] view(float[] img, SplittableRandom r) {
        int dy = r.nextInt(3) - 1;
        int dx = r.nextInt(3) - 1;
        float[] out = new float[64];
        for (int y = 0; y < 8; y++) {
            for (int xx = 0; xx < 8; xx++) {
                float v = img[Math.floorMod(y - dy, 8) * 8 + Math.floorMod(xx - dx, 8)];
                out[y * 8 + xx] = r.nextDouble() < 0.1 ? 0 : v + (float) r.nextGaussian() * 0.1f;
            }
        }
        return out;
    }

    /** The encoder's representation (its hidden layer, before the projection). */
    static double[][] embed(SimpleMlp enc, float[][] x) {
        Tensor h = enc.layer(0, Var.of(Tensor.of(x))).value();
        int d = h.shape().dim(1);
        float[] v = h.toArray();
        double[][] out = new double[x.length][d];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < d; j++) {
                out[i][j] = v[i * d + j];
            }
        }
        return out;
    }

    private static SimpleMlp fit(float[][] x, int[] y, int steps, long seed) {
        SimpleMlp net = SimpleMlp.relu(seed, x[0].length, 64, 10);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int batch = Math.min(64, x.length);
        for (int s = 0; s < steps; s++) {
            float[][] bx = new float[batch][];
            float[] by = new float[batch];
            for (int i = 0; i < batch; i++) {
                int j = r.nextInt(x.length);
                bx[i] = x[j];
                by[i] = y[j];
            }
            Functional.crossEntropy(net.forward(Var.of(Tensor.of(bx))), Tensor.of(Shape.of(batch), by)).backward();
            opt.step(net.parameters());
        }
        return net;
    }

    private static double accuracy(SimpleMlp net, ToyTasks.Data test) {
        return net.predict(test.inputs()).argmax(1).equalTo(test.ids()).mean().item();
    }

    private static double[][] doubles(float[][] x) {
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new double[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] = x[i][j];
            }
        }
        return out;
    }

    private static float[][] rows(float[][] x, int[] idx) {
        float[][] out = new float[idx.length][];
        for (int i = 0; i < idx.length; i++) {
            out[i] = x[idx[i]];
        }
        return out;
    }

    private static int[] rows(int[] y, int[] idx) {
        int[] out = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = y[idx[i]];
        }
        return out;
    }
}
