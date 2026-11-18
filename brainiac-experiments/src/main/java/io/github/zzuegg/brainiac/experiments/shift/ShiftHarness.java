package io.github.zzuegg.brainiac.experiments.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.BiFunction;

/**
 * Shared pieces of the shift experiments: minibatch training of a {@link FeatureClassifier},
 * conversions to probability rows, and the per-seed cache of the trained OOD classifiers that
 * {@link OodShootout} and {@link SelectivePrediction} score post hoc (trained once per seed, so
 * each post-hoc score costs only inference).
 */
public final class ShiftHarness {

    /**
     * A trained OOD set-up.
     *
     * @param suite the data
     * @param model the classifier trained on its in-distribution training rows
     */
    public record OodSetup(ShiftSuites.OodSuite suite, FeatureClassifier model) {
    }

    private static final Map<String, OodSetup> CACHE = new ConcurrentHashMap<>();

    private ShiftHarness() {
    }

    /**
     * The MLP used throughout: 64 inputs, two ReLU layers of 64, a linear head.
     *
     * @param inputs the input features
     * @param classes the number of classes
     * @param seed the seed
     * @return the classifier
     */
    public static FeatureClassifier mlp(int inputs, int classes, long seed) {
        return FeatureClassifier.mlp(inputs, new int[] {64, 64}, classes, seed);
    }

    /**
     * Minibatch cross-entropy training.
     *
     * @param model the classifier
     * @param data the training rows
     * @param steps Adam steps
     * @param lr the step size
     * @param seed the batch-sampling seed
     */
    public static void train(FeatureClassifier model, ShiftSuites.Split data, int steps, float lr, long seed) {
        train(model, data, steps, lr, seed, (logits, y) -> Functional.crossEntropy(logits, y));
    }

    /**
     * Minibatch training with any loss on the logits.
     *
     * @param model the classifier
     * @param data the training rows
     * @param steps Adam steps
     * @param lr the step size
     * @param seed the batch-sampling seed
     * @param loss maps logits and label ids to the loss
     */
    public static void train(FeatureClassifier model, ShiftSuites.Split data, int steps, float lr, long seed,
            BiFunction<Var, Tensor, Var> loss) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            ShiftSuites.Split b = batch(data, 64, r);
            loss.apply(model.logits(Var.of(b.x()), true), b.ids()).backward();
            opt.step(model.parameters());
        }
    }

    /**
     * A random minibatch (with replacement).
     *
     * @param data the rows
     * @param size the batch size
     * @param r the random source
     * @return the batch
     */
    public static ShiftSuites.Split batch(ShiftSuites.Split data, int size, SplittableRandom r) {
        int[] rows = new int[size];
        for (int i = 0; i < size; i++) {
            rows[i] = r.nextInt(data.size());
        }
        return data.rows(rows);
    }

    /**
     * Probability rows from logits.
     *
     * @param logits {@code [n, classes]}
     * @return the softmax rows
     */
    public static double[][] probabilities(Tensor logits) {
        int c = logits.shape().dim(1);
        double[] p = logits.sub(logits.logsumexp(1, true)).exp().toDoubleArray();
        double[][] out = new double[p.length / c][c];
        for (int i = 0; i < out.length; i++) {
            System.arraycopy(p, i * c, out[i], 0, c);
        }
        return out;
    }

    /**
     * Rows from a tensor.
     *
     * @param x {@code [n, d]}
     * @return the rows as doubles
     */
    public static double[][] rows(Tensor x) {
        int d = x.shape().dim(1);
        double[] v = x.toDoubleArray();
        double[][] out = new double[v.length / d][d];
        for (int i = 0; i < out.length; i++) {
            System.arraycopy(v, i * d, out[i], 0, d);
        }
        return out;
    }

    /**
     * A tensor from rows.
     *
     * @param rows the rows
     * @return {@code [n, d]}
     */
    public static Tensor tensor(double[][] rows) {
        int d = rows[0].length;
        float[] v = new float[rows.length * d];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < d; j++) {
                v[i * d + j] = (float) rows[i][j];
            }
        }
        return Tensor.of(Shape.of(rows.length, d), v);
    }

    /**
     * Accuracy of predictions.
     *
     * @param predicted the predictions
     * @param labels the truth
     * @return the share right
     */
    public static double accuracy(int[] predicted, int[] labels) {
        int ok = 0;
        for (int i = 0; i < labels.length; i++) {
            ok += predicted[i] == labels[i] ? 1 : 0;
        }
        return ok / (double) labels.length;
    }

    /**
     * The argmax of each probability row.
     *
     * @param probs the rows
     * @return the classes
     */
    public static int[] argmax(double[][] probs) {
        int[] out = new int[probs.length];
        for (int i = 0; i < probs.length; i++) {
            int b = 0;
            for (int c = 1; c < probs[i].length; c++) {
                if (probs[i][c] > probs[i][b]) {
                    b = c;
                }
            }
            out[i] = b;
        }
        return out;
    }

    /**
     * The OOD classifier of a family and seed, trained once (1,200 rows, 600 Adam steps) and
     * cached.
     *
     * @param family shapes or digits
     * @param seed the seed
     * @return the suite and its trained classifier
     */
    public static OodSetup ood(ShiftSuites.Family family, long seed) {
        return CACHE.computeIfAbsent(family + "/" + seed, k -> {
            ShiftSuites.OodSuite suite = ShiftSuites.ood(family, 1200, 400, seed);
            FeatureClassifier m = mlp(ShiftSuites.PIXELS, suite.classes(), seed);
            train(m, suite.train(), 600, 3e-3f, seed);
            return new OodSetup(suite, m);
        });
    }
}
