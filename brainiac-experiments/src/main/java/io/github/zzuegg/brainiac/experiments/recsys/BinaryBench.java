package io.github.zzuegg.brainiac.experiments.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.function.UnaryOperator;

/**
 * The shared training and scoring loop of the click and tabular experiments: Adam on binary
 * cross-entropy with batches of 64 rows at the same budget for every model, then test accuracy,
 * ROC-AUC and log-loss.
 */
public final class BinaryBench {

    /**
     * Test metrics of a binary classifier.
     *
     * @param accuracy the share of correct labels at probability 0.5
     * @param auc the ROC-AUC
     * @param logLoss the mean binary cross-entropy
     * @param millis the training time
     */
    public record Result(double accuracy, double auc, double logLoss, long millis) {
    }

    private BinaryBench() {
    }

    /**
     * Trains a one-logit model.
     *
     * @param model features to logits {@code [batch, 1]}, with the training flag
     * @param params its parameters
     * @param train training rows
     * @param test test rows
     * @param steps gradient steps of batch 64
     * @param lr the Adam learning rate
     * @param seed the batch seed
     * @return the test metrics
     */
    public static Result train(BiFunction<Var, Boolean, Var> model, List<Var> params, SyntheticRecsys.Table train,
            SyntheticRecsys.Table test, int steps, float lr, long seed) {
        return train(x -> model.apply(x, true), params, train, test, steps, lr, seed,
                x -> Var.noGrad(() -> model.apply(x, false)));
    }

    /**
     * Trains a model whose training loss is computed from a different output than its
     * prediction, such as an ensemble trained per member.
     *
     * @param trainLogits features to training logits {@code [batch, members]} (each column is
     *     scored against the label and the losses averaged)
     * @param params its parameters
     * @param train training rows
     * @param test test rows
     * @param steps gradient steps of batch 64
     * @param lr the Adam learning rate
     * @param seed the batch seed
     * @param predict features to one test logit per row
     * @return the test metrics
     */
    public static Result train(UnaryOperator<Var> trainLogits, List<Var> params, SyntheticRecsys.Table train,
            SyntheticRecsys.Table test, int steps, float lr, long seed, UnaryOperator<Var> predict) {
        long start = System.currentTimeMillis();
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int cols = train.features().shape().dim(1);
        float[] x = train.features().toArray();
        float[] y = train.labels().toArray();
        int batch = 64;
        for (int s = 0; s < steps; s++) {
            float[] bx = new float[batch * cols];
            float[] by = new float[batch];
            for (int i = 0; i < batch; i++) {
                int j = r.nextInt(train.rows());
                System.arraycopy(x, j * cols, bx, i * cols, cols);
                by[i] = y[j];
            }
            Var logit = trainLogits.apply(Var.of(Tensor.of(Shape.of(batch, cols), bx)));
            logit.softplus().sub(logit.mul(Tensor.of(Shape.of(batch, 1), by))).mean().backward();
            opt.step(params);
        }
        long millis = System.currentTimeMillis() - start;
        float[] logits = predict.apply(Var.of(test.features())).value().toArray();
        return score(logits, test.labels().toArray(), millis);
    }

    /**
     * Scores logits against labels.
     *
     * @param logits one logit per row
     * @param labels 0/1 labels
     * @param millis the training time to report
     * @return the metrics
     */
    public static Result score(float[] logits, float[] labels, long millis) {
        double[] p = new double[logits.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = 1 / (1 + Math.exp(-logits[i]));
        }
        return scoreProbabilities(p, labels, millis);
    }

    /**
     * Scores probabilities against labels.
     *
     * @param p the predicted probability of label 1 per row
     * @param labels 0/1 labels
     * @param millis the training time to report
     * @return the metrics
     */
    public static Result scoreProbabilities(double[] p, float[] labels, long millis) {
        boolean[] pos = new boolean[p.length];
        int right = 0;
        double ll = 0;
        for (int i = 0; i < p.length; i++) {
            pos[i] = labels[i] > 0.5f;
            right += (p[i] > 0.5) == pos[i] ? 1 : 0;
            double q = Math.min(1 - 1e-7, Math.max(1e-7, p[i]));
            ll -= pos[i] ? Math.log(q) : Math.log(1 - q);
        }
        return new Result(right / (double) p.length, MlMetrics.auroc(p, pos), ll / p.length, millis);
    }

    /**
     * @param t a table
     * @return its features as {@code double[rows][cols]}
     */
    public static double[][] features(SyntheticRecsys.Table t) {
        int rows = t.rows();
        int cols = t.features().shape().dim(1);
        float[] f = t.features().toArray();
        double[][] out = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                out[i][j] = f[i * cols + j];
            }
        }
        return out;
    }

    /**
     * @param t a table
     * @return its labels as 0/1 ints
     */
    public static int[] labels(SyntheticRecsys.Table t) {
        float[] y = t.labels().toArray();
        int[] out = new int[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = y[i] > 0.5f ? 1 : 0;
        }
        return out;
    }
}
