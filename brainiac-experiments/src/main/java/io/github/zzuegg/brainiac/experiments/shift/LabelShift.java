package io.github.zzuegg.brainiac.experiments.shift;

import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.experimental.robust.TestTimeAdaptation;
import io.github.zzuegg.brainiac.experimental.shift.LabelShiftEstimator;
import io.github.zzuegg.brainiac.experimental.train.Calibration;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 10 item 10.2.8: label-shift estimation and correction ({@link LabelShiftEstimator}) on
 * six noisy shapes. A batch-norm MLP is trained on uniform priors; a held-out source set gives
 * the confusion matrix and the calibration (temperature scaling from {@link Calibration}, or
 * bias-corrected temperature scaling). Target sets are resampled to tweak-one, long-tailed and
 * Dirichlet priors.
 *
 * <p>{@link #estimate} scores each estimator by the mean squared error of its importance
 * weights and the accuracy after correcting the posteriors, against no correction, the oracle
 * weights, and test-time adaptation (BN statistics, TENT), which wave 8 found to fail under
 * label shift. {@link #stream} follows a prior that drifts over a stream of small batches: the
 * online estimator against per-batch MLLS and no correction.
 */
public final class LabelShift {

    /** The estimators. */
    public enum Method {
        /** the source model as it is */
        NONE,
        /** the true weights */
        ORACLE,
        /** black-box shift estimation */
        BBSE,
        /** regularised BBSE */
        RLLS,
        /** EM with temperature scaling */
        MLLS_TS,
        /** EM with bias-corrected temperature scaling */
        MLLS_BCTS,
        /** MAP EM with a Dirichlet(3) prior and BCTS */
        MAP_DIRICHLET,
        /** test-batch BN statistics (accuracy only) */
        BN_STATS,
        /** TENT (accuracy only) */
        TENT
    }

    /**
     * One estimator on one target set.
     *
     * @param weightMse mean squared error of the importance weights (NaN for TTA)
     * @param accuracy target accuracy after correction
     */
    public record Score(double weightMse, double accuracy) {
    }

    /**
     * A trained source model with its held-out calibration.
     *
     * @param model the classifier
     * @param pool a labelled target pool to resample from (fresh images)
     * @param confusion the held-out joint confusion matrix
     * @param temperature the TS temperature
     * @param bcts the BCTS calibrator
     */
    public record Source(FeatureClassifier model, ShiftSuites.Split pool, double[][] confusion, double temperature,
            LabelShiftEstimator.Bcts bcts) {
    }

    private static final int K = 6;
    private static final double[] UNIFORM = {1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6, 1.0 / 6};

    private LabelShift() {
    }

    /**
     * Trains the source model.
     *
     * @param steps Adam steps
     * @param seed the seed
     * @return the model and calibration
     */
    public static Source source(int steps, long seed) {
        int[] all = {0, 1, 2, 3, 4, 5};
        ShiftSuites.Split train = ShiftSuites.shapes(2000, all, ShiftSuites.Style.PLAIN, 0.28, seed);
        ShiftSuites.Split val = ShiftSuites.shapes(1500, all, ShiftSuites.Style.PLAIN, 0.28, seed + 1);
        ShiftSuites.Split pool = ShiftSuites.shapes(3000, all, ShiftSuites.Style.PLAIN, 0.28, seed + 2);
        Model body = Model.sequential().input(ShiftSuites.PIXELS).dense(64).batchNorm().activation(Activation.RELU)
                .dense(64).batchNorm().activation(Activation.RELU).seed(seed).build();
        FeatureClassifier m = new FeatureClassifier(body, new Dense(64, K, Activation.IDENTITY, Init.XAVIER,
                new SplittableRandom(seed)));
        ShiftHarness.train(m, train, steps, 3e-3f, seed);
        Tensor logits = m.logits(val.x());
        double[][] confusion = LabelShiftEstimator.confusion(logits.argmax(1).toIntArray(), val.y(), K);
        double t = Calibration.fitTemperature(ShiftHarness.rows(logits), val.y());
        LabelShiftEstimator.Bcts bcts = LabelShiftEstimator.Bcts.fit(logits, val.y(), 300);
        return new Source(m, pool, confusion, t, bcts);
    }

    /**
     * Scores every method on one target set.
     *
     * @param src the source model
     * @param priors the target priors
     * @param n the target sample size
     * @param seed the resampling seed
     * @return the scores per method
     */
    public static Map<Method, Score> estimate(Source src, double[] priors, int n, long seed) {
        ShiftSuites.Split target = ShiftSuites.withPriors(src.pool(), priors, n, seed);
        Tensor logits = src.model().logits(target.x());
        double[][] raw = ShiftHarness.probabilities(logits);
        double[][] ts = Calibration.withTemperature(ShiftHarness.rows(logits), src.temperature());
        double[][] bc = src.bcts().probabilities(logits);
        double[] trueW = LabelShiftEstimator.weights(priors, UNIFORM);
        double[] mu = LabelShiftEstimator.predictedFrequencies(logits.argmax(1).toIntArray(), K);
        Map<Method, Score> out = new LinkedHashMap<>();
        out.put(Method.NONE, score(raw, ones(), trueW, target.y()));
        out.put(Method.ORACLE, score(raw, trueW, trueW, target.y()));
        out.put(Method.BBSE, score(raw, LabelShiftEstimator.bbse(src.confusion(), mu), trueW, target.y()));
        out.put(Method.RLLS, score(raw, LabelShiftEstimator.rlls(src.confusion(), mu, 1e-3), trueW, target.y()));
        out.put(Method.MLLS_TS, score(ts, LabelShiftEstimator.weights(LabelShiftEstimator.mlls(ts, UNIFORM, 200),
                UNIFORM), trueW, target.y()));
        out.put(Method.MLLS_BCTS, score(bc, LabelShiftEstimator.weights(LabelShiftEstimator.mlls(bc, UNIFORM, 200),
                UNIFORM), trueW, target.y()));
        double[] alpha = new double[K];
        Arrays.fill(alpha, 3);
        out.put(Method.MAP_DIRICHLET, score(bc, LabelShiftEstimator.weights(LabelShiftEstimator.mapDirichlet(bc,
                UNIFORM, alpha, 200), UNIFORM), trueW, target.y()));
        for (Method m : new Method[] {Method.BN_STATS, Method.TENT}) {
            TestTimeAdaptation tta = TestTimeAdaptation.of(src.model().model(), m == Method.TENT
                    ? TestTimeAdaptation.Method.TENT : TestTimeAdaptation.Method.BN_STATS);
            int ok = 0;
            for (int from = 0; from < n; from += 64) {
                int to = Math.min(n, from + 64);
                ShiftSuites.Split b = target.range(from, to);
                int[] p = tta.predict(b.x()).argmax(1).toIntArray();
                for (int i = 0; i < p.length; i++) {
                    ok += p[i] == b.y()[i] ? 1 : 0;
                }
            }
            tta.reset();
            out.put(m, new Score(Double.NaN, ok / (double) n));
        }
        return out;
    }

    private static Score score(double[][] probs, double[] w, double[] trueW, int[] labels) {
        double mse = 0;
        for (int i = 0; i < w.length; i++) {
            mse += (w[i] - trueW[i]) * (w[i] - trueW[i]) / w.length;
        }
        double[][] c = LabelShiftEstimator.correct(probs, w);
        return new Score(mse, ShiftHarness.accuracy(ShiftHarness.argmax(c), labels));
    }

    private static double[] ones() {
        double[] o = new double[K];
        Arrays.fill(o, 1);
        return o;
    }

    /**
     * The named prior shifts of the experiment.
     *
     * @param seed the Dirichlet seed
     * @return name to priors
     */
    public static Map<String, double[]> shifts(long seed) {
        Map<String, double[]> m = new LinkedHashMap<>();
        m.put("tweak-one 0.6", ShiftSuites.tweakOne(K, 0, 0.6));
        m.put("long-tailed 50", ShiftSuites.longTailed(K, 50));
        for (double a : new double[] {0.1, 1, 10}) {
            m.put("Dirichlet " + a, ShiftSuites.dirichlet(a, K, seed + (long) (10 * a)));
        }
        return m;
    }

    /**
     * Stream results.
     *
     * @param noCorrection accuracy without correction
     * @param online accuracy with the online estimator's weights
     * @param perBatch accuracy with MLLS refitted on each batch alone
     * @param onlineError mean absolute prior error of the online estimate
     * @param perBatchError mean absolute prior error of per-batch MLLS
     */
    public record StreamScores(double noCorrection, double online, double perBatch, double onlineError,
            double perBatchError) {
    }

    /**
     * A stream whose prior drifts linearly from one long-tailed prior to its reverse.
     *
     * @param src the source model
     * @param batches the number of batches
     * @param batchSize rows per batch
     * @param step the online step size
     * @param seed the seed
     * @return the scores
     */
    public static StreamScores stream(Source src, int batches, int batchSize, double step, long seed) {
        double[] a = ShiftSuites.longTailed(K, 20);
        double[] b = a.clone();
        for (int i = 0; i < K; i++) {
            b[i] = a[K - 1 - i];
        }
        double[] alpha = new double[K];
        Arrays.fill(alpha, 1);
        LabelShiftEstimator.Online online = new LabelShiftEstimator.Online(UNIFORM, alpha, step);
        int okNone = 0;
        int okOnline = 0;
        int okBatch = 0;
        double errOnline = 0;
        double errBatch = 0;
        for (int t = 0; t < batches; t++) {
            double f = t / (double) Math.max(1, batches - 1);
            double[] q = new double[K];
            for (int i = 0; i < K; i++) {
                q[i] = (1 - f) * a[i] + f * b[i];
            }
            ShiftSuites.Split batch = ShiftSuites.withPriors(src.pool(), q, batchSize, seed + t);
            double[][] p = src.bcts().probabilities(src.model().logits(batch.x()));
            double[] est = online.priors();   // predict with the estimate from earlier batches, then update
            double[] wOnline = LabelShiftEstimator.weights(est, UNIFORM);
            double[] perBatch = LabelShiftEstimator.mlls(p, UNIFORM, 100);
            okNone += hits(LabelShiftEstimator.correct(p, ones()), batch.y());
            okOnline += hits(LabelShiftEstimator.correct(p, wOnline), batch.y());
            okBatch += hits(LabelShiftEstimator.correct(p, LabelShiftEstimator.weights(perBatch, UNIFORM)), batch.y());
            for (int i = 0; i < K; i++) {
                errOnline += Math.abs(est[i] - q[i]) / (K * batches);
                errBatch += Math.abs(perBatch[i] - q[i]) / (K * batches);
            }
            online.update(p);
        }
        double total = batches * (double) batchSize;
        return new StreamScores(okNone / total, okOnline / total, okBatch / total, errOnline, errBatch);
    }

    private static int hits(double[][] p, int[] y) {
        int[] a = ShiftHarness.argmax(p);
        int ok = 0;
        for (int i = 0; i < y.length; i++) {
            ok += a[i] == y[i] ? 1 : 0;
        }
        return ok;
    }

    /**
     * The long-budget run: three seeds, target sizes 100 to 2,000, and the stream.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        for (long seed = 1; seed <= 3; seed++) {
            Source src = source(1500, seed);
            for (int n : new int[] {100, 500, 2000}) {
                for (Map.Entry<String, double[]> e : shifts(seed).entrySet()) {
                    Map<Method, Score> s = estimate(src, e.getValue(), n, seed);
                    StringBuilder sb = new StringBuilder();
                    s.forEach((m, v) -> sb.append(String.format(" %s %.3f/%.3f", m, v.weightMse(), v.accuracy())));
                    System.out.printf("seed %d n %d %s:%s%n", seed, n, e.getKey(), sb);
                }
            }
            System.out.println("stream " + stream(src, 100, 16, 0.1, seed));
        }
    }
}
