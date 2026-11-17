package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.nn.Crosscoder;
import io.github.zzuegg.brainiac.experimental.nn.LayerHooks;
import io.github.zzuegg.brainiac.experimental.nn.SparseAutoencoder;
import io.github.zzuegg.brainiac.experimental.train.MachineUnlearning;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.SplittableRandom;

/**
 * Model diffing with a crosscoder. A base classifier learns a linear rule over twelve sparse
 * binary input features and never reads an extra "trigger" input; a fine-tuned copy
 * additionally learns a planted behaviour, "when the trigger is on, answer class 0". A
 * {@link Crosscoder} is trained on the two models' first hidden layer for the same inputs;
 * features whose decoder lives
 * almost only in the fine-tuned model (relative decoder norm above 0.9) are its
 * fine-tune-only features, and one of them should detect the trigger. The baseline diffs
 * two separately trained sparse autoencoders: a fine-tuned-model feature with no base
 * feature within cosine 0.7 counts as new. The outcome counts the candidates each method
 * flags and how well the best of them detects the trigger (AUROC).
 *
 * <p>Lindsey et al. 2024, "Sparse Crosscoders for Cross-Layer Features and Model Diffing",
 * <a href="https://transformer-circuits.pub/2024/crosscoders/index.html">Transformer Circuits</a>;
 * Minder et al. 2025, "Robustly identifying concepts introduced during chat fine-tuning
 * using crosscoders", <a href="https://arxiv.org/abs/2504.02922">arXiv:2504.02922</a>.
 */
public final class ModelDiffing {

    /**
     * The outcome.
     *
     * @param planted the fine-tuned model's accuracy on the planted behaviour
     * @param crosscoderExclusive features the crosscoder calls fine-tune-only
     * @param crosscoderBestAuroc the best trigger AUROC among them
     * @param crosscoderPrecision the share of them that detect the trigger (AUROC above 0.8, either sign)
     * @param saeNew fine-tuned SAE features with no base match
     * @param saeBestAuroc the best trigger AUROC among them
     * @param saePrecision the share of them that detect the trigger
     */
    public record Outcome(double planted, int crosscoderExclusive, double crosscoderBestAuroc,
            double crosscoderPrecision, int saeNew, double saeBestAuroc, double saePrecision) {
    }

    private static final int CLASSES = 4;
    private static final int DIM = 12;

    private ModelDiffing() {
    }

    /** Independent sparse binary features; the class is the argmax of a fixed linear score. */
    private static float[][][] data(int n, SplittableRandom r, double triggerRate, boolean planted, float[][] score) {
        float[][] x = new float[n][DIM + 1];
        float[][] y = new float[n][CLASSES];
        for (int i = 0; i < n; i++) {
            for (int d = 0; d < DIM; d++) {
                x[i][d] = r.nextDouble() < 0.25 ? 1 : 0;
            }
            int c = 0;
            double best = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < CLASSES; k++) {
                double s = 0;
                for (int d = 0; d < DIM; d++) {
                    s += score[k][d] * x[i][d];
                }
                if (s > best) {
                    best = s;
                    c = k;
                }
            }
            boolean trig = r.nextDouble() < triggerRate;
            x[i][DIM] = trig ? 1 : 0;
            y[i][planted && trig ? 0 : c] = 1;
        }
        return new float[][][] {x, y};
    }

    /**
     * Runs the experiment.
     *
     * @param steps dictionary training steps
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] centres = new float[CLASSES][DIM];
        for (float[] c : centres) {
            for (int d = 0; d < DIM; d++) {
                c[d] = (float) r.nextGaussian();
            }
        }
        Model base = Model.sequential().input(DIM + 1).dense(16, Activation.TANH).dense(16, Activation.TANH)
                .dense(CLASSES, Activation.SOFTMAX).seed(seed).build();
        float[][][] pre = data(1200, r, 0, false, centres);
        Trainer.of(base).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.01f)).batchSize(32).epochs(15).seed(seed)
                .fit(Dataset.of(pre[0], pre[1]));
        // the base model never saw the trigger: it does not read that input at all
        Var first = ((Dense) base.layers().get(0)).weight();
        float[] w0 = first.value().to(DType.FLOAT32).toArray();
        int width = first.shape().dim(1);
        java.util.Arrays.fill(w0, DIM * width, (DIM + 1) * width, 0f);
        first.assign(Tensor.of(first.shape(), w0));
        Model tuned = base.copy(base.device());
        float[][][] ft = data(1200, r, 0.2, true, centres);
        Trainer.of(tuned).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.005f)).batchSize(32).epochs(10)
                .seed(seed + 1).fit(Dataset.of(ft[0], ft[1]));
        float[][][] test = data(800, r, 0.5, true, centres);
        double planted = 0;
        int triggered = 0;
        float[][] p = tuned.predict(test[0]);
        for (int i = 0; i < test[0].length; i++) {
            if (test[0][i][DIM] > 0) {
                triggered++;
                planted += argmax(p[i]) == 0 ? 1 : 0;
            }
        }
        planted /= triggered;
        float[][] hb = hidden(base, test[0]);
        float[][] ht = hidden(tuned, test[0]);
        boolean[] trigger = new boolean[test[0].length];
        for (int i = 0; i < trigger.length; i++) {
            trigger[i] = test[0][i][DIM] > 0;
        }
        int features = 32;
        Crosscoder cc = Crosscoder.of(new int[] {hb[0].length, ht[0].length}, features, 0.3f, seed + 2);
        cc.train(new float[][][] {hb, ht}, steps, 128, 3e-3f, seed + 3);
        double[] rel = cc.relativeDecoderNorm(1);
        float[][] codes = cc.codes(new float[][][] {hb, ht});
        int exclusive = 0;
        int ccHits = 0;
        double ccBest = 0.5;
        for (int f = 0; f < features; f++) {
            if (rel[f] > 0.9 && fires(codes, f)) {
                exclusive++;
                double a = auroc(codes, f, trigger);
                ccBest = Math.max(ccBest, a);
                ccHits += a > 0.8 ? 1 : 0;
            }
        }
        SparseAutoencoder saeBase = SparseAutoencoder.l1(hb[0].length, features, 0.3f, seed + 2);
        saeBase.train(hb, steps, 128, 3e-3f, seed + 3);
        SparseAutoencoder saeTuned = SparseAutoencoder.l1(ht[0].length, features, 0.3f, seed + 2);
        saeTuned.train(ht, steps, 128, 3e-3f, seed + 3);
        float[][] tc = saeTuned.codes(ht);
        int fresh = 0;
        int saeHits = 0;
        double saeBest = 0.5;
        for (int f = 0; f < features; f++) {
            if (!fires(tc, f)) {
                continue;
            }
            double best = 0;
            for (int g = 0; g < features; g++) {
                best = Math.max(best, cosine(saeTuned.direction(f), saeBase.direction(g)));
            }
            if (best < 0.7) {
                fresh++;
                double a = auroc(tc, f, trigger);
                saeBest = Math.max(saeBest, a);
                saeHits += a > 0.8 ? 1 : 0;
            }
        }
        return new Outcome(planted, exclusive, ccBest, exclusive == 0 ? 0 : ccHits / (double) exclusive, fresh, saeBest,
                fresh == 0 ? 0 : saeHits / (double) fresh);
    }

    private static boolean fires(float[][] codes, int f) {
        for (float[] row : codes) {
            if (row[f] > 0) {
                return true;
            }
        }
        return false;
    }

    /** How well a feature separates triggered from untriggered samples: the AUROC, in either direction. */
    private static double auroc(float[][] codes, int f, boolean[] trigger) {
        int n = 0;
        int m = 0;
        for (boolean t : trigger) {
            n += t ? 1 : 0;
            m += t ? 0 : 1;
        }
        double[] pos = new double[n];
        double[] neg = new double[m];
        int i = 0;
        int j = 0;
        for (int s = 0; s < codes.length; s++) {
            if (trigger[s]) {
                pos[i++] = -codes[s][f];
            } else {
                neg[j++] = -codes[s][f];
            }
        }
        double a = MachineUnlearning.auc(pos, neg);
        return Math.max(a, 1 - a);
    }

    private static float[][] hidden(Model m, float[][] x) {
        Tensor h = LayerHooks.trace(m, Var.of(Tensor.of(x))).get(0).value();
        int d = h.shape().dim(-1);
        float[] flat = h.to(DType.FLOAT32).toArray();
        float[][] out = new float[x.length][d];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(flat, i * d, out[i], 0, d);
        }
        return out;
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return na == 0 || nb == 0 ? 0 : dot / Math.sqrt(na * nb);
    }

    private static int argmax(float[] p) {
        int b = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[b]) {
                b = i;
            }
        }
        return b;
    }
}
