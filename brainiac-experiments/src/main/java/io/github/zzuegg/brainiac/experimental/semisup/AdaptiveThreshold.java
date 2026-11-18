package io.github.zzuegg.brainiac.experimental.semisup;

import java.util.Arrays;

/**
 * Adaptive confidence thresholds for {@link FixMatch}: a fixed 0.95 threshold wastes most
 * unlabelled data early in training and favours easy classes.
 *
 * <ul>
 *   <li>{@link Mode#FLEX} (FlexMatch; Zhang et al., NeurIPS 2021,
 *       <a href="https://arxiv.org/abs/2110.08263">arXiv:2110.08263</a>): curriculum pseudo-labelling.
 *       Each class's learning status {@code sigma_c} counts the unlabelled examples currently
 *       predicted as {@code c} above {@code tau}; {@code beta_c = sigma_c / max(max_k sigma_k,
 *       unused)} (the warm-up form), and the class threshold is {@code M(beta_c) tau} with the
 *       convex mapping {@code M(x) = x / (2 - x)}.</li>
 *   <li>{@link Mode#FREE} (FreeMatch; Wang et al., ICLR 2023,
 *       <a href="https://arxiv.org/abs/2205.07246">arXiv:2205.07246</a>): a self-adaptive global
 *       threshold, the EMA of the mean top confidence, modulated per class by the max-normalised
 *       EMA of the mean predicted distribution.</li>
 *   <li>{@link Mode#SOFT} (SoftMatch; Chen et al., ICLR 2023,
 *       <a href="https://arxiv.org/abs/2301.10921">arXiv:2301.10921</a>): no hard cut; a
 *       truncated-Gaussian weight {@code exp(-(c - mu)^2 / (2 sigma^2))} for confidences
 *       {@code c} below the EMA mean {@code mu} and 1 above it.</li>
 * </ul>
 * FreeMatch's self-adaptive fairness term and SoftMatch's uniform alignment are not included.
 */
public final class AdaptiveThreshold implements FixMatch.Mask {

    /** Which adaptive rule. */
    public enum Mode {
        /** FlexMatch's per-class curriculum thresholds. */
        FLEX,
        /** FreeMatch's self-adaptive global and local thresholds. */
        FREE,
        /** SoftMatch's truncated-Gaussian weights. */
        SOFT
    }

    private final Mode mode;
    private final int classes;
    private final float tau;
    private final double ema;
    private final int[] prediction;
    private double globalConf;
    private final double[] classConf;
    private double mu;
    private double var;

    /**
     * Creates the rule.
     *
     * @param mode the rule
     * @param classes the number of classes
     * @param pool the size of the unlabelled pool (FlexMatch remembers every example's prediction)
     * @param tau the upper threshold for FlexMatch, such as 0.95; unused by the others
     * @param ema the EMA momentum of FreeMatch and SoftMatch, such as 0.999 (0.99 for short runs)
     */
    public AdaptiveThreshold(Mode mode, int classes, int pool, float tau, double ema) {
        this.mode = mode;
        this.classes = classes;
        this.tau = tau;
        this.ema = ema;
        this.prediction = new int[pool];
        Arrays.fill(prediction, -1);
        this.globalConf = 1.0 / classes;
        this.classConf = new double[classes];
        Arrays.fill(classConf, 1.0 / classes);
        this.mu = 1.0 / classes;
        this.var = 1.0;
    }

    @Override
    public float[] weights(float[][] probs, int[] indices) {
        int n = probs.length;
        int[] hard = SoftTargets.argmax(probs);
        float[] conf = new float[n];
        for (int i = 0; i < n; i++) {
            conf[i] = probs[i][hard[i]];
        }
        float[] w = new float[n];
        switch (mode) {
            case FLEX -> {
                double[] t = classThresholds();
                for (int i = 0; i < n; i++) {
                    w[i] = conf[i] >= t[hard[i]] ? 1 : 0;
                    if (conf[i] >= tau) {
                        prediction[indices[i]] = hard[i];
                    }
                }
            }
            case FREE -> {
                double meanConf = 0;
                double[] meanProb = new double[classes];
                for (int i = 0; i < n; i++) {
                    meanConf += conf[i] / n;
                    for (int c = 0; c < classes; c++) {
                        meanProb[c] += probs[i][c] / n;
                    }
                }
                globalConf = ema * globalConf + (1 - ema) * meanConf;
                double max = 0;
                for (int c = 0; c < classes; c++) {
                    classConf[c] = ema * classConf[c] + (1 - ema) * meanProb[c];
                    max = Math.max(max, classConf[c]);
                }
                for (int i = 0; i < n; i++) {
                    w[i] = conf[i] >= globalConf * classConf[hard[i]] / max ? 1 : 0;
                }
            }
            case SOFT -> {
                double m = 0;
                for (float c : conf) {
                    m += c / n;
                }
                double v = 0;
                for (float c : conf) {
                    v += (c - m) * (c - m) / Math.max(1, n - 1);
                }
                mu = ema * mu + (1 - ema) * m;
                var = ema * var + (1 - ema) * v;
                for (int i = 0; i < n; i++) {
                    double d = conf[i] - mu;
                    w[i] = conf[i] >= mu ? 1 : (float) Math.exp(-d * d / (2 * var + 1e-12));
                }
            }
            default -> throw new IllegalStateException();
        }
        return w;
    }

    /** @return FlexMatch's current per-class thresholds (for the other modes, the effective ones) */
    public double[] classThresholds() {
        double[] t = new double[classes];
        switch (mode) {
            case FLEX -> {
                int[] sigma = new int[classes];
                int unused = 0;
                for (int p : prediction) {
                    if (p < 0) {
                        unused++;
                    } else {
                        sigma[p]++;
                    }
                }
                int max = unused;
                for (int s : sigma) {
                    max = Math.max(max, s);
                }
                for (int c = 0; c < classes; c++) {
                    double beta = max == 0 ? 0 : sigma[c] / (double) max;
                    t[c] = beta / (2 - beta) * tau;
                }
            }
            case FREE -> {
                double max = Arrays.stream(classConf).max().orElse(1);
                for (int c = 0; c < classes; c++) {
                    t[c] = globalConf * classConf[c] / max;
                }
            }
            default -> Arrays.fill(t, mu);
        }
        return t;
    }
}
