package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Value functions trained by classification, with the HL-Gauss target (Farebrother et al.,
 * "Stop Regressing", <a href="https://arxiv.org/abs/2403.03950">2403.03950</a>). The critic
 * outputs logits over {@code bins} equal bins covering {@code [min, max]}; its value is the
 * expected bin centre. A scalar regression target {@code y} becomes the distribution that a
 * Gaussian {@code N(y, sigma^2)} puts on the bins (differences of its CDF at the bin edges),
 * and the loss is the cross-entropy to it. Cross-entropy gradients do not grow with the TD
 * error and the smoothed target spreads information across nearby values, which makes
 * value learning more robust to noisy targets and to scaling up the network than an MSE
 * critic. {@code sigma} = 0.75 bin widths is the paper's default.
 */
public final class HlGaussValue {
    private final float min;
    private final float max;
    private final int bins;
    private final float sigma;
    private final Tensor centres;

    private HlGaussValue(float min, float max, int bins, float sigma) {
        if (!(max > min) || bins < 2 || !(sigma > 0)) {
            throw new IllegalArgumentException("need min < max, bins >= 2 and sigma > 0");
        }
        this.min = min;
        this.max = max;
        this.bins = bins;
        this.sigma = sigma;
        float[] c = new float[bins];
        float w = (max - min) / bins;
        for (int i = 0; i < bins; i++) {
            c[i] = min + (i + 0.5f) * w;
        }
        this.centres = Tensor.of(Shape.of(bins, 1), c);
    }

    /**
     * Creates the head with sigma = 0.75 bin widths.
     *
     * @param min the lowest representable value
     * @param max the highest
     * @param bins the number of bins
     * @return the head
     */
    public static HlGaussValue of(float min, float max, int bins) {
        return new HlGaussValue(min, max, bins, 0.75f * (max - min) / bins);
    }

    /** @return the number of bins, the critic's output size */
    public int bins() {
        return bins;
    }

    /**
     * The expected value under the critic's distribution.
     *
     * @param logits {@code [batch, bins]}
     * @return {@code [batch, 1]}
     */
    public Var value(Var logits) {
        return RlTensors.softmax(logits).matmul(Var.of(centres.to(logits.value().device())));
    }

    /**
     * The HL-Gauss target distribution for scalar targets.
     *
     * @param targets {@code [batch, 1]} or {@code [batch]}
     * @return {@code [batch, bins]}, rows summing to 1
     */
    public Tensor target(Tensor targets) {
        float[] y = targets.toArray();
        float[] p = new float[y.length * bins];
        float w = (max - min) / bins;
        for (int i = 0; i < y.length; i++) {
            double yi = Math.max(min, Math.min(max, y[i]));
            double lo = cdf((min - yi) / sigma);
            double hi = cdf((max - yi) / sigma);
            double z = Math.max(1e-12, hi - lo);
            double prev = lo;
            for (int b = 0; b < bins; b++) {
                double edge = cdf((min + (b + 1) * w - yi) / sigma);
                p[i * bins + b] = (float) ((edge - prev) / z);
                prev = edge;
            }
        }
        return Tensor.of(Shape.of(y.length, bins), p);
    }

    /**
     * The cross-entropy loss towards the HL-Gauss target.
     *
     * @param logits {@code [batch, bins]}
     * @param targets scalar targets {@code [batch, 1]}
     * @return the mean loss, a scalar
     */
    public Var loss(Var logits, Tensor targets) {
        Tensor t = target(targets).to(logits.value().device());
        return RlTensors.logSoftmax(logits).mul(t).sum(1, true).neg().mean();
    }

    private static double cdf(double x) {
        return 0.5 * (1 + erf(x / Math.sqrt(2)));
    }

    /** The error function (Abramowitz and Stegun 7.1.26, error below 1.5e-7). */
    static double erf(double x) {
        double s = Math.signum(x);
        double a = Math.abs(x);
        double t = 1 / (1 + 0.3275911 * a);
        double y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
                * t * Math.exp(-a * a);
        return s * y;
    }
}
