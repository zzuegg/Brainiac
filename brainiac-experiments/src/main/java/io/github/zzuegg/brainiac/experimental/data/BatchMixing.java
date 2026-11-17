package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Batch-level augmentations that mix pairs of samples <em>and</em> their targets, which the
 * core per-sample {@code Augmentation} cannot express:
 *
 * <ul>
 *   <li>{@link #mixup} (Zhang et al. 2018, <a href="https://arxiv.org/abs/1710.09412">arXiv:1710.09412</a>):
 *       {@code x = lam x_i + (1 - lam) x_j}, same for the one-hot targets, with
 *       {@code lam ~ Beta(alpha, alpha)}.</li>
 *   <li>{@link #cutMix} (Yun et al. 2019, <a href="https://arxiv.org/abs/1905.04899">arXiv:1905.04899</a>):
 *       pastes a random rectangle of another image and mixes the targets by its area.</li>
 * </ul>
 *
 * <p>The mixed targets are soft, which the core {@code Loss.CROSS_ENTROPY} accepts.
 */
public final class BatchMixing {

    private BatchMixing() {
    }

    /**
     * Mixup of a batch with a shuffled copy of itself.
     *
     * @param x {@code [batch, ...]} inputs
     * @param y {@code [batch, classes]} one-hot or soft targets
     * @param alpha the Beta distribution's parameter, for example 0.2
     * @param random the random source
     * @return {@code [mixed x, mixed y]}
     */
    public static Tensor[] mixup(Tensor x, Tensor y, float alpha, SplittableRandom random) {
        int n = x.shape().dim(0);
        int[] perm = permutation(n, random);
        float lam = (float) beta(alpha, random);
        Tensor xs = gatherRows(x, perm);
        Tensor ys = gatherRows(y, perm);
        return new Tensor[] {x.mul(lam).add(xs.mul(1 - lam)), y.mul(lam).add(ys.mul(1 - lam))};
    }

    /**
     * CutMix for images stored as {@code [batch, channels, height, width]} (or flat rows of
     * that many values).
     *
     * @param x the images
     * @param y {@code [batch, classes]} targets
     * @param channels channels per image
     * @param height the image height
     * @param width the image width
     * @param alpha the Beta parameter, for example 1
     * @param random the random source
     * @return {@code [mixed x, mixed y]}
     */
    public static Tensor[] cutMix(Tensor x, Tensor y, int channels, int height, int width, float alpha,
            SplittableRandom random) {
        int n = x.shape().dim(0);
        int[] perm = permutation(n, random);
        double lam = beta(alpha, random);
        int cutH = (int) Math.round(height * Math.sqrt(1 - lam));
        int cutW = (int) Math.round(width * Math.sqrt(1 - lam));
        int cy = random.nextInt(height);
        int cx = random.nextInt(width);
        int y0 = Math.max(0, cy - cutH / 2);
        int y1 = Math.min(height, cy + (cutH + 1) / 2);
        int x0 = Math.max(0, cx - cutW / 2);
        int x1 = Math.min(width, cx + (cutW + 1) / 2);
        float[] v = x.toArray();
        float[] out = v.clone();
        int plane = height * width;
        int size = channels * plane;
        for (int i = 0; i < n; i++) {
            for (int c = 0; c < channels; c++) {
                for (int r = y0; r < y1; r++) {
                    for (int q = x0; q < x1; q++) {
                        int off = c * plane + r * width + q;
                        out[i * size + off] = v[perm[i] * size + off];
                    }
                }
            }
        }
        float kept = 1 - (float) ((y1 - y0) * (x1 - x0)) / plane;
        Tensor ys = gatherRows(y, perm);
        return new Tensor[] {Tensor.of(x.shape(), out), y.mul(kept).add(ys.mul(1 - kept))};
    }

    private static int[] permutation(int n, SplittableRandom random) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    private static Tensor gatherRows(Tensor t, int[] rows) {
        float[] ids = new float[rows.length];
        for (int i = 0; i < rows.length; i++) {
            ids[i] = rows[i];
        }
        return t.gather(0, Tensor.of(Shape.of(rows.length), ids));
    }

    /** A Beta(alpha, alpha) sample from two Gamma samples (Marsaglia-Tsang). */
    static double beta(double alpha, SplittableRandom r) {
        double a = gamma(alpha, r);
        double b = gamma(alpha, r);
        return a / (a + b);
    }

    private static double gamma(double shape, SplittableRandom r) {
        if (shape < 1) {
            return gamma(shape + 1, r) * Math.pow(r.nextDouble(), 1 / shape);
        }
        double d = shape - 1.0 / 3;
        double c = 1 / Math.sqrt(9 * d);
        while (true) {
            double x = r.nextGaussian();
            double v = 1 + c * x;
            if (v <= 0) {
                continue;
            }
            v = v * v * v;
            double u = r.nextDouble();
            if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
                return d * v;
            }
        }
    }
}
