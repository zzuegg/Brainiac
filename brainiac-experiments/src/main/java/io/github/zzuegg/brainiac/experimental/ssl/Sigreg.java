package io.github.zzuegg.brainiac.experimental.ssl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * SIGReg, sketched isotropic Gaussian regularisation (Balestriero and LeCun, LeJEPA, 2025): a
 * differentiable test that a batch of embeddings is distributed as an isotropic standard
 * Gaussian, used as the only anti-collapse term of {@link LeJepa}.
 *
 * <p>The batch {@code [n, d]} is projected onto {@code M} random unit directions (resampled each
 * call, the "sketch"). For each 1-D projection the empirical characteristic function
 * {@code phi_n(t) = mean_i exp(i t x_i)} is compared with the standard normal's
 * {@code exp(-t^2 / 2)} by the Epps-Pulley statistic
 * {@code n * integral |phi_n(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt}, integrated by the trapezoid
 * rule on {@code points} knots over {@code [-5, 5]} (17 in the paper). The statistics are
 * averaged over the directions. By the Cramer-Wold theorem a distribution whose every 1-D
 * projection is standard normal is the isotropic Gaussian, so the loss is minimal exactly
 * there; collapse (all rows alike) is far from it. The cost is linear in the batch size, and
 * the characteristic function is bounded, so the gradients stay bounded even for outliers.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2511.08544">arXiv:2511.08544</a> (Algorithm 1).
 */
public final class Sigreg {

    /** The paper's pseudocode default number of slices. */
    public static final int DEFAULT_SLICES = 256;
    /** The paper's number of integration knots on {@code [-5, 5]}. */
    public static final int DEFAULT_POINTS = 17;

    private Sigreg() {
    }

    /**
     * SIGReg with the default 256 slices and 17 knots.
     *
     * @param z the embeddings {@code [n, d]}
     * @param random the source of the random directions
     * @return the mean Epps-Pulley statistic over the slices, a scalar
     */
    public static Var loss(Var z, RandomGenerator random) {
        return loss(z, DEFAULT_SLICES, DEFAULT_POINTS, random);
    }

    /**
     * SIGReg: the Epps-Pulley statistic of {@code slices} random 1-D projections, averaged.
     *
     * @param z the embeddings {@code [n, d]}
     * @param slices the number of random unit directions {@code M}
     * @param points the trapezoid knots on {@code [-5, 5]}, at least 2
     * @param random the source of the random directions
     * @return a scalar to minimise
     */
    public static Var loss(Var z, int slices, int points, RandomGenerator random) {
        int d = z.shape().dim(1);
        float[] a = new float[d * slices];
        for (int m = 0; m < slices; m++) {
            double norm = 0;
            for (int j = 0; j < d; j++) {
                float g = (float) random.nextGaussian();
                a[j * slices + m] = g;
                norm += g * g;
            }
            float inv = (float) (1 / Math.sqrt(norm + 1e-12));
            for (int j = 0; j < d; j++) {
                a[j * slices + m] *= inv;
            }
        }
        Var projections = z.matmul(Var.of(Tensor.of(Shape.of(d, slices), a, z.value().device())));
        return eppsPulley(projections, points).mean();
    }

    /**
     * The Epps-Pulley normality statistic of every column of {@code x}, scaled by the number
     * of rows as in the paper.
     *
     * @param x samples {@code [n, m]}: {@code m} 1-D samples of size {@code n}
     * @param points the trapezoid knots on {@code [-5, 5]}
     * @return the statistic per column {@code [m]}
     */
    public static Var eppsPulley(Var x, int points) {
        if (points < 2) {
            throw new IllegalArgumentException("need at least 2 knots: " + points);
        }
        int n = x.shape().dim(0);
        int m = x.shape().dim(1);
        float[] t = new float[points];
        float[] target = new float[points];
        float[] weight = new float[points];
        float dt = 10f / (points - 1);
        for (int k = 0; k < points; k++) {
            t[k] = -5 + k * dt;
            target[k] = (float) Math.exp(-0.5 * t[k] * t[k]);
            // Gaussian window times trapezoid weights
            weight[k] = target[k] * dt * (k == 0 || k == points - 1 ? 0.5f : 1f);
        }
        var device = x.value().device();
        Var xt = x.reshape(n, m, 1).mul(Tensor.of(Shape.of(1, 1, points), t, device));
        Var re = xt.cos().mean(0, false);
        Var im = xt.sin().mean(0, false);
        Var err = re.sub(Tensor.of(Shape.of(1, points), target, device)).square().add(im.square());
        return err.mul(Tensor.of(Shape.of(1, points), weight, device)).sum(1, false).mul(n);
    }

    /**
     * The Epps-Pulley statistic of one 1-D sample, in plain doubles (for checks and reports).
     *
     * @param sample the values
     * @param points the knots on {@code [-5, 5]}
     * @return {@code n * integral |phi_n - phi|^2 w}
     */
    public static double statistic(double[] sample, int points) {
        double dt = 10.0 / (points - 1);
        double total = 0;
        for (int k = 0; k < points; k++) {
            double t = -5 + k * dt;
            double re = 0;
            double im = 0;
            for (double v : sample) {
                re += Math.cos(t * v) / sample.length;
                im += Math.sin(t * v) / sample.length;
            }
            double phi = Math.exp(-0.5 * t * t);
            double err = (re - phi) * (re - phi) + im * im;
            total += err * phi * dt * (k == 0 || k == points - 1 ? 0.5 : 1);
        }
        return total * sample.length;
    }
}
