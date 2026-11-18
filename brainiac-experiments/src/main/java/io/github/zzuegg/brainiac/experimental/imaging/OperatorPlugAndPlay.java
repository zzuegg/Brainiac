package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.restore.ImageDenoiser;
import io.github.zzuegg.brainiac.experimental.restore.PlugAndPlay;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Plug-and-play half-quadratic splitting and ADMM for any {@link LinearOperator}, the
 * generalisation of {@link PlugAndPlay}'s blur-only solvers to tomography, MRI, phase retrieval
 * surrogates, inpainting and super-resolution: the data step
 * {@code argmin_x |A x - y|^2 + alpha |x - z|^2} is solved by conjugate gradients, the prior
 * step is any denoiser ({@link Prior}, or an {@link ImageDenoiser} via {@link #of}), with the
 * denoiser's level decreasing log-uniformly from {@code 49/255} to the noise level and
 * {@code alpha_k = lambda sigma^2 / sigma_k^2} (the DPIR schedule {@link PlugAndPlay} uses).
 */
public final class OperatorPlugAndPlay {

    /** A denoiser on one flat image. */
    @FunctionalInterface
    public interface Prior {

        /**
         * Denoises.
         *
         * @param x the image
         * @param sigma the noise level
         * @return the estimate
         */
        double[] denoise(double[] x, double sigma);
    }

    private OperatorPlugAndPlay() {
    }

    /**
     * Wraps a batch denoiser for single square images.
     *
     * @param denoiser the denoiser
     * @param side the image side
     * @return the prior
     */
    public static Prior of(ImageDenoiser denoiser, int side) {
        return (x, sigma) -> {
            float[] f = ImageMath.toFloat(x);
            Tensor t = Tensor.of(Shape.of(1, 1, side, side), f);
            float[] out = Var.noGrad(() -> denoiser.denoise(Var.of(t), (float) sigma)).value().toArray();
            return ImageMath.toDouble(out);
        };
    }

    /**
     * Solves {@code (A^T A + alpha I) x = b} by conjugate gradients.
     *
     * @param a the operator
     * @param b the right-hand side
     * @param alpha the shift
     * @param start the starting point
     * @param iterations the iterations
     * @return the solution
     */
    public static double[] conjugateGradient(LinearOperator a, double[] b, double alpha, double[] start,
            int iterations) {
        double[] x = start.clone();
        double[] ax = a.adjoint(a.apply(x));
        double[] r = new double[b.length];
        for (int i = 0; i < b.length; i++) {
            r[i] = b[i] - ax[i] - alpha * x[i];
        }
        double[] p = r.clone();
        double rr = ImageMath.dot(r, r);
        for (int it = 0; it < iterations && rr > 1e-20; it++) {
            double[] ap = a.adjoint(a.apply(p));
            for (int i = 0; i < ap.length; i++) {
                ap[i] += alpha * p[i];
            }
            double step = rr / ImageMath.dot(p, ap);
            for (int i = 0; i < x.length; i++) {
                x[i] += step * p[i];
                r[i] -= step * ap[i];
            }
            double next = ImageMath.dot(r, r);
            for (int i = 0; i < p.length; i++) {
                p[i] = r[i] + next / rr * p[i];
            }
            rr = next;
        }
        return x;
    }

    private static double[] sigmas(int iterations, double noise) {
        double[] s = new double[iterations];
        double end = Math.max(noise, 0.005);
        for (int i = 0; i < iterations; i++) {
            double t = iterations == 1 ? 1 : i / (double) (iterations - 1);
            s[i] = Math.exp(Math.log(49.0 / 255) * (1 - t) + Math.log(end) * t);
        }
        return s;
    }

    /**
     * Plug-and-play HQS.
     *
     * @param a the forward model
     * @param y the measurements
     * @param noise the measurement noise level (scaled to the image domain by the caller)
     * @param prior the denoiser
     * @param iterations the outer iterations
     * @param lambda the trade-off
     * @param start the starting image
     * @param cgIterations conjugate-gradient iterations per data step
     * @return the estimate
     */
    public static double[] hqs(LinearOperator a, double[] y, double noise, Prior prior, int iterations, double lambda,
            double[] start, int cgIterations) {
        double[] s = sigmas(iterations, noise);
        double[] aty = a.adjoint(y);
        double[] z = start.clone();
        double[] x = start.clone();
        double n2 = Math.max(noise, 0.005) * Math.max(noise, 0.005);
        for (int k = 0; k < iterations; k++) {
            double alpha = lambda * n2 / (s[k] * s[k]);
            x = conjugateGradient(a, ImageMath.axpy(aty, alpha, z), alpha, x, cgIterations);
            z = prior.denoise(x, s[k]);
        }
        return z;
    }

    /**
     * Plug-and-play ADMM on the same schedule.
     *
     * @param a the forward model
     * @param y the measurements
     * @param noise the noise level
     * @param prior the denoiser
     * @param iterations the outer iterations
     * @param lambda the trade-off
     * @param start the starting image
     * @param cgIterations conjugate-gradient iterations per data step
     * @return the estimate
     */
    public static double[] admm(LinearOperator a, double[] y, double noise, Prior prior, int iterations,
            double lambda, double[] start, int cgIterations) {
        double[] s = sigmas(iterations, noise);
        double[] aty = a.adjoint(y);
        double[] z = start.clone();
        double[] x = start.clone();
        double[] u = new double[z.length];
        double n2 = Math.max(noise, 0.005) * Math.max(noise, 0.005);
        for (int k = 0; k < iterations; k++) {
            double alpha = lambda * n2 / (s[k] * s[k]);
            double[] zu = ImageMath.axpy(z, -1, u);
            x = conjugateGradient(a, ImageMath.axpy(aty, alpha, zu), alpha, x, cgIterations);
            z = prior.denoise(ImageMath.axpy(x, 1, u), s[k]);
            for (int i = 0; i < u.length; i++) {
                u[i] += x[i] - z[i];
            }
        }
        return z;
    }
}
