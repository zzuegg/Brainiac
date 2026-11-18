package io.github.zzuegg.brainiac.experimental.imaging;

/**
 * The first-order primal-dual hybrid gradient method of Chambolle and Pock (JMIV 2011) for
 * total-variation imaging, {@code min_x G(x) + lambda TV(x)}: a dual ascent step on the
 * gradient field projected onto the ball {@code |p| <= lambda}, a proximal step on the data
 * term {@code G}, and an over-relaxation {@code xbar = 2 x_new - x}, with step sizes
 * {@code tau sigma |K|^2 < 1} ({@code |grad|^2 <= 8}). The data term enters only through its
 * proximal operator ({@link DataTerm}), so the same loop solves
 *
 * <ul>
 *   <li>{@link Model#TV_L2} denoising ({@code G = 1/2 |x - f|^2}, Gaussian noise; ROF),</li>
 *   <li>{@link Model#TV_L1} denoising ({@code G = |x - f|_1}, impulse noise),</li>
 *   <li>deblurring ({@link #deblur}, the proximal step solved in the Fourier domain) and
 *       inpainting ({@link #inpaint}),</li>
 *   <li>{@link Model#TGV} (second-order total generalised variation, Bredies, Kunisch &amp; Pock
 *       2010), {@code min_{x, w} G(x) + alpha1 |grad x - w| + alpha0 |E w|}, which keeps ramps
 *       instead of staircasing them,</li>
 *   <li>and any linear forward model ({@link #linear}, for example tomography or single-pixel
 *       imaging) with {@code K = [A; grad]} and a dual data variable.</li>
 * </ul>
 */
public final class PrimalDual {

    /** The built-in denoising models. */
    public enum Model {
        /** Total variation with a quadratic data term. */
        TV_L2,
        /** Total variation with an absolute data term. */
        TV_L1,
        /** Second-order total generalised variation with a quadratic data term. */
        TGV
    }

    /** The proximal operator of a separable data term. */
    public interface DataTerm {

        /**
         * {@code argmin_x G(x) + |x - v|^2 / (2 tau)}.
         *
         * @param v the point
         * @param tau the step
         * @return the proximal point
         */
        double[] prox(double[] v, double tau);

        /**
         * The value {@code G(x)}.
         *
         * @param x the image
         * @return the value
         */
        double value(double[] x);
    }

    private PrimalDual() {
    }

    /**
     * {@code G(x) = 1/2 |x - f|^2}.
     *
     * @param f the observation
     * @return the data term
     */
    public static DataTerm l2(double[] f) {
        return new DataTerm() {
            @Override
            public double[] prox(double[] v, double tau) {
                double[] x = new double[v.length];
                for (int i = 0; i < v.length; i++) {
                    x[i] = (v[i] + tau * f[i]) / (1 + tau);
                }
                return x;
            }

            @Override
            public double value(double[] x) {
                return 0.5 * ImageMath.mse(x, f) * x.length;
            }
        };
    }

    /**
     * {@code G(x) = |x - f|_1}.
     *
     * @param f the observation
     * @return the data term
     */
    public static DataTerm l1(double[] f) {
        return new DataTerm() {
            @Override
            public double[] prox(double[] v, double tau) {
                double[] x = new double[v.length];
                for (int i = 0; i < v.length; i++) {
                    double d = v[i] - f[i];
                    x[i] = f[i] + Math.signum(d) * Math.max(0, Math.abs(d) - tau);
                }
                return x;
            }

            @Override
            public double value(double[] x) {
                double s = 0;
                for (int i = 0; i < x.length; i++) {
                    s += Math.abs(x[i] - f[i]);
                }
                return s;
            }
        };
    }

    /**
     * {@code G(x) = 1/2 |k * x - y|^2} for a circular blur, its proximal step solved exactly in
     * the Fourier domain.
     *
     * @param y the blurred image
     * @param kernel the kernel
     * @param k its side
     * @param side the image side
     * @return the data term
     */
    public static DataTerm deblur(double[] y, double[] kernel, int k, int side) {
        double[][] h = ImageMath.transfer(kernel, k, side);
        double[] yr = y.clone();
        double[] yi = new double[y.length];
        ImageMath.fft2(yr, yi, side, side, false);
        return new DataTerm() {
            @Override
            public double[] prox(double[] v, double tau) {
                double[] vr = v.clone();
                double[] vi = new double[v.length];
                ImageMath.fft2(vr, vi, side, side, false);
                for (int i = 0; i < v.length; i++) {
                    double kr = h[0][i];
                    double ki = h[1][i];
                    double nr = vr[i] + tau * (kr * yr[i] + ki * yi[i]);
                    double ni = vi[i] + tau * (kr * yi[i] - ki * yr[i]);
                    double den = 1 + tau * (kr * kr + ki * ki);
                    vr[i] = nr / den;
                    vi[i] = ni / den;
                }
                ImageMath.fft2(vr, vi, side, side, true);
                return vr;
            }

            @Override
            public double value(double[] x) {
                double[] r = ImageMath.convolve(x, side, kernel, k);
                double s = 0;
                for (int i = 0; i < r.length; i++) {
                    s += (r[i] - y[i]) * (r[i] - y[i]);
                }
                return 0.5 * s;
            }
        };
    }

    /**
     * {@code G(x) = 1/2 |M (x - y)|^2} for a pixel mask (inpainting).
     *
     * @param y the observation, any value where not kept
     * @param keep the observed pixels
     * @return the data term
     */
    public static DataTerm inpaint(double[] y, boolean[] keep) {
        return new DataTerm() {
            @Override
            public double[] prox(double[] v, double tau) {
                double[] x = v.clone();
                for (int i = 0; i < v.length; i++) {
                    if (keep[i]) {
                        x[i] = (v[i] + tau * y[i]) / (1 + tau);
                    }
                }
                return x;
            }

            @Override
            public double value(double[] x) {
                double s = 0;
                for (int i = 0; i < x.length; i++) {
                    if (keep[i]) {
                        s += (x[i] - y[i]) * (x[i] - y[i]);
                    }
                }
                return 0.5 * s;
            }
        };
    }

    /**
     * Solves a built-in denoising model.
     *
     * @param model the model
     * @param f the noisy image
     * @param side its side
     * @param lambda the TV weight ({@code alpha1} for TGV, with {@code alpha0 = 2 lambda})
     * @param iterations the iterations
     * @return the solution
     */
    public static double[] denoise(Model model, double[] f, int side, double lambda, int iterations) {
        return switch (model) {
            case TV_L2 -> tv(l2(f), f, side, lambda, iterations);
            case TV_L1 -> tv(l1(f), f, side, lambda, iterations);
            case TGV -> tgv(l2(f), f, side, lambda, 2 * lambda, iterations);
        };
    }

    /**
     * Chambolle-Pock for {@code min_x G(x) + lambda TV(x)}.
     *
     * @param g the data term
     * @param start the starting image
     * @param side the side
     * @param lambda the TV weight
     * @param iterations the iterations
     * @return the solution
     */
    public static double[] tv(DataTerm g, double[] start, int side, double lambda, int iterations) {
        int n = side * side;
        double step = 0.99 / Math.sqrt(8);
        double[] x = start.clone();
        double[] xbar = x.clone();
        double[] px = new double[n];
        double[] py = new double[n];
        for (int it = 0; it < iterations; it++) {
            double[] gx = ImageMath.gradX(xbar, side);
            double[] gy = ImageMath.gradY(xbar, side);
            for (int i = 0; i < n; i++) {
                double a = px[i] + step * gx[i];
                double b = py[i] + step * gy[i];
                double scale = Math.max(1, Math.hypot(a, b) / lambda);
                px[i] = a / scale;
                py[i] = b / scale;
            }
            double[] d = ImageMath.div(px, py, side);
            double[] v = new double[n];
            for (int i = 0; i < n; i++) {
                v[i] = x[i] + step * d[i];
            }
            double[] next = g.prox(v, step);
            for (int i = 0; i < n; i++) {
                xbar[i] = 2 * next[i] - x[i];
            }
            x = next;
        }
        return x;
    }

    /**
     * The objective {@code G(x) + lambda TV(x)}.
     *
     * @param g the data term
     * @param x the image
     * @param side the side
     * @param lambda the TV weight
     * @return the objective
     */
    public static double objective(DataTerm g, double[] x, int side, double lambda) {
        return g.value(x) + lambda * ImageMath.totalVariation(x, side);
    }

    /**
     * Chambolle-Pock for second-order TGV, {@code min G(x) + alpha1 |grad x - w| + alpha0 |E w|}
     * with {@code E w} the symmetrised gradient.
     *
     * @param g the data term
     * @param start the starting image
     * @param side the side
     * @param alpha1 the first-order weight
     * @param alpha0 the second-order weight
     * @param iterations the iterations
     * @return the solution
     */
    public static double[] tgv(DataTerm g, double[] start, int side, double alpha1, double alpha0, int iterations) {
        int n = side * side;
        double step = 0.99 / Math.sqrt(12);
        double[] x = start.clone();
        double[] xbar = x.clone();
        double[] w1 = new double[n];
        double[] w2 = new double[n];
        double[] w1bar = new double[n];
        double[] w2bar = new double[n];
        double[] p1 = new double[n];
        double[] p2 = new double[n];
        double[] q11 = new double[n];
        double[] q22 = new double[n];
        double[] q12 = new double[n];
        for (int it = 0; it < iterations; it++) {
            double[] gx = ImageMath.gradX(xbar, side);
            double[] gy = ImageMath.gradY(xbar, side);
            for (int i = 0; i < n; i++) {
                double a = p1[i] + step * (gx[i] - w1bar[i]);
                double b = p2[i] + step * (gy[i] - w2bar[i]);
                double scale = Math.max(1, Math.hypot(a, b) / alpha1);
                p1[i] = a / scale;
                p2[i] = b / scale;
            }
            double[] e11 = ImageMath.gradX(w1bar, side);
            double[] e22 = ImageMath.gradY(w2bar, side);
            double[] e1y = ImageMath.gradY(w1bar, side);
            double[] e2x = ImageMath.gradX(w2bar, side);
            for (int i = 0; i < n; i++) {
                double a = q11[i] + step * e11[i];
                double b = q22[i] + step * e22[i];
                double c = q12[i] + step * 0.5 * (e1y[i] + e2x[i]);
                double scale = Math.max(1, Math.sqrt(a * a + b * b + 2 * c * c) / alpha0);
                q11[i] = a / scale;
                q22[i] = b / scale;
                q12[i] = c / scale;
            }
            double[] d = ImageMath.div(p1, p2, side);
            double[] v = new double[n];
            for (int i = 0; i < n; i++) {
                v[i] = x[i] + step * d[i];
            }
            double[] next = g.prox(v, step);
            // E^T q = (Dx^T q11 + Dy^T q12, Dy^T q22 + Dx^T q12), Dx^T = -div(., 0)
            double[] zero = new double[n];
            double[] t1 = ImageMath.axpy(ImageMath.div(q11, zero, side), 1, ImageMath.div(zero, q12, side));
            double[] t2 = ImageMath.axpy(ImageMath.div(zero, q22, side), 1, ImageMath.div(q12, zero, side));
            for (int i = 0; i < n; i++) {
                double nw1 = w1[i] + step * (p1[i] + t1[i]);
                double nw2 = w2[i] + step * (p2[i] + t2[i]);
                w1bar[i] = 2 * nw1 - w1[i];
                w2bar[i] = 2 * nw2 - w2[i];
                w1[i] = nw1;
                w2[i] = nw2;
                xbar[i] = 2 * next[i] - x[i];
            }
            x = next;
        }
        return x;
    }

    /**
     * Diagonally preconditioned Chambolle-Pock (Pock &amp; Chambolle, ICCV 2011) for
     * {@code min_x 1/2 |A x - y|^2 + lambda TV(x)} with an entrywise non-negative {@code A} (the
     * Radon transform, a blur, a mask): per-row dual steps {@code 1 / sum_j A_ij} and per-pixel
     * primal steps {@code 1 / (sum_i A_ij + 4)}, which converges far faster than the scalar steps
     * of {@link #linear} on badly conditioned operators such as sparse-view tomography.
     *
     * @param a the forward model, entrywise non-negative
     * @param y the measurements
     * @param side the image side
     * @param lambda the TV weight
     * @param nonNegative whether to project onto {@code x >= 0}
     * @param iterations the iterations
     * @param start the starting image, or {@code null} for zeros
     * @return the solution
     */
    public static double[] linearPreconditioned(LinearOperator a, double[] y, int side, double lambda,
            boolean nonNegative, int iterations, double[] start) {
        int n = side * side;
        double[] onesX = new double[n];
        java.util.Arrays.fill(onesX, 1);
        double[] onesY = new double[y.length];
        java.util.Arrays.fill(onesY, 1);
        double[] rowAbs = a.apply(onesX);
        double[] colAbs = a.adjoint(onesY);
        double[] sigmaQ = new double[y.length];
        for (int i = 0; i < sigmaQ.length; i++) {
            sigmaQ[i] = rowAbs[i] > 0 ? 1 / rowAbs[i] : 0;
        }
        double[] tau = new double[n];
        for (int j = 0; j < n; j++) {
            tau[j] = 1 / (colAbs[j] + 4);
        }
        double sigmaP = 0.5;
        double[] x = start == null ? new double[n] : start.clone();
        double[] xbar = x.clone();
        double[] q = new double[y.length];
        double[] px = new double[n];
        double[] py = new double[n];
        for (int it = 0; it < iterations; it++) {
            double[] ax = a.apply(xbar);
            for (int i = 0; i < q.length; i++) {
                q[i] = (q[i] + sigmaQ[i] * (ax[i] - y[i])) / (1 + sigmaQ[i]);
            }
            double[] gx = ImageMath.gradX(xbar, side);
            double[] gy = ImageMath.gradY(xbar, side);
            for (int i = 0; i < n; i++) {
                double u = px[i] + sigmaP * gx[i];
                double v = py[i] + sigmaP * gy[i];
                double scale = Math.max(1, Math.hypot(u, v) / lambda);
                px[i] = u / scale;
                py[i] = v / scale;
            }
            double[] atq = a.adjoint(q);
            double[] d = ImageMath.div(px, py, side);
            double[] next = new double[n];
            for (int i = 0; i < n; i++) {
                double v = x[i] - tau[i] * (atq[i] - d[i]);
                next[i] = nonNegative ? Math.max(0, v) : v;
                xbar[i] = 2 * next[i] - x[i];
            }
            x = next;
        }
        return x;
    }

    /**
     * Chambolle-Pock for {@code min_x 1/2 |A x - y|^2 + lambda TV(x)} (optionally with
     * {@code x >= 0}) for any linear operator, with {@code K = [A / s; grad]} after rescaling
     * {@code A} by {@code s = |A|} so both blocks have comparable norms.
     *
     * @param a the forward model
     * @param y the measurements
     * @param side the image side
     * @param lambda the TV weight
     * @param nonNegative whether to project onto {@code x >= 0}
     * @param iterations the iterations
     * @param start the starting image, or {@code null} for zeros
     * @return the solution
     */
    public static double[] linear(LinearOperator a, double[] y, int side, double lambda, boolean nonNegative,
            int iterations, double[] start) {
        int n = side * side;
        double s = Math.sqrt(a.normSquared(20));
        double lam = lambda / (s * s);
        double step = 0.99 / Math.sqrt(1 + 8);
        double[] ys = y.clone();
        for (int i = 0; i < ys.length; i++) {
            ys[i] /= s;
        }
        double[] x = start == null ? new double[n] : start.clone();
        double[] xbar = x.clone();
        double[] q = new double[y.length];
        double[] px = new double[n];
        double[] py = new double[n];
        for (int it = 0; it < iterations; it++) {
            double[] ax = a.apply(xbar);
            for (int i = 0; i < q.length; i++) {
                q[i] = (q[i] + step * (ax[i] / s - ys[i])) / (1 + step);
            }
            double[] gx = ImageMath.gradX(xbar, side);
            double[] gy = ImageMath.gradY(xbar, side);
            for (int i = 0; i < n; i++) {
                double u = px[i] + step * gx[i];
                double v = py[i] + step * gy[i];
                double scale = Math.max(1, Math.hypot(u, v) / lam);
                px[i] = u / scale;
                py[i] = v / scale;
            }
            double[] atq = a.adjoint(q);
            double[] d = ImageMath.div(px, py, side);
            double[] next = new double[n];
            for (int i = 0; i < n; i++) {
                double v = x[i] - step * (atq[i] / s - d[i]);
                next[i] = nonNegative ? Math.max(0, v) : v;
                xbar[i] = 2 * next[i] - x[i];
            }
            x = next;
        }
        return x;
    }
}
