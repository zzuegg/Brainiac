package io.github.zzuegg.brainiac.experimental.multitask;

/**
 * Multi-objective descent directions from per-task gradients ({@link TaskGradients#of}).
 *
 * <ul>
 *   <li>{@link #mgda} (Desideri 2012; Sener and Koltun, NeurIPS 2018): the min-norm point of
 *       the convex hull of the task gradients, found by Frank-Wolfe on their Gram matrix. It is
 *       a common descent direction, zero exactly at Pareto-stationary points. The "upper
 *       bound" variant (MGDA-UB) is this applied to gradients of the shared representation
 *       instead of the shared weights; callers choose which gradients to pass.</li>
 *   <li>{@link #caGrad} (Liu et al., NeurIPS 2021): conflict-averse descent maximises the worst
 *       task's improvement inside a ball of radius {@code c |g0|} around the average gradient
 *       {@code g0}. Its dual is a minimisation over the simplex,
 *       {@code min_w g_w . g0 + c |g0| |g_w|}, solved here by exponentiated gradient; the
 *       update is {@code (g0 + c |g0| g_w / |g_w|) / (1 + c^2)}.</li>
 *   <li>{@link #nashMtl} (Navon et al., ICML 2022): the Nash bargaining solution
 *       {@code d = sum_i alpha_i g_i} with {@code (G G^T alpha)_i = 1 / alpha_i}, which is
 *       invariant to the scale of each task's loss. That condition is the stationarity of the
 *       convex {@code 1/2 alpha^T K alpha - sum log alpha_i}, minimised here by damped Newton
 *       steps (the paper uses a sequence of convex programs).</li>
 * </ul>
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1810.04650">arXiv:1810.04650</a>,
 * <a href="https://arxiv.org/abs/2110.14048">arXiv:2110.14048</a>,
 * <a href="https://arxiv.org/abs/2202.01017">arXiv:2202.01017</a>.
 */
public final class ParetoDescent {

    private ParetoDescent() {
    }

    /**
     * MGDA's min-norm weights on the simplex, by Frank-Wolfe.
     *
     * @param g per-task gradients
     * @return the weights, summing to one
     */
    public static double[] mgdaWeights(double[][] g) {
        double[][] k = TaskGradients.gram(g);
        int t = g.length;
        double[] w = new double[t];
        java.util.Arrays.fill(w, 1.0 / t);
        for (int it = 0; it < 100; it++) {
            double[] kw = multiply(k, w);
            int best = 0;
            for (int i = 1; i < t; i++) {
                best = kw[i] < kw[best] ? i : best;
            }
            // line search between w and the vertex e_best for min |sum w g|^2
            double wkw = dot(w, kw);
            double ekw = kw[best];
            double eke = k[best][best];
            double denom = wkw - 2 * ekw + eke;
            double gamma = denom <= 1e-20 ? 0 : Math.max(0, Math.min(1, (wkw - ekw) / denom));
            if (gamma < 1e-9) {
                break;
            }
            for (int i = 0; i < t; i++) {
                w[i] = (1 - gamma) * w[i] + (i == best ? gamma : 0);
            }
        }
        return w;
    }

    /**
     * MGDA's direction, scaled by the number of tasks so that it is comparable to a sum.
     *
     * @param g per-task gradients
     * @return {@code T sum_i w_i g_i}
     */
    public static double[] mgda(double[][] g) {
        double[] d = TaskGradients.combine(g, mgdaWeights(g));
        for (int i = 0; i < d.length; i++) {
            d[i] *= g.length;
        }
        return d;
    }

    /**
     * CAGrad's direction, scaled by the number of tasks.
     *
     * @param g per-task gradients
     * @param c the ball radius factor, 0.4 to 0.5 in the paper
     * @return {@code T (g0 + c |g0| g_w / |g_w|) / (1 + c^2)}
     */
    public static double[] caGrad(double[][] g, double c) {
        int t = g.length;
        double[][] k = TaskGradients.gram(g);
        double[] uniform = new double[t];
        java.util.Arrays.fill(uniform, 1.0 / t);
        double g0norm = Math.sqrt(Math.max(0, dot(uniform, multiply(k, uniform))));
        double radius = c * g0norm;
        double[] kb = multiply(k, uniform);
        double[] w = uniform.clone();
        double scale = 0;
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < t; j++) {
                scale = Math.max(scale, Math.abs(k[i][j]));
            }
        }
        double lr = scale > 0 ? 1.0 / scale : 1;
        for (int it = 0; it < 200; it++) {
            double[] kw = multiply(k, w);
            double gw = Math.sqrt(Math.max(1e-20, dot(w, kw)));
            double[] grad = new double[t];
            for (int i = 0; i < t; i++) {
                grad[i] = kb[i] + radius * kw[i] / gw;
            }
            double sum = 0;
            for (int i = 0; i < t; i++) {
                w[i] *= Math.exp(-lr * grad[i] * 2);
                sum += w[i];
            }
            for (int i = 0; i < t; i++) {
                w[i] /= sum;
            }
        }
        double[] gw = TaskGradients.combine(g, w);
        double[] g0 = TaskGradients.combine(g, uniform);
        double ngw = TaskGradients.norm(gw);
        double lambda = ngw > 0 ? radius / ngw : 0;
        double[] d = new double[g0.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = t * (g0[i] + lambda * gw[i]) / (1 + c * c);
        }
        return d;
    }

    /**
     * Nash-MTL's bargaining weights: the positive solution of {@code alpha_i (K alpha)_i = 1}.
     *
     * @param g per-task gradients
     * @return the weights {@code alpha} (not normalised: their scale sets the step size)
     */
    public static double[] nashWeights(double[][] g) {
        int t = g.length;
        double[][] k = TaskGradients.gram(g);
        double trace = 0;
        for (int i = 0; i < t; i++) {
            trace += k[i][i];
        }
        // scale K to unit trace for conditioning; alpha scales back by 1/sqrt(s)
        double s = trace > 0 ? trace / t : 1;
        double[][] ks = new double[t][t];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < t; j++) {
                ks[i][j] = k[i][j] / s + (i == j ? 1e-6 : 0);
            }
        }
        double[] a = new double[t];
        java.util.Arrays.fill(a, 1);
        for (int it = 0; it < 50; it++) {
            double[] ka = multiply(ks, a);
            double[] grad = new double[t];
            double[][] h = new double[t][t];
            for (int i = 0; i < t; i++) {
                grad[i] = ka[i] - 1 / a[i];
                for (int j = 0; j < t; j++) {
                    h[i][j] = ks[i][j] + (i == j ? 1 / (a[i] * a[i]) : 0);
                }
            }
            double[] step = solve(h, grad);
            double eta = 1;
            while (eta > 1e-8) {
                boolean positive = true;
                for (int i = 0; i < t; i++) {
                    positive &= a[i] - eta * step[i] > 0;
                }
                if (positive && objective(ks, a, step, eta) <= objective(ks, a, step, 0)) {
                    break;
                }
                eta /= 2;
            }
            double change = 0;
            for (int i = 0; i < t; i++) {
                a[i] -= eta * step[i];
                change = Math.max(change, Math.abs(eta * step[i]));
            }
            if (change < 1e-10) {
                break;
            }
        }
        double back = 1 / Math.sqrt(s);
        for (int i = 0; i < t; i++) {
            a[i] *= back;
        }
        return a;
    }

    /**
     * Nash-MTL's direction {@code sum_i alpha_i g_i}, rescaled to the norm of the plain sum so
     * the optimiser's step size means the same as for the other methods.
     *
     * @param g per-task gradients
     * @return the direction
     */
    public static double[] nashMtl(double[][] g) {
        double[] d = TaskGradients.combine(g, nashWeights(g));
        double[] sum = TaskGradients.combine(g, ones(g.length));
        double nd = TaskGradients.norm(d);
        double ns = TaskGradients.norm(sum);
        if (nd > 0) {
            for (int i = 0; i < d.length; i++) {
                d[i] *= ns / nd;
            }
        }
        return d;
    }

    private static double objective(double[][] k, double[] a, double[] step, double eta) {
        int t = a.length;
        double[] x = new double[t];
        for (int i = 0; i < t; i++) {
            x[i] = a[i] - eta * step[i];
            if (x[i] <= 0) {
                return Double.POSITIVE_INFINITY;
            }
        }
        double f = 0.5 * dot(x, multiply(k, x));
        for (double v : x) {
            f -= Math.log(v);
        }
        return f;
    }

    private static double[] ones(int t) {
        double[] w = new double[t];
        java.util.Arrays.fill(w, 1);
        return w;
    }

    private static double[] multiply(double[][] k, double[] w) {
        double[] out = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w.length; j++) {
                out[i] += k[i][j] * w[j];
            }
        }
        return out;
    }

    private static double dot(double[] a, double[] b) {
        return TaskGradients.dot(a, b);
    }

    /** Solves {@code A x = b} by Gaussian elimination with partial pivoting. */
    private static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[][] m = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, m[i], 0, n);
            m[i][n] = b[i];
        }
        for (int col = 0; col < n; col++) {
            int pivot = col;
            for (int r = col + 1; r < n; r++) {
                pivot = Math.abs(m[r][col]) > Math.abs(m[pivot][col]) ? r : pivot;
            }
            double[] tmp = m[col];
            m[col] = m[pivot];
            m[pivot] = tmp;
            double p = Math.abs(m[col][col]) < 1e-300 ? 1e-300 : m[col][col];
            for (int r = 0; r < n; r++) {
                if (r == col) {
                    continue;
                }
                double f = m[r][col] / p;
                for (int c = col; c <= n; c++) {
                    m[r][c] -= f * m[col][c];
                }
            }
        }
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = m[i][n] / (Math.abs(m[i][i]) < 1e-300 ? 1e-300 : m[i][i]);
        }
        return x;
    }
}
