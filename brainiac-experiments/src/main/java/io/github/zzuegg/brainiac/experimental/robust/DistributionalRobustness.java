package io.github.zzuegg.brainiac.experimental.robust;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Lbfgs;

import java.util.Arrays;
import java.util.List;

/**
 * Distributionally robust optimisation without group labels: minimise the worst-case expected loss
 * over distributions near the training distribution.
 * <ul>
 *   <li>f-divergence balls around the empirical distribution, as minibatch reweightings with
 *       closed-form (or one-dimensional) duals, following the large-scale estimators of Levy,
 *       Carmon, Duchi and Sidford (NeurIPS 2020, <a href="https://arxiv.org/abs/2010.05893">arXiv:2010.05893</a>)
 *       and Duchi and Namkoong (Annals of Statistics 2021,
 *       <a href="https://arxiv.org/abs/1810.08750">arXiv:1810.08750</a>):
 *       <ul>
 *         <li>{@link Kind#CVAR}: the average of the worst {@code alpha} share of losses (the
 *             conditional value at risk; weight {@code 1 / (alpha n)} on the top losses);</li>
 *         <li>{@link Kind#CHI_SQUARE}: the {@code chi^2}-penalised worst case
 *             {@code max_p sum p_i l_i - lambda D_chi2(p || uniform)}, whose maximiser is
 *             {@code p_i = (1 + (l_i - eta) / lambda)_+ / n} with {@code eta} set by bisection so
 *             the weights sum to one;</li>
 *         <li>{@link Kind#KL}: the KL-penalised worst case, {@code p_i ~ exp(l_i / tau)}, a soft
 *             maximum of the losses.</li>
 *       </ul>
 *       The weights are computed from the batch losses and held fixed (detached) for the gradient,
 *       which is the gradient of the robust objective by Danskin's theorem.</li>
 *   <li>{@link #wassersteinLogistic}: Wasserstein DRO for logistic regression against covariate
 *       perturbations under an L2 transport cost (Shafieezadeh-Abadeh, Mohajerin Esfahani and Kuhn,
 *       NeurIPS 2015): with labels not allowed to move, the worst case over a ball of radius
 *       {@code epsilon} equals the empirical logistic loss plus {@code epsilon ||w||_2}, a convex
 *       program solved here by L-BFGS ({@link Lbfgs}).</li>
 * </ul>
 */
public final class DistributionalRobustness {

    /** The uncertainty set. */
    public enum Kind {
        /** Plain empirical risk (uniform weights). */
        ERM,
        /** Conditional value at risk at level {@code alpha}. */
        CVAR,
        /** Chi-squared-penalised worst case with penalty {@code lambda}. */
        CHI_SQUARE,
        /** KL-penalised worst case with temperature {@code tau}. */
        KL
    }

    private DistributionalRobustness() {
    }

    /**
     * The worst-case weights of a batch.
     *
     * @param kind the uncertainty set
     * @param losses the per-example losses
     * @param parameter {@code alpha} (CVaR, in (0, 1]), {@code lambda} (chi-squared, positive) or
     *     {@code tau} (KL, positive); ignored for ERM
     * @return weights summing to one
     */
    public static double[] weights(Kind kind, double[] losses, double parameter) {
        int n = losses.length;
        double[] w = new double[n];
        switch (kind) {
            case ERM -> Arrays.fill(w, 1.0 / n);
            case CVAR -> {
                Integer[] order = new Integer[n];
                for (int i = 0; i < n; i++) {
                    order[i] = i;
                }
                Arrays.sort(order, (a, b) -> Double.compare(losses[b], losses[a]));
                double mass = parameter * n;
                for (int k = 0; k < n && mass > 0; k++) {
                    double take = Math.min(1, mass);
                    w[order[k]] = take / (parameter * n);
                    mass -= take;
                }
            }
            case CHI_SQUARE -> {
                double lo = Arrays.stream(losses).min().orElse(0) - parameter;
                double hi = Arrays.stream(losses).max().orElse(0) + parameter;
                for (int it = 0; it < 100; it++) {
                    double eta = 0.5 * (lo + hi);
                    double s = 0;
                    for (double l : losses) {
                        s += Math.max(0, 1 + (l - eta) / parameter) / n;
                    }
                    if (s > 1) {
                        lo = eta;
                    } else {
                        hi = eta;
                    }
                }
                double eta = 0.5 * (lo + hi);
                double s = 0;
                for (int i = 0; i < n; i++) {
                    w[i] = Math.max(0, 1 + (losses[i] - eta) / parameter) / n;
                    s += w[i];
                }
                for (int i = 0; i < n; i++) {
                    w[i] /= s;
                }
            }
            case KL -> {
                double max = Arrays.stream(losses).max().orElse(0);
                double s = 0;
                for (int i = 0; i < n; i++) {
                    w[i] = Math.exp((losses[i] - max) / parameter);
                    s += w[i];
                }
                for (int i = 0; i < n; i++) {
                    w[i] /= s;
                }
            }
            default -> throw new IllegalArgumentException(kind.toString());
        }
        return w;
    }

    /**
     * The robust loss of a batch: per-example losses weighted by the (detached) worst-case weights.
     *
     * @param kind the uncertainty set
     * @param perExample per-example losses {@code [b]}
     * @param parameter the set's parameter (see {@link #weights})
     * @return the scalar robust loss
     */
    public static Var loss(Kind kind, Var perExample, double parameter) {
        float[] l = perExample.value().toArray();
        double[] d = new double[l.length];
        for (int i = 0; i < l.length; i++) {
            d[i] = l[i];
        }
        double[] w = weights(kind, d, parameter);
        float[] wf = new float[w.length];
        for (int i = 0; i < w.length; i++) {
            wf[i] = (float) w[i];
        }
        return perExample.reshape(l.length).mul(Tensor.of(Shape.of(l.length), wf)).sum();
    }

    /**
     * A linear classifier {@code sign(w . x + b)}.
     *
     * @param w the weights
     * @param b the bias
     */
    public record Linear(double[] w, double b) {

        /**
         * @param x a row
         * @return the margin {@code w . x + b}
         */
        public double margin(double[] x) {
            double s = b;
            for (int j = 0; j < w.length; j++) {
                s += w[j] * x[j];
            }
            return s;
        }

        /** @return {@code ||w||_2} */
        public double norm() {
            double s = 0;
            for (double v : w) {
                s += v * v;
            }
            return Math.sqrt(s);
        }

        /**
         * Accuracy under the worst L2 perturbation of size {@code epsilon} per row (for a linear
         * model the attack moves each row by {@code epsilon} against the normal, so the margin drops
         * by {@code epsilon ||w||}).
         *
         * @param x rows
         * @param y labels in {0, 1}
         * @param epsilon the perturbation radius
         * @return the robust accuracy
         */
        public double robustAccuracy(double[][] x, int[] y, double epsilon) {
            double norm = norm();
            int right = 0;
            for (int i = 0; i < x.length; i++) {
                double s = (2 * y[i] - 1) * margin(x[i]);
                right += s - epsilon * norm > 0 ? 1 : 0;
            }
            return right / (double) x.length;
        }
    }

    /**
     * Wasserstein-DRO logistic regression: minimises
     * {@code mean log(1 + exp(-y_i (w . x_i + b))) + epsilon ||w||_2 + l2 ||w||^2}.
     *
     * @param x rows
     * @param y labels in {0, 1}
     * @param epsilon the Wasserstein radius (0 gives plain logistic regression)
     * @param l2 an optional ridge term, such as 1e-4
     * @param iterations L-BFGS iterations
     * @return the classifier
     */
    public static Linear wassersteinLogistic(double[][] x, int[] y, double epsilon, double l2, int iterations) {
        int n = x.length;
        int d = x[0].length;
        float[] xv = new float[n * d];
        float[] sv = new float[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                xv[i * d + j] = (float) x[i][j];
            }
            sv[i] = 2 * y[i] - 1;
        }
        Tensor xt = Tensor.of(Shape.of(n, d), xv);
        Tensor st = Tensor.of(Shape.of(n, 1), sv);
        Var w = Var.param(Tensor.zeros(Shape.of(d, 1)));
        Var b = Var.param(Tensor.zeros(Shape.of(1)));
        new Lbfgs(10, 1e-7).minimize(List.of(w, b), () -> {
            Var margin = Var.of(xt).matmul(w).add(b).mul(st);
            Var loss = margin.neg().softplus().mean();
            Var penalty = w.square().sum().add(1e-8f).sqrt().mul((float) epsilon);
            return loss.add(penalty).add(w.square().sum().mul((float) l2));
        }, iterations);
        float[] wv = w.value().toArray();
        double[] out = new double[d];
        for (int j = 0; j < d; j++) {
            out[j] = wv[j];
        }
        return new Linear(out, b.value().toArray()[0]);
    }
}
