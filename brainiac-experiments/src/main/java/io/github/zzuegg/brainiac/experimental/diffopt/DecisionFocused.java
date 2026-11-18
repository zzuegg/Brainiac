package io.github.zzuegg.brainiac.experimental.diffopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.combopt.BranchAndBound;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;
import io.github.zzuegg.brainiac.experimental.ml.LinearProgram;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Decision-focused (predict-then-optimise) losses for a predictor of the cost vector of a linear
 * program {@code min_{w in W} c . w}.
 *
 * <p>When predictions {@code c^} only choose decisions {@code w*(c^)}, what matters is the regret
 * {@code c . w*(c^) - c . w*(c)}, not the prediction error. The losses, all on autograd
 * {@link Var}s of predicted costs {@code [batch, d]} with the realised costs as targets:
 * <ul>
 *   <li>{@link Loss#TWO_STAGE_MSE}: plain least squares, the "predict, then optimise" baseline;</li>
 *   <li>{@link Loss#SPO_PLUS} (Elmachtoub and Grigas, Management Science 2022,
 *       <a href="https://arxiv.org/abs/1710.08005">arXiv:1710.08005</a>): the convex surrogate
 *       {@code -min_w (2 c^ - c) . w + 2 c^ . w*(c) - c . w*(c)}, with subgradient
 *       {@code 2 (w*(c) - w*(2 c^ - c))}: one extra solver call;</li>
 *   <li>{@link Loss#PG_BACKWARD} and {@link Loss#PG_CENTRAL} (Huang and Gupta, NeurIPS 2024,
 *       <a href="https://arxiv.org/abs/2402.03256">arXiv:2402.03256</a>): with the plug-in value
 *       {@code V(t) = min_w t . w}, Danskin's theorem gives {@code d/dh V(c^ + h c) = c . w*(c^ + h c)},
 *       the decision loss; the backward difference {@code (V(c^) - V(c^ - h c)) / h} and the central
 *       {@code (V(c^ + h c) - V(c^ - h c)) / 2h} approximate it with gradients
 *       {@code (w*(c^) - w*(c^ - h c)) / h} and {@code (w*(c^ + h c) - w*(c^ - h c)) / 2h};</li>
 *   <li>{@link Loss#PERTURBED_FY}: the perturbed Fenchel-Young loss of {@link PerturbedOptimizers}
 *       with the true decision as its target.</li>
 * </ul>
 * Problems: monotone shortest paths on a grid DAG (dynamic programming), multi-dimensional
 * knapsacks (the experimental {@link BranchAndBound}) and capped portfolios
 * ({@link LinearProgram}).
 */
public final class DecisionFocused {

    /** The training losses. */
    public enum Loss {
        /** Mean squared prediction error. */
        TWO_STAGE_MSE,
        /** The SPO+ surrogate. */
        SPO_PLUS,
        /** The backward-difference perturbation-gradient loss. */
        PG_BACKWARD,
        /** The central-difference perturbation-gradient loss. */
        PG_CENTRAL,
        /** The perturbed Fenchel-Young loss. */
        PERTURBED_FY
    }

    /** A linear minimisation oracle {@code argmin_{w in W} c . w}. */
    @FunctionalInterface
    public interface Problem {

        /**
         * Solves the problem.
         *
         * @param cost the cost vector
         * @return the minimiser
         */
        double[] solve(double[] cost);
    }

    private DecisionFocused() {
    }

    /**
     * A decision loss averaged over the batch.
     *
     * @param loss the loss
     * @param predicted predicted costs {@code [batch, d]}
     * @param costs realised costs {@code [batch][d]}
     * @param problem the oracle
     * @param h the PG step, or the Fenchel-Young noise scale
     * @param random the random source (Fenchel-Young only)
     * @return the scalar loss
     */
    public static Var loss(Loss loss, Var predicted, double[][] costs, Problem problem, double h,
            SplittableRandom random) {
        Objects.requireNonNull(loss, "loss");
        int batch = predicted.shape().dim(0);
        int d = predicted.shape().dim(1);
        float[] flat = new float[batch * d];
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < d; i++) {
                flat[b * d + i] = (float) costs[b][i];
            }
        }
        Tensor target = Tensor.of(Shape.of(batch, d), flat);
        if (loss == Loss.TWO_STAGE_MSE) {
            return predicted.sub(target).square().mean();
        }
        if (loss == Loss.PERTURBED_FY) {
            float[] decisions = new float[batch * d];
            for (int b = 0; b < batch; b++) {
                double[] w = problem.solve(costs[b]);
                for (int i = 0; i < d; i++) {
                    decisions[b * d + i] = (float) w[i];
                }
            }
            PerturbedOptimizers.Solver max = theta -> problem.solve(negate(theta));
            return PerturbedOptimizers.fenchelYoungLoss(predicted.neg(), Tensor.of(Shape.of(batch, d), decisions),
                    max, h, 8, PerturbedOptimizers.Noise.GAUSSIAN, random);
        }
        double[][] grads = new double[batch][];
        return Var.custom(in -> {
            double[] pv = in[0].toDoubleArray();
            double value = 0;
            for (int b = 0; b < batch; b++) {
                double[] chat = java.util.Arrays.copyOfRange(pv, b * d, (b + 1) * d);
                double[] c = costs[b];
                double[] g = new double[d];
                switch (loss) {
                    case SPO_PLUS -> {
                        double[] wc = problem.solve(c);
                        double[] t = new double[d];
                        for (int i = 0; i < d; i++) {
                            t[i] = 2 * chat[i] - c[i];
                        }
                        double[] wt = problem.solve(t);
                        value += -dot(t, wt) + 2 * dot(chat, wc) - dot(c, wc);
                        for (int i = 0; i < d; i++) {
                            g[i] = 2 * (wc[i] - wt[i]);
                        }
                    }
                    case PG_BACKWARD -> {
                        double[] lo = axpy(chat, -h, c);
                        double[] w0 = problem.solve(chat);
                        double[] w1 = problem.solve(lo);
                        value += (dot(chat, w0) - dot(lo, w1)) / h;
                        for (int i = 0; i < d; i++) {
                            g[i] = (w0[i] - w1[i]) / h;
                        }
                    }
                    case PG_CENTRAL -> {
                        double[] hi = axpy(chat, h, c);
                        double[] lo = axpy(chat, -h, c);
                        double[] w0 = problem.solve(hi);
                        double[] w1 = problem.solve(lo);
                        value += (dot(hi, w0) - dot(lo, w1)) / (2 * h);
                        for (int i = 0; i < d; i++) {
                            g[i] = (w0[i] - w1[i]) / (2 * h);
                        }
                    }
                    default -> throw new IllegalStateException(loss.name());
                }
                grads[b] = g;
            }
            return Tensor.of(Shape.of(1), (float) (value / batch));
        }, (inputs, output, grad) -> {
            double gs = grad.toDoubleArray()[0];
            float[] out = new float[batch * d];
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < d; i++) {
                    out[b * d + i] = (float) (gs * grads[b][i] / batch);
                }
            }
            return new Tensor[] {Tensor.of(Shape.of(batch, d), out)};
        }, predicted);
    }

    private static double[] negate(double[] x) {
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = -x[i];
        }
        return y;
    }

    private static double[] axpy(double[] x, double a, double[] y) {
        double[] z = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            z[i] = x[i] + a * y[i];
        }
        return z;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    /**
     * The normalised regret {@code sum (c . w*(c^) - c . w*(c)) / sum |c . w*(c)|} over a set.
     *
     * @param problem the oracle
     * @param predicted predicted costs {@code [n][d]}
     * @param costs realised costs {@code [n][d]}
     * @return the normalised regret
     */
    public static double normalisedRegret(Problem problem, double[][] predicted, double[][] costs) {
        double regret = 0;
        double optimum = 0;
        for (int i = 0; i < costs.length; i++) {
            double best = dot(costs[i], problem.solve(costs[i]));
            regret += dot(costs[i], problem.solve(predicted[i])) - best;
            optimum += Math.abs(best);
        }
        return regret / optimum;
    }

    /**
     * Monotone shortest paths from the top-left to the bottom-right node of a {@code g x g} grid
     * whose edges point right or down, solved exactly by dynamic programming. Edges are indexed as
     * in {@code DecisionInstances}: the {@code g (g - 1)} right edges row by row, then the down
     * edges.
     *
     * @param grid nodes per side
     * @return the oracle, whose output is the 0/1 edge indicator
     */
    public static Problem gridPaths(int grid) {
        int horizontal = grid * (grid - 1);
        return cost -> {
            double[][] dist = new double[grid][grid];
            boolean[][] fromLeft = new boolean[grid][grid];
            for (int r = 0; r < grid; r++) {
                for (int c = 0; c < grid; c++) {
                    if (r == 0 && c == 0) {
                        continue;
                    }
                    double viaLeft = c > 0 ? dist[r][c - 1] + cost[r * (grid - 1) + c - 1] : Double.POSITIVE_INFINITY;
                    double viaUp = r > 0 ? dist[r - 1][c] + cost[horizontal + (r - 1) * grid + c]
                            : Double.POSITIVE_INFINITY;
                    fromLeft[r][c] = viaLeft <= viaUp;
                    dist[r][c] = Math.min(viaLeft, viaUp);
                }
            }
            double[] w = new double[cost.length];
            int r = grid - 1;
            int c = grid - 1;
            while (r > 0 || c > 0) {
                if (fromLeft[r][c]) {
                    w[r * (grid - 1) + c - 1] = 1;
                    c--;
                } else {
                    w[horizontal + (r - 1) * grid + c] = 1;
                    r--;
                }
            }
            return w;
        };
    }

    /**
     * Multi-dimensional knapsacks: minimise {@code cost . w} (costs are negated values) over binary
     * {@code w} with {@code A w <= b}, by branch and bound.
     *
     * @param weights {@code A [dims][items]}
     * @param capacity {@code b}
     * @return the oracle
     */
    public static Problem knapsack(double[][] weights, double[] capacity) {
        BranchAndBound solver = BranchAndBound.of(BranchAndBound.Rule.MOST_FRACTIONAL);
        return cost -> {
            double[] value = negate(cost);
            boolean any = false;
            for (double v : value) {
                any |= v > 0;
            }
            double[] w = new double[cost.length];
            if (!any) {
                return w;
            }
            BranchAndBound.Result r = solver.solve(new CombinatorialInstances.Ilp("mkp", value, weights, capacity, 0,
                    Double.NaN));
            for (int i = 0; i < w.length; i++) {
                w[i] = r.x()[i];
            }
            return w;
        };
    }

    /**
     * Capped portfolios: minimise {@code cost . w} (costs are negated returns) with
     * {@code sum w = 1} and {@code 0 <= w <= cap}, by the simplex method.
     *
     * @param assets the number of assets
     * @param cap the per-asset cap, at least {@code 1 / assets}
     * @return the oracle
     */
    public static Problem portfolio(int assets, double cap) {
        double[][] a = new double[assets + 1][assets];
        double[] b = new double[assets + 1];
        for (int i = 0; i < assets; i++) {
            a[i][i] = 1;
            b[i] = cap;
            a[assets][i] = 1;
        }
        b[assets] = 1;
        return cost -> {
            // sum w >= 1 has a negative bound, which the simplex code rejects: shift every return
            // by a constant M so the budget binds at the optimum, then solve with sum w <= 1
            double shift = 0;
            for (double c : cost) {
                shift = Math.max(shift, c);
            }
            shift += 1;
            double[] obj = new double[assets];
            for (int i = 0; i < assets; i++) {
                obj[i] = shift - cost[i];
            }
            return LinearProgram.maximize(obj, a, b).x();
        };
    }
}
