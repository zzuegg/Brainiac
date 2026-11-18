package io.github.zzuegg.brainiac.experimental.diffopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.SplittableRandom;

/**
 * Learning through combinatorial solvers: perturbed optimisers, their Fenchel-Young losses and
 * blackbox differentiation, on autograd {@link Var}s.
 *
 * <p>A solver {@code y*(theta) = argmax_{y in C} theta . y} over a finite set {@code C} (paths,
 * permutations, matchings) is piecewise constant in {@code theta}. Two ways to train a network
 * that produces {@code theta}:
 * <ul>
 *   <li>Perturbed optimisers (Berthet et al. 2020, <a href="https://arxiv.org/abs/2002.08676">
 *       arXiv:2002.08676</a>): {@code y_s(theta) = E[y*(theta + s Z)]} is smooth, with the Monte
 *       Carlo Jacobian {@code E[y*(theta + s Z) grad nu(Z)^T] / s} ({@code nu = -log density}:
 *       {@code Z} for Gaussian noise, {@code 1 - exp(-Z)} for Gumbel). Its Fenchel-Young loss
 *       (Blondel, Martins and Niculae 2020, <a href="https://arxiv.org/abs/1901.02324">
 *       arXiv:1901.02324</a>) {@code L(theta; y) = F_s(theta) - theta . y + Omega(y)} is convex with
 *       gradient {@code y_s(theta) - y}, so supervising by the solution alone needs only solver
 *       calls;</li>
 *   <li>Blackbox differentiation (Vlastelica, Paulus et al. 2020, <a href="https://arxiv.org/abs/1912.02175">
 *       arXiv:1912.02175</a>): the forward pass is the exact solution; the backward pass calls the
 *       solver once more at {@code theta' = theta - lambda dL/dy} and returns
 *       {@code (y*(theta) - y*(theta')) / lambda}, the gradient of a piecewise-linear interpolation
 *       of the loss.</li>
 * </ul>
 * Solvers provided: the cheapest 8- or 4-connected path between opposite corners of a vertex-weighted
 * grid (Dijkstra; the maximisation is over {@code theta = -cost}), ranking (sort) and linear
 * assignment (core {@link Linalg#linearAssignment}). The sparsemax loss of the experimental
 * {@code Sparsemax} is the Fenchel-Young loss of the simplex with a quadratic regulariser.
 */
public final class PerturbedOptimizers {

    /** A linear maximisation oracle {@code argmax_{y in C} theta . y}. */
    @FunctionalInterface
    public interface Solver {

        /**
         * Solves the problem.
         *
         * @param theta the objective
         * @return the maximiser, as a vector of the same length
         */
        double[] solve(double[] theta);
    }

    /** The perturbation distribution. */
    public enum Noise {
        /** Standard normal. */
        GAUSSIAN,
        /** Standard Gumbel. */
        GUMBEL
    }

    private PerturbedOptimizers() {
    }

    private static double draw(Noise noise, SplittableRandom r) {
        if (noise == Noise.GAUSSIAN) {
            return r.nextGaussian();
        }
        double u = Math.min(1 - 1e-12, Math.max(1e-12, r.nextDouble()));
        return -Math.log(-Math.log(u));
    }

    private static double score(Noise noise, double z) {
        return noise == Noise.GAUSSIAN ? z : 1 - Math.exp(-z);
    }

    /**
     * The perturbed maximiser {@code E[y*(theta + s Z)]} by Monte Carlo, with the samples it used.
     *
     * @param solver the oracle
     * @param theta the objective
     * @param sigma the noise scale {@code s}
     * @param samples Monte Carlo samples
     * @param noise the noise
     * @param random the random source
     * @return the expectation
     */
    public static double[] perturbedArgmax(Solver solver, double[] theta, double sigma, int samples, Noise noise,
            SplittableRandom random) {
        int d = theta.length;
        double[] mean = new double[d];
        double[] t = new double[d];
        for (int m = 0; m < samples; m++) {
            for (int i = 0; i < d; i++) {
                t[i] = theta[i] + sigma * draw(noise, random);
            }
            double[] y = solver.solve(t);
            for (int i = 0; i < d; i++) {
                mean[i] += y[i] / samples;
            }
        }
        return mean;
    }

    /**
     * The perturbed maximiser as a differentiable layer.
     *
     * @param theta the objectives {@code [batch, d]}
     * @param solver the oracle
     * @param sigma the noise scale
     * @param samples Monte Carlo samples per row
     * @param noise the noise
     * @param random the random source
     * @return {@code y_s(theta)} as {@code [batch, d]}
     */
    public static Var perturbed(Var theta, Solver solver, double sigma, int samples, Noise noise,
            SplittableRandom random) {
        Objects.requireNonNull(solver, "solver");
        int batch = theta.shape().dim(0);
        int d = theta.shape().dim(1);
        double[][][] ys = new double[batch][samples][];
        double[][][] zs = new double[batch][samples][d];
        return Var.custom(in -> {
            double[] th = in[0].toDoubleArray();
            float[] out = new float[batch * d];
            double[] t = new double[d];
            for (int b = 0; b < batch; b++) {
                for (int m = 0; m < samples; m++) {
                    for (int i = 0; i < d; i++) {
                        zs[b][m][i] = draw(noise, random);
                        t[i] = th[b * d + i] + sigma * zs[b][m][i];
                    }
                    ys[b][m] = solver.solve(t);
                    for (int i = 0; i < d; i++) {
                        out[b * d + i] += (float) (ys[b][m][i] / samples);
                    }
                }
            }
            return Tensor.of(Shape.of(batch, d), out);
        }, (inputs, output, grad) -> {
            double[] g = grad.toDoubleArray();
            float[] gt = new float[batch * d];
            for (int b = 0; b < batch; b++) {
                for (int m = 0; m < samples; m++) {
                    double yg = 0;
                    for (int i = 0; i < d; i++) {
                        yg += ys[b][m][i] * g[b * d + i];
                    }
                    for (int i = 0; i < d; i++) {
                        gt[b * d + i] += (float) (yg * score(noise, zs[b][m][i]) / (sigma * samples));
                    }
                }
            }
            return new Tensor[] {Tensor.of(Shape.of(batch, d), gt)};
        }, theta);
    }

    /**
     * The perturbed Fenchel-Young loss, averaged over the batch. Its value is
     * {@code E[max_y (theta + s Z) . y] - theta . target} (the regulariser {@code Omega(target)} is
     * constant in {@code theta} and left out) and its gradient {@code y_s(theta) - target}.
     *
     * @param theta the objectives {@code [batch, d]}
     * @param target the target solutions {@code [batch, d]}
     * @param solver the oracle
     * @param sigma the noise scale
     * @param samples Monte Carlo samples per row
     * @param noise the noise
     * @param random the random source
     * @return the scalar loss
     */
    public static Var fenchelYoungLoss(Var theta, Tensor target, Solver solver, double sigma, int samples,
            Noise noise, SplittableRandom random) {
        int batch = theta.shape().dim(0);
        int d = theta.shape().dim(1);
        double[] tg = target.toDoubleArray();
        double[][] mean = new double[batch][];
        return Var.custom(in -> {
            double[] th = in[0].toDoubleArray();
            double value = 0;
            double[] t = new double[d];
            for (int b = 0; b < batch; b++) {
                mean[b] = new double[d];
                for (int m = 0; m < samples; m++) {
                    for (int i = 0; i < d; i++) {
                        t[i] = th[b * d + i] + sigma * draw(noise, random);
                    }
                    double[] y = solver.solve(t);
                    for (int i = 0; i < d; i++) {
                        mean[b][i] += y[i] / samples;
                        value += t[i] * y[i] / samples;
                    }
                }
                for (int i = 0; i < d; i++) {
                    value -= th[b * d + i] * tg[b * d + i];
                }
            }
            return Tensor.of(Shape.of(1), (float) (value / batch));
        }, (inputs, output, grad) -> {
            double g = grad.toDoubleArray()[0];
            float[] gt = new float[batch * d];
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < d; i++) {
                    gt[b * d + i] = (float) (g * (mean[b][i] - tg[b * d + i]) / batch);
                }
            }
            return new Tensor[] {Tensor.of(Shape.of(batch, d), gt)};
        }, theta);
    }

    /**
     * The solver as a layer differentiated by blackbox differentiation.
     *
     * @param theta the objectives {@code [batch, d]}
     * @param solver the oracle
     * @param lambda the interpolation strength {@code lambda}
     * @return {@code y*(theta)} as {@code [batch, d]}
     */
    public static Var blackbox(Var theta, Solver solver, double lambda) {
        int batch = theta.shape().dim(0);
        int d = theta.shape().dim(1);
        double[][] ys = new double[batch][];
        return Var.custom(in -> {
            double[] th = in[0].toDoubleArray();
            float[] out = new float[batch * d];
            for (int b = 0; b < batch; b++) {
                ys[b] = solver.solve(Arrays.copyOfRange(th, b * d, (b + 1) * d));
                for (int i = 0; i < d; i++) {
                    out[b * d + i] = (float) ys[b][i];
                }
            }
            return Tensor.of(Shape.of(batch, d), out);
        }, (inputs, output, grad) -> {
            double[] th = inputs[0].toDoubleArray();
            double[] g = grad.toDoubleArray();
            float[] gt = new float[batch * d];
            double[] shifted = new double[d];
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < d; i++) {
                    shifted[i] = th[b * d + i] - lambda * g[b * d + i];
                }
                double[] y2 = solver.solve(shifted);
                for (int i = 0; i < d; i++) {
                    gt[b * d + i] = (float) ((ys[b][i] - y2[i]) / lambda);
                }
            }
            return new Tensor[] {Tensor.of(Shape.of(batch, d), gt)};
        }, theta);
    }

    /**
     * The cheapest path between the top-left and bottom-right cells of a vertex-weighted grid,
     * as a maximiser over {@code theta = -cost} (costs are clamped below at {@code minCost} so
     * perturbed objectives stay valid for Dijkstra).
     *
     * @param side cells per side
     * @param eightConnected diagonal moves allowed
     * @param minCost the smallest admissible cell cost, for example 1e-3
     * @return the solver, whose output is the 0/1 indicator of the path's cells
     */
    public static Solver gridShortestPath(int side, boolean eightConnected, double minCost) {
        return theta -> {
            double[] cost = new double[theta.length];
            for (int i = 0; i < cost.length; i++) {
                cost[i] = Math.max(minCost, -theta[i]);
            }
            return shortestPathCells(cost, side, eightConnected);
        };
    }

    /**
     * Dijkstra on a vertex-weighted grid from cell 0 to the last cell.
     *
     * @param cost non-negative cell costs, row-major
     * @param side cells per side
     * @param eightConnected diagonal moves allowed
     * @return the 0/1 indicator of the cheapest path's cells (both ends included)
     */
    public static double[] shortestPathCells(double[] cost, int side, boolean eightConnected) {
        int n = side * side;
        double[] dist = new double[n];
        int[] prev = new int[n];
        Arrays.fill(dist, Double.POSITIVE_INFINITY);
        Arrays.fill(prev, -1);
        dist[0] = cost[0];
        PriorityQueue<double[]> queue = new PriorityQueue<>((a, b) -> Double.compare(a[0], b[0]));
        queue.add(new double[] {dist[0], 0});
        boolean[] done = new boolean[n];
        while (!queue.isEmpty()) {
            double[] top = queue.poll();
            int u = (int) top[1];
            if (done[u]) {
                continue;
            }
            done[u] = true;
            if (u == n - 1) {
                break;
            }
            int r = u / side;
            int c = u % side;
            for (int dr = -1; dr <= 1; dr++) {
                for (int dc = -1; dc <= 1; dc++) {
                    if (dr == 0 && dc == 0 || !eightConnected && dr != 0 && dc != 0) {
                        continue;
                    }
                    int rr = r + dr;
                    int cc = c + dc;
                    if (rr < 0 || cc < 0 || rr >= side || cc >= side) {
                        continue;
                    }
                    int v = rr * side + cc;
                    double nd = dist[u] + cost[v];
                    if (nd < dist[v]) {
                        dist[v] = nd;
                        prev[v] = u;
                        queue.add(new double[] {nd, v});
                    }
                }
            }
        }
        double[] y = new double[n];
        for (int v = n - 1; v >= 0; v = prev[v]) {
            y[v] = 1;
            if (v == 0) {
                break;
            }
        }
        return y;
    }

    /**
     * Ranking: the maximiser of {@code theta . y} over permutations of {@code (1, ..., n)}, which
     * gives rank {@code n} to the largest score.
     *
     * @return the solver
     */
    public static Solver ranking() {
        return theta -> {
            int[] order = SoftSorting.argsortDescending(theta);
            double[] y = new double[theta.length];
            for (int i = 0; i < order.length; i++) {
                y[order[i]] = theta.length - i;
            }
            return y;
        };
    }

    /**
     * Linear assignment: {@code theta} is a flattened {@code n x n} score matrix and the output the
     * flattened permutation matrix of maximal total score (core Hungarian algorithm).
     *
     * @param n the size
     * @return the solver
     */
    public static Solver assignment(int n) {
        return theta -> {
            float[] cost = new float[n * n];
            for (int i = 0; i < n * n; i++) {
                cost[i] = (float) -theta[i];
            }
            int[] match = Linalg.linearAssignment(Tensor.of(Shape.of(n, n), cost));
            double[] y = new double[n * n];
            for (int i = 0; i < n; i++) {
                y[i * n + match[i]] = 1;
            }
            return y;
        };
    }
}
