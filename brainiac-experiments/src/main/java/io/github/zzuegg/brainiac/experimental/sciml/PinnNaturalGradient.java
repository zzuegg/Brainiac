package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Energy natural gradient descent for PINNs (ENGD, Müller and Zeinhofer, ICML 2023,
 * <a href="https://arxiv.org/abs/2302.13163">arXiv:2302.13163</a>). The PINN loss is a sum of
 * squared residuals {@code L = |r(theta)|^2} of the PDE operator and the boundary conditions,
 * each term scaled by one over the square root of its point count. Its natural gradient in
 * function space, pulled back to the parameters, is preconditioned by the Gram matrix of the
 * operator applied to the network's tangent space; for the quadratic energy this Gram matrix
 * is exactly the Gauss-Newton matrix {@code G = J^T J} of the residual Jacobian {@code J}.
 * Each iteration forms {@code J} (one vector-Jacobian product per residual, so for networks
 * of up to a couple of thousand parameters), solves {@code (G + mu I) d = J^T r} by Cholesky
 * and takes the best step {@code theta - eta d} over a grid line search
 * {@code eta = 2^-k}. On small networks it reaches errors orders of magnitude below Adam.
 *
 * <p>{@link Mode#LEVENBERG_MARQUARDT} replaces the line search with Marquardt's adaptive
 * damping {@code lambda diag(G)} (accept and shrink, or reject and grow), a baseline built here
 * because the generic {@code LevenbergMarquardt} evaluates its trial step without a graph, and
 * a PINN residual needs one for its input derivatives.
 */
public final class PinnNaturalGradient {

    /** The step rule. */
    public enum Mode {
        /** Gauss-Newton direction with fixed small damping and a grid line search (ENGD). */
        ENGD,
        /** Gauss-Newton with Marquardt's adaptive diagonal damping. */
        LEVENBERG_MARQUARDT
    }

    private final Mode mode;
    private double damping;
    private int lineSearch = 14;

    private PinnNaturalGradient(Mode mode, double damping) {
        this.mode = mode;
        this.damping = damping;
    }

    /**
     * ENGD.
     *
     * @param damping the Tikhonov damping {@code mu}, for example 1e-8 to 1e-4
     * @return the optimiser
     */
    public static PinnNaturalGradient engd(double damping) {
        return new PinnNaturalGradient(Mode.ENGD, damping);
    }

    /**
     * Levenberg-Marquardt on the same residuals.
     *
     * @param initialLambda the starting damping, for example 1e-2
     * @return the optimiser
     */
    public static PinnNaturalGradient levenbergMarquardt(double initialLambda) {
        return new PinnNaturalGradient(Mode.LEVENBERG_MARQUARDT, initialLambda);
    }

    /**
     * @param steps the number of halvings tried in the line search
     * @return this optimiser
     */
    public PinnNaturalGradient lineSearch(int steps) {
        this.lineSearch = steps;
        return this;
    }

    /** @return the step rule */
    public Mode mode() {
        return mode;
    }

    /**
     * The outcome of a second-order run.
     *
     * @param loss the final loss
     * @param millis the wall time
     * @param losses the loss after every iteration
     * @param monitor the monitored value (for example the error against a reference) after
     *     every iteration, or empty
     * @param times the elapsed milliseconds after every iteration
     */
    public record Result(double loss, double millis, double[] losses, double[] monitor, double[] times) {
    }

    /**
     * The scaled residual vector of a problem: the PDE residual at the collocation points and
     * every condition term, each divided by the square root of its size, stacked into one
     * column, so its squared norm is the sum of the terms' mean squares (the unweighted PINN
     * loss).
     *
     * @param pinn the network
     * @param problem the problem
     * @param points the collocation points {@code [n, d]}
     * @return {@code [m, 1]}
     */
    public static Var residuals(PhysicsInformedNetwork pinn, PinnTraining.Problem problem, Tensor points) {
        List<Var> parts = new ArrayList<>();
        parts.add(column(pinn.residual(points, problem.residual())));
        for (PinnTraining.Term t : problem.conditions()) {
            parts.add(column(t.errors().apply(pinn)));
        }
        return Var.concat(0, parts.toArray(new Var[0]));
    }

    private static Var column(Var e) {
        long n = e.shape().size();
        return e.reshape((int) n, 1).mul((float) (1 / Math.sqrt(n)));
    }

    static double squaredNorm(Var r) {
        double s = 0;
        for (float v : r.value().toArray()) {
            s += (double) v * v;
        }
        return s;
    }

    /**
     * Trains a PINN.
     *
     * @param pinn the network, trained in place
     * @param problem the problem
     * @param collocation collocation points (drawn once, Latin hypercube)
     * @param iterations iterations
     * @param monitor evaluated after every iteration, or {@code null}
     * @param seed the seed
     * @return the outcome
     */
    public Result train(PhysicsInformedNetwork pinn, PinnTraining.Problem problem, int collocation, int iterations,
            ToDoubleFunction<PhysicsInformedNetwork> monitor, long seed) {
        long start = System.nanoTime();
        Tensor points = CollocationSampler.tensor(problem.domain().latinHypercube(collocation,
                new SplittableRandom(seed)));
        List<Var> params = pinn.parameters();
        double[] losses = new double[iterations];
        double[] mon = new double[monitor == null ? 0 : iterations];
        double[] times = new double[iterations];
        double loss = Double.NaN;
        for (int it = 0; it < iterations; it++) {
            Var r = residuals(pinn, problem, points);
            float[] rv = r.value().toArray();
            double l0 = squaredNorm(r);
            double[][] j = CurvatureProducts.jacobian(r, params);
            int p = j[0].length;
            double[][] g = new double[p][p];
            double[] grad = new double[p];
            for (int k = 0; k < rv.length; k++) {
                double[] row = j[k];
                for (int a = 0; a < p; a++) {
                    double ra = row[a];
                    if (ra == 0) {
                        continue;
                    }
                    grad[a] += ra * rv[k];
                    for (int b = a; b < p; b++) {
                        g[a][b] += ra * row[b];
                    }
                }
            }
            for (int a = 0; a < p; a++) {
                for (int b = 0; b < a; b++) {
                    g[a][b] = g[b][a];
                }
            }
            double[] theta = CurvatureProducts.values(params);
            loss = mode == Mode.ENGD ? engdStep(pinn, problem, points, params, theta, g, grad, l0)
                    : marquardtStep(pinn, problem, points, params, theta, g, grad, l0);
            losses[it] = loss;
            if (monitor != null) {
                mon[it] = monitor.applyAsDouble(pinn);
            }
            times[it] = (System.nanoTime() - start) / 1e6;
            if (!Double.isFinite(loss)) {
                break;
            }
        }
        return new Result(loss, (System.nanoTime() - start) / 1e6, losses, mon, times);
    }

    private double lossAt(PhysicsInformedNetwork pinn, PinnTraining.Problem problem, Tensor points, List<Var> params,
            double[] theta) {
        CurvatureProducts.assign(params, theta);
        return squaredNorm(residuals(pinn, problem, points));
    }

    private double engdStep(PhysicsInformedNetwork pinn, PinnTraining.Problem problem, Tensor points,
            List<Var> params, double[] theta, double[][] g, double[] grad, double l0) {
        int p = theta.length;
        double trace = 0;
        for (int a = 0; a < p; a++) {
            trace += g[a][a];
        }
        double mu = damping * Math.max(1e-12, trace / p);
        double[][] sys = new double[p][];
        for (int a = 0; a < p; a++) {
            sys[a] = g[a].clone();
            sys[a][a] += mu + 1e-14;
        }
        double[] d = CurvatureProducts.solveSpd(sys, grad);
        double best = l0;
        double bestEta = 0;
        double eta = 1;
        for (int k = 0; k < lineSearch; k++) {
            double l = lossAt(pinn, problem, points, params, CurvatureProducts.axpy(theta, -eta, d));
            if (Double.isFinite(l) && l < best) {
                best = l;
                bestEta = eta;
            }
            eta /= 2;
        }
        CurvatureProducts.assign(params, CurvatureProducts.axpy(theta, -bestEta, d));
        return best;
    }

    private double marquardtStep(PhysicsInformedNetwork pinn, PinnTraining.Problem problem, Tensor points,
            List<Var> params, double[] theta, double[][] g, double[] grad, double l0) {
        int p = theta.length;
        for (int attempt = 0; attempt < 12; attempt++) {
            double[][] sys = new double[p][];
            for (int a = 0; a < p; a++) {
                sys[a] = g[a].clone();
                sys[a][a] += damping * Math.max(g[a][a], 1e-9) + 1e-12;
            }
            double[] d = CurvatureProducts.solveSpd(sys, grad);
            double l = lossAt(pinn, problem, points, params, CurvatureProducts.axpy(theta, -1, d));
            if (Double.isFinite(l) && l < l0) {
                damping = Math.max(1e-9, damping / 3);
                return l;
            }
            damping = Math.min(1e9, damping * 3);
        }
        CurvatureProducts.assign(params, theta);
        return l0;
    }

    @Override
    public String toString() {
        return "PinnNaturalGradient(" + mode + ", damping " + damping + ")";
    }
}
