package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Iterator;
import java.util.List;

/**
 * L-BFGS (Liu and Nocedal 1989, <a href="https://link.springer.com/article/10.1007/BF01589116">doi</a>):
 * a quasi-Newton method whose inverse-Hessian approximation is implicit in the last
 * {@code m} pairs {@code s = delta theta, y = delta grad}, applied by the two-loop recursion.
 *
 * <ul>
 *   <li>{@link #fullBatch}: the classic deterministic method with a strong-Wolfe line search
 *       (Nocedal and Wright, Algorithms 3.5 and 3.6) through {@link Optimizer.Reevaluation},
 *       for small smooth full-batch fits (neural fields, PINN-like problems).</li>
 *   <li>{@link #online} (oLBFGS, Schraudolph, Yu and Guenter 2007,
 *       <a href="https://proceedings.mlr.press/v2/schraudolph07a.html">AISTATS</a>): a fixed
 *       step size, and {@code y} measured on the <em>same</em> minibatch by re-evaluating the
 *       gradient after the step, so each curvature pair is consistent; {@code y + lambda s}
 *       keeps the pairs positive.</li>
 * </ul>
 *
 * <p>Both need {@link #step(List, Optimizer.Reevaluation)} (core {@code TrainingLoop} passes it);
 * plain {@link #step(List)} falls back to the two-loop direction with pairs from consecutive
 * calls, which is noisy on minibatches.
 */
public final class Lbfgs implements Optimizer {

    private float learningRate;
    private final int history;
    private final boolean lineSearch;
    private final double lambda;
    private final Deque<double[][]> pairs = new ArrayDeque<>();
    private double[] lastX;
    private double[] lastG;
    private Double cachedLoss;
    private int evaluations;

    private Lbfgs(float learningRate, int history, boolean lineSearch, double lambda) {
        this.learningRate = learningRate;
        this.history = history;
        this.lineSearch = lineSearch;
        this.lambda = lambda;
    }

    /**
     * Full-batch L-BFGS with a strong-Wolfe line search.
     *
     * @param initialStep the first trial step of each line search, 1 for the usual scaling
     * @param history the pairs kept, for example 10
     * @return the optimizer
     */
    public static Lbfgs fullBatch(float initialStep, int history) {
        return new Lbfgs(initialStep, history, true, 0);
    }

    /**
     * Online L-BFGS for minibatches.
     *
     * @param learningRate the fixed step size along the quasi-Newton direction, for example 0.1
     * @param history the pairs kept
     * @param lambda the curvature regularizer added as {@code y + lambda s}
     * @return the optimizer
     */
    public static Lbfgs online(float learningRate, int history, double lambda) {
        return new Lbfgs(learningRate, history, false, lambda);
    }

    /** @return gradient evaluations requested through {@link Optimizer.Reevaluation} so far */
    public int evaluations() {
        return evaluations;
    }

    /** Clears the curvature pairs and the cached loss. */
    public void reset() {
        pairs.clear();
        lastX = null;
        lastG = null;
        cachedLoss = null;
    }

    /**
     * The two-loop recursion: {@code -H g} for the current pairs.
     *
     * @param g the gradient
     * @return the quasi-Newton direction
     */
    public double[] direction(double[] g) {
        double[] q = g.clone();
        int k = pairs.size();
        double[] alpha = new double[k];
        double[] rho = new double[k];
        int i = 0;
        for (Iterator<double[][]> it = pairs.descendingIterator(); it.hasNext(); i++) {
            double[][] p = it.next();
            rho[i] = 1 / CurvatureProducts.dot(p[1], p[0]);
            alpha[i] = rho[i] * CurvatureProducts.dot(p[0], q);
            q = CurvatureProducts.axpy(q, -alpha[i], p[1]);
        }
        double gamma = 1;
        if (k > 0) {
            double[][] last = pairs.peekLast();
            gamma = CurvatureProducts.dot(last[0], last[1]) / CurvatureProducts.dot(last[1], last[1]);
        }
        double[] r = new double[q.length];
        for (int j = 0; j < q.length; j++) {
            r[j] = gamma * q[j];
        }
        i = k - 1;
        for (Iterator<double[][]> it = pairs.iterator(); it.hasNext(); i--) {
            double[][] p = it.next();
            double beta = rho[i] * CurvatureProducts.dot(p[1], r);
            r = CurvatureProducts.axpy(r, alpha[i] - beta, p[0]);
        }
        for (int j = 0; j < r.length; j++) {
            r[j] = -r[j];
        }
        return r;
    }

    private void remember(double[] s, double[] y) {
        if (CurvatureProducts.dot(s, y) <= 1e-10 * Math.sqrt(CurvatureProducts.dot(y, y) * CurvatureProducts.dot(s, s))) {
            return;
        }
        pairs.addLast(new double[][] {s, y});
        if (pairs.size() > history) {
            pairs.removeFirst();
        }
    }

    @Override
    public void step(List<Var> parameters) {
        double[] x = CurvatureProducts.values(parameters);
        double[] g = CurvatureProducts.grads(parameters);
        if (lastX != null) {
            double[] s = CurvatureProducts.axpy(x, -1, lastX);
            double[] y = CurvatureProducts.axpy(g, -1, lastG);
            remember(s, CurvatureProducts.axpy(y, lambda, s));
        }
        double[] d = direction(g);
        lastX = x;
        lastG = g;
        CurvatureProducts.assign(parameters, CurvatureProducts.axpy(x, learningRate, d));
        clear(parameters);
    }

    @Override
    public void step(List<Var> parameters, Reevaluation reevaluate) {
        double[] x = CurvatureProducts.values(parameters);
        double[] g = CurvatureProducts.grads(parameters);
        double[] d = direction(g);
        double slope = CurvatureProducts.dot(g, d);
        if (!(slope < 0)) {
            pairs.clear();
            d = direction(g);
            slope = CurvatureProducts.dot(g, d);
        }
        if (!lineSearch) {
            double[] xNew = CurvatureProducts.axpy(x, learningRate, d);
            CurvatureProducts.assign(parameters, xNew);
            reevaluate.lossAndGradients();
            evaluations++;
            double[] gNew = CurvatureProducts.grads(parameters);
            double[] s = CurvatureProducts.axpy(xNew, -1, x);
            remember(s, CurvatureProducts.axpy(CurvatureProducts.axpy(gNew, -1, g), lambda, s));
            clear(parameters);
            return;
        }
        double f0;
        if (cachedLoss == null) {
            f0 = reevaluate.lossAndGradients();
            evaluations++;
            g = CurvatureProducts.grads(parameters);
            d = direction(g);
            slope = CurvatureProducts.dot(g, d);
        } else {
            f0 = cachedLoss;
        }
        double a0 = pairs.isEmpty() ? Math.min(1, 1 / Math.max(1e-12, norm1(g))) * learningRate : learningRate;
        LineResult r = strongWolfe(parameters, reevaluate, x, d, f0, slope, a0);
        double[] s = CurvatureProducts.axpy(r.x, -1, x);
        remember(s, CurvatureProducts.axpy(r.g, -1, g));
        CurvatureProducts.assign(parameters, r.x);
        cachedLoss = r.f;
        clear(parameters);
    }

    private record LineResult(double[] x, double f, double[] g) {
    }

    private record Point(double alpha, double f, double slope, double[] x, double[] g) {
    }

    private Point evaluate(List<Var> params, Reevaluation reev, double[] x0, double[] d, double alpha) {
        double[] x = CurvatureProducts.axpy(x0, alpha, d);
        CurvatureProducts.assign(params, x);
        double f = reev.lossAndGradients();
        evaluations++;
        double[] g = CurvatureProducts.grads(params);
        return new Point(alpha, Double.isFinite(f) ? f : Double.MAX_VALUE, CurvatureProducts.dot(g, d), x, g);
    }

    private LineResult strongWolfe(List<Var> params, Reevaluation reev, double[] x0, double[] d, double f0,
            double slope0, double a1) {
        final double c1 = 1e-4;
        final double c2 = 0.9;
        Point prev = new Point(0, f0, slope0, x0, null);
        double alpha = a1;
        for (int i = 0; i < 20; i++) {
            Point cur = evaluate(params, reev, x0, d, alpha);
            if (cur.f > f0 + c1 * alpha * slope0 || (i > 0 && cur.f >= prev.f)) {
                return zoom(params, reev, x0, d, f0, slope0, prev, cur);
            }
            if (Math.abs(cur.slope) <= -c2 * slope0) {
                return new LineResult(cur.x, cur.f, cur.g);
            }
            if (cur.slope >= 0) {
                return zoom(params, reev, x0, d, f0, slope0, cur, prev);
            }
            prev = cur;
            alpha *= 2;
        }
        return new LineResult(prev.x, prev.f, prev.g == null ? new double[x0.length] : prev.g);
    }

    private LineResult zoom(List<Var> params, Reevaluation reev, double[] x0, double[] d, double f0, double slope0,
            Point lo, Point hi) {
        final double c1 = 1e-4;
        final double c2 = 0.9;
        Point best = lo;
        for (int i = 0; i < 20; i++) {
            double a = cubicMinimum(lo, hi);
            Point cur = evaluate(params, reev, x0, d, a);
            if (cur.f > f0 + c1 * a * slope0 || cur.f >= lo.f) {
                hi = cur;
            } else {
                if (Math.abs(cur.slope) <= -c2 * slope0) {
                    return new LineResult(cur.x, cur.f, cur.g);
                }
                if (cur.slope * (hi.alpha - lo.alpha) >= 0) {
                    hi = lo;
                }
                lo = cur;
                best = cur;
            }
            if (Math.abs(hi.alpha - lo.alpha) < 1e-10) {
                break;
            }
        }
        if (best.g == null) {
            // nothing better than the start: take a tiny step so the pair stays informative
            Point tiny = evaluate(params, reev, x0, d, Math.min(lo.alpha, hi.alpha) + 1e-6);
            return tiny.f <= f0 ? new LineResult(tiny.x, tiny.f, tiny.g)
                    : new LineResult(x0, f0, evaluate(params, reev, x0, d, 0).g);
        }
        return new LineResult(best.x, best.f, best.g);
    }

    /** The minimizer of the cubic through two points with slopes, safeguarded into the interval. */
    private static double cubicMinimum(Point p, Point q) {
        double lo = Math.min(p.alpha, q.alpha);
        double hi = Math.max(p.alpha, q.alpha);
        double d1 = p.slope + q.slope - 3 * (p.f - q.f) / (p.alpha - q.alpha);
        double rad = d1 * d1 - p.slope * q.slope;
        double a = 0.5 * (lo + hi);
        if (rad >= 0) {
            double d2 = Math.signum(q.alpha - p.alpha) * Math.sqrt(rad);
            double c = q.alpha - (q.alpha - p.alpha) * (q.slope + d2 - d1) / (q.slope - p.slope + 2 * d2);
            if (Double.isFinite(c)) {
                a = c;
            }
        }
        double margin = 0.1 * (hi - lo);
        return Math.min(hi - margin, Math.max(lo + margin, a));
    }

    private static double norm1(double[] g) {
        double s = 0;
        for (double v : g) {
            s += Math.abs(v);
        }
        return s;
    }

    private static void clear(List<Var> parameters) {
        for (Var p : parameters) {
            p.zeroGrad();
        }
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void learningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public String toString() {
        return (lineSearch ? "L-BFGS" : "oLBFGS") + "(m=" + history + ")";
    }
}
