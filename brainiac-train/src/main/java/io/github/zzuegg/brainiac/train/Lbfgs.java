package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;
import java.util.function.Supplier;

/**
 * Limited-memory BFGS for a list of parameters, as {@code torch.optim.LBFGS} with a
 * backtracking (Armijo) line search: for full-batch problems with smooth losses, such as
 * fitting a small model exactly, style transfer, or physics-informed networks after Adam. The
 * parameters are gathered to the host each iteration, so it suits models up to a few million
 * values.
 */
public final class Lbfgs {

    /**
     * The result of a run.
     *
     * @param loss the final loss
     * @param iterations the iterations taken
     * @param evaluations the loss evaluations, line search included
     * @param converged whether the gradient or the change fell below the tolerance
     */
    public record Result(double loss, int iterations, int evaluations, boolean converged) {
    }

    private final int history;
    private final double tolerance;

    /**
     * @param history the curvature pairs kept, such as 10
     * @param tolerance stop when the largest gradient entry, or the loss change, is below it
     */
    public Lbfgs(int history, double tolerance) {
        if (history < 1) {
            throw new IllegalArgumentException("history must be at least 1");
        }
        this.history = history;
        this.tolerance = tolerance;
    }

    /** @return an optimizer keeping 10 pairs, stopping at a gradient of 1e-7 */
    public static Lbfgs defaults() {
        return new Lbfgs(10, 1e-7);
    }

    /**
     * Minimizes a loss over the parameters, updating them in place.
     *
     * @param parameters the parameters
     * @param loss computes the loss (a scalar) at the parameters' current values
     * @param maxIterations the most iterations
     * @return what happened
     */
    public Result minimize(List<Var> parameters, Supplier<Var> loss, int maxIterations) {
        Point p = minimize(gather(parameters), (x, g) -> evaluate(parameters, loss, x, g), maxIterations);
        scatter(parameters, p.x());
        for (Var v : parameters) {
            v.zeroGrad();
        }
        return p.result();
    }

    /** A function of a vector with its gradient, in double precision. */
    @FunctionalInterface
    public interface Objective {

        /**
         * Evaluates the function.
         *
         * @param x the point; not to be changed
         * @param gradient filled with the gradient at {@code x}
         * @return the value at {@code x}
         */
        double value(double[] x, double[] gradient);
    }

    /**
     * Where a minimization of an {@link Objective} ended.
     *
     * @param x the point reached
     * @param result the loss, iterations, evaluations and whether it converged
     */
    public record Point(double[] x, Result result) {
    }

    /**
     * Minimizes a function in double precision, such as to polish a fit that a float model
     * brought close: losses whose differences matter below float's 7 digits.
     *
     * @param x0 the start; not changed
     * @param f the function and its gradient
     * @param maxIterations the most iterations
     * @return the point reached
     */
    public Point minimize(double[] x0, Objective f, int maxIterations) {
        int evaluations = 0;
        double[] x = x0.clone();
        double[] g = new double[x.length];
        double fx = f.value(x, g);
        evaluations++;
        Deque<double[][]> pairs = new ArrayDeque<>();   // {s, y, {rho}}
        int it = 0;
        boolean converged = maxAbs(g) <= tolerance;
        while (!converged && it < maxIterations) {
            it++;
            double[] d = direction(g, pairs);
            double slope = dot(g, d);
            if (slope >= 0) {   // not a descent direction: restart from steepest descent
                pairs.clear();
                d = negate(g);
                slope = dot(g, d);
            }
            double step = it == 1 ? Math.min(1, 1 / Math.max(1e-12, sum(g))) : 1;
            double[] xn = new double[x.length];
            double[] gn = new double[x.length];
            double fn;
            int tries = 0;
            while (true) {
                for (int i = 0; i < x.length; i++) {
                    xn[i] = x[i] + step * d[i];
                }
                fn = f.value(xn, gn);
                evaluations++;
                if (fn <= fx + 1e-4 * step * slope || ++tries > 30) {
                    break;
                }
                step *= 0.5;
            }
            double[] sv = new double[x.length];
            double[] yv = new double[x.length];
            for (int i = 0; i < x.length; i++) {
                sv[i] = xn[i] - x[i];
                yv[i] = gn[i] - g[i];
            }
            double sy = dot(sv, yv);
            if (sy > 1e-16 * Math.max(1e-300, Math.sqrt(dot(sv, sv) * dot(yv, yv)))) {
                pairs.addLast(new double[][] {sv, yv, {1 / sy}});
                if (pairs.size() > history) {
                    pairs.removeFirst();
                }
            }
            double change = Math.abs(fx - fn);
            x = xn;
            g = gn;
            fx = fn;
            converged = maxAbs(g) <= tolerance || change <= tolerance * Math.max(1, Math.abs(fx));
        }
        return new Point(x, new Result(fx, it, evaluations, converged));
    }

    // the two-loop recursion: -H g
    private static double[] direction(double[] g, Deque<double[][]> pairs) {
        double[] q = g.clone();
        double[] alpha = new double[pairs.size()];
        var back = pairs.descendingIterator();
        int k = pairs.size() - 1;
        while (back.hasNext()) {
            double[][] p = back.next();
            alpha[k] = p[2][0] * dot(p[0], q);
            axpy(-alpha[k], p[1], q);
            k--;
        }
        if (!pairs.isEmpty()) {
            double[][] last = pairs.getLast();
            double gamma = dot(last[0], last[1]) / dot(last[1], last[1]);
            for (int i = 0; i < q.length; i++) {
                q[i] *= gamma;
            }
        }
        k = 0;
        for (double[][] p : pairs) {
            double beta = p[2][0] * dot(p[1], q);
            axpy(alpha[k] - beta, p[0], q);
            k++;
        }
        return negate(q);
    }

    private static double evaluate(List<Var> parameters, Supplier<Var> loss, double[] x, double[] g) {
        scatter(parameters, x);
        for (Var p : parameters) {
            p.zeroGrad();
        }
        Var l = loss.get();
        l.backward();
        int at = 0;
        for (Var p : parameters) {
            int n = Math.toIntExact(p.value().size());
            if (p.grad() == null) {
                java.util.Arrays.fill(g, at, at + n, 0);
            } else {
                System.arraycopy(p.grad().toDoubleArray(), 0, g, at, n);
            }
            at += n;
        }
        return l.value().toDoubleArray()[0];
    }

    private static double[] gather(List<Var> parameters) {
        int n = 0;
        for (Var p : parameters) {
            n += Math.toIntExact(p.value().size());
        }
        double[] x = new double[n];
        int at = 0;
        for (Var p : parameters) {
            double[] v = p.value().toDoubleArray();
            System.arraycopy(v, 0, x, at, v.length);
            at += v.length;
        }
        return x;
    }

    // keeps each parameter's dtype: FLOAT64 parameters are optimized in full double precision
    private static void scatter(List<Var> parameters, double[] x) {
        int at = 0;
        for (Var p : parameters) {
            Tensor old = p.value();
            int n = Math.toIntExact(old.size());
            double[] v = java.util.Arrays.copyOfRange(x, at, at + n);
            at += n;
            p.assign(Tensor.of(old.shape(), v, old.dtype(), old.device()));
        }
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static void axpy(double a, double[] x, double[] y) {
        for (int i = 0; i < x.length; i++) {
            y[i] += a * x[i];
        }
    }

    private static double[] negate(double[] a) {
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = -a[i];
        }
        return out;
    }

    private static double sum(double[] a) {
        double s = 0;
        for (double v : a) {
            s += Math.abs(v);
        }
        return s;
    }

    private static double maxAbs(double[] a) {
        double m = 0;
        for (double v : a) {
            m = Math.max(m, Math.abs(v));
        }
        return m;
    }
}
