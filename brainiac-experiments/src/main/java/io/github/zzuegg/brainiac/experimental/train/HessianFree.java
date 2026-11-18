package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.List;
import java.util.function.Function;

/**
 * Hessian-free optimization (Martens 2010,
 * <a href="https://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf">ICML</a>), a
 * truncated Newton method: each step minimizes the local quadratic model
 * {@code q(d) = g.d + 1/2 d.(G + lambda I)d} with conjugate gradients, using exact
 * Gauss-Newton-vector products ({@link CurvatureProducts#gaussNewtonVector}) and never
 * forming {@code G}. The CG solve is warm-started from a decayed copy of the previous step,
 * and the damping {@code lambda} follows the Levenberg-Marquardt rule on the reduction ratio
 * {@code rho = (f(theta + d) - f(theta)) / q(d)}: grown by 3/2 when {@code rho < 1/4}, shrunk
 * by 2/3 when {@code rho > 3/4}. Steps that do not lower the loss are backtracked.
 *
 * <p>Second-order methods need the model's outputs, not only the scalar loss that core
 * {@code Optimizer.Reevaluation} gives, so this takes the output function and the loss'
 * output-space curvature directly.
 */
public final class HessianFree {

    private double lambda;
    private final int maxCg;
    private final double warmStart;
    private double[] previous;
    private int cgIterations;

    /**
     * @param initialDamping the starting {@code lambda}, for example 1
     * @param maxCg the most CG iterations per step, for example 50
     * @param warmStart the decay of the previous step used as CG's start, 0.95 in the paper
     */
    public HessianFree(double initialDamping, int maxCg, double warmStart) {
        this.lambda = initialDamping;
        this.maxCg = maxCg;
        this.warmStart = warmStart;
    }

    /** @return the current damping */
    public double damping() {
        return lambda;
    }

    /** @return CG iterations run so far */
    public int cgIterations() {
        return cgIterations;
    }

    /**
     * Takes one Hessian-free step.
     *
     * @param params the parameters
     * @param outputs computes the network's outputs from the parameters
     * @param loss maps those outputs to the scalar loss
     * @param curvature the loss' curvature in output space
     * @return the loss after the step
     */
    public double step(List<Var> params, Function<List<Var>, Var> outputs, Function<Var, Var> loss,
            CurvatureProducts.OutputCurvature curvature) {
        double[] x = CurvatureProducts.values(params);
        Var out = outputs.apply(params);
        Var l0 = loss.apply(out);
        double f0 = l0.value().item();
        double[] g = CurvatureProducts.gradient(l0, params);
        double[] b = g.clone();
        for (int i = 0; i < b.length; i++) {
            b[i] = -b[i];
        }
        double[] start = null;
        if (previous != null) {
            start = previous.clone();
            for (int i = 0; i < start.length; i++) {
                start[i] *= warmStart;
            }
        }
        double damp = lambda;
        CurvatureProducts.Solution sol = CurvatureProducts.conjugateGradient(
                v -> CurvatureProducts.axpy(CurvatureProducts.gaussNewtonVector(out, curvature, params, v), damp, v),
                b, start, null, maxCg, 5e-4);
        cgIterations += sol.iterations();
        double[] d = sol.x();
        double[] gd = CurvatureProducts.gaussNewtonVector(out, curvature, params, d);
        double model = CurvatureProducts.dot(g, d) + 0.5 * (CurvatureProducts.dot(d, gd) + damp * CurvatureProducts.dot(d, d));
        double scale = 1;
        double f = Double.NaN;
        for (int k = 0; k < 12; k++) {
            CurvatureProducts.assign(params, CurvatureProducts.axpy(x, scale, d));
            f = Var.noGrad(() -> loss.apply(outputs.apply(params))).value().item();
            if (Double.isFinite(f) && f <= f0 + 1e-2 * scale * CurvatureProducts.dot(g, d)) {
                break;
            }
            scale *= 0.8;
        }
        if (!(Double.isFinite(f) && f <= f0)) {
            CurvatureProducts.assign(params, x);
            f = f0;
            scale = 0;
        }
        double rho = model < 0 ? (f - f0) / model : 0;
        if (rho < 0.25) {
            lambda *= 1.5;
        } else if (rho > 0.75) {
            lambda *= 2.0 / 3;
        }
        previous = d;
        if (scale == 0) {
            previous = null;
        }
        return f;
    }

    @Override
    public String toString() {
        return "HessianFree(lambda=" + lambda + ")";
    }
}
