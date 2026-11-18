package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Integrators for Ito stochastic differential equations {@code dX = f(X) dt + g(X) dW} with
 * diagonal noise:
 *
 * <ul>
 *   <li>Euler-Maruyama, {@code X' = X + f dt + g dW}, strong order 1/2;
 *   <li>Milstein, which adds {@code 1/2 g g' (dW^2 - dt)} (with {@code g'} by central
 *       differences), strong order 1.
 * </ul>
 *
 * <p>{@link #eulerMaruyama(Drift, Drift, Var, float, Tensor)} is the differentiable step used to
 * train SDE models by backpropagating through the solver.
 */
public final class SdeSolver {

    private SdeSolver() {
    }

    /** A differentiable drift or diffusion. */
    @FunctionalInterface
    public interface Drift {

        /**
         * @param x states {@code [b, d]}
         * @return values {@code [b, d]}
         */
        Var apply(Var x);
    }

    /**
     * One differentiable Euler-Maruyama step with given Brownian increments.
     *
     * @param drift {@code f}
     * @param diffusion {@code g} (diagonal)
     * @param x states {@code [b, d]}
     * @param dt the step
     * @param dw Brownian increments {@code [b, d]} with variance {@code dt}
     * @return the next states
     */
    public static Var eulerMaruyama(Drift drift, Drift diffusion, Var x, float dt, Tensor dw) {
        return x.add(drift.apply(x).mul(dt)).add(diffusion.apply(x).mul(Var.of(dw)));
    }

    /**
     * Simulates paths by Euler-Maruyama or Milstein.
     *
     * @param drift {@code f}
     * @param diffusion {@code g} (diagonal)
     * @param x0 the initial state
     * @param dt the step
     * @param steps the steps
     * @param milstein whether to add the Milstein correction
     * @param random the noise source
     * @return states {@code [steps + 1][d]}
     */
    public static double[][] simulate(UnaryOperator<double[]> drift, UnaryOperator<double[]> diffusion, double[] x0,
            double dt, int steps, boolean milstein, SplittableRandom random) {
        int d = x0.length;
        double[][] out = new double[steps + 1][];
        out[0] = x0.clone();
        double sq = Math.sqrt(dt);
        for (int t = 1; t <= steps; t++) {
            double[] x = out[t - 1];
            double[] f = drift.apply(x);
            double[] g = diffusion.apply(x);
            double[] dg = null;
            if (milstein) {
                dg = new double[d];
                double h = 1e-5;
                for (int i = 0; i < d; i++) {
                    double[] xp = x.clone();
                    double[] xm = x.clone();
                    xp[i] += h;
                    xm[i] -= h;
                    dg[i] = (diffusion.apply(xp)[i] - diffusion.apply(xm)[i]) / (2 * h);
                }
            }
            double[] y = new double[d];
            for (int i = 0; i < d; i++) {
                double dw = sq * random.nextGaussian();
                y[i] = x[i] + f[i] * dt + g[i] * dw;
                if (milstein) {
                    y[i] += 0.5 * g[i] * dg[i] * (dw * dw - dt);
                }
            }
            out[t] = y;
        }
        return out;
    }
}
