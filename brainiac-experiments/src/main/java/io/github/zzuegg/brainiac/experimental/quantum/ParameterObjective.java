package io.github.zzuegg.brainiac.experimental.quantum;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Lbfgs;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;

/**
 * A scalar objective over a flat parameter vector with a hand-computed gradient (a circuit's
 * adjoint gradient, a variational Monte Carlo force), exposed as an autograd {@link Var} so
 * that the core optimizers drive it: any {@link Optimizer} through {@link Optimizer#minimize},
 * or core {@link Lbfgs}. The parameters live in a single {@code float} parameter tensor; the
 * objective itself is evaluated in double precision.
 */
@FunctionalInterface
public interface ParameterObjective {

    /**
     * Evaluates the objective and its gradient.
     *
     * @param x the parameters
     * @param gradient receives the gradient (same length)
     * @return the value
     */
    double evaluate(double[] x, double[] gradient);

    /**
     * The objective as a differentiable scalar of a parameter {@code Var}.
     *
     * @param parameters a 1-D parameter
     * @return the loss
     */
    default Var loss(Var parameters) {
        double[][] last = new double[1][];
        return Var.custom(in -> {
            double[] x = toDoubles(in[0]);
            double[] g = new double[x.length];
            double v = evaluate(x, g);
            last[0] = g;
            return Tensor.of(Shape.of(1), (float) v);
        }, (in, out, grad) -> {
            double[] g = last[0];
            float scale = grad.toArray()[0];
            float[] f = new float[g.length];
            for (int i = 0; i < f.length; i++) {
                f[i] = (float) (g[i] * scale);
            }
            return new Tensor[] {Tensor.of(in[0].shape(), f)};
        }, parameters).sum();
    }

    /**
     * Minimises with a core optimizer.
     *
     * @param optimizer the optimizer, such as {@code Optimizer.adam(0.05f)}
     * @param x0 the start
     * @param steps the steps
     * @param trace receives the value before each step, or null
     * @return the final parameters
     */
    default double[] minimize(Optimizer optimizer, double[] x0, int steps, double[] trace) {
        Var p = parameter(x0);
        for (int s = 0; s < steps; s++) {
            Var l = optimizer.minimize(List.of(p), () -> loss(p));
            if (trace != null && s < trace.length) {
                trace[s] = l.value().item();
            }
        }
        return toDoubles(p.value());
    }

    /**
     * Minimises with core L-BFGS.
     *
     * @param x0 the start
     * @param maxIterations the most iterations
     * @return the final parameters
     */
    default double[] minimizeLbfgs(double[] x0, int maxIterations) {
        Var p = parameter(x0);
        new Lbfgs(10, 1e-9).minimize(List.of(p), () -> loss(p), maxIterations);
        return toDoubles(p.value());
    }

    /**
     * @param x values
     * @return a 1-D float parameter holding them
     */
    static Var parameter(double[] x) {
        float[] f = new float[x.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) x[i];
        }
        return Var.param(Tensor.of(Shape.of(x.length), f));
    }

    /**
     * @param t a tensor
     * @return its values as doubles
     */
    static double[] toDoubles(Tensor t) {
        float[] f = t.toArray();
        double[] d = new double[f.length];
        for (int i = 0; i < f.length; i++) {
            d[i] = f[i];
        }
        return d;
    }
}
