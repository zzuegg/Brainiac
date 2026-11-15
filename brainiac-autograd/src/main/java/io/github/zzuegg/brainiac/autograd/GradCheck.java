package io.github.zzuegg.brainiac.autograd;

import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Verifies gradients numerically: compares what {@link Var#backward()} computes with
 * central differences {@code (f(x + h) - f(x - h)) / 2h}. Every operation and layer in
 * Brainiac is tested this way.
 *
 * <pre>{@code
 * GradCheck.Result r = GradCheck.check(
 *         xs -> xs.get(0).matmul(xs.get(1)).tanh().sum(),
 *         List.of(inputs, weights));
 * assertTrue(r.passed(), r.toString());
 * }</pre>
 *
 * <p>Values are float32, so use inputs of moderate size (around -3..3) and avoid points
 * where a function is not smooth, such as 0 for ReLU.
 */
public final class GradCheck {

    /** Default step size for the finite differences. */
    public static final float DEFAULT_STEP = 1e-2f;

    /** Default tolerance, applied to both absolute and relative error. */
    public static final float DEFAULT_TOLERANCE = 1e-2f;

    private GradCheck() {
    }

    /**
     * The outcome of a gradient check.
     *
     * @param passed whether every gradient value was within tolerance
     * @param maxError the largest error seen, combining absolute and relative error
     * @param input index of the input with the largest error
     * @param element index of the element with the largest error, in row-major order
     * @param analytic the gradient computed by backward() at that element
     * @param numeric the gradient estimated by finite differences at that element
     */
    public record Result(boolean passed, double maxError, int input, int element, double analytic, double numeric) {
        @Override
        public String toString() {
            return (passed ? "passed" : "FAILED") + ": max error " + maxError + " at input " + input
                    + " element " + element + " (backward " + analytic + ", numeric " + numeric + ")";
        }
    }

    /**
     * Checks gradients with the default step and tolerance.
     *
     * @param f a function from parameter values to a scalar
     * @param inputs the points at which to check; one parameter is made per input
     * @return the result
     */
    public static Result check(Function<List<Var>, Var> f, List<Tensor> inputs) {
        return check(f, inputs, DEFAULT_STEP, DEFAULT_TOLERANCE);
    }

    /**
     * Checks gradients.
     *
     * @param f a function from parameter values to a scalar
     * @param inputs the points at which to check; one parameter is made per input
     * @param step the finite-difference step {@code h}
     * @param tolerance the allowed error: an element passes if its absolute error or its
     *     relative error is at most this
     * @return the result
     */
    public static Result check(Function<List<Var>, Var> f, List<Tensor> inputs, float step, float tolerance) {
        List<Var> params = new ArrayList<>();
        for (Tensor t : inputs) {
            params.add(Var.param(t));
        }
        Var out = f.apply(params);
        if (out.value().size() != 1) {
            throw new IllegalArgumentException("the function must return a scalar, got shape " + out.shape());
        }
        out.backward();

        double worst = 0;
        int worstInput = -1;
        int worstElement = -1;
        double worstAnalytic = 0;
        double worstNumeric = 0;
        for (int i = 0; i < inputs.size(); i++) {
            Tensor x = inputs.get(i);
            Tensor g = params.get(i).grad();
            float[] analytic = g == null ? new float[(int) x.size()] : g.toArray();
            float[] values = x.toArray();
            for (int e = 0; e < values.length; e++) {
                float original = values[e];
                values[e] = original + step;
                double plus = evaluate(f, inputs, i, x, values);
                values[e] = original - step;
                double minus = evaluate(f, inputs, i, x, values);
                values[e] = original;
                double numeric = (plus - minus) / (2.0 * step);
                double abs = Math.abs(numeric - analytic[e]);
                double rel = abs / Math.max(Math.abs(numeric), Math.abs(analytic[e]));
                double error = Math.min(abs, Double.isNaN(rel) ? 0 : rel);
                if (error > worst || worstInput < 0) {
                    worst = error;
                    worstInput = i;
                    worstElement = e;
                    worstAnalytic = analytic[e];
                    worstNumeric = numeric;
                }
            }
        }
        return new Result(worst <= tolerance, worst, worstInput, worstElement, worstAnalytic, worstNumeric);
    }

    /**
     * Checks a custom operation's {@link Op#backward} against finite differences, with
     * the default step and tolerance. The output is weighted with fixed random values
     * before summing, so every output element contributes a different amount.
     *
     * @param op the operation
     * @param inputs the input values at which to check
     * @return the result
     */
    public static Result check(Op op, Tensor... inputs) {
        return check(op, DEFAULT_STEP, DEFAULT_TOLERANCE, inputs);
    }

    /**
     * Checks a custom operation's gradient.
     *
     * @param op the operation
     * @param step the finite-difference step {@code h}
     * @param tolerance the allowed error
     * @param inputs the input values at which to check
     * @return the result
     */
    public static Result check(Op op, float step, float tolerance, Tensor... inputs) {
        Tensor out = op.forward(inputs.clone());
        Tensor weights = Tensor.uniform(out.shape(), 0.5f, 1.5f, new SplittableRandom(42)).to(out.device());
        return check(v -> Var.apply(op, v.toArray(Var[]::new)).mul(weights).sum(), List.of(inputs), step, tolerance);
    }

    private static double evaluate(Function<List<Var>, Var> f, List<Tensor> inputs, int index, Tensor x, float[] values) {
        List<Var> vars = new ArrayList<>();
        for (int j = 0; j < inputs.size(); j++) {
            Tensor t = j == index ? Tensor.of(x.shape(), values, x.device()) : inputs.get(j);
            vars.add(Var.of(t));
        }
        return f.apply(vars).value().item();
    }
}
