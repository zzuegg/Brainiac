package io.github.zzuegg.brainiac.experiments.synthesis;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.evolve.SymbolicRegression;
import io.github.zzuegg.brainiac.experimental.nn.NeuralArithmeticUnit;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Symbolic regression on Feynman-style equations: genetic programming
 * ({@link SymbolicRegression}) with epsilon-lexicase selection against its tournament
 * ablation, an MLP and, for the product, a NALU. Each model sees 64 noise-free samples
 * from a training range and is scored on an extrapolation range outside it. Scores:
 * whether the equation was recovered exactly (relative extrapolation error below
 * {@code 1e-4}) and the mean relative extrapolation error.
 */
public final class SymbolicRegressionBenchmark {

    /** The target equations. */
    public enum Equation {
        /** {@code x0 * x1 / (1 + x2)} on {@code [0, 2]^3}, tested on {@code [2, 4]^3}. */
        RATIO(3, 0, 2, 2, 4, x -> x[0] * x[1] / (1 + x[2])),
        /** {@code sin(x0) * exp(x0)} on {@code [-1, 1]}, tested on {@code [1, 2]}. */
        SIN_EXP(1, -1, 1, 1, 2, x -> Math.sin(x[0]) * Math.exp(x[0])),
        /** {@code x0 * x1} on {@code [0.1, 10]^2}, tested on {@code [10, 1000]^2} (ArithmeticExtrapolation). */
        PRODUCT(2, 0.1, 10, 10, 1000, x -> x[0] * x[1]),
        /** {@code x0^2 - x1 + 1} on {@code [-2, 2]^2}, tested on {@code [2, 5]^2}. */
        POLY(2, -2, 2, 2, 5, x -> x[0] * x[0] - x[1] + 1);

        final int vars;
        final double low;
        final double high;
        final double testLow;
        final double testHigh;
        final ToDoubleFunction<double[]> f;

        Equation(int vars, double low, double high, double testLow, double testHigh, ToDoubleFunction<double[]> f) {
            this.vars = vars;
            this.low = low;
            this.high = high;
            this.testLow = testLow;
            this.testHigh = testHigh;
            this.f = f;
        }
    }

    /** The compared models. */
    public enum Learner {
        /** GP with epsilon-lexicase selection. */
        GP_LEXICASE,
        /** GP with tournament selection (the ablation). */
        GP_TOURNAMENT,
        /** MLP with two hidden layers of 32 SiLU units. */
        MLP,
        /** A single NALU layer (meaningful for the product only). */
        NALU
    }

    /**
     * One fit's scores.
     *
     * @param recovered whether the extrapolation error is below {@code 1e-4}
     * @param extrapolationError the mean relative error on the extrapolation range
     * @param expression the found expression, or the model's name
     */
    public record Score(boolean recovered, double extrapolationError, String expression) {
    }

    private SymbolicRegressionBenchmark() {
    }

    private static double[][] sample(Equation eq, int n, double low, double high, SplittableRandom r) {
        double[][] x = new double[n][eq.vars];
        for (double[] row : x) {
            for (int j = 0; j < eq.vars; j++) {
                row[j] = low + (high - low) * r.nextDouble();
            }
        }
        return x;
    }

    /**
     * Fits one learner to one equation.
     *
     * @param eq the equation
     * @param learner the learner
     * @param generations GP generations (MLP and NALU get 40 Adam steps per generation)
     * @param seed the seed
     * @return the scores
     */
    public static Score run(Equation eq, Learner learner, int generations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = sample(eq, 64, eq.low, eq.high, r);
        double[] y = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            y[i] = eq.f.applyAsDouble(x[i]);
        }
        double[][] tx = sample(eq, 200, eq.testLow, eq.testHigh, new SplittableRandom(seed + 9));
        ToDoubleFunction<double[]> model;
        String name;
        switch (learner) {
            case GP_LEXICASE, GP_TOURNAMENT -> {
                SymbolicRegression gp = new SymbolicRegression(learner == Learner.GP_LEXICASE
                        ? SymbolicRegression.Selection.LEXICASE : SymbolicRegression.Selection.TOURNAMENT,
                        300, generations, 25, 1e-6);
                SymbolicRegression.Expr e = gp.fit(x, y, seed).best();
                model = e::eval;
                name = e.toString();
            }
            default -> {
                Model m = learner == Learner.MLP
                        ? Model.sequential().input(eq.vars).dense(32, Activation.SILU).dense(32, Activation.SILU)
                                .dense(1, Activation.IDENTITY).seed(seed).build()
                        : Model.sequential().input(eq.vars).add(NeuralArithmeticUnit.spec(1, NeuralArithmeticUnit.Kind.NALU))
                                .seed(seed).build();
                Optimizer opt = Optimizer.adam(learner == Learner.MLP ? 0.01f : 0.05f);
                Tensor tx0 = tensor(x);
                Tensor ty = Tensor.of(Shape.of(y.length, 1), toFloat(y));
                for (int s = 0; s < 40 * generations; s++) {
                    m.forward(Var.of(tx0), true).sub(Var.of(ty)).square().mean().backward();
                    opt.step(m.parameters());
                }
                model = v -> m.forward(Var.of(Tensor.of(Shape.of(1, v.length), toFloat(v)))).value().item();
                name = learner.name();
            }
        }
        double err = 0;
        for (double[] v : tx) {
            double t = eq.f.applyAsDouble(v);
            double e = Math.abs(model.applyAsDouble(v) - t) / Math.max(1e-9, Math.abs(t));
            err += Double.isFinite(e) ? Math.min(e, 1e6) : 1e6;
        }
        err /= tx.length;
        return new Score(err < 1e-4, err, name);
    }

    private static Tensor tensor(double[][] x) {
        float[] v = new float[x.length * x[0].length];
        for (int i = 0; i < x.length; i++) {
            for (int j = 0; j < x[0].length; j++) {
                v[i * x[0].length + j] = (float) x[i][j];
            }
        }
        return Tensor.of(Shape.of(x.length, x[0].length), v);
    }

    private static float[] toFloat(double[] d) {
        float[] f = new float[d.length];
        for (int i = 0; i < d.length; i++) {
            f[i] = (float) d[i];
        }
        return f;
    }
}
