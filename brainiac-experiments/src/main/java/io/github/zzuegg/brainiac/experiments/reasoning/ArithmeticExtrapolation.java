package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.KanLayer;
import io.github.zzuegg.brainiac.experimental.nn.NeuralArithmeticUnit;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Wave 6 item 2.10: arithmetic extrapolation. Models learn {@code a + b} or {@code a * b} from
 * pairs drawn in {@code [0.1, 10]} and are tested on pairs in {@code [10, 1000]}, far outside
 * the training range. Compared: a {@link NeuralArithmeticUnit} (one NAC for addition, one NALU
 * for both), two-hidden-layer MLPs with ReLU and SiLU, and a {@link KanLayer} network. Reported:
 * the mean relative error {@code |y_hat - y| / |y|} on the test range.
 */
public final class ArithmeticExtrapolation {

    /** The target operation. */
    public enum Operation {
        /** {@code a + b} */
        ADD,
        /** {@code a * b} */
        MULTIPLY
    }

    /** The compared models. */
    public enum Architecture {
        /** a single NAC layer */
        NAC,
        /** a single NALU layer */
        NALU,
        /** MLP 2-32-32-1 with ReLU */
        MLP_RELU,
        /** MLP 2-32-32-1 with SiLU */
        MLP_SILU,
        /** KAN 2-8-1 */
        KAN
    }

    private ArithmeticExtrapolation() {
    }

    private static Model build(Architecture a, long seed) {
        Model.Builder b = Model.sequential().input(2).seed(seed);
        switch (a) {
            case NAC -> b.add(NeuralArithmeticUnit.spec(1, NeuralArithmeticUnit.Kind.NAC));
            case NALU -> b.add(NeuralArithmeticUnit.spec(1, NeuralArithmeticUnit.Kind.NALU));
            case MLP_RELU -> b.dense(32, Activation.RELU).dense(32, Activation.RELU).dense(1, Activation.IDENTITY);
            case MLP_SILU -> b.dense(32, Activation.SILU).dense(32, Activation.SILU).dense(1, Activation.IDENTITY);
            case KAN -> b.add(KanLayer.spec(8, 6)).add(KanLayer.spec(1, 6));
        }
        return b.build();
    }

    private static Tensor[] data(Operation op, int n, double low, double high, SplittableRandom r) {
        float[] x = new float[2 * n];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            float a = (float) (low + (high - low) * r.nextDouble());
            float c = (float) (low + (high - low) * r.nextDouble());
            x[2 * i] = a;
            x[2 * i + 1] = c;
            y[i] = op == Operation.ADD ? a + c : a * c;
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 2), x), Tensor.of(Shape.of(n, 1), y)};
    }

    /**
     * Trains on {@code [0.1, 10]} and tests on {@code [10, 1000]}.
     *
     * @param a the model
     * @param op the operation
     * @param steps Adam steps of 64 pairs
     * @param seed the seed
     * @return {@code {mean relative error in range, mean relative error on the extrapolation range}}
     */
    public static double[] run(Architecture a, Operation op, int steps, long seed) {
        Model m = build(a, seed);
        boolean arithmetic = a == Architecture.NAC || a == Architecture.NALU;
        Optimizer opt = Optimizer.adam(arithmetic ? 0.05f : 0.01f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Tensor[] d = data(op, 64, 0.1, 10, r);
            m.forward(Var.of(d[0]), true).sub(Var.of(d[1])).square().mean().backward();
            opt.step(m.parameters());
        }
        SplittableRandom t = new SplittableRandom(seed + 9);
        return new double[] {relativeError(m, data(op, 500, 0.1, 10, t)), relativeError(m, data(op, 500, 10, 1000, t))};
    }

    private static double relativeError(Model m, Tensor[] d) {
        float[] p = m.forward(Var.of(d[0])).value().toArray();
        float[] y = d[1].toArray();
        double e = 0;
        for (int i = 0; i < y.length; i++) {
            double v = Math.abs(p[i] - y[i]) / Math.abs(y[i]);
            e += Double.isFinite(v) ? Math.min(v, 1e6) : 1e6;
        }
        return e / y.length;
    }
}
