package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.KanLayer;
import io.github.zzuegg.brainiac.experimental.nn.NeuralArithmeticUnit;
import io.github.zzuegg.brainiac.experimental.nn.NeuralGpu;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 6 item 2.10: arithmetic extrapolation. Models learn {@code a + b} or {@code a * b} from
 * pairs drawn in {@code [0.1, 10]} and are tested on pairs in {@code [10, 1000]}, far outside
 * the training range. Compared: a {@link NeuralArithmeticUnit} (one NAC for addition, one NALU
 * for both), two-hidden-layer MLPs with ReLU and SiLU, and a {@link KanLayer} network. Reported:
 * the mean relative error {@code |y_hat - y| / |y|} on the test range.
 *
 * <p>Wave 7 item 7.6.10 adds digit-level length generalisation ({@link #binaryLength}):
 * binary addition (or multiplication) learned from numbers of up to a few bits and tested
 * on much longer ones, comparing the {@link NeuralGpu} with a transformer encoder over the
 * same per-bit encoding (sinusoidal absolute positions, two blocks). Reported: the share
 * of test sums whose every output bit is right, per test length.
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

    /** Digit-level models for {@link #binaryLength}. */
    public enum BitModel {
        /** The Neural GPU (state 24, two CGRUs per step). */
        NEURAL_GPU,
        /** Two transformer blocks of width 32 with sinusoidal positions. */
        TRANSFORMER
    }

    /** One bit-level batch: inputs {@code [b, width, 5]} and target bits {@code [b * width]}. */
    private static Tensor[] bits(Operation op, int b, int n, SplittableRandom r) {
        int width = op == Operation.ADD ? n + 1 : 2 * n;
        float[] x = new float[b * width * 5];
        float[] y = new float[b * width];
        for (int i = 0; i < b; i++) {
            java.math.BigInteger a = new java.math.BigInteger(n, new java.util.Random(r.nextLong()));
            java.math.BigInteger c = new java.math.BigInteger(n, new java.util.Random(r.nextLong()));
            java.math.BigInteger res = op == Operation.ADD ? a.add(c) : a.multiply(c);
            for (int p = 0; p < width; p++) {
                int ab = (p < n && a.testBit(p) ? 1 : 0) + 2 * (p < n && c.testBit(p) ? 1 : 0);
                x[(i * width + p) * 5 + (p < n ? ab : 4)] = 1;
                y[i * width + p] = res.testBit(p) ? 1 : 0;
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(b, width, 5), x), Tensor.of(Shape.of(b * width), y)};
    }

    /**
     * Trains a bit-level model on numbers of 1 to {@code trainBits} bits (a length
     * curriculum: the longest length grows by one whenever a batch is learned) and tests
     * exact accuracy at each test length.
     *
     * @param model the model
     * @param op addition or multiplication
     * @param trainBits the longest training numbers
     * @param steps Adam steps of 32 problems
     * @param testBits the test lengths
     * @param seed the seed
     * @return the share of test problems with every output bit right, per test length
     */
    public static double[] binaryLength(BitModel model, Operation op, int trainBits, int steps, int[] testBits,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Layer core = model == BitModel.NEURAL_GPU ? new NeuralGpu(5, 24, 2, r)
                : null;
        int width = model == BitModel.NEURAL_GPU ? 24 : 32;
        Var inProj = Var.param(Tensor.normal(Shape.of(5, width), 0, 0.5f, r));
        List<Layer> blocks = new ArrayList<>();
        if (model == BitModel.TRANSFORMER) {
            for (int i = 0; i < 2; i++) {
                blocks.add(TransformerBlock.of(4).ffnSize(2 * width).create(Shape.of(8, width), r));
            }
        }
        Var head = Var.param(Tensor.normal(Shape.of(width, 2), 0, 0.3f, r));
        List<Var> params = new ArrayList<>(List.of(head));
        if (core != null) {
            params.addAll(core.parameters());
        } else {
            params.add(inProj);
            blocks.forEach(b -> params.addAll(b.parameters()));
        }
        java.util.function.BiFunction<Tensor, Boolean, Var> logits = (x, training) -> {
            Var h;
            if (core != null) {
                h = core.forward(Var.of(x), training);
            } else {
                int w = x.shape().dim(1);
                h = Var.of(x).matmul(inProj).add(Functional.sinusoidal(w, width, 10000f, x.device()));
                for (Layer b : blocks) {
                    h = b.forward(h, training);
                }
            }
            return h.matmul(head).reshape(x.shape().dim(0) * x.shape().dim(1), 2);
        };
        Optimizer opt = Optimizer.adam(model == BitModel.NEURAL_GPU ? 5e-3f : 2e-3f);
        int reach = Math.min(3, trainBits);
        for (int s = 0; s < steps; s++) {
            Tensor[] d = bits(op, 32, 1 + r.nextInt(reach), r);
            Var z = logits.apply(d[0], true);
            Functional.crossEntropy(z, d[1]).backward();
            opt.step(params);
            if (reach < trainBits && z.value().argmax(1).equalTo(d[1]).mean().item() > 0.98f) {
                reach++;
            }
        }
        double[] out = new double[testBits.length];
        for (int k = 0; k < testBits.length; k++) {
            Tensor[] d = bits(op, 50, testBits[k], new SplittableRandom(seed + 100 + k));
            float[] pred = Var.noGrad(() -> logits.apply(d[0], false)).value().argmax(1).toArray();
            float[] y = d[1].toArray();
            int w = d[0].shape().dim(1);
            int right = 0;
            for (int i = 0; i < 50; i++) {
                boolean all = true;
                for (int p = 0; p < w; p++) {
                    all &= pred[i * w + p] == y[i * w + p];
                }
                right += all ? 1 : 0;
            }
            out[k] = right / 50.0;
        }
        return out;
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
