package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.QuartetDense;
import io.github.zzuegg.brainiac.experimental.nn.QuartetDense.Precision;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Native low-precision training, the training-time twin of {@link LowBitQuantization}: a
 * three-layer MLP of {@link QuartetDense} layers learns a teacher's ten-way classification
 * of inputs with a few outlier channels (eight times larger than the rest, as in large
 * models' activations), with every matrix product of the forward and backward pass in the
 * given {@link Precision}. The measure is the held-out loss gap to float32 at the same
 * number of steps; Quartet (Hadamard rotations, stochastic rounding in the backward pass)
 * against naive FP4 and per-tensor FP8.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class Fp4Training {

    private Fp4Training() {
    }

    private static Tensor inputs(int n, SplittableRandom r) {
        float[] s = new float[64];
        java.util.Arrays.fill(s, 1);
        for (int j = 0; j < 64; j += 16) {
            s[j] = 8;   // four outlier channels
        }
        return Tensor.normal(Shape.of(n, 64), 0, 1, r).mul(Tensor.of(Shape.of(1, 64), s));
    }

    /**
     * Trains the classifier.
     *
     * @param precision the precision of every product
     * @param steps Adam steps of batch 32
     * @param seed the seed
     * @return the held-out cross-entropy at the end
     */
    public static float train(Precision precision, int steps, long seed) {
        SplittableRandom tr = new SplittableRandom(99);
        Tensor teacher1 = Tensor.normal(Shape.of(64, 32), 0, 0.25f, tr);
        Tensor teacher2 = Tensor.normal(Shape.of(32, 10), 0, 1, tr);
        Model model = Model.sequential().input(64)
                .add(QuartetDense.spec(64, Activation.RELU, precision))
                .add(QuartetDense.spec(64, Activation.RELU, precision))
                .add(QuartetDense.spec(10, Activation.IDENTITY, precision))
                .seed(seed).build();
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            Tensor x = inputs(32, r);
            Tensor y = x.matmul(teacher1).tanh().matmul(teacher2).argmax(1);
            Functional.crossEntropy(model.forward(Var.of(x), true), y).backward();
            opt.step(model.parameters());
        }
        Tensor xt = inputs(1024, new SplittableRandom(seed + 2));
        Tensor yt = xt.matmul(teacher1).tanh().matmul(teacher2).argmax(1);
        return Functional.crossEntropy(model.forward(Var.of(xt)), yt).value().item();
    }
}
