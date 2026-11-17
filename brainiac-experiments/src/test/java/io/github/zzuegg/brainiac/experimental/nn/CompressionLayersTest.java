package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.tensor.ThinSvd;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class CompressionLayersTest {

    /** Inputs whose features differ in scale by up to 30x and are correlated, as LLM activations are. */
    private static Tensor inputs(SplittableRandom r, int n, int in) {
        float[] x = new float[n * in];
        for (int s = 0; s < n; s++) {
            double shared = r.nextGaussian();
            for (int i = 0; i < in; i++) {
                double scale = i % 8 == 0 ? 30 : 1;
                x[s * in + i] = (float) (scale * (0.6 * shared + 0.8 * r.nextGaussian()));
            }
        }
        return Tensor.of(Shape.of(n, in), x);
    }

    private static double error(Tensor x, Tensor w, Tensor pruned) {
        return x.matmul(w).sub(x.matmul(pruned)).square().sum().item();
    }

    @Test
    void sparseGptBeatsWandaBeatsMagnitude() {
        SplittableRandom r = new SplittableRandom(1);
        Tensor x = inputs(r, 256, 32);
        Tensor w = Tensor.normal(Shape.of(32, 16), 0, 1, r);
        Tensor mag = WeightPruning.magnitude(w, 0.5);
        Tensor wanda = WeightPruning.wanda(w, x, 0.5);
        Tensor sgpt = WeightPruning.sparseGpt(w, x, 0.5, 0, 0, 8, 0.01);
        assertEquals(0.5, WeightPruning.sparsity(mag), 1e-9);
        assertEquals(0.5, WeightPruning.sparsity(wanda), 1e-9);
        assertEquals(0.5, WeightPruning.sparsity(sgpt), 0.01);
        double em = error(x, w, mag);
        double ew = error(x, w, wanda);
        double es = error(x, w, sgpt);
        assertTrue(ew < em, "wanda " + ew + " vs magnitude " + em);
        assertTrue(es < ew, "sparsegpt " + es + " vs wanda " + ew);
        Tensor twoFour = WeightPruning.semiStructured(w, x, 2, 4);
        Tensor sgpt24 = WeightPruning.sparseGpt(w, x, 0, 2, 4, 8, 0.01);
        assertTrue(WeightPruning.isNofM(twoFour, 2, 4));
        assertTrue(WeightPruning.isNofM(sgpt24, 2, 4));
        assertTrue(!WeightPruning.isNofM(w, 2, 4));
        assertTrue(error(x, w, sgpt24) < error(x, w, twoFour));
    }

    @Test
    void svdFactorsAreOptimalAndActivationAwareHelpsOnSkewedInputs() {
        SplittableRandom r = new SplittableRandom(2);
        Tensor a = Tensor.normal(Shape.of(12, 3), 0, 1, r);
        Tensor b = Tensor.normal(Shape.of(3, 9), 0, 1, r);
        Tensor lowRank = a.matmul(b);
        ThinSvd svd = ThinSvd.of(lowRank);
        assertEquals(3, svd.rankFor(0.999999));
        Dense d = new Dense(12, 9, Activation.IDENTITY, lowRank, Tensor.zeros(Shape.of(9)));
        LowRankDense f = LowRankDense.factorize(d, 3);
        assertTrue(f.weight().sub(lowRank).square().sum().item() < 1e-6);
        assertTrue(f.parameters().get(0).shape().size() + f.parameters().get(1).shape().size() < 12 * 9);

        Dense full = new Dense(32, 16, Activation.IDENTITY, Init.XAVIER, r);
        Tensor x = inputs(r, 256, 32);
        LowRankDense plain = LowRankDense.factorize(full, 4);
        LowRankDense aware = LowRankDense.activationAware(full, x, 4);
        Tensor want = full.forward(Var.of(x)).value();
        double ep = plain.forward(Var.of(x)).value().sub(want).square().sum().item();
        double ea = aware.forward(Var.of(x)).value().sub(want).square().sum().item();
        assertTrue(ea < ep, "activation-aware " + ea + " vs plain " + ep);
    }
}
