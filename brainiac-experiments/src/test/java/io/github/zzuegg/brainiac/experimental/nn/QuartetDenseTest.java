package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class QuartetDenseTest {

    @Test
    void stochasticMxfp4RoundingIsUnbiased() {
        SplittableRandom r = new SplittableRandom(1);
        float[] a = new float[64];
        for (int i = 0; i < a.length; i++) {
            a[i] = (float) r.nextGaussian();
        }
        double[] mean = new double[64];
        int trials = 400;
        for (int t = 0; t < trials; t++) {
            float[] q = QuartetDense.mxfp4(a, 1, 64, false, null, r);
            for (int i = 0; i < 64; i++) {
                mean[i] += q[i] / (double) trials;
            }
        }
        float[] nearest = QuartetDense.mxfp4(a, 1, 64, false, null, null);
        double srBias = 0;
        double rtnError = 0;
        for (int i = 0; i < 64; i++) {
            srBias += Math.abs(mean[i] - a[i]) / 64;
            rtnError += Math.abs(nearest[i] - a[i]) / 64;
        }
        System.out.printf("MXFP4: mean |bias| of stochastic rounding over %d draws %.4f vs one round-to-nearest "
                + "error %.4f%n", trials, srBias, rtnError);
        assertTrue(srBias < 0.3 * rtnError, "stochastic rounding averages to the true values");
    }

    @Test
    void gradientsFlowThroughEveryPrecision() {
        Tensor w = Tensor.normal(Shape.of(40, 8), 0, 0.3f, new SplittableRandom(2));
        Tensor x = Tensor.normal(Shape.of(16, 40), 0, 1, new SplittableRandom(3));
        QuartetDense ref = new QuartetDense(w, Activation.IDENTITY, QuartetDense.Precision.FP32, 1);
        Var xr = Var.param(x);
        ref.forward(xr).square().sum().backward();
        for (QuartetDense.Precision p : QuartetDense.Precision.values()) {
            QuartetDense layer = new QuartetDense(w, Activation.IDENTITY, p, 1);
            Var xv = Var.param(x);
            layer.forward(xv).square().sum().backward();
            assertEquals(8, layer.forward(Var.of(x)).shape().dim(1));
            double num = layer.weight().grad().mul(ref.weight().grad()).sum().item();
            double den = Math.sqrt(layer.weight().grad().square().sum().item() * ref.weight().grad().square().sum()
                    .item());
            assertTrue(num / den > 0.9, p + " weight gradient points the same way as float32's");
        }
    }
}
