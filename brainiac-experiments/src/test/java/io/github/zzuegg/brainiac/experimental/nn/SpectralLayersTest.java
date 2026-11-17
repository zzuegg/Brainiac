package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** FFT long convolution (Hyena) and the Fourier Neural Operator. */
class SpectralLayersTest {

    @Test
    void fftConvolutionMatchesDirectConvolutionAndGradients() {
        Tensor u = Tensor.uniform(Shape.of(2, 3, 7), -1, 1, new SplittableRandom(1));
        Tensor h = Tensor.uniform(Shape.of(3, 7), -1, 1, new SplittableRandom(2));
        float[] y = MixerOps.causalConvolution(Var.of(u), Var.of(h)).value().toArray();
        float[] uv = u.toArray();
        float[] hv = h.toArray();
        float[] direct = new float[y.length];
        for (int r = 0; r < 6; r++) {
            int c = r % 3;
            for (int t = 0; t < 7; t++) {
                for (int s = 0; s <= t; s++) {
                    direct[r * 7 + t] += hv[c * 7 + t - s] * uv[r * 7 + s];
                }
            }
        }
        assertArrayEquals(direct, y, 1e-4f);
        Tensor w = Tensor.uniform(Shape.of(2, 3, 7), 0.5f, 1.5f, new SplittableRandom(3));
        GradCheck.Result r = GradCheck.check(v -> MixerOps.causalConvolution(v.get(0), v.get(1)).mul(w).sum(),
                List.of(u, h));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void spectralConvolutionKeepsOnlyLowModesAndHasCorrectGradients() {
        // with weight 1 on mode 0 only, the result is the mean of the input
        Tensor u = Tensor.uniform(Shape.of(1, 1, 8), -1, 1, new SplittableRandom(4));
        Tensor wr = Tensor.of(Shape.of(1, 1, 1), 1f);
        Tensor wi = Tensor.zeros(Shape.of(1, 1, 1));
        float[] y = MixerOps.spectralConvolution(Var.of(u), Var.of(wr), Var.of(wi)).value().toArray();
        float mean = u.mean().item();
        for (float v : y) {
            assertTrue(Math.abs(v - mean) < 1e-5);
        }
        Tensor x = Tensor.uniform(Shape.of(2, 2, 8), -1, 1, new SplittableRandom(5));
        Tensor a = Tensor.uniform(Shape.of(3, 2, 3), -1, 1, new SplittableRandom(6));
        Tensor b = Tensor.uniform(Shape.of(3, 2, 3), -1, 1, new SplittableRandom(7));
        Tensor w = Tensor.uniform(Shape.of(2, 3, 8), 0.5f, 1.5f, new SplittableRandom(8));
        GradCheck.Result r = GradCheck.check(v -> MixerOps.spectralConvolution(v.get(0), v.get(1), v.get(2)).mul(w).sum(),
                List.of(x, a, b));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void hyenaLearnsRecall() {
        double acc = MixerHarness.train(new HyenaOperator(16, 2, 16, new SplittableRandom(9)),
                MixerHarness.recall(4, 4, 2), 16, 300, 0.01f, 10);
        assertTrue(acc > 0.7, "accuracy " + acc);
    }

    /** Samples of a random smooth periodic function and its antiderivative (zero mean). */
    private static Tensor[] antiderivative(SplittableRandom r, int batch, int n) {
        float[] f = new float[batch * n];
        float[] g = new float[batch * n];
        for (int i = 0; i < batch; i++) {
            double[] a = new double[3];
            double[] b = new double[3];
            for (int k = 0; k < 3; k++) {
                a[k] = r.nextDouble() * 2 - 1;
                b[k] = r.nextDouble() * 2 - 1;
            }
            for (int j = 0; j < n; j++) {
                double x = 2 * Math.PI * j / n;
                double fv = 0;
                double gv = 0;
                for (int k = 0; k < 3; k++) {
                    fv += a[k] * Math.cos((k + 1) * x) + b[k] * Math.sin((k + 1) * x);
                    gv += (a[k] * Math.sin((k + 1) * x) - b[k] * Math.cos((k + 1) * x)) / (k + 1);
                }
                f[i * n + j] = (float) fv;
                g[i * n + j] = (float) gv;
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(batch, n, 1), f), Tensor.of(Shape.of(batch, n, 1), g)};
    }

    @Test
    void fourierNeuralOperatorLearnsAntiderivativeAndTransfersAcrossResolutions() {
        SplittableRandom r = new SplittableRandom(11);
        FourierNeuralOperator1d fno = new FourierNeuralOperator1d(1, 1, 8, 4, 2, r);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int step = 0; step < 300; step++) {
            Tensor[] d = antiderivative(r, 16, 16);
            Var loss = fno.forward(Var.of(d[0])).sub(d[1]).square().mean();
            loss.backward();
            opt.step(fno.parameters());
        }
        for (int n : new int[] {16, 64}) {
            Tensor[] d = antiderivative(new SplittableRandom(12), 32, n);
            double mse = fno.forward(Var.of(d[0])).sub(Var.of(d[1])).square().mean().value().item();
            double var = d[1].square().mean().item();
            assertTrue(mse / var < 0.1, "resolution " + n + ": relative mse " + mse / var);
        }
    }
}
