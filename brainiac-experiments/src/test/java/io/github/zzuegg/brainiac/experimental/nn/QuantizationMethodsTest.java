package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class QuantizationMethodsTest {

    /** Correlated calibration inputs with a few strong channels, like real activations. */
    private static Tensor activations(long seed, int n, int d) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[n * d];
        for (int i = 0; i < n; i++) {
            double shared = r.nextGaussian();
            for (int j = 0; j < d; j++) {
                double scale = j % 7 == 0 ? 8 : 1;
                x[i * d + j] = (float) (scale * (0.7 * shared + 0.7 * r.nextGaussian()));
            }
        }
        return Tensor.of(Shape.of(n, d), x);
    }

    @Test
    void zLossAddsTheSquaredNormalizer() {
        Tensor logits = Tensor.of(Shape.of(2, 3), 1f, 2f, 3f, 0f, 0f, 0f);
        Tensor ids = Tensor.of(2f, 0f);
        float ce = new ZLoss(0).compute(Var.of(logits), ids).value().item();
        float z = new ZLoss(1).compute(Var.of(logits), ids).value().item();
        double lse1 = Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3));
        double lse2 = Math.log(3);
        assertEquals((lse1 - 3 + lse2) / 2, ce, 1e-5);
        assertEquals(ce + (lse1 * lse1 + lse2 * lse2) / 2, z, 1e-4);
        GradCheck.Result r = GradCheck.check(v -> new ZLoss(0.1f).compute(v.get(0), ids), List.of(logits));
        assertTrue(r.passed(), r.toString());
        // training with z-loss keeps the normalizer near zero
        Var w = Var.param(Tensor.of(Shape.of(2, 3), 5f, 5f, 5f, 5f, 5f, 5f));
        Optimizer opt = Optimizer.sgd(0.1f);
        for (int i = 0; i < 200; i++) {
            new ZLoss(0.5f).compute(w, ids).backward();
            opt.step(List.of(w));
        }
        assertTrue(Math.abs(ZLoss.logSumExp(w).value().mean().item()) < 1f);
    }

    @Test
    void fp8RoundsToTheFormatGrids() {
        assertEquals(448f, Fp8Format.E4M3.round(1e6f));
        assertEquals(-57344f, Fp8Format.E5M2.round(-1e9f));
        assertEquals(1.125f, Fp8Format.E4M3.round(1.1f));     // spacing 1/8 near 1
        assertEquals(1.0f, Fp8Format.E5M2.round(1.1f));       // spacing 1/4 near 1
        assertEquals((float) Math.scalb(1.0, -9), Fp8Format.E4M3.minSubnormal());
        assertEquals(Fp8Format.E4M3.minSubnormal(), Fp8Format.E4M3.round(0.6f * Fp8Format.E4M3.minSubnormal()));
        Tensor t = Tensor.normal(Shape.of(1000), 0, 3, new SplittableRandom(1));
        Tensor e4 = Fp8Format.E4M3.quantize(t);
        Tensor e5 = Fp8Format.E5M2.quantize(t);
        float err4 = e4.sub(t).square().mean().item();
        float err5 = e5.sub(t).square().mean().item();
        assertTrue(err4 < err5, "E4M3 is more precise: " + err4 + " vs " + err5);
        assertTrue(err4 < 1e-2f * t.square().mean().item());
        Var x = Var.param(t);
        Fp8Format.E4M3.fakeQuantize(x).sum().backward();
        assertArrayEquals(Tensor.ones(Shape.of(1000)).toArray(), x.grad().toArray());
    }

    @Test
    void gptqBeatsRoundToNearest() {
        Tensor w = Tensor.normal(Shape.of(32, 16), 0, 1, new SplittableRandom(2));
        Tensor x = activations(3, 256, 32);
        double rtn = Gptq.outputError(x, w, Gptq.roundToNearest(w, 3));
        double gptq = Gptq.outputError(x, w, new Gptq(3).quantize(w, x));
        double ordered = Gptq.outputError(x, w, new Gptq(3).actOrder(true).quantize(w, x));
        assertTrue(gptq < 0.7 * rtn, "GPTQ " + gptq + " vs RTN " + rtn);
        assertTrue(ordered < 0.7 * rtn, "act-order GPTQ " + ordered + " vs RTN " + rtn);
        // every column stays on its per-channel integer grid
        float[] q = new Gptq(3).quantize(w, x).toArray();
        float[] wv = w.toArray();
        for (int c = 0; c < 16; c++) {
            double max = 0;
            for (int r = 0; r < 32; r++) {
                max = Math.max(max, Math.abs(wv[r * 16 + c]));
            }
            double step = max / 3;
            for (int r = 0; r < 32; r++) {
                double k = q[r * 16 + c] / step;
                assertEquals(Math.rint(k), k, 1e-3);
                assertTrue(Math.abs(k) <= 3 + 1e-6);
            }
        }
    }

    @Test
    void awqProtectsSalientChannels() {
        Tensor w = Tensor.normal(Shape.of(32, 16), 0, 1, new SplittableRandom(4));
        Tensor x = activations(5, 256, 32);
        double rtn = Gptq.outputError(x, w, Gptq.roundToNearest(w, 3));
        Awq.Result r = Awq.quantize(w, x, 3);
        assertTrue(r.error() < 0.9 * rtn, "AWQ " + r.error() + " vs RTN " + rtn + " at alpha " + r.alpha());
        assertTrue(r.alpha() > 0);
        assertEquals(r.error(), Gptq.outputError(x, w, r.weight()), 1e-4 * r.error() + 1e-6);
    }

    @Test
    void smoothQuantMakesOutlierActivationsQuantizable() {
        SplittableRandom rnd = new SplittableRandom(6);
        Tensor w = Tensor.normal(Shape.of(16, 8), 0, 0.5f, rnd);
        float[] xv = new float[128 * 16];
        for (int i = 0; i < 128; i++) {
            for (int j = 0; j < 16; j++) {
                xv[i * 16 + j] = (float) rnd.nextGaussian() * (j == 3 ? 60 : 1);
            }
        }
        Tensor x = Tensor.of(Shape.of(128, 16), xv);
        double plain = SmoothQuant.quantizedError(x, w, null, 8);
        double smoothed = SmoothQuant.quantizedError(x, w, SmoothQuant.scales(x, w, 0.5f), 8);
        assertTrue(smoothed < 0.5 * plain, "smoothed " + smoothed + " vs plain " + plain);
    }

    @Test
    void lsqLearnsItsStepSize() {
        // the step gradient: inside the range round(v) - v, outside the clip level
        Var w = Var.param(Tensor.of(0.24f, -0.9f, 5f));
        Var s = Var.param(Tensor.of(Shape.of(1), 0.5f));
        LsqDense.quantize(w, s, 2).sum().backward();
        float g = (float) (1 / Math.sqrt(3 * 1.0));
        // 2 bits: levels {-1, 0, 1}; v = (0.48, -1.8, 10): only the first is inside
        assertEquals((-0.48f - 1f + 1f) * g, s.grad().item(), 1e-4f);
        assertArrayEquals(new float[] {1, 0, 0}, w.grad().toArray());

        // 2-bit regression: learning the step beats freezing it at its initial value
        SplittableRandom rnd = new SplittableRandom(7);
        Tensor x = Tensor.normal(Shape.of(128, 8), 0, 1, rnd);
        Tensor teacher = Tensor.normal(Shape.of(8, 4), 0, 1, rnd);
        Tensor y = x.matmul(teacher);
        float learned = trainLsq(x, y, teacher, true);
        float frozen = trainLsq(x, y, teacher, false);
        assertTrue(learned < 0.8f * frozen, "learned step " + learned + " vs frozen " + frozen);
    }

    private static float trainLsq(Tensor x, Tensor y, Tensor init, boolean learnStep) {
        LsqDense layer = new LsqDense(init.mul(0.3f), null, Activation.IDENTITY, 2, 0);
        Optimizer opt = Optimizer.adam(0.01f);
        List<Var> params = learnStep ? layer.parameters()
                : layer.parameters().stream().filter(p -> p != layer.weightStep()).toList();
        float loss = 0;
        for (int i = 0; i < 300; i++) {
            Var l = layer.forward(Var.of(x)).sub(y).square().mean();
            loss = l.value().item();
            l.backward();
            opt.step(params);
            layer.weightStep().zeroGrad();
        }
        return loss;
    }
}
