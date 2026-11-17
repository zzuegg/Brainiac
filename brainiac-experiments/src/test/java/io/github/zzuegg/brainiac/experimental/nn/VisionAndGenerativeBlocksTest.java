package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class VisionAndGenerativeBlocksTest {

    private static Tensor randn(long seed, int... dims) {
        return Tensor.normal(Shape.of(dims), 0, 1, new SplittableRandom(seed));
    }

    @Test
    void patchifyCutsRowMajorPatches() {
        float[] img = new float[16];
        for (int i = 0; i < 16; i++) {
            img[i] = i;
        }
        float[] p = PatchEmbedding.patchify(Var.of(Tensor.of(Shape.of(1, 1, 4, 4), img)), 2).value().toArray();
        assertArrayEquals(new float[] {0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15}, p);
    }

    @Test
    void visionModelsWorkInSequentialModels() {
        Tensor x = randn(1, 2, 1, 8, 8);
        Model vit = Model.sequential().input(1, 8, 8).add(VisionTransformer.of(4, 16, 1, 2).classes(3)).build();
        assertEquals(Shape.of(2, 3), vit.forward(Var.of(x)).shape());
        Model rope = Model.sequential().input(1, 8, 8)
                .add(VisionTransformer.of(2, 16, 1, 2).rope2d(true).classToken(false).classes(3)).build();
        assertEquals(Shape.of(2, 3), rope.forward(Var.of(x)).shape());
        Model mixer = Model.sequential().input(1, 8, 8).add(PatchEmbedding.spec(2, 8, false))
                .add(MlpMixerBlock.spec(0.5f, 2f)).build();
        assertEquals(Shape.of(2, 16, 8), mixer.forward(Var.of(x)).shape());
        Model convnext = Model.sequential().input(1, 8, 8).conv2d(4, 1, io.github.zzuegg.brainiac.nn.Activation.IDENTITY)
                .add(ConvNeXtBlock.spec(3)).add(DeformableConv2d.spec(2, 3)).build();
        assertEquals(Shape.of(2, 2, 8, 8), convnext.forward(Var.of(x)).shape());
        for (Model m : List.of(vit, rope, mixer, convnext)) {
            m.forward(Var.of(x), true).square().sum().backward();
            for (Var p : m.parameters()) {
                assertTrue(p.grad() != null, m + ": parameter without gradient " + p.shape());
            }
        }
    }

    @Test
    void deformableConvWithZeroOffsetsIsAPlainConvolutionAndOffsetsShiftSampling() {
        DeformableConv2d d = new DeformableConv2d(1, 1, 3, new SplittableRandom(2));
        Tensor x = randn(3, 1, 1, 5, 5);
        Var w = d.parameters().get(0);   // [taps, 1]
        Tensor kernel = w.value().reshape(1, 1, 3, 3);
        Var plain = io.github.zzuegg.brainiac.nn.Functional.conv(Var.of(x), Var.of(kernel), null, new int[] {1, 1},
                new int[] {1, 1}, new int[] {1, 1}, new int[] {1, 1}, 1);
        assertArrayEquals(plain.value().toArray(), d.forward(Var.of(x)).value().toArray(), 1e-5f);
        // an offset of exactly one column on every tap shifts the sampling grid one pixel right
        float[] off = new float[18 * 25];
        for (int t = 0; t < 9; t++) {
            for (int q = 0; q < 25; q++) {
                off[(2 * t + 1) * 25 + q] = 1f;
            }
        }
        float[] shifted = d.sample(Var.of(x), Var.of(Tensor.of(Shape.of(1, 18, 5, 5), off))).value().toArray();
        float[] base = plain.value().toArray();
        for (int r = 0; r < 5; r++) {
            for (int c = 0; c < 3; c++) {
                assertEquals(base[r * 5 + c + 1], shifted[r * 5 + c], 1e-4f);
            }
        }
        // fractional offsets get gradients through the bilinear weights
        Var offsets = Var.param(Tensor.full(Shape.of(1, 18, 5, 5), 0.3f));
        d.sample(Var.of(x), offsets).square().sum().backward();
        assertTrue(offsets.grad().abs().sum().item() > 0);
    }

    @Test
    void realNvpIsInvertibleAndLearnsADensity() {
        SplittableRandom rnd = new SplittableRandom(4);
        RealNvpFlow flow = new RealNvpFlow(2, 4, 16, rnd);
        // data: a stretched, shifted, correlated Gaussian
        Tensor e = Tensor.normal(Shape.of(256, 2), 0, 1, rnd);
        Tensor data = e.matmul(Tensor.of(Shape.of(2, 2), 2f, 0.5f, 0f, 0.3f)).add(Tensor.of(3f, -1f));
        float before = flow.logProb(Var.of(data)).mean().value().item();
        Optimizer opt = Optimizer.adam(0.01f);
        for (int i = 0; i < 300; i++) {
            flow.logProb(Var.of(data)).mean().neg().backward();
            opt.step(flow.parameters());
        }
        float after = flow.logProb(Var.of(data)).mean().value().item();
        assertTrue(after > before + 3, before + " -> " + after);
        // the entropy bound for this data is about -2.33 nats; the flow comes close
        assertTrue(after > -2.6f, "log-likelihood " + after);
        Tensor z = flow.forward(Var.of(data))[0].value();
        assertArrayEquals(data.toArray(), flow.inverse(z).toArray(), 1e-3f);
        Tensor samples = flow.sample(512, rnd);
        float[] mean = samples.mean(0, false).toArray();
        assertEquals(3f, mean[0], 0.4f);
        assertEquals(-1f, mean[1], 0.4f);
    }

    @Test
    void pixelCnnIsCausalAndLearnsStripes() {
        PixelCnn net = new PixelCnn(8, 1, new SplittableRandom(5));
        // causality: the logit of pixel (2, 2) must not depend on pixel (2, 2) or later pixels
        Var x = Var.param(randn(6, 1, 1, 5, 5));
        net.forward(x).slice(2, 2, 3).slice(3, 2, 3).sum().backward();
        float[] g = x.grad().toArray();
        for (int i = 2 * 5 + 2; i < 25; i++) {
            assertEquals(0f, g[i], "pixel " + i + " leaks into pixel 12");
        }
        assertTrue(Math.abs(g[2 * 5 + 1]) > 0, "the left neighbour is visible");
        // data: vertical stripes with a random phase: each row repeats the first
        SplittableRandom rnd = new SplittableRandom(7);
        float[] imgs = new float[32 * 25];
        for (int s = 0; s < 32; s++) {
            int phase = rnd.nextInt(2);
            for (int i = 0; i < 25; i++) {
                imgs[s * 25 + i] = (i % 5 + phase) % 2;
            }
        }
        Tensor data = Tensor.of(Shape.of(32, 1, 5, 5), imgs);
        float before = net.nll(data).value().item();
        Optimizer opt = Optimizer.adam(0.02f);
        for (int i = 0; i < 80; i++) {
            net.nll(data).backward();
            opt.step(net.parameters());
        }
        float after = net.nll(data).value().item();
        assertTrue(after < 0.3f * before, before + " -> " + after + " nats");
        float[] sample = net.sample(2, 5, 5, rnd).toArray();
        int alternations = 0;
        for (int i = 0; i < 2 * 25; i++) {
            if (i % 5 != 0 && sample[i] != sample[i - 1]) {
                alternations++;
            }
        }
        assertTrue(alternations >= 30, "samples are mostly stripes: " + alternations + "/40");
    }

    @Test
    void waveNetIsCausalAndPredictsAPeriodicSignal() {
        WaveNet net = new WaveNet(1, 8, 1, 3, new SplittableRandom(8));
        assertEquals(8, net.receptiveField());
        Var x = Var.param(randn(9, 1, 1, 20));
        net.forward(x).slice(2, 10, 11).sum().backward();
        float[] g = x.grad().toArray();
        for (int t = 11; t < 20; t++) {
            assertEquals(0f, g[t], "step " + t + " leaks into step 10");
        }
        assertTrue(Math.abs(g[10]) + Math.abs(g[9]) > 0);
        // next-step prediction of a period-5 pattern
        float[] pattern = {0f, 1f, 0.5f, -1f, -0.5f};
        float[] seq = new float[4 * 30];
        for (int s = 0; s < 4; s++) {
            for (int t = 0; t < 30; t++) {
                seq[s * 30 + t] = pattern[(t + s) % 5];
            }
        }
        Tensor all = Tensor.of(Shape.of(4, 1, 30), seq);
        Var in = Var.of(all.slice(2, 0, 29));
        Tensor target = all.slice(2, 1, 30);
        Optimizer opt = Optimizer.adam(0.02f);
        float first = 0;
        float last = 0;
        for (int i = 0; i < 150; i++) {
            Var loss = net.forward(in).sub(target).slice(2, 8, 29).square().mean();
            last = loss.value().item();
            first = i == 0 ? last : first;
            loss.backward();
            opt.step(net.parameters());
        }
        assertTrue(last < 0.05f * first, first + " -> " + last);
    }

    @Test
    void gradientPenaltyIsExactForALinearCritic() {
        // D(x) = x . w: grad_x D = w for every x, so the WGAN-GP penalty is (||w|| - 1)^2
        Var w = Var.param(Tensor.of(Shape.of(3, 1), 1f, 2f, 2f));
        java.util.function.UnaryOperator<Var> critic = x -> x.matmul(w);
        Tensor real = randn(10, 4, 3);
        Tensor fake = randn(11, 4, 3);
        w.setGrad(Tensor.of(Shape.of(3, 1), 7f, 7f, 7f));
        Var p = GradientPenalty.wgan(critic, List.of(w), real, fake, new SplittableRandom(12));
        assertArrayEquals(new float[] {7f, 7f, 7f}, w.grad().toArray(), "existing gradients are kept");
        w.zeroGrad();
        assertEquals(4f, p.value().item(), 1e-3f);
        p.backward();
        // d/dw (||w|| - 1)^2 = 2 (||w|| - 1) w / ||w|| = 2 * 2 * w / 3
        assertArrayEquals(new float[] {4f / 3, 8f / 3, 8f / 3}, w.grad().toArray(), 1e-2f);
        w.zeroGrad();
        Var r1 = GradientPenalty.r1(critic, List.of(w), real);
        assertEquals(9f, r1.value().item(), 1e-2f);
        // a non-linear critic: compare with the numeric norm of its input gradient
        Var w2 = Var.param(randn(13, 3, 1));
        java.util.function.UnaryOperator<Var> tanhCritic = x -> x.matmul(w2).tanh();
        float[] norms = GradientPenalty.gradientNorm(tanhCritic, List.of(w2), real, 1e-2f).value().toArray();
        float[] wv = w2.value().toArray();
        float[] rv = real.toArray();
        double wn = Math.sqrt(wv[0] * wv[0] + wv[1] * wv[1] + wv[2] * wv[2]);
        for (int i = 0; i < 4; i++) {
            double a = rv[i * 3] * wv[0] + rv[i * 3 + 1] * wv[1] + rv[i * 3 + 2] * wv[2];
            double t = Math.tanh(a);
            assertEquals((1 - t * t) * wn, norms[i], 1e-3);
        }
    }

    /** The penalty's weight gradient matches finite differences of the exact gradient norm. */
    @Test
    void gradientPenaltyWeightGradientMatchesDoubleBackprop() {
        SplittableRandom rnd = new SplittableRandom(20);
        Var w1 = Var.param(Tensor.normal(Shape.of(2, 8), 0, 1, rnd));
        Var w2 = Var.param(Tensor.normal(Shape.of(8, 1), 0, 1, rnd));
        java.util.function.UnaryOperator<Var> critic = x -> x.matmul(w1).tanh().matmul(w2);
        Tensor x = randn(21, 5, 2);
        GradientPenalty.gradientNorm(critic, List.of(w1, w2), x, 1e-3f).sub(1f).square().mean().backward();
        float[] analytic = w1.grad().toArray();
        float[] base = w1.value().toArray();
        for (int i = 0; i < base.length; i++) {
            float[] plus = base.clone();
            float[] minus = base.clone();
            plus[i] += 1e-2f;
            minus[i] -= 1e-2f;
            double numeric = (exactPenalty(critic, w1, plus, x) - exactPenalty(critic, w1, minus, x)) / 2e-2;
            assertEquals(numeric, analytic[i], 2e-2 * Math.max(1, Math.abs(numeric)), "weight " + i);
            w1.assign(Tensor.of(Shape.of(2, 8), base));
        }
    }

    private static double exactPenalty(java.util.function.UnaryOperator<Var> critic, Var w1, float[] values,
            Tensor x) {
        w1.assign(Tensor.of(Shape.of(2, 8), values));
        Var xv = Var.param(x);
        critic.apply(xv).sum().backward();
        float[] g = xv.grad().toArray();
        double total = 0;
        for (int i = 0; i < 5; i++) {
            double n = Math.sqrt(g[2 * i] * g[2 * i] + g[2 * i + 1] * g[2 * i + 1]);
            total += (n - 1) * (n - 1) / 5;
        }
        return total;
    }
}
