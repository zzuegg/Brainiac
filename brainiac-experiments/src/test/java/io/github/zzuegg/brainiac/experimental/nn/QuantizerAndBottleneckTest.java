package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class QuantizerAndBottleneckTest {

    static Tensor clusters(int n, SplittableRandom r) {
        float[] v = new float[n * 2];
        for (int i = 0; i < n; i++) {
            double a = 2 * Math.PI * r.nextInt(8) / 8;
            v[2 * i] = (float) (Math.cos(a) + 0.02 * r.nextGaussian());
            v[2 * i + 1] = (float) (Math.sin(a) + 0.02 * r.nextGaussian());
        }
        return Tensor.of(Shape.of(n, 2), v);
    }

    static double error(Tensor x, Tensor q) {
        return x.sub(q).square().sum(1).sqrt().mean().item();
    }

    @Test
    void straightThroughPassesTheGradientUnchanged() {
        VectorQuantizer vq = VectorQuantizer.of(4, 3, 0.25f, 1);
        Var x = Var.param(Tensor.normal(Shape.of(5, 3), 0, 1, new SplittableRandom(2)));
        Tensor w = Tensor.normal(Shape.of(5, 3), 0, 1, new SplittableRandom(3));
        Var out = vq.forward(x);
        int[] codes = vq.nearest(x.value());
        assertArrayEquals(vq.lookup(codes).value().toArray(), out.value().toArray(), 1e-6f);
        out.mul(w).sum().backward();
        assertArrayEquals(w.toArray(), x.grad().toArray(), 1e-6f);
    }

    @Test
    void codebookLearnsClustersWithDeadCodeRestarts() {
        SplittableRandom r = new SplittableRandom(4);
        VectorQuantizer vq = VectorQuantizer.of(8, 2, 0.25f, 5);
        Optimizer opt = Optimizer.adam(0.05f);
        Tensor probe = clusters(256, r);
        double before = error(probe, vq.forward(Var.of(probe)).value());
        vq.initialize(clusters(256, r));
        for (int step = 0; step < 150; step++) {
            Tensor x = clusters(64, r);
            vq.quantize(Var.of(x)).loss().backward();
            opt.step(vq.parameters());
            if (step % 25 == 24 && step < 100) {
                vq.restartDeadCodes(x);
            }
        }
        double after = error(probe, vq.forward(Var.of(probe)).value());
        assertTrue(after < 0.15 && after < before / 5, "quantization error " + before + " -> " + after);
    }

    @Test
    void residualStagesRefineTheCode() {
        SplittableRandom r = new SplittableRandom(6);
        ResidualVectorQuantizer rvq = ResidualVectorQuantizer.of(3, 16, 4, 0.25f, 7);
        Optimizer opt = Optimizer.adam(0.02f);
        for (int step = 0; step < 200; step++) {
            Tensor x = Tensor.normal(Shape.of(64, 4), 0, 1, r);
            rvq.quantizeWithDropout(Var.of(x), r).loss().backward();
            opt.step(rvq.parameters());
            if (step % 40 == 39) {
                rvq.restartDeadCodes(x);
            }
        }
        Tensor probe = Tensor.normal(Shape.of(256, 4), 0, 1, r);
        double[] err = new double[3];
        for (int used = 1; used <= 3; used++) {
            ResidualVectorQuantizer.Result res = rvq.quantize(Var.of(probe), used);
            err[used - 1] = error(probe, res.output().value());
            assertArrayEquals(res.output().value().toArray(), rvq.decode(res.codes()).toArray(), 1e-5f);
        }
        assertTrue(err[1] < err[0] * 0.85 && err[2] < err[1] * 0.85, java.util.Arrays.toString(err));
    }

    @Test
    void klMatchesTheClosedFormAndFreeBitsClamp() {
        // mean 1, logvar 0 -> KL 0.5; mean 0, logvar 0 -> 0
        Tensor stats = Tensor.of(Shape.of(1, 4), 1, 0, 0, 0);
        assertEquals(0.5f, VariationalBottleneck.kl(Var.of(stats), 0).value().item(), 1e-6f);
        assertEquals(0.5f + 0.2f, VariationalBottleneck.kl(Var.of(stats), 0.2f).value().item(), 1e-6f);
        GradCheck.Result g = GradCheck.check(v -> VariationalBottleneck.kl(v.get(0), 0),
                List.of(Tensor.normal(Shape.of(3, 4), 0, 1, new SplittableRandom(1))));
        assertTrue(g.passed(), g.toString());
        // a dimension under its free bits gets no gradient
        Var p = Var.param(Tensor.of(Shape.of(1, 4), 1, 0.01f, 0, 0));
        VariationalBottleneck.kl(p, 0.2f).backward();
        float[] grad = p.grad().toArray();
        assertEquals(1f, grad[0], 1e-6f);
        assertEquals(0f, grad[1], 0);
    }

    @Test
    void reparameterizedSamplesFollowTheGaussian() {
        VariationalBottleneck vb = VariationalBottleneck.of(1, 3);
        float[] s = new float[2000 * 2];
        for (int i = 0; i < 2000; i++) {
            s[2 * i] = 2;
            s[2 * i + 1] = (float) Math.log(0.25);   // std 0.5
        }
        Tensor z = vb.forward(Var.of(Tensor.of(Shape.of(2000, 2), s)), true).value();
        double mean = z.mean().item();
        double std = Math.sqrt(z.sub((float) mean).square().mean().item());
        assertEquals(2, mean, 0.05);
        assertEquals(0.5, std, 0.05);
        assertEquals(2f, vb.forward(Var.of(Tensor.of(Shape.of(1, 2), 2, 0))).value().item(), 0);
    }
}
