package io.github.zzuegg.brainiac.experimental.ssl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SslBlocksTest {

    private static Var gaussian(int n, int d, double scale, double shift, long seed) {
        return Var.param(Tensor.normal(Shape.of(n, d), (float) shift, (float) scale, new SplittableRandom(seed)));
    }

    @Test
    void sigregIsSmallForIsotropicGaussianAndLargeForCollapse() {
        double gauss = Sigreg.loss(gaussian(256, 16, 1, 0, 1), 64, 17, new SplittableRandom(2)).value().item();
        double collapsed = Sigreg.loss(gaussian(256, 16, 0.01, 0, 1), 64, 17, new SplittableRandom(2)).value().item();
        double shifted = Sigreg.loss(gaussian(256, 16, 1, 2, 1), 64, 17, new SplittableRandom(2)).value().item();
        System.out.printf("SIGReg: N(0,I) %.3f, collapsed %.3f, shifted %.3f%n", gauss, collapsed, shifted);
        assertTrue(gauss < 3, "a standard Gaussian passes");
        assertTrue(collapsed > 20 * gauss && shifted > 20 * gauss, "collapse and shift are penalised");
        double[] sample = new double[256];
        SplittableRandom r = new SplittableRandom(3);
        for (int i = 0; i < sample.length; i++) {
            sample[i] = r.nextGaussian();
        }
        Var one = Var.of(Tensor.of(Shape.of(256, 1), toFloat(sample)));
        assertEquals(Sigreg.statistic(sample, 17), Sigreg.eppsPulley(one, 17).value().item(), 1e-3,
                "the Var and double statistics agree");
    }

    private static float[] toFloat(double[] x) {
        float[] f = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            f[i] = (float) x[i];
        }
        return f;
    }

    @Test
    void sigregGradientSpreadsACollapsedBatch() {
        Var z = gaussian(128, 8, 0.05, 0, 4);
        SplittableRandom r = new SplittableRandom(5);
        double before = Sigreg.loss(z, 32, 17, r).value().item();
        for (int i = 0; i < 200; i++) {
            Var l = Sigreg.loss(z, 32, 17, r);
            List<Var> g = Var.gradients(l, List.of(z), false);
            z.assign(z.value().sub(g.get(0).value().mul(0.05f)));
        }
        double after = Sigreg.loss(z, 32, 17, new SplittableRandom(6)).value().item();
        System.out.printf("SIGReg descent on a collapsed batch: %.2f -> %.3f%n", before, after);
        assertTrue(after < before / 5);
    }

    @Test
    void koleoPrefersSpreadBatchesAndSimSiamStopsGradients() {
        double spread = KoLeo.loss(gaussian(64, 8, 1, 0, 7)).value().item();
        double clumped = KoLeo.loss(gaussian(64, 8, 0.05, 3, 7)).value().item();
        assertTrue(spread < clumped, "spread " + spread + " clumped " + clumped);
        assertTrue(KoLeo.uniformity(gaussian(64, 8, 1, 0, 7).value(), 2)
                < KoLeo.uniformity(gaussian(64, 8, 0.05, 3, 7).value(), 2));
        Var p = gaussian(8, 4, 1, 0, 8);
        Var z = gaussian(8, 4, 1, 0, 9);
        Tensor gz = Var.gradients(SimSiam.loss(p, z, p, z, true), List.of(z), false).get(0).value();
        assertEquals(0, gz.abs().sum().item(), 1e-9, "the targets get no gradient");
    }

    @Test
    void swavCodesAreBalancedAndMocoQueueCycles() {
        SwappedPrediction swav = new SwappedPrediction(8, 4, 0.1f, 0.05, 3, 1);
        // every row prefers prototype 0, yet the codes must spread the batch over all four
        Tensor scores = Tensor.of(Shape.of(8, 4), 0.9f, 0.1f, 0.0f, 0.0f, 0.8f, 0.2f, 0.1f, 0f, 0.9f, 0.0f, 0.2f, 0.1f,
                0.7f, 0.1f, 0.0f, 0.3f, 0.9f, 0.3f, 0.1f, 0.0f, 0.8f, 0.0f, 0.4f, 0.1f, 0.9f, 0.1f, 0.1f, 0.2f, 0.6f,
                0.2f, 0.1f, 0.3f);
        float[] q = swav.codes(scores).toArray();
        double[] column = new double[4];
        for (int i = 0; i < 8; i++) {
            for (int k = 0; k < 4; k++) {
                column[k] += q[i * 4 + k] / 8;
            }
        }
        System.out.println("SwAV code mass per prototype (arg-max would give 1, 0, 0, 0): "
                + java.util.Arrays.toString(column));
        for (double c : column) {
            assertTrue(c > 0.1 && c < 0.5, "near equipartition: " + java.util.Arrays.toString(column));
        }
        MomentumContrast moco = new MomentumContrast(4, 6, 0.2f, 1);
        moco.enqueue(Tensor.ones(Shape.of(4, 4)));
        moco.enqueue(Tensor.ones(Shape.of(4, 4)));
        assertEquals(6, moco.filled());
        float l = moco.loss(Var.param(Tensor.ones(Shape.of(2, 4))), Tensor.ones(Shape.of(2, 4))).value().item();
        assertTrue(Float.isFinite(l));
    }
}
