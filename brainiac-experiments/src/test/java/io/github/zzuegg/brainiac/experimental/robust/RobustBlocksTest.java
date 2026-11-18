package io.github.zzuegg.brainiac.experimental.robust;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class RobustBlocksTest {

    @Test
    void boundsAreSoundAndCrownIsTighterThanIbp() {
        SimpleMlp net = SimpleMlp.relu(3, 6, 16, 16, 4);
        List<BoundPropagation.Affine> layers = BoundPropagation.of(net);
        SplittableRandom r = new SplittableRandom(1);
        Tensor x = Tensor.uniform(Shape.of(5, 6), -1, 1, r);
        int[] labels = {0, 1, 2, 3, 0};
        float eps = 0.05f;
        Tensor ibp = BoundPropagation.ibpMargins(layers, Var.of(x), eps, labels).value();
        Tensor crown = BoundPropagation.crown(layers, Var.of(x), eps, labels).value();
        Tensor crownIbp = BoundPropagation.crownIbp(layers, Var.of(x), eps, labels).value();
        float[] li = ibp.toArray();
        float[] lc = crown.toArray();
        double gapIbp = 0;
        double gapCrown = 0;
        // sampled margins never fall below either bound
        for (int s = 0; s < 300; s++) {
            Tensor noise = Tensor.uniform(Shape.of(5, 6), -eps, eps, r);
            float[] z = BoundPropagation.forward(layers, Var.of(x.add(noise))).value().toArray();
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 4; j++) {
                    double m = z[i * 4 + labels[i]] - z[i * 4 + j];
                    assertTrue(m >= li[i * 4 + j] - 1e-4, "IBP unsound");
                    assertTrue(m >= lc[i * 4 + j] - 1e-4, "CROWN unsound");
                }
            }
        }
        for (int k = 0; k < li.length; k++) {
            gapIbp -= li[k];
            gapCrown -= lc[k];
        }
        System.out.printf("Bounds 6-16-16-4 eps 0.05: summed margin lower bounds IBP %.3f, CROWN-IBP %.3f, CROWN %.3f%n",
                -gapIbp, sum(crownIbp), -gapCrown);
        assertTrue(gapCrown < gapIbp, "CROWN should be tighter");
    }

    private static double sum(Tensor t) {
        double s = 0;
        for (float v : t.toArray()) {
            s += v;
        }
        return s;
    }

    @Test
    void convolutionUnrollsToItsDenseMatrix() {
        SplittableRandom r = new SplittableRandom(2);
        Var k = Var.param(Tensor.normal(Shape.of(3, 2, 3, 3), 0, 0.3f, r));
        Var b = Var.param(Tensor.normal(Shape.of(3), 0, 0.3f, r));
        Tensor x = Tensor.normal(Shape.of(4, 2, 5, 5), 0, 1, r);
        Tensor conv = Functional.conv(Var.of(x), k, b, new int[] {1, 1}, new int[] {1, 1}, new int[] {1, 1},
                new int[] {1, 1}, 1).value();
        BoundPropagation.Affine a = BoundPropagation.conv(k, b, 2, 5, 5, 1, 1);
        Tensor dense = Var.of(x.reshape(4, 50)).matmul(a.weight()).add(a.bias()).value();
        float[] c = conv.toArray();
        float[] d = dense.toArray();
        assertEquals(c.length, d.length);
        for (int i = 0; i < c.length; i++) {
            assertEquals(c[i], d[i], 1e-4);
        }
    }

    @Test
    void lipschitzLayersDoNotExpandDistances() {
        SplittableRandom r = new SplittableRandom(3);
        Layer[] layers = {new LipschitzLayers.BjorckDense(16, 16, 20, r), new LipschitzLayers.AolDense(16, 8, r),
            new LipschitzLayers.GroupSort()};
        for (Layer layer : layers) {
            double worst = 0;
            for (int t = 0; t < 50; t++) {
                Tensor a = Tensor.normal(Shape.of(1, 16), 0, 1, r);
                Tensor b = a.add(Tensor.normal(Shape.of(1, 16), 0, 0.1f, r));
                double num = norm(layer.forward(Var.of(a)).value().sub(layer.forward(Var.of(b)).value()));
                worst = Math.max(worst, num / norm(a.sub(b)));
            }
            System.out.printf("Lipschitz %-20s largest empirical ratio %.4f%n", layer.getClass().getSimpleName(), worst);
            assertTrue(worst <= 1.001, layer + " expands distances: " + worst);
        }
        LipschitzLayers.AolConv conv = new LipschitzLayers.AolConv(2, 4, 3, r);
        double worst = 0;
        for (int t = 0; t < 20; t++) {
            Tensor a = Tensor.normal(Shape.of(1, 2, 6, 6), 0, 1, r);
            Tensor b = a.add(Tensor.normal(Shape.of(1, 2, 6, 6), 0, 0.1f, r));
            worst = Math.max(worst, norm(conv.forward(Var.of(a)).value().sub(conv.forward(Var.of(b)).value()))
                    / norm(a.sub(b)));
        }
        System.out.printf("Lipschitz AolConv             largest empirical ratio %.4f%n", worst);
        assertTrue(worst <= 1.001, "AolConv expands distances: " + worst);
        Tensor w = ((LipschitzLayers.BjorckDense) layers[0]).orthonormal().value();
        Tensor gram = w.transpose().matmul(w);
        float[] g = gram.toArray();
        for (int i = 0; i < 16; i++) {
            assertEquals(1, g[i * 16 + i], 1e-3, "Björck weight is orthonormal");
        }
    }

    private static double norm(Tensor t) {
        return Math.sqrt(t.square().sum().item());
    }

    @Test
    void clopperPearsonAndSmoothingCertificates() {
        double lower = RandomizedSmoothing.clopperPearsonLower(990, 1000, 0.001);
        System.out.printf("Clopper-Pearson 990/1000 at 99.9%%: %.4f%n", lower);
        assertTrue(lower > 0.97 && lower < 0.99, "lower " + lower);
        // a linear classifier sign(x0): the exact certified radius at x0 = 0.5 is 0.5
        RandomizedSmoothing s = RandomizedSmoothing.of(batch -> {
            float[] v = batch.toArray();
            int n = batch.shape().dim(0);
            int d = v.length / n;
            int[] out = new int[n];
            for (int i = 0; i < n; i++) {
                out[i] = v[i * d] > 0 ? 1 : 0;
            }
            return out;
        }, 0.25f, 500);
        RandomizedSmoothing.Certificate c = s.certify(Tensor.of(Shape.of(2), 0.5f, 0f), 2, 100, 5000, 0.001,
                new SplittableRandom(4));
        System.out.printf("Smoothing sign(x0) at 0.5, sigma 0.25: class %d radius %.3f (exact 0.5)%n", c.label(),
                c.radius());
        assertEquals(1, c.label());
        assertTrue(c.radius() > 0.4 && c.radius() <= 0.5, c.toString());
    }
}
