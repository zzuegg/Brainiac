package io.github.zzuegg.brainiac.experimental.timeseries;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

/** Shapes, gradients and basic behaviour of the learnable front ends and DPRNN. */
class AudioBlocksTest {

    private static Var tones(int batch, int samples, double hz) {
        float[] x = new float[batch * samples];
        SplittableRandom r = new SplittableRandom(1);
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < samples; t++) {
                x[b * samples + t] = (float) (0.5 * Math.sin(2 * Math.PI * hz * t / 8000) + 0.01 * r.nextGaussian());
            }
        }
        return Var.of(Tensor.of(Shape.of(batch, samples), x));
    }

    private static double gradNorm(Layer l) {
        double s = 0;
        for (Var p : l.parameters()) {
            for (float g : p.grad().toArray()) {
                s += g * g;
            }
        }
        return Math.sqrt(s);
    }

    @Test
    void sincFilterbankPicksTheToneBand() {
        SincFilterbank sinc = new SincFilterbank(16, 65, 2, 128, 8000);
        Var e = sinc.energies(tones(2, 1024, 1000));
        assertEquals(Shape.of(2, 8, 16), e.shape());
        float[] v = e.value().toArray();
        int best = 0;
        for (int c = 1; c < 16; c++) {
            best = v[4 * 16 + c] > v[4 * 16 + best] ? c : best;
        }
        double[] band = sinc.cutoffs()[best];
        System.out.printf("sinc filterbank: 1000 Hz tone peaks in band %.0f-%.0f Hz%n", band[0], band[1]);
        assertTrue(band[0] <= 1000 && band[1] >= 1000, band[0] + "-" + band[1]);
        sinc.forward(tones(2, 1024, 1000), true).sum().backward();
        assertTrue(gradNorm(sinc) > 0);
    }

    @Test
    void pcenSuppressesStationaryLevels() {
        Pcen pcen = new Pcen(1, 32, 0.2);
        float[] e = new float[32];
        for (int t = 0; t < 32; t++) {
            e[t] = t == 20 ? 10f : 1f;                 // a steady floor and one onset
        }
        float[] loud = new float[32];
        for (int t = 0; t < 32; t++) {
            loud[t] = 100 * e[t];
        }
        float[] a = pcen.forward(Var.of(Tensor.of(Shape.of(1, 32, 1), e))).value().toArray();
        float[] b = pcen.forward(Var.of(Tensor.of(Shape.of(1, 32, 1), loud))).value().toArray();
        System.out.printf("pcen: floor %.3f onset %.3f; 100x louder floor %.3f onset %.3f%n", a[10], a[20], b[10],
                b[20]);
        assertTrue(a[20] > 3 * a[10], a[20] + " vs " + a[10]);
        assertTrue(Math.abs(b[10] - a[10]) < 0.1 * a[20], "gain invariance " + a[10] + " " + b[10]);
    }

    @Test
    void leafAndDprnnShapesAndGradients() {
        LearnableFrontEnd leaf = new LearnableFrontEnd(8, 33, 2, 128, 8, 8000);
        Var y = leaf.forward(tones(2, 1024, 500), true);
        assertEquals(Shape.of(2, 8, 8), y.shape());
        y.sum().backward();
        assertTrue(gradNorm(leaf) > 0);
        DualPathRnn net = new DualPathRnn(16, 16, 8, 8, 1, 2, 512, new SplittableRandom(3));
        Var out = net.forward(tones(2, 512, 300), true);
        assertEquals(Shape.of(2, 2, 512), out.shape());
        out.square().mean().backward();
        System.out.printf("leaf and dprnn: grad norms %.3g and %.3g%n", gradNorm(leaf), gradNorm(net));
        assertTrue(gradNorm(net) > 0);
    }
}
