package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.vision.Fft2d;
import io.github.zzuegg.brainiac.experimental.vision.SpectralConv;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SpectralConvolutionTest {

    @Test
    void fftConvolutionCrossoverAndSpectralPooling() {
        List<SpectralConvolution.TimingRow> rows = SpectralConvolution.timing(new int[] {16, 32, 64},
                new int[] {3, 7, 11, 15}, 4, 3, 1);
        StringBuilder line = new StringBuilder("11.5.17 conv ms direct/im2col/fft/core (size,k):");
        double maxErr = 0;
        for (SpectralConvolution.TimingRow r : rows) {
            line.append(String.format(" %d,%d %.2f/%.2f/%.2f/%.2f;", r.size(), r.kernel(), r.directMs(),
                    r.im2colMs(), r.fftMs(), r.coreMs()));
            maxErr = Math.max(maxErr, r.fftError());
        }
        line.append(String.format(" winograd 32,3 %.2f; max fft err %.1e", rows.get(4).winogradMs(), maxErr));
        System.out.println(line);
        assertTrue(maxErr < 1e-3, line.toString());

        List<SpectralConvolution.PoolingRow> pool = SpectralConvolution.pooling(200, 3);
        StringBuilder p = new StringBuilder("11.5.17 ShapeImages CNN accuracy by pooling:");
        for (SpectralConvolution.PoolingRow r : pool) {
            p.append(String.format(" %s %.3f (%.0f ms);", r.pooling(), r.accuracy(), r.trainMs()));
        }
        System.out.println(p);
        assertTrue(pool.get(2).accuracy() > 0.3, p.toString());
    }

    @Test
    void fftConvGradientAndTransforms() {
        SplittableRandom r = new SplittableRandom(5);
        Tensor x = Tensor.normal(Shape.of(1, 2, 5, 5), 0, 1, r);
        Tensor w = Tensor.normal(Shape.of(2, 2, 3, 3), 0, 1, r);
        GradCheck.Result g = GradCheck.check(v -> SpectralConv.conv(v.get(0), v.get(1), 1).square().sum(),
                List.of(x, w));
        assertTrue(g.passed(), g.toString());
        // 2-D FFT round trip and agreement of the Var transform with the array one
        float[] img = x.slice(0, 0, 1).toArray();
        double[][] spec = Fft2d.forward(java.util.Arrays.copyOf(img, 16), 4, 4, 4, 4);
        Var z = Fft2d.fft2d(Fft2d.complex(Var.of(Tensor.of(Shape.of(4, 4), java.util.Arrays.copyOf(img, 16)))), false);
        float[] zv = z.value().toArray();
        double err = 0;
        for (int i = 0; i < 16; i++) {
            err = Math.max(err, Math.abs(zv[2 * i] - spec[0][i]) + Math.abs(zv[2 * i + 1] - spec[1][i]));
        }
        assertTrue(err < 1e-3, "Var and array FFT differ by " + err);
    }
}
