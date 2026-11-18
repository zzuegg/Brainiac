package io.github.zzuegg.brainiac.experimental.restore;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DegradedImages;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class RestoreBlocksTest {

    @Test
    void superResolutionShapesAndCost() {
        Var x = Var.of(Tensor.normal(Shape.of(2, 1, 8, 8), 0, 1, new SplittableRandom(1)));
        long srcnn = 0;
        long espcn = 0;
        for (SuperResolution.Kind k : SuperResolution.Kind.values()) {
            SuperResolution sr = new SuperResolution(k, 2, 16, 2, true, new SplittableRandom(2));
            assertEquals(Shape.of(2, 1, 16, 16), sr.forward(x).shape(), k.name());
            if (k == SuperResolution.Kind.SRCNN) {
                srcnn = sr.macs(8, 8);
            } else if (k == SuperResolution.Kind.ESPCN) {
                espcn = sr.macs(8, 8);
            }
        }
        System.out.printf("MACs on 8x8 -> 16x16: SRCNN %d, ESPCN %d%n", srcnn, espcn);
        assertTrue(espcn * 3 < srcnn, "sub-pixel convolution is much cheaper");
    }

    @Test
    void dataStepInvertsANoiselessBlur() {
        int side = 16;
        float[] clean = DegradedImages.scenes(1, side, 3).get(0);
        float[] k = DegradedImages.gaussianKernel(5, 0.7);
        float[] blurred = DegradedImages.blur(clean, side, side, k, 5);
        Tensor t = PlugAndPlay.stackTransfers(List.of(PlugAndPlay.transfer(k, 5, side, side)));
        float[] est = PlugAndPlay.wiener(DegradedImages.stack(List.of(blurred), side), t, 1e-6f).toArray();
        double p = DegradedImages.psnr(est, clean);
        System.out.printf("Wiener on a noiseless 0.7-px Gaussian blur: %.1f dB (blurred %.1f dB)%n", p,
                DegradedImages.psnr(blurred, clean));
        assertTrue(p > 40, "the FFT data step inverts the circular blur: " + p);
    }

    @Test
    void selfSupervisedSamplers() {
        Tensor x = Tensor.normal(Shape.of(2, 1, 8, 8), 0, 1, new SplittableRandom(4));
        Tensor[] m = SelfSupervisedDenoising.neighborMasks(x.shape(), new SplittableRandom(5));
        assertEquals(32f, m[0].sum().item(), 1e-4);
        assertEquals(0f, m[0].mul(m[1]).sum().item(), 1e-6);
        SelfSupervisedDenoising.BlindSpot b = SelfSupervisedDenoising.blindSpot(x, 0.2, new SplittableRandom(6));
        assertTrue(b.mask().sum().item() > 5);
        float[][] pair = SelfSupervisedDenoising.medianPair(new float[81], 9, 4, new SplittableRandom(7));
        assertEquals(9, pair[0].length);
    }

    @Test
    void ffdnetTakesANoiseMap() {
        Denoisers.FfdNet f = new Denoisers.FfdNet(3, 8, true, new SplittableRandom(8));
        Var noisy = Var.of(Tensor.normal(Shape.of(2, 1, 8, 8), 0, 1, new SplittableRandom(9)));
        assertEquals(Shape.of(2, 1, 8, 8), f.denoise(noisy, 0.1f).shape());
    }
}
