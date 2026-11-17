package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GaussianSplats2d;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class NeuralFieldsTest {
    static final int SIDE = 24;
    static final float[] PICTURE = NeuralFields.picture(SIDE, 6);

    @Test
    void fourierFeaturesBeatRawCoordinates() {
        double raw = NeuralFields.fit(NeuralFields.coordinateMlp(0, 0, 1), PICTURE, SIDE, 300, 5e-3f);
        double fourier = NeuralFields.fit(NeuralFields.coordinateMlp(64, 3f, 1), PICTURE, SIDE, 300, 5e-3f);
        assertTrue(fourier > raw + 4 && fourier > 20, "PSNR raw " + raw + " dB vs Fourier features " + fourier + " dB");
    }

    @Test
    void gaussianSplatsFitThePicture() {
        GaussianSplats2d splats = GaussianSplats2d.of(48, 1, 2);
        double before = NeuralFields.psnr(splats.forward(Var.of(GaussianSplats2d.grid(SIDE, SIDE))).value().toArray(),
                PICTURE);
        double after = NeuralFields.fit(splats, PICTURE, SIDE, 300, 2e-2f);
        assertTrue(after > before + 6 && after > 15, "PSNR " + before + " -> " + after);
    }
}
