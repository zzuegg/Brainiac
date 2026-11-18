package io.github.zzuegg.brainiac.experiments.generative;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.diffusion.ConditionalDenoiser;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import io.github.zzuegg.brainiac.experimental.nn.diffusion.UNetDenoiser;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class UNetDenoiserTest {

    private static double[] trainAndSample(ConditionalDenoiser net, int iterations) {
        TinyDdpm ddpm = TinyDdpm.of(net, 50, true, 3e-3f, 0.15f, 2);
        SplittableRandom r = new SplittableRandom(3);
        for (int i = 0; i < iterations; i++) {
            float[][] x = new float[32][];
            int[] l = new int[32];
            for (int b = 0; b < 32; b++) {
                l[b] = r.nextInt(ToyImages.CLASSES);
                x[b] = ToyImages.sample(l[b], r);
            }
            ddpm.trainStep(x, l);
        }
        int[] labels = new int[24];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = i % ToyImages.CLASSES;
        }
        return TinyDdpmTest.score(ddpm.sampleDdim(labels, 20, 3f, 7, null), labels);
    }

    @Test
    void unetDenoiserDropsIntoTinyDdpm() {
        double[] unet = trainAndSample(UNetDenoiser.of(ToyImages.SIDE, ToyImages.CLASSES, 8, 2, 1)
                .timeScale(1f / 50), 250);
        double[] mlp = trainAndSample(MlpDenoiser.of(ToyImages.PIXELS, ToyImages.CLASSES, 128, 2, 1)
                .timeScale(1f / 50), 250);
        double[] untrained = trainAndSample(UNetDenoiser.of(ToyImages.SIDE, ToyImages.CLASSES, 8, 2, 1)
                .timeScale(1f / 50), 0);
        System.out.printf("DDPM nearest-real distance / class match after 250 steps: U-Net %.3f/%.3f, MLP %.3f/%.3f, "
                + "untrained %.3f%n", unet[0], unet[1], mlp[0], mlp[1], untrained[0]);
        assertTrue(unet[0] < untrained[0] / 2, "the U-Net denoiser learns the images");
    }
}
