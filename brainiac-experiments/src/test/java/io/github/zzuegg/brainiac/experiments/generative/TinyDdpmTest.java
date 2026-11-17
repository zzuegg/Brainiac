package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.experimental.nn.diffusion.MlpDenoiser;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TinyDdpmTest {

    static TinyDdpm trained(int iterations) {
        TinyDdpm ddpm = TinyDdpm.of(MlpDenoiser.of(ToyImages.PIXELS, ToyImages.CLASSES, 128, 2, 1).timeScale(1f / 50), 50, true,
                3e-3f, 0.15f, 2);
        SplittableRandom r = new SplittableRandom(3);
        for (int i = 0; i < iterations; i++) {
            float[][] x = new float[64][];
            int[] l = new int[64];
            for (int b = 0; b < 64; b++) {
                l[b] = r.nextInt(ToyImages.CLASSES);
                x[b] = ToyImages.sample(l[b], r);
            }
            ddpm.trainStep(x, l);
        }
        return ddpm;
    }

    static double[] score(float[][] samples, int[] labels) {
        double dist = 0;
        int correct = 0;
        for (int i = 0; i < samples.length; i++) {
            double[] nn = ToyImages.nearest(samples[i]);
            dist += nn[1];
            correct += (int) nn[0] == labels[i] ? 1 : 0;
        }
        return new double[] {dist / samples.length, (double) correct / samples.length};
    }

    @Test
    void schedulesDecayToNoise() {
        TinyDdpm ddpm = TinyDdpm.of(MlpDenoiser.of(4, 0, 8, 1, 1), 100, true, 1e-3f, 0, 1);
        assertTrue(ddpm.alphaBar(0) > 0.99);
        assertTrue(ddpm.alphaBar(99) < 1e-3);
        for (int t = 1; t < 100; t++) {
            assertTrue(ddpm.alphaBar(t) < ddpm.alphaBar(t - 1));
        }
    }

    @Test
    void learnsShapesAndFollowsGuidance() {
        TinyDdpm ddpm = trained(600);
        TinyDdpm untrained = trained(0);
        int[] labels = new int[24];
        for (int i = 0; i < labels.length; i++) {
            labels[i] = i % ToyImages.CLASSES;
        }
        List<float[][]> trajectory = new ArrayList<>();
        double[] ddim = score(ddpm.sampleDdim(labels, 20, 3f, 7, trajectory), labels);
        double[] ddpmScore = score(ddpm.sampleDdpm(labels, 3f, 7, null), labels);
        double[] base = score(untrained.sampleDdim(labels, 20, 3f, 7, null), labels);
        System.out.printf("DDPM toy shapes: nearest-real distance DDIM %.3f, DDPM %.3f, untrained %.3f; class match "
                + "DDIM %.3f, DDPM %.3f (chance 0.25)%n", ddim[0], ddpmScore[0], base[0], ddim[1], ddpmScore[1]);
        assertEquals(20, trajectory.size());
        byte[] gif = TinyDdpm.denoisingGif(trajectory, ToyImages.SIDE, 4);
        assertEquals('G', gif[0]);
        assertTrue(gif.length > 1000);
        // a random 8x8 {-1,1} image is ~64 squared distance from the nearest real one, a good sample ~0
        assertTrue(ddim[0] < base[0] / 3, "DDIM " + ddim[0] + " vs untrained " + base[0]);
        assertTrue(ddpmScore[0] < base[0] / 3, "DDPM " + ddpmScore[0] + " vs untrained " + base[0]);
        assertTrue(ddim[1] > 0.5, "guided samples match their class (chance 0.25): " + ddim[1]);
    }

    @Test
    void ddimWithoutNoiseIsDeterministic() {
        TinyDdpm ddpm = trained(5);
        int[] labels = {0, 1};
        assertArrayEquals(ddpm.sampleDdim(labels, 5, 1, 9, null)[1], ddpm.sampleDdim(labels, 5, 1, 9, null)[1]);
    }
}
