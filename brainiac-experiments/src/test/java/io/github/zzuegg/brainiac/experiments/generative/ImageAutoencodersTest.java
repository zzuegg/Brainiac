package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ImageAutoencodersTest {

    static float[][] batch(int n, SplittableRandom r, int[] labels) {
        float[][] x = new float[n][];
        for (int i = 0; i < n; i++) {
            int c = r.nextInt(ToyImages.CLASSES);
            if (labels != null) {
                labels[i] = c;
            }
            x[i] = ToyImages.sample(c, r);
        }
        return x;
    }

    @Test
    void vaeLearnsAndFreeBitsKeepLatentsAlive() {
        SplittableRandom r = new SplittableRandom(1);
        ImageAutoencoders.Vae plain = ImageAutoencoders.vae(8, 0, 2);
        ImageAutoencoders.Vae free = ImageAutoencoders.vae(8, 0.5f, 2);
        double untrained = TinyDdpmTest.score(free.sample(32, 3), new int[32])[0];
        float[] first = null;
        float[] last = null;
        for (int step = 0; step < 400; step++) {
            float[][] x = batch(64, r, null);
            plain.trainStep(x);
            last = free.trainStep(x);
            if (first == null) {
                first = last;
            }
        }
        float[][] probe = batch(256, r, null);
        int activePlain = plain.activeLatents(probe);
        int activeFree = free.activeLatents(probe);
        assertTrue(last[0] < first[0] / 3, "reconstruction " + first[0] + " -> " + last[0]);
        assertTrue(activeFree >= activePlain && activeFree >= 3,
                "active latents: free bits " + activeFree + ", plain " + activePlain);
        double sampled = TinyDdpmTest.score(free.sample(32, 3), new int[32])[0];
        System.out.printf("VAE: reconstruction %.3f -> %.3f, active latents free bits %d vs plain %d, prior-sample "
                + "distance %.3f vs untrained %.3f%n", first[0], last[0], activeFree, activePlain, sampled, untrained);
        assertTrue(sampled < untrained / 2, "prior samples " + sampled + " vs untrained " + untrained);
    }

    @Test
    void vqVaeCompressesEachImageIntoOneCode() {
        SplittableRandom r = new SplittableRandom(4);
        ImageAutoencoders.VqVae vq = ImageAutoencoders.vqVae(96, 8, 5);
        vq.initialize(batch(256, r, null));
        for (int step = 0; step < 300; step++) {
            vq.trainStep(batch(64, r, null));
            if (step % 50 == 49 && step < 200) {
                vq.restart(batch(256, r, null));
            }
        }
        int[] labels = new int[64];
        float[][] probe = batch(64, r, labels);
        float[][] round = vq.decode(vq.encode(probe));
        double[] s = TinyDdpmTest.score(round, labels);
        System.out.printf("VQ-VAE: decoded codes keep the class %.3f, nearest-real distance %.3f%n", s[1], s[0]);
        assertTrue(s[1] > 0.9, "decoded codes keep the class: " + s[1]);
        assertTrue(s[0] < 8, "decoded codes are close to real images: " + s[0]);
    }
}
