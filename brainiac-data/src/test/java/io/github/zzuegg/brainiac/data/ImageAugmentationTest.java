package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ImageAugmentationTest {

    private static final int C = 3;
    private static final int H = 12;
    private static final int W = 16;

    private static float[] image(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] f = new float[5 + C * H * W];   // an offset in front
        for (int i = 5; i < f.length; i++) {
            f[i] = (float) r.nextDouble();
        }
        return f;
    }

    private static void inRange(float[] f) {
        for (int i = 5; i < f.length; i++) {
            assertTrue(f[i] >= 0 && f[i] <= 1, "value " + f[i]);
        }
    }

    @Test
    void flipTwiceIsIdentityAndMirrors() {
        float[] f = image(1);
        float[] orig = f.clone();
        Augmentation flip = Augmentation.horizontalFlip(C, H, W);
        for (long seed = 0; seed < 4; seed++) {
            float[] g = orig.clone();
            flip.apply(g, 5, new SplittableRandom(seed));
            assertTrue(g[5] == orig[5] || g[5] == orig[5 + W - 1]);
        }
        float[] g = orig.clone();
        flip.apply(g, 5, new SplittableRandom(3));
        flip.apply(g, 5, new SplittableRandom(3));
        assertArrayEquals(orig, g);
    }

    @Test
    void cropKeepsConstantImagesAndRange() {
        float[] flat = new float[5 + C * H * W];
        java.util.Arrays.fill(flat, 5, flat.length, 0.4f);
        Augmentation crop = Augmentation.randomResizedCrop(C, H, W, 0.08, 1);
        crop.apply(flat, 5, new SplittableRandom(2));
        for (int i = 5; i < flat.length; i++) {
            assertEquals(0.4f, flat[i], 1e-6f);
        }
        float[] f = image(2);
        crop.apply(f, 5, new SplittableRandom(4));
        inRange(f);
        assertEquals(0f, f[0]);
    }

    @Test
    void jitterErasingAndRandAugmentStayInRange() {
        Augmentation all = Augmentation.colorJitter(C, H, W, 0.4f, 0.4f, 0.4f)
                .then(Augmentation.randomErasing(C, H, W, 1, 0.1, 0.3))
                .then(Augmentation.randAugment(C, H, W, 2, 9));
        for (long s = 0; s < 20; s++) {
            float[] f = image(s);
            all.apply(f, 5, new SplittableRandom(s));
            inRange(f);
        }
        float[] a = image(7);
        float[] b = image(7);
        all.apply(a, 5, new SplittableRandom(11));
        all.apply(b, 5, new SplittableRandom(11));
        assertArrayEquals(a, b);
    }

    @Test
    void erasingZeroesARectangle() {
        float[] f = new float[5 + C * H * W];
        java.util.Arrays.fill(f, 5, f.length, 1f);
        Augmentation.randomErasing(C, H, W, 1, 0.1, 0.2).apply(f, 5, new SplittableRandom(9));
        int zeros = 0;
        for (int i = 5; i < 5 + H * W; i++) {
            zeros += f[i] == 0 ? 1 : 0;
        }
        assertTrue(zeros >= 0.05 * H * W && zeros <= 0.3 * H * W, "zeros " + zeros);
    }
}
