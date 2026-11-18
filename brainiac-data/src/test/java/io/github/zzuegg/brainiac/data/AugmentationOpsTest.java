package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** The single operations, batch application and RandAugment without its flip. */
class AugmentationOpsTest {

    @Test
    void affineRotatesAndShifts() {
        float[] img = {1, 2, 3, 4, 5, 6, 7, 8, 9};   // one 3x3 channel
        float[] r = img.clone();
        Augmentation.affine(1, 3, 3, 90, 0, 0).apply(r, 0, new SplittableRandom(1));
        float[] cw = {3, 6, 9, 2, 5, 8, 1, 4, 7};
        float[] ccw = {7, 4, 1, 8, 5, 2, 9, 6, 3};
        assertTrue(Arrays.equals(cw, r) || Arrays.equals(ccw, r), Arrays.toString(r));
        float[] s = img.clone();
        Augmentation.affine(1, 3, 3, 0, 1, 0).apply(s, 0, new SplittableRandom(1));
        assertArrayEquals(new float[] {0, 1, 2, 0, 4, 5, 0, 7, 8}, s);
        float[] p = {0.1f, 0.6f, 0.9f};
        Augmentation.solarize(3, 0.5f).then(Augmentation.posterize(3, 1)).apply(p, 0, new SplittableRandom(1));
        assertArrayEquals(new float[] {0, 0, 0}, p, 1e-6f);
    }

    @Test
    void aBatchGetsTheSameDrawsAsOneByOne() {
        Augmentation a = Augmentation.translate(1, 4, 4, 0.5);
        float[] batch = new float[3 * 16];
        for (int i = 0; i < batch.length; i++) {
            batch[i] = i;
        }
        float[] one = batch.clone();
        a.applyBatch(batch, 16, new SplittableRandom(5));
        SplittableRandom r = new SplittableRandom(5);
        for (int k = 0; k < 3; k++) {
            a.apply(one, k * 16, r);
        }
        assertArrayEquals(one, batch);
    }

    @Test
    void randAugmentCanLeaveOutTheFlip() {
        // at magnitude 0 every operation but the flip leaves this image as it is
        float[] img = new float[16];
        for (int i = 0; i < 16; i++) {
            img[i] = (i % 4) / 256f;
        }
        boolean flipped = false;
        Augmentation without = Augmentation.randAugment(1, 4, 4, 3, 0, false);
        Augmentation with = Augmentation.randAugment(1, 4, 4, 3, 0, true);
        SplittableRandom r = new SplittableRandom(7);
        for (int k = 0; k < 50; k++) {
            float[] a = img.clone();
            without.apply(a, 0, r);
            assertArrayEquals(img, a, "trial " + k);
            float[] b = img.clone();
            with.apply(b, 0, r);
            flipped |= !Arrays.equals(img, b);
        }
        assertTrue(flipped);
        assertFalse(Arrays.equals(new float[0], img));
    }
}
