package io.github.zzuegg.brainiac.experimental.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class MixingAndCurriculumTest {

    @Test
    void mixupMixesInputsAndTargetsWithTheSameWeight() {
        Tensor x = Tensor.of(new float[][] {{0, 0}, {10, 10}, {20, 20}, {30, 30}});
        Tensor y = Tensor.of(new float[][] {{1, 0}, {0, 1}, {1, 0}, {0, 1}});
        Tensor[] m = BatchMixing.mixup(x, y, 0.4f, new SplittableRandom(3));
        float[] mx = m[0].toArray();
        float[] my = m[1].toArray();
        for (int i = 0; i < 4; i++) {
            assertEquals(1, my[2 * i] + my[2 * i + 1], 1e-5, "targets stay distributions");
            assertEquals(mx[2 * i], mx[2 * i + 1], 1e-4);
        }
        double mean = 0;
        int n = 4000;
        SplittableRandom r = new SplittableRandom(1);
        for (int i = 0; i < n; i++) {
            mean += BatchMixing.beta(0.4, r) / n;
        }
        assertEquals(0.5, mean, 0.03, "Beta(a, a) has mean 1/2");
    }

    @Test
    void cutMixPastesAPatchAndMixesTargetsByArea() {
        Tensor x = Tensor.concat(0, Tensor.zeros(Shape.of(1, 1, 8, 8)), Tensor.full(Shape.of(1, 1, 8, 8), 1f));
        Tensor y = Tensor.of(new float[][] {{1, 0}, {0, 1}});
        Tensor[] m = BatchMixing.cutMix(x, y, 1, 8, 8, 1f, new SplittableRandom(5));
        float[] v = m[0].toArray();
        float[] t = m[1].toArray();
        float ones = 0;
        for (int i = 0; i < 64; i++) {
            ones += v[i];
        }
        // sample 0 (all zeros) got a patch of ones only if it was paired with sample 1
        assertEquals(t[1], ones / 64, 1e-5, "the label share equals the pasted area");
        assertEquals(1, t[0] + t[1], 1e-5);
    }

    @Test
    void curriculumStartsEasyAndWidens() {
        float[] difficulty = new float[100];
        for (int i = 0; i < 100; i++) {
            difficulty[i] = 99 - i;   // sample 99 is easiest
        }
        Curriculum c = new Curriculum(difficulty, 0.1, 1000, Curriculum.Pacing.LINEAR);
        assertEquals(10, c.available(0));
        assertEquals(55, c.available(500));
        assertEquals(100, c.available(5000));
        for (int i : c.batch(0, 50, new SplittableRandom(1))) {
            assertTrue(i >= 90, "only the easiest at the start: " + i);
        }
        Curriculum root = new Curriculum(difficulty, 0.1, 1000, Curriculum.Pacing.ROOT);
        assertTrue(root.available(250) > c.available(250));
        float[][] f = new float[100][1];
        float[][] t = new float[100][1];
        for (int i = 0; i < 100; i++) {
            f[i][0] = i;
        }
        Dataset sub = c.subset(Dataset.of(f, t), 0);
        assertEquals(10, sub.size());
        assertArrayEquals(new float[] {99}, sub.featuresOf(0));
    }
}
