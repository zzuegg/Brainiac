package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

class DatasetTest {

    private static Dataset counting(int n) {
        float[][] x = new float[n][];
        float[][] y = new float[n][];
        for (int i = 0; i < n; i++) {
            x[i] = new float[] {i, -i};
            y[i] = new float[] {i * 10};
        }
        return Dataset.of(x, y);
    }

    @Test
    void storesRowsAndExposesTensors() {
        Dataset d = counting(3);
        assertEquals(3, d.size());
        assertEquals(2, d.featureCount());
        assertEquals(1, d.targetCount());
        assertEquals(Shape.of(3, 2), d.features().shape());
        assertArrayEquals(new float[] {0, 10, 20}, d.targets().toArray());
        assertArrayEquals(new float[] {2, -2}, d.featuresOf(2));
        assertArrayEquals(new float[] {10}, d.targetsOf(1));
    }

    @Test
    void rejectsInconsistentRows() {
        assertThrows(IllegalArgumentException.class, () -> Dataset.of(new float[][] {{1}}, new float[][] {{1}, {2}}));
        assertThrows(IllegalArgumentException.class, () -> Dataset.of(new float[][] {{1}, {1, 2}}, new float[][] {{1}, {2}}));
    }

    @Test
    void shufflesReproduciblyAndKeepsPairsTogether() {
        Dataset d = counting(20);
        Dataset a = d.shuffle(1);
        Dataset b = d.shuffle(1);
        assertArrayEquals(a.features().toArray(), b.features().toArray());
        assertNotEquals(Arrays.toString(d.features().toArray()), Arrays.toString(a.features().toArray()));
        for (int i = 0; i < a.size(); i++) {
            assertEquals(a.featuresOf(i)[0] * 10, a.targetsOf(i)[0]);
        }
    }

    @Test
    void splitsAndSlices() {
        Dataset.Split s = counting(10).split(0.8);
        assertEquals(8, s.train().size());
        assertEquals(2, s.test().size());
        assertArrayEquals(new float[] {8, -8}, s.test().featuresOf(0));
        assertThrows(IllegalArgumentException.class, () -> counting(10).split(1.0));
        assertThrows(IllegalArgumentException.class, () -> counting(2).split(0.1));
    }

    @Test
    void batchesWithASmallerLastBatch() {
        List<Batch> batches = counting(5).batches(2);
        assertEquals(3, batches.size());
        assertEquals(2, batches.get(0).size());
        assertEquals(1, batches.get(2).size());
        assertArrayEquals(new float[] {4, -4}, batches.get(2).features().toArray());
        assertArrayEquals(new float[] {40}, batches.get(2).targets().toArray());
    }

    @Test
    void standardizesWithStatisticsFromTraining() {
        Dataset train = Dataset.of(new float[][] {{1, 5}, {3, 5}}, new float[][] {{0}, {1}});
        Standardizer s = Standardizer.fit(train);
        assertArrayEquals(new float[] {2, 5}, s.mean());
        assertArrayEquals(new float[] {1, 1}, s.std()); // constant column keeps scale 1
        assertArrayEquals(new float[] {-1, 0, 1, 0}, train.standardize(s).features().toArray());
        assertArrayEquals(new float[] {3, 1}, s.apply(new float[] {5, 6}));
    }
}
