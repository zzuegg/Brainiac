package io.github.zzuegg.brainiac.experimental.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class DataPrunerTest {

    @Test
    void scoresEl2nAndForgettingAndKeepsTheMiddle() {
        DataPruner pruner = new DataPruner(3);
        Tensor labels = Tensor.of(0, 1, 0);
        // example 0 always right and confident; 1 right then wrong (one forgetting event); 2 never right
        pruner.record(Tensor.of(new float[][] {{20, 0}, {0, 20}, {0, 20}}), labels);
        pruner.record(Tensor.of(new float[][] {{20, 0}, {20, 0}, {0, 20}}), labels);
        double[] el2n = pruner.el2n();
        assertEquals(0, el2n[0], 1e-6);
        assertEquals(Math.sqrt(2) / 2, el2n[1], 1e-6);
        assertEquals(Math.sqrt(2), el2n[2], 1e-6);
        assertArrayEquals(new double[] {0, 1, 2}, pruner.forgettingScores());
        assertEquals(2, pruner.recordings());
        assertArrayEquals(new int[] {1}, DataPruner.keep(el2n, 0.34, 0.34), "drop easiest and hardest third");
        assertArrayEquals(new int[] {1, 2}, DataPruner.keep(el2n, 0.67, 0));
        assertEquals(5, DataPruner.random(10, 0.5, 1).length);
        assertArrayEquals(new float[] {5, 6, 1, 2}, DataPruner.select(Tensor.of(new float[][] {{1, 2}, {3, 4},
            {5, 6}}), new int[] {2, 0}).toArray());
    }
}
