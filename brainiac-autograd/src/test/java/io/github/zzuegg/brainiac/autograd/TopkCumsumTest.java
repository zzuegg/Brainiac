package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class TopkCumsumTest {

    @Test
    void topkPicksTheLargestFirst() {
        Tensor x = Tensor.of(Shape.of(2, 4), new float[] {3, -1, 7, 2, -5, -2, -9, -2});
        Tensor[] top = x.topk(3);
        assertArrayEquals(new float[] {7, 3, 2, -2, -2, -5}, top[0].toArray());
        assertArrayEquals(new float[] {2, 0, 3, 1, 3, 0}, top[1].toArray());
    }

    @Test
    void cumsumRunsAlongAnyAxis() {
        Tensor x = Tensor.of(Shape.of(2, 3), new float[] {1, 2, 3, 4, 5, 6});
        assertArrayEquals(new float[] {1, 3, 6, 4, 9, 15}, x.cumsum(1).toArray());
        assertArrayEquals(new float[] {1, 2, 3, 5, 7, 9}, x.cumsum(0).toArray());
    }
}
