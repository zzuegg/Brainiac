package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class CreatedTest {

    @Test
    void releasingWhatAComputationMadeLeavesViewsOfEarlierTensors() {
        Tensor weight = Tensor.of(Shape.of(2, 2), 1, 2, 3, 4);
        Tensor[] out = new Tensor[1];
        List<Tensor> made = Tensor.created(() -> out[0] = weight.reshape(4).mul(2f));
        Tensor.releaseCreated(made, List.of(out[0]));
        assertArrayEquals(new float[] {1, 2, 3, 4}, weight.toArray());
        assertArrayEquals(new float[] {2, 4, 6, 8}, out[0].toArray());
    }

    @Test
    void tensorsWithoutValuesHaveAShapeOnly() {
        Tensor t = Tensor.withoutValues(() -> Tensor.normal(Shape.of(1000, 1000), 0, 1, new SplittableRandom(1)));
        assertEquals(Shape.of(1000, 1000), t.shape());
        Tensor z = Tensor.withoutValues(() -> Tensor.zeros(Shape.of(3)));
        assertEquals(Shape.of(3), z.shape());
        assertEquals(Shape.of(3), Tensor.zeros(Shape.of(3)).shape());
    }
}
