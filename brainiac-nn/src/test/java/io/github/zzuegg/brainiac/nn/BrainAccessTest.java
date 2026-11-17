package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Reading every region's activity and which neurons are active. */
class BrainAccessTest {

    @Test
    void regionsActivityAndMasks() {
        Brain b = Brain.builder()
                .region("a", 8).initialNeurons(3).activation(Activation.TANH)
                .region("b", 4).initialNeurons(4).activation(Activation.TANH)
                .connect(Brain.INPUT, "a", 1).connect("a", "b", 1).output("b").ticks(2).build()
                .create(Shape.of(5), new SplittableRandom(1));
        Var x = Var.of(Tensor.uniform(Shape.of(6, 5), -1, 1, new SplittableRandom(2)));
        Map<String, Var> regions = b.forwardRegions(x, false);
        assertEquals(List.of("a", "b"), List.copyOf(regions.keySet()));
        assertEquals(Shape.of(6, 8), regions.get("a").shape());
        assertArrayEquals(b.forward(x).value().toArray(), regions.get("b").value().toArray());
        boolean[] active = b.activeNeurons("a");
        int count = 0;
        float[] a = regions.get("a").value().toArray();
        for (int k = 0; k < 8; k++) {
            count += active[k] ? 1 : 0;
            for (int row = 0; row < 6; row++) {
                if (!active[k]) {
                    assertEquals(0, a[row * 8 + k]);
                }
            }
        }
        assertEquals(3, count);
        b.track(true);
        b.forward(x, true);
        double[] activity = b.activity("a");
        for (int k = 0; k < 8; k++) {
            assertTrue(active[k] ? activity[k] >= 0 : activity[k] == 0);
        }
    }
}
