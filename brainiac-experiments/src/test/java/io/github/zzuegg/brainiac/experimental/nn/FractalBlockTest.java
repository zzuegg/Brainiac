package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class FractalBlockTest {
    @Test
    void structureAndColumns() {
        // C columns hold 1 + 2 + ... + 2^(C-1) = 2^C - 1 units
        FractalBlock f = FractalBlock.of(3, Dense.spec(4, Activation.IDENTITY, Init.XAVIER), Shape.of(4), 0.2, 0.5, 1);
        assertEquals(2 * 7, f.parameters().size());
        f = FractalBlock.of(2, Dense.spec(4, Activation.IDENTITY, Init.XAVIER), Shape.of(4), 0.2, 0.5, 1);
        Var x = Var.of(Tensor.normal(Shape.of(2, 4), 0, 1, new SplittableRandom(2)));
        // two columns meet in one join: the full block is the mean of the two standalone columns
        float[][] cols = new float[2][];
        for (int c = 1; c <= 2; c++) {
            f.useColumn(c);
            cols[c - 1] = f.forward(x).value().toArray();
        }
        f.useColumn(0);
        float[] all = f.forward(x).value().toArray();
        float[] mean = new float[all.length];
        for (int i = 0; i < all.length; i++) {
            mean[i] = (cols[0][i] + cols[1][i]) / 2;
        }
        assertArrayEquals(mean, all, 1e-5f);
        // evaluation is deterministic; training with drop-path is not
        assertArrayEquals(all, f.forward(x, false).value().toArray(), 0);
        boolean differs = false;
        for (int t = 0; t < 10 && !differs; t++) {
            differs = !java.util.Arrays.equals(all, f.forward(x, true).value().toArray());
        }
        assertEquals(true, differs);
    }
}
