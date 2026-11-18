package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class BeaconCompressorTest {

    @Test
    void eachChunkIsCompressedFromItsOwnTokensOnly() {
        BeaconCompressor c = BeaconCompressor.beacon(4, 2, 3, new SplittableRandom(1));
        SplittableRandom r = new SplittableRandom(2);
        float[] x = new float[2 * 8 * 3];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) r.nextGaussian();
        }
        float[] out = c.forward(Var.of(Tensor.of(Shape.of(2, 8, 3), x))).value().toArray();
        assertEquals(2 * 4 * 3, out.length);
        // changing the second chunk of the first sequence leaves its first chunk's beacons alone
        float[] y = x.clone();
        for (int i = 4 * 3; i < 8 * 3; i++) {
            y[i] += 1;
        }
        float[] out2 = c.forward(Var.of(Tensor.of(Shape.of(2, 8, 3), y))).value().toArray();
        assertArrayEquals(java.util.Arrays.copyOf(out, 6), java.util.Arrays.copyOf(out2, 6), 1e-5f);
        // each beacon of a chunk reads that chunk: the same input chunk in another sequence gives the same beacons
        float[] z = x.clone();
        System.arraycopy(x, 0, z, 8 * 3, 4 * 3);
        float[] out3 = c.forward(Var.of(Tensor.of(Shape.of(2, 8, 3), z))).value().toArray();
        assertArrayEquals(java.util.Arrays.copyOf(out3, 6), java.util.Arrays.copyOfRange(out3, 12, 18), 1e-5f);
        // gradients reach every parameter
        Var loss = c.forward(Var.of(Tensor.of(Shape.of(2, 8, 3), x))).square().sum();
        loss.backward();
        List<Var> ps = c.parameters();
        for (Var p : ps) {
            float s = 0;
            for (float g : p.grad().toArray()) {
                s += Math.abs(g);
            }
            assertEquals(true, s > 0, p.toString());
        }
        assertEquals(4, c.compressedLength(8));
    }
}
