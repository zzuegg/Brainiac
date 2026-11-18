package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class PermutationInvariantLossTest {

    @Test
    void siSnrIsScaleInvariantAndUpitIgnoresOutputOrder() {
        SplittableRandom r = new SplittableRandom(3);
        int n = 256;
        float[] refs = new float[2 * n];
        float[] swapped = new float[2 * n];
        for (int i = 0; i < n; i++) {
            refs[i] = (float) Math.sin(0.1 * i);
            refs[n + i] = (float) r.nextGaussian();
        }
        for (int i = 0; i < n; i++) {
            // outputs in the wrong order, rescaled, with a little noise
            swapped[i] = -3 * refs[n + i] + 0.05f * (float) r.nextGaussian();
            swapped[n + i] = 0.5f * refs[i] + 0.05f * (float) r.nextGaussian();
        }
        Var ref = Var.of(Tensor.of(Shape.of(1, 2, n), refs));
        double loss = PermutationInvariantLoss.loss(Var.of(Tensor.of(Shape.of(1, 2, n), swapped)), ref).value().item();
        double direct = PermutationInvariantLoss.siSnr(slice(swapped, 1, n), slice(refs, 0, n));
        double plain = (PermutationInvariantLoss.siSnr(slice(swapped, 1, n), slice(refs, 0, n))
                + PermutationInvariantLoss.siSnr(slice(swapped, 0, n), slice(refs, 1, n))) / 2;
        System.out.printf("uPIT loss %.2f dB, best-assignment SI-SNR %.2f dB (source 1 alone %.2f dB)%n", loss,
                plain, direct);
        assertEquals(-plain, loss, 1e-2);
        assertTrue(plain > 15, "a rescaled, reordered estimate scores high: " + plain);
    }

    private static float[] slice(float[] x, int i, int n) {
        float[] y = new float[n];
        System.arraycopy(x, i * n, y, 0, n);
        return y;
    }
}
