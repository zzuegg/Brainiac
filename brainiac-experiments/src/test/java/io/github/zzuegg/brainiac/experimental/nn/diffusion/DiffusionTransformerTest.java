package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DiffusionTransformerTest {

    @Test
    void patchifyRoundTrips() {
        DiffusionTransformer dit = DiffusionTransformer.builder(4, 6).patch(2).width(8).heads(2).build();
        Tensor x = Tensor.arange(24).reshape(1, 24);
        Var p = dit.patchify(Var.of(x));
        assertEquals(Shape.of(1, 6, 4), p.shape());
        // the first patch is the top-left 2x2 square: pixels 0, 1, 6, 7
        assertArrayEquals(new float[] {0, 1, 6, 7}, p.value().slice(1, 0, 1).toArray());
        assertArrayEquals(x.toArray(), dit.unpatchify(p).value().toArray());
    }

    @Test
    void adaLnZeroStartsAsZeroButStillLearns() {
        DiffusionTransformer dit = DiffusionTransformer.builder(8, 8).patch(2).width(32).depth(2).heads(2).classes(3)
                .seed(1).build();
        Tensor x = Tensor.normal(Shape.of(4, 64), 0, 1, new SplittableRandom(2));
        Var out = dit.apply(Var.of(x), Tensor.of(Shape.of(4), 0.1f, 0.5f, 0.9f, 0.3f), new int[] {0, 1, 2, -1});
        assertEquals(0, out.value().abs().max().item(), 0);
        out.sub(Var.of(x)).square().mean().backward();
        // only the zero-initialised output maps get a gradient at first; that is enough to start
        assertNotNull(dit.parameters().get(dit.parameters().size() - 2).grad());
    }

    @Test
    void learnsToDenoiseImages() {
        DiffusionTransformer dit = DiffusionTransformer.builder(8, 8).patch(2).width(32).depth(2).heads(2).classes(2)
                .seed(1).build();
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(3);
        float first = 0;
        float last = 0;
        for (int step = 0; step < 150; step++) {
            float[] clean = new float[16 * 64];
            int[] labels = new int[16];
            for (int i = 0; i < 16; i++) {
                labels[i] = r.nextInt(2);
                int pos = 1 + r.nextInt(6);
                for (int k = 0; k < 64; k++) {
                    boolean on = labels[i] == 0 ? k / 8 == pos : k % 8 == pos;   // a row or a column
                    clean[i * 64 + k] = on ? 1 : -1;
                }
            }
            Tensor x0 = Tensor.of(Shape.of(16, 64), clean);
            Tensor noise = Tensor.normal(Shape.of(16, 64), 0, 0.5f, r);
            float[] t = new float[16];
            java.util.Arrays.fill(t, 0.5f);
            Var loss = dit.apply(Var.of(x0.add(noise)), Tensor.of(Shape.of(16), t), labels).sub(Var.of(noise)).square().mean();
            if (step < 10) {
                first += loss.value().item() / 10;
            }
            if (step >= 140) {
                last += loss.value().item() / 10;
            }
            loss.backward();
            opt.step(dit.parameters());
        }
        assertTrue(last < first * 0.6, "noise-prediction loss " + first + " -> " + last);
    }
}
