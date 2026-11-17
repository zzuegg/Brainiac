package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LandmarkAttentionTest {

    static final int KEYS = 40;
    static final int VALUES = 8;
    static final int F = KEYS + VALUES + 1;

    @Test
    void causalAndRetrievingEveryBlockIsTheFullLayer() {
        LandmarkAttention layer = new LandmarkAttention(6, 2, 4, 3, new SplittableRandom(1));
        Tensor x = Tensor.normal(Shape.of(2, 11, 6), 0, 1, new SplittableRandom(2));
        float[] full = layer.forward(Var.of(x)).value().toArray();
        float[] changed = x.toArray();
        for (int f = 0; f < 6; f++) {
            changed[8 * 6 + f] += 3;                                // token 8 of the first sequence
        }
        float[] out = layer.forward(Var.of(Tensor.of(x.shape(), changed))).value().toArray();
        for (int i = 0; i < 8 * 6; i++) {
            assertEquals(full[i], out[i], 1e-6, "position " + i / 6 + " cannot see token 8");
        }
        assertArrayEquals(full, layer.retrieve(4).forward(Var.of(x)).value().toArray(), 1e-6f);
        float[] top1 = layer.retrieve(1).forward(Var.of(x)).value().toArray();
        assertArrayEquals(Arrays.copyOf(full, 6 * 6), Arrays.copyOf(top1, 6 * 6), 1e-6f); // blocks 0-1 see <= 1 landmark
        layer.retrieve(0);
    }

    /** {@code pairs} blocks [key, value], then the query key alone; target its value. */
    static Tensor[] batch(int b, int pairs, SplittableRandom r) {
        int steps = 2 * pairs + 1;
        float[] x = new float[b * steps * F];
        float[] target = new float[b];
        for (int s = 0; s < b; s++) {
            int[] keys = new int[KEYS];
            for (int i = 0; i < KEYS; i++) {
                keys[i] = i;
            }
            for (int i = 0; i < pairs; i++) {
                int j = i + r.nextInt(KEYS - i);
                int t = keys[i];
                keys[i] = keys[j];
                keys[j] = t;
            }
            int[] values = new int[pairs];
            for (int p = 0; p < pairs; p++) {
                values[p] = r.nextInt(VALUES);
                x[(s * steps + 2 * p) * F + keys[p]] = 1;
                x[(s * steps + 2 * p + 1) * F + KEYS + values[p]] = 1;
            }
            int asked = r.nextInt(pairs);
            x[(s * steps + steps - 1) * F + keys[asked]] = 1;
            x[(s * steps + steps - 1) * F + F - 1] = 1;
            target[s] = values[asked];
        }
        return new Tensor[] {Tensor.of(Shape.of(b, steps, F), x), Tensor.of(Shape.of(b), target)};
    }

    static Var logits(LandmarkAttention layer, Var out, Tensor x) {
        int b = x.shape().dim(0);
        int steps = x.shape().dim(1);
        Var y = layer.forward(Var.of(x)).add(x).slice(1, steps - 1, steps).reshape(b, F);
        return y.matmul(out);
    }

    static double accuracy(LandmarkAttention layer, Var out, int pairs) {
        Tensor[] b = batch(64, pairs, new SplittableRandom(77));
        float[] p = logits(layer, out, b[0]).value().toArray();
        float[] t = b[1].toArray();
        int right = 0;
        for (int i = 0; i < t.length; i++) {
            int best = 0;
            for (int v = 1; v < VALUES; v++) {
                if (p[i * VALUES + v] > p[i * VALUES + best]) {
                    best = v;
                }
            }
            right += best == (int) t[i] ? 1 : 0;
        }
        return right / (double) t.length;
    }

    @Test
    void trainedShortRetrievesFromLongContexts() {
        SplittableRandom r = new SplittableRandom(4);
        LandmarkAttention layer = new LandmarkAttention(F, 1, 16, 2, r);
        Var out = Var.param(Init.XAVIER.create(Shape.of(F, VALUES), F, VALUES, r));
        List<Var> params = new ArrayList<>(layer.parameters());
        params.add(out);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int step = 0; step < 400; step++) {
            Tensor[] b = batch(32, 6, r);
            Functional.crossEntropy(logits(layer, out, b[0]), b[1]).backward();
            opt.step(params);
        }
        double shortAcc = accuracy(layer, out, 6);
        double longTop1 = accuracy(layer.retrieve(1), out, 24);
        assertTrue(shortAcc > 0.9, "trained length: " + shortAcc);
        assertTrue(longTop1 > 0.85, "4x longer, reading one retrieved block: " + longTop1);
    }
}
