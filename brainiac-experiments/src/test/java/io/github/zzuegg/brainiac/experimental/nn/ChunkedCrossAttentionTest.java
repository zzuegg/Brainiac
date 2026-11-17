package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ChunkedCrossAttentionTest {

    static final int V = 8;
    static final int M = 4;
    static final int C = 4;
    static final int T = C * M;
    static final int F = V + M;

    @Test
    void aChunksNeighboursReachOnlyTheTokensFromItsLastTokenOn() {
        ChunkedCrossAttention cca = new ChunkedCrossAttention(6, 2, 4, M, new SplittableRandom(1));
        SplittableRandom r = new SplittableRandom(2);
        Tensor h = Tensor.normal(Shape.of(1, T, 6), 0, 1, r);
        Tensor nb = Tensor.normal(Shape.of(1, C, 5, 6), 0, 1, r);
        float[] base = cca.forward(List.of(Var.of(h), Var.of(nb)), false).value().toArray();
        float[] changed = nb.toArray();
        for (int j = 0; j < 5 * 6; j++) {
            changed[2 * 5 * 6 + j] += 1;                          // chunk 2's neighbours
        }
        float[] out = cca.forward(List.of(Var.of(h), Var.of(Tensor.of(nb.shape(), changed))), false).value()
                .toArray();
        float[] in = h.toArray();
        for (int i = 0; i < T; i++) {
            boolean reads = i >= 2 * M + M - 1 && i < 3 * M + M - 1;
            for (int f = 0; f < 6; f++) {
                int at = i * 6 + f;
                if (!reads) {
                    assertEquals(base[at], out[at], 1e-6, "position " + i);
                }
                if (i < M - 1) {
                    assertEquals(in[at], out[at], 1e-6, "the first m-1 tokens read nothing");
                }
            }
            if (reads) {
                float diff = 0;
                for (int f = 0; f < 6; f++) {
                    diff += Math.abs(base[i * 6 + f] - out[i * 6 + f]);
                }
                assertTrue(diff > 1e-4, "position " + i + " reads chunk 2");
            }
        }
        assertEquals(-1, ChunkedCrossAttention.chunkOf(M - 2, M));
        assertEquals(0, ChunkedCrossAttention.chunkOf(M - 1, M));
        assertEquals(1, ChunkedCrossAttention.chunkOf(2 * M - 1, M));
    }

    /** One batch: hidden = position-in-chunk one-hot, neighbours = the continuation tokens. */
    static Tensor[] batch(int b, SplittableRandom r, boolean useful) {
        float[] h = new float[b * T * F];
        float[] nb = new float[b * C * M * F];
        float[] target = new float[b * T];
        for (int s = 0; s < b; s++) {
            int[] x = new int[T + M];
            int[] other = new int[T + M];
            for (int i = 0; i < x.length; i++) {
                x[i] = r.nextInt(V);
                other[i] = r.nextInt(V);
            }
            for (int i = 0; i < T; i++) {
                h[(s * T + i) * F + V + (i + 1) % M] = 1;
                target[s * T + i] = i < M - 1 ? -1 : x[i + 1];
            }
            int[] src = useful ? x : other;
            for (int c = 0; c < C; c++) {
                for (int j = 0; j < M; j++) {
                    int row = ((s * C + c) * M + j) * F;
                    nb[row + src[(c + 1) * M + j]] = 1;
                    nb[row + V + j] = 1;
                }
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(b, T, F), h), Tensor.of(Shape.of(b, C, M, F), nb),
            Tensor.of(Shape.of(b * T), target)};
    }

    static double train(boolean useful) {
        SplittableRandom r = new SplittableRandom(5);
        ChunkedCrossAttention cca = new ChunkedCrossAttention(F, 1, 16, M, r);
        Var out = Var.param(Init.XAVIER.create(Shape.of(F, V), F, V, r));
        List<Var> params = new ArrayList<>(cca.parameters());
        params.add(out);
        Optimizer opt = Optimizer.adam(0.02f);
        for (int step = 0; step < 300; step++) {
            Tensor[] b = batch(16, r, useful);
            Var y = cca.forward(List.of(Var.of(b[0]), Var.of(b[1])), true).reshape(16 * T, F).matmul(out);
            Functional.crossEntropy(y, b[2]).backward();
            opt.step(params);
        }
        Tensor[] b = batch(64, new SplittableRandom(99), useful);
        float[] p = cca.forward(List.of(Var.of(b[0]), Var.of(b[1])), false).reshape(64 * T, F).matmul(out).value()
                .toArray();
        float[] t = b[2].toArray();
        int right = 0;
        int all = 0;
        for (int i = 0; i < t.length; i++) {
            if (t[i] < 0) {
                continue;
            }
            int best = 0;
            for (int v = 1; v < V; v++) {
                if (p[i * V + v] > p[i * V + best]) {
                    best = v;
                }
            }
            right += best == (int) t[i] ? 1 : 0;
            all++;
        }
        return right / (double) all;
    }

    @Test
    void retrievedContinuationsPredictTheNextTokens() {
        double useful = train(true);
        double random = train(false);
        assertTrue(useful > 0.9, "with continuation neighbours: " + useful);
        assertTrue(random < 0.25, "with unrelated neighbours: " + random);
    }

    @Test
    void specKeepsTheHiddenShape() {
        var spec = ChunkedCrossAttention.spec(2, 4, M);
        List<Shape> in = List.of(Shape.of(T, 6), Shape.of(C, 3, 6));
        assertArrayEquals(new int[] {T, 6}, spec.outputShape(in).toArray());
        assertEquals(4, spec.create(List.of(Shape.of(1, T, 6), Shape.of(1, C, 3, 6)), new SplittableRandom(1))
                .parameters().size());
    }
}
