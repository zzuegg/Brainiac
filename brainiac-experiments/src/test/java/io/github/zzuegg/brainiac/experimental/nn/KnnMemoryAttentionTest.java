package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class KnnMemoryAttentionTest {

    private static final int K = 4;

    /** Segment one shows key/value pairs; segment two asks for the value of each key. */
    private static float[] facts(int[] perm) {
        float[] x = new float[K * 2 * K];
        for (int i = 0; i < K; i++) {
            x[i * 2 * K + i] = 1;
            x[i * 2 * K + K + perm[i]] = 1;
        }
        return x;
    }

    private static float[] questions(int[] order) {
        float[] x = new float[K * 2 * K];
        for (int i = 0; i < K; i++) {
            x[i * 2 * K + order[i]] = 1;
        }
        return x;
    }

    private static int[] shuffled(SplittableRandom r) {
        int[] p = new int[K];
        for (int i = 0; i < K; i++) {
            p[i] = i;
        }
        for (int i = K - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    /**
     * Accuracy recalling facts from a previous segment with or without memory: trains while
     * reading all memories, then answers reading only the single nearest one.
     */
    private static double recall(boolean useMemory) {
        SplittableRandom random = new SplittableRandom(7);
        KnnMemoryAttention attention = KnnMemoryAttention.create(2 * K, 1, 2 * K, K, random.split());
        Dense head = new Dense(2 * K, K, Activation.IDENTITY, Init.XAVIER, random.split());
        List<Var> params = new ArrayList<>(attention.parameters());
        params.addAll(head.parameters());
        Optimizer opt = Optimizer.adam(0.005f);
        SplittableRandom data = new SplittableRandom(1);
        double correct = 0;
        int train = 800;
        int test = 30;
        for (int step = 0; step < train + test; step++) {
            if (step == train) {
                attention.topK(1);
            }
            int[] perm = shuffled(data);
            int[] order = shuffled(data);
            attention.clearMemory();
            if (useMemory) {
                attention.memorize(true).forward(Var.of(Tensor.of(Shape.of(1, K, 2 * K), facts(perm))));
            }
            Var out = head.forward(attention.memorize(false).forward(Var.of(Tensor.of(Shape.of(1, K, 2 * K),
                    questions(order)))).reshape(K, 2 * K));
            float[] targets = new float[K];
            for (int i = 0; i < K; i++) {
                targets[i] = perm[order[i]];
            }
            if (step >= train) {
                float[] p = out.value().argmax(1).toArray();
                for (int i = 0; i < K; i++) {
                    correct += p[i] == targets[i] ? 1 : 0;
                }
            } else {
                Functional.crossEntropy(out, Tensor.of(targets)).backward();
                opt.step(params);
            }
        }
        return correct / (test * K);
    }

    @Test
    void recallsFactsFromAnEarlierSegment() {
        double with = recall(true);
        double without = recall(false);
        assertTrue(with > 0.9, "with memory " + with);
        assertTrue(without < 0.5, "without memory " + without);
    }

    @Test
    void memoryPersistsAcrossSessions(@TempDir Path dir) throws Exception {
        KnnMemoryAttention a = KnnMemoryAttention.create(8, 2, 4, 2, new SplittableRandom(3));
        Tensor doc = Tensor.normal(Shape.of(1, 5, 8), 0, 1, new SplittableRandom(4));
        a.forward(Var.of(doc));
        assertEquals(5, a.memorySize());
        Tensor query = Tensor.normal(Shape.of(1, 3, 8), 0, 1, new SplittableRandom(5));
        float[] before = a.memorize(false).forward(Var.of(query)).value().toArray();
        Path file = dir.resolve("memory.bin");
        a.saveMemory(file);
        a.clearMemory();
        assertEquals(0, a.memorySize());
        a.loadMemory(file);
        assertEquals(5, a.memorySize());
        assertArrayEquals(before, a.forward(Var.of(query)).value().toArray(), 1e-6f);
    }

    @Test
    void readingAllMemoriesEqualsAttentionOverThem() {
        KnnMemoryAttention a = KnnMemoryAttention.create(8, 2, 4, 100, new SplittableRandom(3));
        List<Var> p = a.parameters();
        p.get(4).assign(Tensor.full(Shape.of(1, 2, 1, 1), 30f));
        Tensor past = Tensor.normal(Shape.of(1, 5, 8), 0, 1, new SplittableRandom(4));
        Tensor now = Tensor.normal(Shape.of(1, 3, 8), 0, 1, new SplittableRandom(5));
        a.forward(Var.of(past));
        float[] got = a.memorize(false).forward(Var.of(now)).value().toArray();
        Var q = Var.of(now).matmul(p.get(0)).reshape(1, 3, 2, 4).permute(0, 2, 1, 3);
        Var k = Var.of(past).matmul(p.get(1)).reshape(1, 5, 2, 4).permute(0, 2, 1, 3);
        Var v = Var.of(past).matmul(p.get(2)).reshape(1, 5, 2, 4).permute(0, 2, 1, 3);
        Var o = Functional.attention(q, k, v, false, 0.5f).permute(0, 2, 1, 3).reshape(1, 3, 8).matmul(p.get(3));
        assertArrayEquals(o.value().toArray(), got, 1e-4f);
    }

    @Test
    void capacityPrunesLeastRecentlyUsed() {
        KnnMemoryAttention a = KnnMemoryAttention.create(8, 1, 4, 2, new SplittableRandom(3)).capacity(6);
        for (int i = 0; i < 4; i++) {
            a.forward(Var.of(Tensor.normal(Shape.of(1, 4, 8), 0, 1, new SplittableRandom(i))));
        }
        assertEquals(6, a.memorySize());
    }
}
