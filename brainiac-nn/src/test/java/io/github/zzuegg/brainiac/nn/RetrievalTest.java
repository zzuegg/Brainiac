package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.Random;
import org.junit.jupiter.api.Test;

class RetrievalTest {

    private static Tensor random(int rows, int cols, Random r) {
        float[] v = new float[rows * cols];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) r.nextGaussian();
        }
        return Tensor.of(Shape.of(rows, cols), v);
    }

    @Test
    void exactSearchFindsTheNearestItems() {
        Random r = new Random(1);
        Tensor items = random(500, 16, r);
        Tensor noise = random(500, 16, r).mul(0.05f);
        Tensor queries = items.add(noise);
        int[] relevant = new int[500];
        for (int i = 0; i < 500; i++) {
            relevant[i] = i;
        }
        Retrieval.Hits hits = new Retrieval.Index(items, true, 64).search(queries, 5);
        assertEquals(1.0, Retrieval.recall(hits, relevant), 1e-9);
        assertEquals(1.0, Retrieval.meanReciprocalRank(hits, relevant), 1e-9);
        assertTrue(hits.scores()[0][0] >= hits.scores()[0][1]);
        // dot product search agrees with a host computation
        Retrieval.Hits dot = new Retrieval.Index(items, false).search(queries.slice(0, 0, 1), 3);
        float[] q = queries.slice(0, 0, 1).toArray();
        float[] it = items.toArray();
        float best = Float.NEGATIVE_INFINITY;
        int bestId = -1;
        for (int i = 0; i < 500; i++) {
            float s = 0;
            for (int d = 0; d < 16; d++) {
                s += q[d] * it[i * 16 + d];
            }
            if (s > best) {
                best = s;
                bestId = i;
            }
        }
        assertEquals(bestId, dot.ids()[0][0]);
        assertEquals(best, dot.scores()[0][0], 1e-3f);
    }

    @Test
    void constantSamplingCorrectionChangesNothing() {
        Random r = new Random(2);
        Var q = Var.of(random(8, 4, r));
        Var it = Var.of(random(8, 4, r));
        float plain = Retrieval.inBatchSoftmax(q, it, null, 0.5f, false).value().item();
        float shifted = Retrieval.inBatchSoftmax(q, it, Tensor.full(Shape.of(8), -3f), 0.5f, false).value().item();
        assertEquals(plain, shifted, 1e-5f);
        float skewed = Retrieval.inBatchSoftmax(q, it, Tensor.of(Shape.of(8), 0, -5, -5, -5, -5, -5, -5, -5), 0.5f,
                false).value().item();
        assertTrue(skewed != plain);
    }

    @Test
    void towersLearnToMatchPairs() {
        Random r = new Random(3);
        Tensor users = random(32, 6, r);
        Tensor itemFeatures = users.matmul(random(6, 6, r));   // each item is a hidden function of its user
        Var wq = Var.param(random(6, 8, r).mul(0.3f));
        Var wi = Var.param(random(6, 8, r).mul(0.3f));
        float first = 0;
        float last = 0;
        for (int step = 0; step < 300; step++) {
            Var loss = Retrieval.inBatchSoftmax(Var.of(users).matmul(wq), Var.of(itemFeatures).matmul(wi), null,
                    0.2f, true);
            loss.backward();
            last = loss.value().item();
            if (step == 0) {
                first = last;
            }
            for (Var w : new Var[] {wq, wi}) {
                w.assign(w.value().sub(w.grad().mul(0.5f)));
                w.zeroGrad();
            }
        }
        assertTrue(last < first * 0.3f, first + " -> " + last);
        Retrieval.Index index = new Retrieval.Index(itemFeatures.matmul(wi.value()), true);
        int[] relevant = new int[32];
        for (int i = 0; i < 32; i++) {
            relevant[i] = i;
        }
        double recall = Retrieval.recall(index.search(users.matmul(wq.value()), 3), relevant);
        assertTrue(recall > 0.8, "recall@3 " + recall);
        assertEquals(32, index.size());
    }
}
