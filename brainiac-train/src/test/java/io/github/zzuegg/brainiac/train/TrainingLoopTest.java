package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TrainingLoopTest {

    static final Tensor X = Tensor.uniform(Shape.of(8, 4), -1, 1, new SplittableRandom(1));
    static final Tensor Y = Tensor.uniform(Shape.of(8, 2), -1, 1, new SplittableRandom(2));

    static Map<String, Var> model(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Map<String, Var> m = new LinkedHashMap<>();
        m.put("w", Var.param(Tensor.uniform(Shape.of(4, 2), -1, 1, r)));
        m.put("b", Var.param(Tensor.uniform(Shape.of(2), -1, 1, r)));
        return m;
    }

    static Supplier<Var> loss(Map<String, Var> m, int from, int to) {
        return () -> Var.of(X.slice(0, from, to)).matmul(m.get("w")).add(m.get("b")).sub(Var.of(Y.slice(0, from, to)))
                .square().mean();
    }

    @Test
    void accumulatingMicroBatchesEqualsOneBigBatch() {
        Map<String, Var> big = model(3);
        TrainingLoop one = TrainingLoop.of(big, Optimizer.adam(0.1f));
        Map<String, Var> small = model(3);
        TrainingLoop four = TrainingLoop.of(small, Optimizer.adam(0.1f)).accumulate(4);
        for (int step = 0; step < 3; step++) {
            one.microBatch(loss(big, 0, 8));
            for (int k = 0; k < 4; k++) {
                four.microBatch(loss(small, 2 * k, 2 * k + 2));
                assertEquals(k == 3, four.stepped());
            }
        }
        assertEquals(3, four.step());
        assertArrayEquals(big.get("w").value().toArray(), small.get("w").value().toArray(), 1e-5f);
        assertArrayEquals(big.get("b").value().toArray(), small.get("b").value().toArray(), 1e-5f);
    }

    @Test
    void clipsTheGradientNorm() {
        Map<String, Var> m = model(4);
        float[] w0 = m.get("w").value().toArray();
        float[] b0 = m.get("b").value().toArray();
        TrainingLoop loop = TrainingLoop.of(m, Optimizer.sgd(1)).clipGradientNorm(0.01f);
        loop.microBatch(loss(m, 0, 8));
        assertTrue(loop.lastGradientNorm() > 0.01f);
        double sq = 0;
        float[] w = m.get("w").value().toArray();
        float[] b = m.get("b").value().toArray();
        for (int i = 0; i < w.length; i++) {
            sq += (w[i] - w0[i]) * (w[i] - w0[i]);
        }
        for (int i = 0; i < b.length; i++) {
            sq += (b[i] - b0[i]) * (b[i] - b0[i]);
        }
        assertEquals(0.01, Math.sqrt(sq), 1e-5);
    }

    @Test
    void resumesExactlyWhereItStopped(@TempDir Path dir) {
        LearningRateSchedule schedule = LearningRateSchedule.warmup(2, LearningRateSchedule.cosine(6, 0.001f));
        Map<String, Var> straight = model(5);
        TrainingLoop a = TrainingLoop.of(straight, Optimizer.adamW(0.05f, 0.1f)).schedule(schedule).ema(0.9f)
                .accumulate(2);
        for (int i = 0; i < 12; i++) {
            a.microBatch(loss(straight, i % 2 * 4, i % 2 * 4 + 4));
        }
        Map<String, Var> first = model(5);
        TrainingLoop b = TrainingLoop.of(first, Optimizer.adamW(0.05f, 0.1f)).schedule(schedule).ema(0.9f).accumulate(2);
        for (int i = 0; i < 6; i++) {
            b.microBatch(loss(first, i % 2 * 4, i % 2 * 4 + 4));
        }
        b.save(dir, Map.of("cursor", 6));
        Map<String, Var> resumed = model(99);   // other weights: the checkpoint replaces them
        TrainingLoop c = TrainingLoop.of(resumed, Optimizer.adamW(0.05f, 0.1f)).schedule(schedule).ema(0.9f)
                .accumulate(2);
        assertTrue(c.resumeIfPresent(dir));
        assertFalse(TrainingLoop.of(model(1), Optimizer.sgd(1)).resumeIfPresent(dir.resolve("none")));
        assertEquals(3, c.step());
        assertEquals(6L, ((Number) c.resume(dir).get("cursor")).longValue());
        for (int i = 6; i < 12; i++) {
            c.microBatch(loss(resumed, i % 2 * 4, i % 2 * 4 + 4));
        }
        assertEquals(6, c.step());
        for (String k : straight.keySet()) {
            assertArrayEquals(straight.get(k).value().toArray(), resumed.get(k).value().toArray(), 1e-6f, k);
        }
        a.ema().swap();
        c.ema().swap();
        for (String k : straight.keySet()) {
            assertArrayEquals(straight.get(k).value().toArray(), resumed.get(k).value().toArray(), 1e-6f, "ema " + k);
        }
    }
}
