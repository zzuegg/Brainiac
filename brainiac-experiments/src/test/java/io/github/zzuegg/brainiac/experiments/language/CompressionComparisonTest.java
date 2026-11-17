package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class CompressionComparisonTest {

    @Test
    void calibratedPruningKeepsTheModelUsable() {
        HookedDecoder base = PeftShootoutTest.base();
        float dense = CountingTasks.evaluate(base, 12, 8, 1, 2);
        Map<CompressionComparison.Method, Float> loss = new EnumMap<>(CompressionComparison.Method.class);
        for (CompressionComparison.Method m : CompressionComparison.Method.values()) {
            HookedDecoder c = CompressionComparison.compress(base, m, 0.5, 1);
            loss.put(m, CountingTasks.evaluate(c, 12, 8, 1, 2));
        }
        System.out.printf("LM loss at 50%% compression: dense %.3f, %s%n", dense, loss);
        assertTrue(loss.get(CompressionComparison.Method.SPARSEGPT) < loss.get(CompressionComparison.Method.MAGNITUDE),
                loss.toString());
        assertTrue(loss.get(CompressionComparison.Method.SPARSEGPT) < dense + 1.0, "dense " + dense + " " + loss);
        assertTrue(loss.get(CompressionComparison.Method.SPARSEGPT_2_4) < loss.get(CompressionComparison.Method.WANDA_2_4),
                loss.toString());
        assertTrue(loss.get(CompressionComparison.Method.ACTIVATION_AWARE_SVD) < loss.get(CompressionComparison.Method.SVD) + 0.5,
                loss.toString());
    }
}
