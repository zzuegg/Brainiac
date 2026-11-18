package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class PeriodicEmbeddingTest {

    @Test
    void periodicEmbeddingsFitHighFrequencies() {
        Map<String, BinaryBench.Result> r = TabularShootout.highFrequency(400, 1);
        String line = TabularShootoutTest.line("sin(6x)", r);
        System.out.println(line);
        assertTrue(r.get("MLP + PLR (sigma 1)").accuracy() > r.get("MLP").accuracy() + 0.1, line);
    }
}
