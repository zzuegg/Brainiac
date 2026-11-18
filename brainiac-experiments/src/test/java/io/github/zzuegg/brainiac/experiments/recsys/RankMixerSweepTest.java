package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class RankMixerSweepTest {

    @Test
    void rankMixerTrainsDenseAndSparse() {
        Map<String, double[]> r = ClickModels.rankMixerSweep(new int[] {16}, 100, 1);
        String line = InteractionOrderTest.line("RankMixer params/AUC/ms-per-batch/active-experts:", r);
        System.out.printf("%s%n", line);
        assertTrue(r.get("RankMixer d=16")[1] > 0.6, line);
        assertTrue(r.get("RankMixer-MoE d=16")[1] > 0.6, line);
        assertTrue(r.get("RankMixer-MoE d=16")[3] <= 4, line);
        assertTrue(r.get("RankMixer d=16")[2] < r.get("FT-Transformer d=16")[2], line);
    }
}
