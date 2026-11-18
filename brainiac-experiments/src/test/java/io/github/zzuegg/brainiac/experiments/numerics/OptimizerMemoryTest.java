package io.github.zzuegg.brainiac.experiments.numerics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

/** The optimizer-state and low-rank splits of {@link MemoryEfficientTraining}. */
class OptimizerMemoryTest {

    static String line(String title, Map<String, MemoryEfficientTraining.Row> r, String memory, String extra) {
        StringBuilder s = new StringBuilder(title + " acc/loss/" + memory + "/ms per step" + (extra == null ? "" : "/" + extra)
                + ":");
        r.forEach((k, v) -> s.append(String.format(" %s %.3f/%.3f/%.1f/%.2f%s;", k, v.accuracy(), v.loss(), v.memory(),
                v.millisPerStep(), extra == null ? "" : String.format("/%.3g", v.extra()))));
        System.out.println(s);
        return s.toString();
    }

    @Test
    void fourBitStatesMatchAdamWhereNaiveFourBitDoesNot() {
        Map<String, MemoryEfficientTraining.Row> r = MemoryEfficientTraining.optimizerStates(120, 1, 2, 3);
        String s = line("Digits MLP optimizer states, 3 seeds,", r, "state bytes per parameter", "diverged");
        MemoryEfficientTraining.Row adam = r.get("fp32 Adam");
        MemoryEfficientTraining.Row four = r.get("4-bit (rank-1, no zero)");
        assertTrue(four.memory() < 1.5, s);
        assertTrue(four.accuracy() > adam.accuracy() - 0.1, s);
        assertTrue(r.get("COAT FP8").accuracy() > adam.accuracy() - 0.1, s);
    }

    @Test
    void reLoraAccumulatesAHigherRankUpdateThanLora() {
        Map<String, MemoryEfficientTraining.Row> r = MemoryEfficientTraining.lowRank(4, 50, 250, 4);
        String s = line("Digits 64-64-64-10 from scratch, rank 4,", r, "optimizer bytes", "effective rank of update");
        assertTrue(r.get("ReLoRA").extra() > r.get("LoRA").extra(), s);
        assertTrue(r.get("ReLoRA").memory() < r.get("full-rank AdamW").memory(), s);
    }
}
