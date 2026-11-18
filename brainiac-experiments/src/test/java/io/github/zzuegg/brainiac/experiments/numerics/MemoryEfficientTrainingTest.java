package io.github.zzuegg.brainiac.experiments.numerics;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class MemoryEfficientTrainingTest {

    private static String line(String title, Map<String, MemoryEfficientTraining.Row> r, String memory, String extra) {
        StringBuilder s = new StringBuilder(title + " acc/loss/" + memory + "/ms per step" + (extra == null ? "" : "/" + extra)
                + ":");
        r.forEach((k, v) -> s.append(String.format(" %s %.3f/%.3f/%.1f/%.2f%s;", k, v.accuracy(), v.loss(), v.memory(),
                v.millisPerStep(), extra == null ? "" : String.format("/%.3g", v.extra()))));
        System.out.println(s);
        return s.toString();
    }

    @Test
    void twoBitActivationsTrainNearlyAsWellWithAFractionOfTheMemory() {
        Map<String, MemoryEfficientTraining.Row> r = MemoryEfficientTraining.activations(150, 1);
        String s = line("Digits MLP activations,", r, "bytes kept for backward", null);
        MemoryEfficientTraining.Row fp32 = r.get("fp32");
        MemoryEfficientTraining.Row two = r.get("ActNN 2-bit");
        assertTrue(two.memory() * 8 < fp32.memory(), s);
        assertTrue(two.accuracy() > fp32.accuracy() - 0.1, s);
    }

    @Test
    void reversibleStackReconstructsItsActivations() {
        Map<String, MemoryEfficientTraining.Row> r = MemoryEfficientTraining.reversible(12, 150, 2);
        String s = line("Digits 12-block reversible,", r, "floats kept per example", "drift");
        MemoryEfficientTraining.Row rev = r.get("reversible");
        MemoryEfficientTraining.Row plain = r.get("ordinary graph");
        assertEquals(plain.accuracy(), rev.accuracy(), 0.05, s);
        assertTrue(rev.extra() < 1e-3, s);
        assertTrue(rev.memory() * 10 < plain.memory(), s);
    }
}
