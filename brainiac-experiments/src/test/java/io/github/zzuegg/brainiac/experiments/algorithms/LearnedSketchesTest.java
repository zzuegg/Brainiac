package io.github.zzuegg.brainiac.experiments.algorithms;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.algorithms.SketchShootout.FreqRow;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/** Item 11.2.15: learned Count-Min, learned Misra-Gries and learned Frequent Directions. */
class LearnedSketchesTest {

    @Test
    void learnedSketchesAndFrequentDirections() {
        Object[] data = SketchShootout.featuredStream(200_000, 10_000, 1.1, 0.3, 5);
        int[] stream = (int[]) data[0];
        float[][] features = (float[][]) data[1];
        List<FreqRow> rows = SketchShootout.learned(stream, features, 1024, 100, new double[] {0.1, 0.5}, 5);
        rows.forEach(System.out::println);
        Map<String, Double> stable = SketchShootout.matrix(2000, 8, 4, false, 6);
        Map<String, Double> drift = SketchShootout.matrix(2000, 8, 4, true, 6);
        System.out.println("Matrix stream, stable: " + stable);
        System.out.println("Matrix stream, drifting: " + drift);
        double cm = SketchShootoutTest.find(rows, "COUNT_MIN").weightedError();
        double lcm = SketchShootoutTest.find(rows, "LEARNED_COUNT_MIN [trained]").weightedError();
        double mg = SketchShootoutTest.find(rows, "MISRA_GRIES").weightedError();
        double lmg = SketchShootoutTest.find(rows, "LEARNED_MISRA_GRIES [trained]").weightedError();
        double lmgBad = SketchShootoutTest.find(rows, "LEARNED_MISRA_GRIES [trained, 50% flipped]").weightedError();
        System.out.printf("11.2.15 weighted err: CM %.2e vs learned CM %.2e; MG %.2e vs learned MG %.2e (50%% flipped %.2e);"
                + " FD cov err stable %.4f vs learned %.4f (random dirs %.4f, SVD %.4f, row sampling %.4f);"
                + " drifting FD %.4f vs learned %.4f%n", cm, lcm, mg, lmg, lmgBad, stable.get("FD"),
                stable.get("learned FD"), stable.get("learned FD, random directions"), stable.get("SVD oracle rank 8"),
                stable.get("row sampling"), drift.get("FD"), drift.get("learned FD"));
        assertTrue(stable.get("FD") <= stable.get("row sampling") + 1e-9 || stable.get("FD") < 0.05,
                "Frequent Directions is at least as good as row sampling");
    }
}
