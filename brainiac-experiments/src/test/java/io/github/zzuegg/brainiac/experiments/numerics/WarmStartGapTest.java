package io.github.zzuegg.brainiac.experiments.numerics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class WarmStartGapTest {

    @Test
    void shrinkAndPerturbClosesTheWarmStartGap() {
        Map<String, Double> r = WarmStartGap.run(4, 250, 0.4f, 0.1f, 1, 2, 3);
        StringBuilder s = new StringBuilder("Digits in 4 chunks, 250 steps per stage, 3 seeds, final test accuracy:");
        r.forEach((k, v) -> s.append(String.format(" %s %.3f;", k, v)));
        System.out.println(s);
        assertTrue(r.get("shrink and perturb") >= r.get("warm start") - 0.02, s.toString());
        assertTrue(r.get("from scratch") > 0.5, s.toString());
    }
}
