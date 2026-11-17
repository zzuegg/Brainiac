package io.github.zzuegg.brainiac.experiments.generative;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class GuidanceComparisonTest {

    @Test
    void autoguidanceKeepsRecallWhereCfgLosesIt() {
        Map<String, double[]> r = GuidanceComparison.run(1500, new float[] {3f}, 1);
        StringBuilder line = new StringBuilder("ToyPoints guidance, precision/recall:");
        r.forEach((k, v) -> line.append(String.format(" %s %.2f/%.2f;", k, v[0], v[1])));
        System.out.println(line);
        double[] cfg = r.get("CFG w=3.0");
        double[] auto = r.get("autoguidance w=3.0");
        double[] apg = r.get("APG w=3.0");
        assertTrue(auto[1] > cfg[1], "autoguidance keeps more of the data's spread than CFG");
        assertTrue(apg[1] >= cfg[1] - 0.05, "APG does not lose more recall than CFG");
        assertTrue(auto[0] > 0.5, "and its samples stay on the data");
    }
}
