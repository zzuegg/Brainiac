package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class TabularStressTest {

    static String table(String title, Map<String, Map<String, Double>> r) {
        StringBuilder line = new StringBuilder(title);
        r.forEach((k, v) -> {
            line.append(" [").append(k).append(":");
            v.forEach((m, a) -> line.append(String.format(" %s %.3f", m, a)));
            line.append("]");
        });
        return line.toString();
    }

    @Test
    void treesWinOnIrregularTargetsUntilRotated() {
        double[] corr = new double[1];
        Map<String, Map<String, Double>> r = TabularShootout.stress(200, 1, corr);
        String line = table("Stress suite accuracy:", r) + String.format(" NAM shape correlation %.3f", corr[0]);
        System.out.println(line);
        assertTrue(r.get("0% noise").get("GBT") > r.get("0% noise").get("MLP"), line);
        assertTrue(corr[0] > 0.7, line);
    }
}
