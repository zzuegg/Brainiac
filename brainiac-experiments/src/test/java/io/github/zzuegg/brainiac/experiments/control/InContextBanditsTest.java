package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class InContextBanditsTest {

    @Test
    void distilledTransformerImprovesInContext() {
        Map<String, double[]> r = InContextBandits.run(30, 500, 300, 60, 1);
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> b.append(String.format("%s %.2f/%.2f/%.2f, ", k, v[0], v[1], v[2])));
        System.out.printf("W6 3.17 bandits, reward per pull first third/last third/all of 30 pulls: %s%n", b);
        double[] ad = r.get("AD (UCB histories)");
        assertTrue(ad[1] > ad[0] + 0.15, "AD must improve in context: " + b);
        assertTrue(ad[2] > r.get("expert distillation")[2] + 0.15, b.toString());
    }
}
