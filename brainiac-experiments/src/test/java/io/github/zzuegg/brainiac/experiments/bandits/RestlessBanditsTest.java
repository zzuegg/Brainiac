package io.github.zzuegg.brainiac.experiments.bandits;

import org.junit.jupiter.api.Test;

import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class RestlessBanditsTest {

    private static String row(Map<String, Double> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(k).append('=').append(String.format(Locale.ROOT, "%.4g ", v)));
        return sb.toString().trim();
    }

    @Test
    void restless() {
        Map<String, Double> r = RestlessBandits.restless(6, 2, 20_000, 200, 1);
        System.out.printf("12.6.15 restless 6 arms budget 2 mean reward per step: %s%n", row(r));
        double whittle = r.get("Whittle index (exact)");
        assertTrue(whittle <= r.get("LP upper bound") + 0.02, "the LP bounds every policy");
        assertTrue(whittle > r.get("random") && whittle > r.get("round robin"), "Whittle beats the index-free baselines");
    }

    @Test
    void knapsacks() {
        Map<String, Double> random = RestlessBandits.knapsacks(5, 2, 0.2, 5_000, 3);
        Map<String, Double> pricing = RestlessBandits.knapsacks(0, 1, 0.1, 5_000, 3);
        System.out.printf("12.6.16 BwK total reward, 5 arms 2 resources B=0.2T: %s | pricing with inventory B=0.1T: %s%n",
                row(random), row(pricing));
        assertTrue(pricing.get("PRIMAL_DUAL_BWK") > pricing.get("UCB1 (ignores budget)"),
                "budget-aware bandits beat budget-blind UCB1 under a tight budget");
        assertTrue(random.get("UCB_BWK") <= random.get("LP benchmark") * 1.05, "the LP bounds the reward");
    }
}
