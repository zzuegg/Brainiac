package io.github.zzuegg.brainiac.experiments.control;

import org.junit.jupiter.api.Test;

import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class TabularExplorationTest {

    static String table(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k).append('=');
            for (int i = 0; i < v.length; i++) {
                sb.append(String.format(Locale.ROOT, i == 0 ? "%.3g" : "/%.3g", v[i]));
            }
            sb.append(' ');
        });
        return sb.toString().trim();
    }

    @Test
    void deepSeaAndRiverSwim() {
        Map<String, double[]> r = TabularExploration.exploration(10, 300, 3);
        System.out.printf("12.6.1 deep sea N=10 regret/first-reward episode: %s%n", table(r));
        double eps = r.get("EPSILON_GREEDY_Q")[0];
        double best = Math.min(Math.min(r.get("RMAX")[0], r.get("UCRL2")[0]),
                Math.min(r.get("PSRL")[0], r.get("UCB_Q_BERNSTEIN")[0]));
        assertTrue(best < eps, "a directed explorer beats epsilon-greedy");
        Map<String, Double> river = TabularExploration.riverSwim(6, 20, 150, 3);
        System.out.printf("12.6.1 RiverSwim-6 mean return per episode: %s%n", river);
        assertTrue(river.get("PSRL") >= river.get("EPSILON_GREEDY_Q") - 1e-9);
    }
}
