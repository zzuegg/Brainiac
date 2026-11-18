package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

/** Wave 7 item 7.1.8. */
class ExpressivityTest {

    @Test
    void beyondWeisfeilerLehmanOnSkipLinkGraphs() {
        Map<String, Double> r = GraphStructureShootout.expressivity(List.of("GIN", "GIN + RWSE", "GIN + RNF", "ESAN"), 2,
                80, 1);
        System.out.printf("7.1.8 CSL accuracy: ESAN %.3f, GIN+RNF %.3f, GIN+RWSE %.3f; GIN %.3f (chance 0.1)%n",
                r.get("ESAN"), r.get("GIN + RNF"), r.get("GIN + RWSE"), r.get("GIN"));
        assertTrue(r.get("GIN") <= 0.15, r.toString());
        assertTrue(r.get("GIN + RWSE") > r.get("GIN") + 0.2, r.toString());
    }
}
