package io.github.zzuegg.brainiac.experiments.metric;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.metric.MetricLearningShootout.Objective;
import io.github.zzuegg.brainiac.experiments.metric.MetricLearningShootout.Scores;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 10.3.10: Proxy-NCA, Proxy-NCA++ and Proxy-Anchor; Recall@1 against steps for Proxy-Anchor and multi-similarity. */
class ProxyLossesTest {

    private static List<Scores> run(Objective... objectives) {
        List<Scores> out = new ArrayList<>();
        for (Objective o : objectives) {
            Scores s = MetricLearningShootout.run(o, 600, 1);
            System.out.println(s);
            out.add(s);
        }
        return out;
    }

    @Test
    void proxiesConvergeFast() {
        List<Scores> s = run(Objective.PROXY_NCA, Objective.PROXY_NCA_PLUS_PLUS, Objective.PROXY_ANCHOR);
        StringBuilder line = new StringBuilder("R@1 at 25/50/100/200 steps:");
        for (Objective o : List.of(Objective.PROXY_ANCHOR, Objective.MULTI_SIMILARITY)) {
            line.append(" ").append(o);
            for (Scores c : MetricLearningShootout.curve(o, new int[] {25, 50, 100, 200}, 1)) {
                line.append(String.format(" %.3f", c.recall1()));
            }
        }
        System.out.println(line);
        assertTrue(s.get(2).recall1() > 0.4, "Proxy-Anchor beats chance (0.25) on unseen classes");
    }
}
