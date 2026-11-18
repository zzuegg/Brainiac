package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Wave 6, 6.12: ProRL reference resets against plain GRPO, by pass@k at large k. */
class ProlongedRlCoverageTest {

    @Test
    void prolongedRlWithResetsKeepsImproving() {
        ProlongedRlCoverage.Result r = ProlongedRlCoverage.run(6, 40, 64, 1);
        int last = r.ks().length - 1;
        StringBuilder sb = new StringBuilder();
        r.arms().forEach((k, a) -> sb.append(String.format(" %s pass@1 %.2f pass@%d %.2f KL %.2f (%d resets);", k,
                a.passAtK()[0], r.ks()[last], a.passAtK()[last], a.klFromBase(), a.resets())));
        System.out.printf("add+mul(6):%s%n", sb);
        double base1 = r.arms().get("base").passAtK()[0];
        double short1 = r.arms().get("GRPO x1").passAtK()[0];
        double prorl1 = r.arms().get("ProRL x5").passAtK()[0];
        assertTrue(short1 > base1, "GRPO improves pass@1");
        assertTrue(prorl1 > short1, "prolonged RL keeps improving pass@1");
        assertTrue(prorl1 >= r.arms().get("GRPO x5").passAtK()[0] - 0.02, "resets do not hurt the long run");
        assertTrue(r.arms().get("ProRL x5").resets() == 4, "four resets in five intervals");
    }
}
