package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class LatticeSamplingTest {

    @Test
    void flowChainIsExactAndReportsItsAcceptance() {
        Map<String, double[]> r = LatticeSampling.run(6, new double[] {6.975, 4.0}, 600, 3000, 1);
        System.out.printf("13.3.6 phi^4 L=6, m^2=-4, lambda={6.975 (critical), 4 (broken)}: %s%n",
                Lines.of(r));
        assertTrue(r.get("flow acceptance")[0] > 0.05, Lines.of(r));
        double[] local = r.get("<|M|> local");
        double[] flow = r.get("<|M|> flow");
        assertTrue(Math.abs(local[0] - flow[0]) < 0.15, Lines.of(r));
    }
}
