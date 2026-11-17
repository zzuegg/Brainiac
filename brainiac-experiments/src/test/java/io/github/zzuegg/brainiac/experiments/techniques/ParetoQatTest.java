package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class ParetoQatTest {

    @Test
    void qatPhaseAfterFloatTrainingBeatsQatFromScratchAndPtqAtLowBits() {
        StringBuilder line = new StringBuilder("ParetoQ 70/30 test loss (ParetoQ | QAT from scratch | PTQ | float):");
        List<ParetoQat.Result> results = new ArrayList<>();
        for (double bits : new double[] {1.58, 2, 3}) {
            ParetoQat.Result r = ParetoQat.run(bits, 300, 0.3, 3);
            results.add(r);
            line.append(String.format(" %s bits %.3f | %.3f | %.3f | %.3f (acc %.2f/%.2f/%.2f);", bits, r.paretoq().loss(),
                    r.qat().loss(), r.ptq().loss(), r.full().loss(), r.paretoq().accuracy(), r.qat().accuracy(),
                    r.ptq().accuracy()));
        }
        System.out.println(line);
        for (ParetoQat.Result r : results) {
            assertTrue(r.paretoq().loss() < r.ptq().loss(), "ParetoQ vs PTQ at " + r.bits());
            if (r.bits() <= 2) {
                assertTrue(r.paretoq().loss() < r.qat().loss(), "ParetoQ vs QAT from scratch at " + r.bits());
                assertTrue(r.paretoq().accuracy() > 0.9f, "ParetoQ accuracy at " + r.bits());
            }
        }
    }
}
