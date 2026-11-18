package io.github.zzuegg.brainiac.experiments.uncertainty;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class EnsembleShootoutDiversityTest {

    @Test
    void divDisAndDbatAgainstErmAndJtt() {
        for (double agreement : new double[] {1.0, 0.5}) {
            List<EnsembleShootout.DiversityRow> rows = EnsembleShootout.diversity(agreement, 200, 1);
            StringBuilder b = new StringBuilder(String.format("11.4.10 coloured shapes agreement %.1f avg/worst/dis:",
                    agreement));
            rows.forEach(r -> b.append(String.format(" %s %.3f/%.3f/%.3f;", r.method(), r.average(), r.worstGroup(),
                    r.headDisagreement())));
            System.out.println(b);
            for (EnsembleShootout.DiversityRow r : rows) {
                assertTrue(r.average() >= 0.3 && r.worstGroup() >= 0, r.toString());
            }
        }
    }

    @Test
    void distillationKeepsOrLosesDisagreement() {
        List<EnsembleShootout.Row> rows = EnsembleShootout.distillation(300, 1);
        System.out.println("11.4.14 distillation:" + EnsembleShootoutSingleRunTest.line(rows));
        for (EnsembleShootout.Row r : rows) {
            assertTrue(r.accuracy() > 0.4, r.toString());
        }
        assertTrue(EnsembleShootoutSingleRunTest.row(rows, "hydra").disagreement() > 0, "Hydra heads differ");
    }
}
