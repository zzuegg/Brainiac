package io.github.zzuegg.brainiac.experiments.uncertainty;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class EnsembleShootoutAuditTest {

    @Test
    void ensembleZooUnderLabelNoise() {
        for (double[] cfg : new double[][] {{0.4, 1}}) {
            List<EnsembleShootout.AuditRow> rows = EnsembleShootout.audit(cfg[0], cfg[1], 400, 1);
            StringBuilder b = new StringBuilder("11.4.16 " + rows.getFirst().setting()
                    + " acc/NLL(TS)/far AUROC/dis/NLL gap+-se:");
            rows.forEach(r -> b.append(String.format(" %s %.3f/%.3f/%.3f/%.3f/%+.3f+-%.3f;", r.method(), r.accuracy(),
                    r.nll(), r.farAuroc(), r.disagreement(), r.nllGapToDeep(), r.nllGapSe())));
            System.out.println(b);
            for (EnsembleShootout.AuditRow r : rows) {
                assertTrue(r.accuracy() > 0.15 && Double.isFinite(r.nll()), r.toString());
            }
        }
    }
}
