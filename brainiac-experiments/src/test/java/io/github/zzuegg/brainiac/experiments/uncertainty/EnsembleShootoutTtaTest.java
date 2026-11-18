package io.github.zzuegg.brainiac.experiments.uncertainty;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class EnsembleShootoutTtaTest {

    @Test
    void testTimeAugmentationAgainstDropoutAndSnapshots() {
        List<EnsembleShootout.TtaRow> rows = EnsembleShootout.testTimeAugmentation(UncertaintyFixture.Task.DIGITS, 1);
        StringBuilder b = new StringBuilder("11.4.9 digits acc / NLL(TS) / fixes-breaks / noisy acc:");
        rows.forEach(r -> b.append(String.format(" %s %.3f / %.3f / %d-%d / %.3f;", r.method(), r.accuracy(), r.nll(),
                r.corrections(), r.corruptions(), r.noisyAccuracy())));
        System.out.println(b);
        for (EnsembleShootout.TtaRow r : rows) {
            assertTrue(r.accuracy() > 0.25 && Double.isFinite(r.nll()), r.toString());
        }
    }
}
