package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ConstrainedFieldTest {

    @Test
    void hardConstraintsHoldExactly() {
        List<PdeShootout.ConstraintRow> tg = PdeShootout.taylorGreen(24, 0.05, new double[] {0.1, 1, 10}, 300, 1);
        List<PdeShootout.ConstraintRow> ad = PdeShootout.advectedDensity(30, 0.02, 300, 1);
        StringBuilder line = new StringBuilder("8.3.17 rel L2 / max violation: [Taylor-Green]");
        tg.forEach(r -> line.append(String.format(" %s %.3f/%.1e;", r.method(), r.error(), r.maxViolation())));
        line.append(" [advected density]");
        ad.forEach(r -> line.append(String.format(" %s %.3f/%.1e;", r.method(), r.error(), r.maxViolation())));
        System.out.println(line);
        assertTrue(tg.get(0).maxViolation() < 1e-3, line.toString());
        assertTrue(ad.get(0).maxViolation() < 1e-3, line.toString());
    }
}
