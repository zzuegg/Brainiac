package io.github.zzuegg.brainiac.experiments.robustness;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.robust.AdversarialTraining;

import org.junit.jupiter.api.Test;

class AutoAttackReevaluationTest {

    @Test
    void autoAttackNeverReportsMoreRobustnessThanFixedStepPgd() {
        double pgdAt = 0;
        double standard = 0;
        for (AdversarialTraining.Method m : AdversarialTraining.Method.values()) {
            AdversarialRobustness.Reevaluated r = AdversarialRobustness.autoAttack(m, 0.1f, 200, 150, 1);
            System.out.printf("AutoAttack re-evaluation eps 0.1 %-9s clean %.3f PGD-10 %.3f PGD-50x3 %.3f "
                    + "AutoAttack %.3f (gap to PGD-10 %.3f)%n", m, r.clean(), r.pgd10(), r.pgd50(), r.autoAttack(),
                    r.pgd10() - r.autoAttack());
            assertTrue(r.autoAttack() <= r.pgd10() + 1e-9, m + " AutoAttack <= PGD-10");
            if (m == AdversarialTraining.Method.PGD_AT) {
                pgdAt = r.autoAttack();
            } else if (m == AdversarialTraining.Method.STANDARD) {
                standard = r.autoAttack();
            }
        }
        assertTrue(pgdAt > standard + 0.2, "ranking kept: PGD-AT " + pgdAt + " vs standard " + standard);
    }
}
