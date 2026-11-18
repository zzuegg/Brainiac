package io.github.zzuegg.brainiac.experiments.privacy;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class PrivacyAttacksPateTest {

    @Test
    void confidentGnMaxSpendsLessThanGnMax() {
        List<PrivacyAttacks.Pate> rows = PrivacyAttacks.pate(40, 4000, 300, 15, 4, 28, 1);
        for (PrivacyAttacks.Pate p : rows) {
            System.out.printf("PATE %-16s eps %6.2f  student acc %.3f  answered %d  label acc %.3f%n", p.method(),
                    p.epsilon(), p.accuracy(), p.answered(), p.labelAccuracy());
        }
        PrivacyAttacks.Pate confident = rows.get(2);
        PrivacyAttacks.Pate dpSgd = rows.get(3);
        assertTrue(confident.epsilon() < rows.get(1).epsilon(), "Confident-GNMax spends less than GNMax");
        assertTrue(confident.labelAccuracy() > 0.8, confident.toString());
        assertTrue(dpSgd.epsilon() <= confident.epsilon() + 1e-3, dpSgd.toString());
    }
}
