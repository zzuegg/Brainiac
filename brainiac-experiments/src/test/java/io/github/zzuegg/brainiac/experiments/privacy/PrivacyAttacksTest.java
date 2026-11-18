package io.github.zzuegg.brainiac.experiments.privacy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class PrivacyAttacksTest {

    @Test
    void singleGradientsLeakAndDefencesStopIt() {
        PrivacyAttacks.Inversion one = PrivacyAttacks.inversion(false, true, PrivacyAttacks.Defence.NONE, 1, 120, 1);
        print(one);
        print(PrivacyAttacks.inversion(false, false, PrivacyAttacks.Defence.NONE, 1, 120, 1));
        print(PrivacyAttacks.inversion(true, true, PrivacyAttacks.Defence.NONE, 1, 120, 1));
        PrivacyAttacks.Inversion four = PrivacyAttacks.inversion(false, true, PrivacyAttacks.Defence.NONE, 4, 120, 1);
        print(four);
        PrivacyAttacks.Inversion[] defended = new PrivacyAttacks.Inversion[PrivacyAttacks.Defence.values().length];
        for (PrivacyAttacks.Defence d : PrivacyAttacks.Defence.values()) {
            if (d != PrivacyAttacks.Defence.NONE) {
                defended[d.ordinal()] = PrivacyAttacks.inversion(false, true, d, 1, 120, 1);
                print(defended[d.ordinal()]);
            }
        }
        assertTrue(one.psnr() > 15, one.toString());
        assertEquals(1, one.labelAccuracy(), one.toString());
        assertTrue(one.psnr() > four.psnr(), "batching makes inversion harder");
        assertTrue(defended[PrivacyAttacks.Defence.TOP_K_1.ordinal()].psnr() < one.psnr() - 10, "top-1% stops it");
        assertTrue(defended[PrivacyAttacks.Defence.DP_NOISE.ordinal()].psnr() < one.psnr() - 10, "DP noise stops it");
        assertTrue(defended[PrivacyAttacks.Defence.SECURE_AGGREGATION.ordinal()].psnr() < one.psnr() - 10,
                "only the sum is visible");
    }

    private static void print(PrivacyAttacks.Inversion r) {
        System.out.printf("Inversion %-9s %-18s batch %2d  PSNR %5.1f dB  labels %.2f  defended training acc %.3f%n",
                r.attack(), r.defence(), r.batch(), r.psnr(), r.labelAccuracy(), r.defendedAccuracy());
    }
}
