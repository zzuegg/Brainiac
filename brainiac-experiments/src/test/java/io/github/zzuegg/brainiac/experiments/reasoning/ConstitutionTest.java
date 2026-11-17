package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Constitutional AI / RLAIF against self-rewarding on a toy refusal rule. */
class ConstitutionTest {

    @Test
    void constitutionLowersViolationsWhereSelfRewardingDoesNot() {
        SelfImprovement.ConstitutionResult cai = SelfImprovement.constitution(150, 6, true, 1);
        SelfImprovement.ConstitutionResult srt = SelfImprovement.constitution(150, 6, false, 1);
        System.out.printf("CAI violation rate %.2f -> %.2f, addition accuracy %.2f -> %.2f; self-rewarding violation"
                + " %.2f -> %.2f, accuracy %.2f -> %.2f%n", cai.violationBefore(), cai.violationAfter(),
                cai.accuracyBefore(), cai.accuracyAfter(), srt.violationBefore(), srt.violationAfter(),
                srt.accuracyBefore(), srt.accuracyAfter());
        assertTrue(cai.violationAfter() < cai.violationBefore(), cai.toString());
        assertTrue(cai.violationAfter() < srt.violationAfter(), cai + " vs " + srt);
    }
}
