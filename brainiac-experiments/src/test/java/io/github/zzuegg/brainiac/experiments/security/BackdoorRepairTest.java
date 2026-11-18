package io.github.zzuegg.brainiac.experiments.security;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.security.Backdoors;

import java.util.List;

import org.junit.jupiter.api.Test;

class BackdoorRepairTest {

    @Test
    void repairsLowerAttackSuccessFromFivePercentCleanData() {
        List<BackdoorShootout.Repair> rows = BackdoorShootout.repair(Backdoors.Kind.PATCH, 0.05, 1);
        double best = 1;
        for (BackdoorShootout.Repair r : rows) {
            System.out.printf("Backdoor repair PATCH 5%% %-14s clean %.3f ASR %.3f%s %s%n", r.method(), r.clean(),
                    r.success(), r.method().equals("NEURAL_CLEANSE") ? "  flagged class " + r.target() : "",
                    r.detail());
            if (!r.method().equals("BACKDOORED") && r.clean() > rows.get(0).clean() - 0.1) {
                best = Math.min(best, r.success());
            }
        }
        BackdoorShootout.Repair bad = rows.get(0);
        System.out.printf("Backdoor repair: backdoored ASR %.3f, fine-tune %.3f, best repair within 10 points of "
                + "clean accuracy %.3f%n", bad.success(), rows.get(1).success(), best);
        assertTrue(bad.success() > 0.8, "backdoor " + bad.success());
        assertTrue(best < bad.success() - 0.2, "best repair " + best);
    }
}
