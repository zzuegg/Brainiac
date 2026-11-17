package io.github.zzuegg.brainiac.experiments.safety;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class PrivateTrainingTest {

    @Test
    void dpSgdTradesAccuracyForPrivacy() {
        List<PrivateTraining.Budget> out = PrivateTraining.budgets(new double[] {1, 3, 8}, 100, 200, 0.2, 1);
        for (PrivateTraining.Budget b : out) {
            System.out.printf("DP-SGD target eps %4.1f  sigma %.2f  spent eps %5.2f  test accuracy %.2f%n",
                    b.targetEpsilon(), b.noiseMultiplier(), b.epsilon(), b.accuracy());
        }
        PrivateTraining.Budget plain = out.get(0);
        PrivateTraining.Budget eps1 = out.get(1);
        PrivateTraining.Budget eps8 = out.get(3);
        assertTrue(plain.accuracy() > 0.9, plain.toString());
        assertTrue(eps8.accuracy() > 0.7 && eps8.epsilon() <= 8.001, eps8.toString());
        assertTrue(eps1.accuracy() <= eps8.accuracy() + 0.02, eps1 + " vs " + eps8);
    }

    @Test
    void oneRunAuditSeparatesPrivateFromNonPrivateTraining() {
        List<PrivateTraining.Audit> out = PrivateTraining.audit(50, 200, 20, 8, 400, 1);
        for (PrivateTraining.Audit a : out) {
            System.out.printf("Audit %-13s theory eps %5.2f  audit %d/%d right -> eps >= %.2f  MIA AUC %.2f  "
                    + "test %.2f%n", a.method(), a.theoreticalEpsilon(), a.audit().correct(), a.audit().guesses(),
                    a.audit().epsilonLowerBound(), a.membershipAuc(), a.accuracy());
        }
        PrivateTraining.Audit plain = out.get(0);
        PrivateTraining.Audit dp = out.get(1);
        assertTrue(plain.audit().epsilonLowerBound() > 1, plain.toString());
        assertTrue(dp.audit().epsilonLowerBound() <= dp.theoreticalEpsilon(), dp.toString());
        assertTrue(dp.audit().epsilonLowerBound() < plain.audit().epsilonLowerBound(), dp + " vs " + plain);
    }
}
