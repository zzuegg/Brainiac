package io.github.zzuegg.brainiac.experiments.privacy;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.privacy.LikelihoodRatioAttack;
import io.github.zzuegg.brainiac.experimental.train.MachineUnlearning;

import org.junit.jupiter.api.Test;

class PrivacyAttacksLiraTest {

    @Test
    void liraBeatsTheLossAttackAtLowFalsePositiveRates() {
        PrivacyAttacks.Membership plain = PrivacyAttacks.lira(PrivacyAttacks.Training.PLAIN, 0, 500, 16, 1);
        print(plain);
        PrivacyAttacks.Membership dp = PrivacyAttacks.lira(PrivacyAttacks.Training.DP_SGD, 4, 300, 8, 1);
        print(dp);
        assertTrue(plain.lira().auc() > 0.6, plain.toString());
        assertTrue(plain.lira().tprAt01() > 10 * Math.max(plain.loss().tprAt01(), 0.001), plain.toString());
        assertTrue(dp.lira().auc() < plain.lira().auc(), dp.toString());
    }

    @Test
    void unlearningAgainstLiraAndTheLossAttack() {
        print(PrivacyAttacks.liraUnlearning(null, 400, 8, 50, 1));
        PrivacyAttacks.Membership npo = PrivacyAttacks.liraUnlearning(MachineUnlearning.Method.NPO, 400, 8, 50, 1);
        print(npo);
        print(PrivacyAttacks.liraUnlearning(MachineUnlearning.Method.GRADIENT_ASCENT, 400, 8, 50, 1));
        assertTrue(npo.lira().auc() > 0.3 && npo.lira().auc() < 1, npo.toString());
    }

    private static void print(PrivacyAttacks.Membership m) {
        LikelihoodRatioAttack.Quality l = m.lira();
        LikelihoodRatioAttack.Quality o = m.offline();
        LikelihoodRatioAttack.Quality b = m.loss();
        System.out.printf("MIA %-25s LiRA AUC %.3f TPR@0.1%% %.3f TPR@1%% %.3f | offline AUC %.3f TPR@1%% %.3f | "
                + "loss AUC %.3f TPR@0.1%% %.3f TPR@1%% %.3f | test acc %.3f%n", m.model(), l.auc(), l.tprAt01(),
                l.tprAt1(), o.auc(), o.tprAt1(), b.auc(), b.tprAt01(), b.tprAt1(), m.testAccuracy());
    }
}
