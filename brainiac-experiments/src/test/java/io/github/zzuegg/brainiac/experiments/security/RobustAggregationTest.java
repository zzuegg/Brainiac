package io.github.zzuegg.brainiac.experiments.security;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.security.ModelPoisoning;
import io.github.zzuegg.brainiac.experiments.security.RobustAggregation.Defence;

import java.util.List;

import org.junit.jupiter.api.Test;

class RobustAggregationTest {

    static final int ROUNDS = 25;
    private static final Defence[] RULES = {Defence.KRUM, Defence.MULTI_KRUM, Defence.MEDIAN, Defence.TRIMMED_MEAN,
        Defence.GEOMETRIC_MEDIAN};

    @Test
    void robustRulesSurviveSignFlipsThatBreakFedAvg() {
        List<RobustAggregation.Cell> base = RobustAggregation.baselines(0.2, 0, ROUNDS, 1);
        base.forEach(RobustAggregationTest::print);
        double oracle = base.get(3).accuracy();
        RobustAggregation.Cell avg = RobustAggregation.run(Defence.FEDAVG, ModelPoisoning.Attack.SIGN_FLIP, 0.2, 0,
                true, ROUNDS, 1);
        print(avg);
        double worstCoordinate = 1;
        for (Defence d : RULES) {
            RobustAggregation.Cell c = RobustAggregation.run(d, ModelPoisoning.Attack.SIGN_FLIP, 0.2, 0, true, ROUNDS,
                    1);
            print(c);
            if (d != Defence.KRUM) {
                worstCoordinate = Math.min(worstCoordinate, c.accuracy());
            }
        }
        System.out.printf("Robust agg IID 20%% sign flip: worst of Multi-Krum/median/TM/GM %.3f, FedAvg %.3f, "
                + "honest oracle %.3f%n", worstCoordinate, avg.accuracy(), oracle);
        assertTrue(oracle > 0.6, "oracle " + oracle);
        assertTrue(worstCoordinate > avg.accuracy() + 0.2, worstCoordinate + " vs FedAvg " + avg.accuracy());
    }

    @Test
    void robustRulesPayForHeterogeneityWithoutAttackers() {
        double fedAvg = RobustAggregation.baselines(0, 0.1, ROUNDS, 1).get(0).accuracy();
        double worst = 1;
        for (Defence d : RULES) {
            RobustAggregation.Cell c = RobustAggregation.run(d, ModelPoisoning.Attack.NONE, 0, 0.1, true, ROUNDS, 1);
            print(c);
            worst = Math.min(worst, c.accuracy());
        }
        System.out.printf("Robust agg a=0.1 no attack: FedAvg %.3f, worst robust rule %.3f (cost %.3f)%n", fedAvg,
                worst, fedAvg - worst);
        assertTrue(fedAvg > 0.4, "FedAvg " + fedAvg);
        assertTrue(worst < fedAvg, "some robust rule loses accuracy under skew");
    }

    static void print(RobustAggregation.Cell c) {
        System.out.printf("Robust agg a=%.1f f=%2.0f%% %-17s %-17s acc %.3f%s%n", c.alpha(), 100 * c.share(),
                c.attack(), c.defence(), c.accuracy(), c.backdoor() >= 0
                        ? String.format("  backdoor %.3f", c.backdoor()) : "");
    }
}
