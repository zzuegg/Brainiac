package io.github.zzuegg.brainiac.experiments.security;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.security.ModelPoisoning;
import io.github.zzuegg.brainiac.experiments.security.RobustAggregation.Defence;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class PreAggregationTest {

    @Test
    void preAggregationRecoversTheHeterogeneityLossUnderAttack() {
        Defence[] defences = {Defence.FEDAVG, Defence.FEDAVG_M, Defence.MEDIAN, Defence.BUCKETING_MEDIAN,
            Defence.NNM_MEDIAN, Defence.CENTERED_CLIP, Defence.LEARNED};
        ModelPoisoning.Attack[] attacks = {ModelPoisoning.Attack.NONE, ModelPoisoning.Attack.SIGN_FLIP,
            ModelPoisoning.Attack.ALIE};
        Map<Defence, Double> worst = new EnumMap<>(Defence.class);
        for (Defence d : defences) {
            double w = 1;
            for (ModelPoisoning.Attack a : attacks) {
                RobustAggregation.Cell c = RobustAggregation.run(d, a, a == ModelPoisoning.Attack.NONE ? 0 : 0.2, 0.1,
                        true, RobustAggregationTest.ROUNDS, 1);
                RobustAggregationTest.print(c);
                w = Math.min(w, c.accuracy());
            }
            worst.put(d, w);
        }
        System.out.printf("Pre-aggregation a=0.1 20%% worst cell: FedAvg %.3f FedAvgM %.3f median %.3f bucketing %.3f "
                + "NNM+median %.3f CC %.3f learned %.3f%n", worst.get(Defence.FEDAVG), worst.get(Defence.FEDAVG_M),
                worst.get(Defence.MEDIAN), worst.get(Defence.BUCKETING_MEDIAN), worst.get(Defence.NNM_MEDIAN),
                worst.get(Defence.CENTERED_CLIP), worst.get(Defence.LEARNED));
        double best = Math.max(worst.get(Defence.NNM_MEDIAN), Math.max(worst.get(Defence.CENTERED_CLIP),
                worst.get(Defence.BUCKETING_MEDIAN)));
        assertTrue(best > worst.get(Defence.FEDAVG) + 0.1, "pre-aggregation " + best);
        assertTrue(best >= worst.get(Defence.MEDIAN) - 0.02, "pre-aggregation " + best + " vs median "
                + worst.get(Defence.MEDIAN));
    }
}
