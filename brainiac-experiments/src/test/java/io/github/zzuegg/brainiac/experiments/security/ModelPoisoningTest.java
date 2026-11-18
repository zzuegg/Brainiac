package io.github.zzuegg.brainiac.experiments.security;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.security.ModelPoisoning;
import io.github.zzuegg.brainiac.experiments.security.RobustAggregation.Defence;

import org.junit.jupiter.api.Test;

class ModelPoisoningTest {

    private static final int ROUNDS = RobustAggregationTest.ROUNDS;

    @Test
    void strongAttacksHurtMedianAndMultiKrumMoreThanSignFlips() {
        double[][] acc = new double[2][];
        Defence[] rules = {Defence.MEDIAN, Defence.MULTI_KRUM};
        for (int k = 0; k < rules.length; k++) {
            RobustAggregation.Cell none = RobustAggregation.run(rules[k], ModelPoisoning.Attack.NONE, 0.2, 0, true,
                    ROUNDS, 1);
            RobustAggregation.Cell flip = RobustAggregation.run(rules[k], ModelPoisoning.Attack.SIGN_FLIP, 0.2, 0, true,
                    ROUNDS, 1);
            RobustAggregation.Cell noise = RobustAggregation.run(rules[k], ModelPoisoning.Attack.GAUSSIAN, 1, 0.2, 0,
                    true, ROUNDS, 1);
            RobustAggregation.Cell alie = RobustAggregation.run(rules[k], ModelPoisoning.Attack.ALIE, 0.2, 0, true,
                    ROUNDS, 1);
            RobustAggregation.Cell alieBlind = RobustAggregation.run(rules[k], ModelPoisoning.Attack.ALIE, 0.2, 0, false,
                    ROUNDS, 1);
            RobustAggregation.Cell ipm = RobustAggregation.run(rules[k], ModelPoisoning.Attack.IPM, 0.2, 0, true,
                    ROUNDS, 1);
            acc[k] = new double[] {none.accuracy(), flip.accuracy(), noise.accuracy(), alie.accuracy(),
                alieBlind.accuracy(), ipm.accuracy()};
            System.out.printf("Poisoning IID 20%% %-10s none %.3f sign flip %.3f same-norm noise %.3f ALIE %.3f "
                    + "ALIE (non-omniscient) %.3f IPM %.3f%n", rules[k], acc[k][0], acc[k][1], acc[k][2], acc[k][3],
                    acc[k][4], acc[k][5]);
        }
        for (double[] a : acc) {
            assertTrue(a[0] > 0.5, "no-attack accuracy " + a[0]);
            assertTrue(Math.min(a[3], a[5]) < a[1], "ALIE or IPM hurts more than sign flips");
        }
    }

    @Test
    void modelReplacementImplantsTheTriggerUnderFedAvgButNotCenteredClipping() {
        RobustAggregation.Cell avg = RobustAggregation.run(Defence.FEDAVG, ModelPoisoning.Attack.MODEL_REPLACEMENT,
                0.1, 0, true, ROUNDS, 1);
        RobustAggregation.Cell clip = RobustAggregation.run(Defence.CENTERED_CLIP,
                ModelPoisoning.Attack.MODEL_REPLACEMENT, 0.1, 0, true, ROUNDS, 1);
        RobustAggregation.Cell median = RobustAggregation.run(Defence.MEDIAN, ModelPoisoning.Attack.MODEL_REPLACEMENT,
                0.1, 0, true, ROUNDS, 1);
        RobustAggregation.Cell none = RobustAggregation.run(Defence.FEDAVG, ModelPoisoning.Attack.NONE, 0.1, 0, true,
                ROUNDS, 1);
        RobustAggregationTest.print(avg);
        RobustAggregationTest.print(clip);
        RobustAggregationTest.print(median);
        System.out.printf("Model replacement 10%%: backdoor success FedAvg %.3f, centred clipping %.3f, median %.3f; "
                + "clean acc FedAvg %.3f vs no attack %.3f%n", avg.backdoor(), clip.backdoor(), median.backdoor(),
                avg.accuracy(), none.accuracy());
        assertTrue(avg.backdoor() > 0.5, "FedAvg backdoor " + avg.backdoor());
        assertTrue(clip.backdoor() < avg.backdoor() - 0.1, "clipping " + clip.backdoor());
    }
}
