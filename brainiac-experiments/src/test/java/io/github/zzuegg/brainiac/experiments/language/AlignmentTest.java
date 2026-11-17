package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.RejectionSamplingFineTuning;
import io.github.zzuegg.brainiac.experimental.llm.RewardModel;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class AlignmentTest {

    @Test
    void lossesHaveTheirClosedForms() {
        Var c = Var.param(Tensor.of(1f, 2f));
        Var r = Var.param(Tensor.of(0f, 2f));
        double bt = (Math.log1p(Math.exp(-1)) + Math.log(2)) / 2;
        assertEquals(bt, PreferenceLosses.bradleyTerry(c, r).value().item(), 1e-5);
        Var zero = Var.of(Tensor.of(0f, 0f));
        // IPO: margins 1 and 0 against a target of 1/(2*0.5) = 1
        assertEquals(0.5, PreferenceLosses.ipo(c, r, zero, zero, 0.5f).value().item(), 1e-5);
        // SimPO with beta 1, gamma 0 equals Bradley-Terry on the mean log-probs
        assertEquals(bt, PreferenceLosses.simpo(c.sub(3), r.sub(3), 1, 0).value().item(), 1e-5);
    }

    @Test
    void preferenceObjectivesRaiseTheChosenMargin() {
        HookedDecoder base = PeftShootoutTest.base();
        for (PreferenceTuning.Objective o : PreferenceTuning.Objective.values()) {
            PreferenceTuning.Result res = PreferenceTuning.run(base, o, 25, 4);
            System.out.printf("%-6s chosen margin %.3f -> %.3f%n", o, res.marginBefore(), res.marginAfter());
            assertTrue(res.marginAfter() > res.marginBefore() + 2, o + ": " + res);
        }
    }

    @Test
    void rewardModelLearnsPreferencesAndRejectionSamplingClimbsIt() {
        HookedDecoder backbone = HookedDecoder.create(HookedDecoder.Settings.tiny(12), 31);
        RewardModel rm = new RewardModel(backbone);
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(2);
        for (int s = 0; s < 60; s++) {
            int[][][] p = PreferenceTuning.pairs(r, 16, 12);
            Var loss = rm.loss(p[0], p[1]);
            loss.backward();
            opt.step(rm.parameters());
            rm.zeroGrad();
        }
        int[][][] test = PreferenceTuning.pairs(new SplittableRandom(77), 64, 12);
        System.out.printf("reward model preference accuracy %.3f%n", rm.accuracy(test[0], test[1]));
        assertTrue(rm.accuracy(test[0], test[1]) > 0.9, "reward model accuracy " + rm.accuracy(test[0], test[1]));

        HookedDecoder policy = PeftShootoutTest.base().copy();
        RejectionSamplingFineTuning rft = new RejectionSamplingFineTuning(policy,
                seq -> rm.score(new int[][] {seq}).value().item(), Optimizer.adam(0.005f), policy.parameters(), 1.5f);
        SplittableRandom pr = new SplittableRandom(5);
        double first = Double.NaN;
        double last = Double.NaN;
        for (int round = 0; round < 4; round++) {
            int[][] prompts = new int[6][2];
            for (int[] p : prompts) {
                p[0] = pr.nextInt(12);
                p[1] = (p[0] + 1) % 12;
            }
            RejectionSamplingFineTuning.Round res = rft.round(prompts, 4, 4, Double.NEGATIVE_INFINITY, 5, pr);
            System.out.printf("rejection sampling round %d: mean reward %.3f, kept reward %.3f, kept %d%n", round,
                    res.meanReward(), res.keptReward(), res.kept());
            first = Double.isNaN(first) ? res.meanReward() : first;
            last = res.meanReward();
            assertTrue(res.keptReward() >= res.meanReward());
        }
        assertTrue(last > first, "mean reward " + first + " -> " + last);
    }
}
