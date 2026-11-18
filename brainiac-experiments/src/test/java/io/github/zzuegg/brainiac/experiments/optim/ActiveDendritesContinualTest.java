package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.train.ContinualBrain;
import io.github.zzuegg.brainiac.experimental.train.ForgettingTracker;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.5: active dendrites on permuted digits with a task context. */
class ActiveDendritesContinualTest {

    @Test
    void dendritesForgetLessThanPlainAndSiNetworks() {
        List<ContinualLearningExperiment.Task> tasks = ContinualLearningExperiment.permutedDigits(5, 30, 1);
        ForgettingTracker dend = ActiveDendritesContinual.run(tasks, ActiveDendritesContinual.Method.DENDRITES, 15, 1);
        ForgettingTracker mlp = ActiveDendritesContinual.run(tasks, ActiveDendritesContinual.Method.MLP, 15, 1);
        ForgettingTracker ctx = ActiveDendritesContinual.run(tasks, ActiveDendritesContinual.Method.MLP_CONTEXT, 15, 1);
        ForgettingTracker si = ActiveDendritesContinual.run(tasks, ActiveDendritesContinual.Method.MLP_SI, 15, 1);
        ForgettingTracker ewc = ContinualLearningExperiment.run(tasks, ContinualBrain.Protection.SHARED_EWC, 32, 15, 1);
        ForgettingTracker frozen = ContinualLearningExperiment.run(tasks, ContinualBrain.Protection.GROW_FROZEN, 32,
                15, 1);
        System.out.printf("permuted digits x5, avg acc / forgetting: dendrites %.2f/%.2f, MLP %.2f/%.2f,"
                + " MLP+context %.2f/%.2f, MLP+SI %.2f/%.2f; multi-head ContinualBrain EWC %.2f/%.2f,"
                + " GROW_FROZEN %.2f/%.2f%n", dend.averageAccuracy(), dend.averageForgetting(), mlp.averageAccuracy(),
                mlp.averageForgetting(), ctx.averageAccuracy(), ctx.averageForgetting(), si.averageAccuracy(),
                si.averageForgetting(), ewc.averageAccuracy(), ewc.averageForgetting(), frozen.averageAccuracy(),
                frozen.averageForgetting());
        assertTrue(dend.averageForgetting() < mlp.averageForgetting(), "dendrites forget less than the MLP");
        assertTrue(dend.averageAccuracy() > mlp.averageAccuracy(), "and keep more on average");
    }
}
