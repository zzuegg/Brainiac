package io.github.zzuegg.brainiac.experiments.learning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.train.PredictiveCodingNetwork;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.train.Optimizer;

import org.junit.jupiter.api.Test;

/** Item 7.6.17: incremental predictive coding against standard PC on the looped Brain graph (the negative row). */
class IncrementalPredictiveCodingTest {

    @Test
    void incrementalPcOnLoopedAndFeedForwardBrains() {
        ToyTasks.Data train = LearningRulesVsBackpropTest.SPIRAL;
        ToyTasks.Data test = LearningRulesVsBackpropTest.SPIRAL_TEST;
        Brain.Spec ff = Brain.builder().region("sensory", 32).region("association", 32)
                .connect("input", "sensory", 1).connect("sensory", "association", 0.5).output("association").build();
        Brain.Spec loop = Brain.builder().region("sensory", 32).region("association", 32)
                .connect("input", "sensory", 1).connect("sensory", "association", 0.5)
                .connect("association", "sensory", 0.3).output("association").build();
        LearnerComparison.Result loopPc = LearnerComparison.run("PC looped", PredictiveCodingNetwork.fromBrain(loop,
                2, 3, 1, Optimizer.adam(0.01f), false), train, test, 30, 50);
        LearnerComparison.Result loopIpc = LearnerComparison.run("iPC looped", PredictiveCodingNetwork.fromBrain(loop,
                2, 3, 1, Optimizer.adam(0.002f), true), train, test, 30, 50);
        LearnerComparison.Result ffPc = LearnerComparison.run("PC feed-forward", PredictiveCodingNetwork.fromBrain(ff,
                2, 3, 1, Optimizer.adam(0.01f), false), train, test, 30, 50);
        LearnerComparison.Result ffIpc = LearnerComparison.run("iPC feed-forward",
                PredictiveCodingNetwork.fromBrain(ff, 2, 3, 1, Optimizer.adam(0.002f), true), train, test, 30, 50);
        LearnerComparison.Result bp = LearnerComparison.backprop(train, test, 30, 50, 0.02f, 1, 32, 32);
        System.out.printf("Spiral test acc, 30 epochs: looped brain PC %.3f (%d ms) vs iPC %.3f (%d ms);"
                + " feed-forward brain PC %.3f (%d ms) vs iPC %.3f (%d ms); backprop MLP %.3f%n",
                loopPc.testAccuracy(), loopPc.millis(), loopIpc.testAccuracy(), loopIpc.millis(), ffPc.testAccuracy(),
                ffPc.millis(), ffIpc.testAccuracy(), ffIpc.millis(), bp.testAccuracy());
        assertTrue(loopIpc.testAccuracy() > loopPc.testAccuracy(), "incremental PC beats PC on the looped graph");
    }
}
