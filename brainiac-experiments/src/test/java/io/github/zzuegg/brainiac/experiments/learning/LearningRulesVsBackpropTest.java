package io.github.zzuegg.brainiac.experiments.learning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.DifferenceTargetPropagation;
import io.github.zzuegg.brainiac.experimental.train.EquilibriumPropagation;
import io.github.zzuegg.brainiac.experimental.train.FeedbackAlignment;
import io.github.zzuegg.brainiac.experimental.train.ForwardForward;
import io.github.zzuegg.brainiac.experimental.train.ForwardGradientLearner;
import io.github.zzuegg.brainiac.experimental.train.MonoForward;
import io.github.zzuegg.brainiac.experimental.train.NoProp;
import io.github.zzuegg.brainiac.experimental.train.PredictiveCodingNetwork;
import io.github.zzuegg.brainiac.experimental.train.SoftHebbClassifier;
import io.github.zzuegg.brainiac.experimental.train.SyntheticGradients;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.train.Optimizer;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Each learning rule against backpropagation on the same spiral or digits task and budget. */
class LearningRulesVsBackpropTest {

    static final ToyTasks.Data SPIRAL = ToyTasks.spiral(100, 3, 0.1, 1);
    static final ToyTasks.Data SPIRAL_TEST = ToyTasks.spiral(50, 3, 0.1, 2);
    static final ToyTasks.Data DIGITS = ToyTasks.digits(10, 30, 8, 0.1, 3);
    static final ToyTasks.Data DIGITS_TEST = ToyTasks.digits(10, 20, 8, 0.1, 3, 99);
    static LearnerComparison.Result spiralBackprop;
    static LearnerComparison.Result digitsBackprop;

    @BeforeAll
    static void baselines() {
        spiralBackprop = LearnerComparison.backprop(SPIRAL, SPIRAL_TEST, 60, 50, 0.02f, 1, 32, 32);
        digitsBackprop = LearnerComparison.backprop(DIGITS, DIGITS_TEST, 20, 50, 0.01f, 1, 64);
        System.out.println("spiral " + spiralBackprop);
        System.out.println("digits " + digitsBackprop);
    }

    private static void close(LearnerComparison.Result rule, LearnerComparison.Result backprop, double gap) {
        System.out.println(rule);
        assertTrue(rule.testAccuracy() >= backprop.testAccuracy() - gap,
                rule + " is more than " + gap + " behind " + backprop);
    }

    @Test
    void feedbackAlignmentFamilyMatchesBackpropOnSpiral() {
        for (FeedbackAlignment.Mode m : FeedbackAlignment.Mode.values()) {
            close(LearnerComparison.run(m.name(), new FeedbackAlignment(SimpleMlp.tanh(1, 2, 32, 32, 3),
                    Optimizer.adam(0.02f), m, 5), SPIRAL, SPIRAL_TEST, 60, 50), spiralBackprop, 0.1);
        }
    }

    @Test
    void forwardForwardWithHardNegativesLearnsDigits() {
        ForwardForward ff = ForwardForward.builder(64, 10).hidden(64, 64)
                .negatives(ForwardForward.NegativeData.hardestLabel()).build();
        close(LearnerComparison.run("forward-forward", ff, DIGITS, DIGITS_TEST, 40, 50), digitsBackprop, 0.1);
    }

    @Test
    void monoForwardMatchesBackpropOnSpiral() {
        close(LearnerComparison.run("mono-forward", new MonoForward(2, 3, Optimizer.adam(0.02f), 1, 32, 32),
                SPIRAL, SPIRAL_TEST, 60, 50), spiralBackprop, 0.05);
    }

    @Test
    void noPropDenoisingBlocksLearnSpiral() {
        close(LearnerComparison.run("noprop", new NoProp(2, 3, 4, 64, Optimizer.adam(0.01f), 1),
                SPIRAL, SPIRAL_TEST, 60, 50), spiralBackprop, 0.12);
    }

    @Test
    void differenceTargetPropagationLearnsSpiral() {
        close(LearnerComparison.run("difference target prop", new DifferenceTargetPropagation(
                SimpleMlp.tanh(1, 2, 32, 32, 3), Optimizer.adam(0.01f), 0.5f, 0.1f, 1), SPIRAL, SPIRAL_TEST, 60, 50),
                spiralBackprop, 0.15);
    }

    @Test
    void syntheticGradientsLearnSpiral() {
        close(LearnerComparison.run("synthetic gradients (cDNI)", new SyntheticGradients(
                SimpleMlp.tanh(1, 2, 32, 32, 3), Optimizer.adam(0.01f), true), SPIRAL, SPIRAL_TEST, 60, 50),
                spiralBackprop, 0.12);
    }

    @Test
    void predictiveCodingLearnsSpiralAndDigits() {
        close(LearnerComparison.run("predictive coding", PredictiveCodingNetwork.layered(1, Optimizer.adam(0.01f),
                2, 32, 32, 3), SPIRAL, SPIRAL_TEST, 60, 50), spiralBackprop, 0.1);
        close(LearnerComparison.run("predictive coding digits", PredictiveCodingNetwork.layered(1,
                Optimizer.adam(0.005f), 64, 64, 10), DIGITS, DIGITS_TEST, 20, 50), digitsBackprop, 0.1);
    }

    @Test
    void predictiveCodingLearnsOnASparseBrainRegionGraph() {
        Brain.Spec spec = Brain.builder().region("sensory", 32).region("association", 32)
                .connect("input", "sensory", 1).connect("sensory", "association", 0.5).output("association").build();
        close(LearnerComparison.run("predictive coding on brain", PredictiveCodingNetwork.fromBrain(spec, 2, 3, 1,
                Optimizer.adam(0.01f)), SPIRAL, SPIRAL_TEST, 60, 50), spiralBackprop, 0.15);
        // with a recurrent feedback projection it still learns, though less well
        Brain.Spec loop = Brain.builder().region("sensory", 32).region("association", 32)
                .connect("input", "sensory", 1).connect("sensory", "association", 0.5)
                .connect("association", "sensory", 0.3).output("association").build();
        LearnerComparison.Result r = LearnerComparison.run("predictive coding on looped brain",
                PredictiveCodingNetwork.fromBrain(loop, 2, 3, 1, Optimizer.adam(0.01f)), SPIRAL, SPIRAL_TEST, 30, 50);
        System.out.println(r);
        assertTrue(r.testAccuracy() > 0.42, r.toString());
    }

    @Test
    void equilibriumPropagationMatchesBackpropOnSpiral() {
        close(LearnerComparison.run("equilibrium propagation", new EquilibriumPropagation(Optimizer.adam(0.01f),
                0.5f, 20, 8, 1, 2, 64, 3), SPIRAL, SPIRAL_TEST, 60, 50), spiralBackprop, 0.05);
    }

    @Test
    void softHebbFeaturesWithALinearReadoutLearnDigits() {
        close(LearnerComparison.run("softhebb + readout", new SoftHebbClassifier(64, 256, 10, 0.05f, 0.1f,
                Optimizer.adam(0.05f), 1), DIGITS, DIGITS_TEST, 40, 50), digitsBackprop, 0.1);
    }

    @Test
    void forwardGradientsWithoutABackwardPassLearnSpiral() {
        close(LearnerComparison.run("forward gradient", new ForwardGradientLearner(SimpleMlp.tanh(1, 2, 32, 32, 3),
                Optimizer.adam(0.01f), 32, 2), SPIRAL, SPIRAL_TEST, 60, 50), spiralBackprop, 0.25);
    }
}
