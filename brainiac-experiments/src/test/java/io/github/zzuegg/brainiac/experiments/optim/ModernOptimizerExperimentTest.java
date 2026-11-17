package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.train.AdamMini;
import io.github.zzuegg.brainiac.experimental.train.Adopt;
import io.github.zzuegg.brainiac.experimental.train.Cautious;
import io.github.zzuegg.brainiac.experimental.train.Mars;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.MarkovSource;
import io.github.zzuegg.brainiac.experiments.optim.ModernOptimizerExperiment.Run;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.function.Function;

import org.junit.jupiter.api.Test;

class ModernOptimizerExperimentTest {

    static final MarkovSource SOURCE = MarkovSource.random(10, 1);
    static final int STEPS = 400;
    static final int EVAL = 25;

    /** {mean final loss, mean steps to the target} over three seeds at the best of the rates. */
    static double[] tuned(Function<Float, Optimizer> make, float target, float... rates) {
        double[] best = {Double.POSITIVE_INFINITY, 0, 0};
        for (float lr : rates) {
            double fin = 0;
            double to = 0;
            for (long seed = 1; seed <= 3; seed++) {
                Run run = ModernOptimizerExperiment.train(() -> make.apply(lr), SOURCE, 64, STEPS, 32, EVAL, seed);
                fin += run.finalLoss() / 3;
                int s = run.stepsTo(target, EVAL);
                to += (s < 0 ? STEPS * 2 : s) / 3.0;
            }
            if (fin < best[0]) {
                best = new double[] {fin, to, lr};
            }
        }
        return best;
    }

    @Test
    void cautiousMasksKeepAdamWAndLionTraining() {
        double[] adamW = tuned(lr -> Optimizer.adamW(lr, 0.01f), 1.2f, 3e-3f, 1e-2f);
        double[] cAdamW = tuned(lr -> Cautious.adamW(lr, 0.01f), 1.2f, 3e-3f, 1e-2f);
        double[] lion = tuned(lr -> Optimizer.lion(lr, 0.1f), 1.2f, 1e-3f, 3e-3f);
        double[] cLion = tuned(lr -> Cautious.lion(lr, 0.1f), 1.2f, 1e-3f, 3e-3f);
        System.out.printf("Cautious (loss @%d steps, tuned lr, entropy %.3f): C-AdamW %.3f vs AdamW %.3f, "
                + "C-Lion %.3f vs Lion %.3f%n", STEPS, SOURCE.entropy(), cAdamW[0], adamW[0], cLion[0], lion[0]);
        assertTrue(cAdamW[0] < 1.05 * adamW[0] && cLion[0] < 1.05 * lion[0], "cautious variants keep up");
        assertTrue(cAdamW[0] < 1.2 && cLion[0] < 1.2, "and learn the source");
    }

    @Test
    void marsReachesTheTargetLoss() {
        double[] adamW = tuned(lr -> Optimizer.adamW(lr, 0.01f), 1.2f, 3e-3f, 1e-2f, 2e-2f);
        double[] mars = tuned(lr -> Mars.adamW(lr, 0.01f), 1.2f, 3e-3f, 1e-2f, 2e-2f);
        System.out.printf("MARS-AdamW: steps to loss 1.2 %.0f vs AdamW %.0f; final %.3f vs %.3f (lr %.3f vs %.3f)%n",
                mars[1], adamW[1], mars[0], adamW[0], mars[2], adamW[2]);
        assertTrue(mars[1] <= STEPS && mars[0] < 1.05 * adamW[0], "MARS reaches the target and keeps up");
    }

    @Test
    void adoptConvergesWhereAdamFailsForAnyBeta2() {
        StringBuilder line = new StringBuilder();
        double worstAdam = -1;
        double worstAdopt = -1;
        for (float b2 : new float[] {0.1f, 0.5f, 0.9f}) {
            double adam = ModernOptimizerExperiment.counterexample(Optimizer.adam(0.01f, 0.9f, b2, 1e-8f), 50, 5000, 1);
            double adopt = ModernOptimizerExperiment.counterexample(new Adopt(0.01f, 0.9f, b2, 1e-6f, 0, false), 50,
                    5000, 1);
            worstAdam = Math.max(worstAdam, adam);
            worstAdopt = Math.max(worstAdopt, adopt);
            line.append(String.format(" b2=%.1f %+.2f/%+.2f", b2, adopt, adam));
        }
        double adamLm = 0;
        double adoptLm = 0;
        for (float b2 : new float[] {0.5f, 0.9f, 0.99f, 0.999f}) {
            adamLm = Math.max(adamLm, ModernOptimizerExperiment.train(() -> Optimizer.adam(3e-3f, 0.9f, b2, 1e-8f),
                    SOURCE, 64, STEPS, 32, EVAL, 1).finalLoss());
            adoptLm = Math.max(adoptLm, ModernOptimizerExperiment.train(() -> Adopt.of(3e-3f, b2), SOURCE, 64, STEPS,
                    32, EVAL, 1).finalLoss());
        }
        System.out.printf("ADOPT: counterexample theta (optimum -1) ADOPT/Adam%s; worst LM loss over beta2 sweep "
                + "%.3f vs Adam %.3f%n", line, adoptLm, adamLm);
        assertTrue(worstAdopt < 0.2 && worstAdam > 0.5, "ADOPT heads for the optimum, Adam drifts the wrong way");
        assertTrue(adoptLm < 1.05 * adamLm, "and ADOPT trains the LM at every beta2");
    }

    @Test
    void adamMiniHalvesOptimizerStateAtTheSameLoss() {
        Run adamW = ModernOptimizerExperiment.train(() -> Optimizer.adamW(3e-3f, 0.01f), SOURCE, 64, STEPS, 32, EVAL, 1);
        Run mini = ModernOptimizerExperiment.train(() -> AdamMini.of(3e-3f, 0.01f), SOURCE, 64, STEPS, 32, EVAL, 1);
        Run eight = ModernOptimizerExperiment.train(() -> Optimizer.adam8bit(3e-3f, 0.9f, 0.999f, 1e-8f, 0.01f),
                SOURCE, 64, STEPS, 32, EVAL, 1);
        System.out.printf("Adam-mini: loss %.3f with %d state bytes vs AdamW %.3f / %d B, Adam8bit %.3f / %d B%n",
                mini.finalLoss(), mini.stateBytes(), adamW.finalLoss(), adamW.stateBytes(), eight.finalLoss(),
                eight.stateBytes());
        assertTrue(mini.stateBytes() < 0.55 * adamW.stateBytes(), "about half of AdamW's state");
        assertTrue(mini.finalLoss() < 1.05 * adamW.finalLoss(), "at about AdamW's loss");
        assertEquals(adamW.lossCurve().length, mini.lossCurve().length);
    }
}
