package io.github.zzuegg.brainiac.experiments.multitask;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Method;
import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Problem;
import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Result;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

/** Item 10.3.16: FAMO against equal weights and Nash-MTL, in Delta_m and cost per step. */
class FamoTest {

    @Test
    void famoBalancesAtTheCostOfOneBackward() {
        for (Problem p : new Problem[] {Problem.DIGITS, Problem.SCALES}) {
            double[] single = GradientBalancing.singleTask(p, 250, 1);
            Result equal = GradientBalancing.run(p, Method.EQUAL, 250, 1, single);
            Result famo = GradientBalancing.run(p, Method.FAMO, 250, 1, single);
            Result nash = GradientBalancing.run(p, Method.NASH_MTL, 250, 1, single);
            System.out.println(p + " single-task " + Arrays.toString(single) + " | " + equal + " | " + famo + " | "
                    + nash);
            assertTrue(Double.isFinite(famo.deltaM()), famo.toString());
            assertTrue(famo.millisPerStep() < nash.millisPerStep(), "FAMO is cheaper than per-task gradients");
        }
    }
}
