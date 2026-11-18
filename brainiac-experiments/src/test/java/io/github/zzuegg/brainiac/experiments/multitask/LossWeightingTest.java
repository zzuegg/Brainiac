package io.github.zzuegg.brainiac.experiments.multitask;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.multitask.LossWeighting;
import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Method;
import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Problem;
import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Result;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

/** Item 10.3.13: uncertainty weighting, GradNorm and DWA against equal weights and single-task models. */
class LossWeightingTest {

    @Test
    void weightingsOnScaleMismatchAndDigits() {
        for (Problem p : new Problem[] {Problem.SCALES, Problem.DIGITS}) {
            double[] single = GradientBalancing.singleTask(p, 250, 1);
            System.out.println(p + " single-task " + Arrays.toString(single));
            for (Method m : new Method[] {Method.EQUAL, Method.UNCERTAINTY, Method.GRADNORM, Method.DWA}) {
                Result r = GradientBalancing.run(p, m, 250, 1, single);
                System.out.println(r);
                assertTrue(Double.isFinite(r.deltaM()), r.toString());
                assertTrue(r.deltaM() > -100, "the shared model learns every task: " + r);
            }
        }
    }

    @Test
    void mmoeOnClicks() {
        double[] single = GradientBalancing.singleTask(Problem.CLICKS, 250, 1);
        Result equal = GradientBalancing.run(Problem.CLICKS, Method.EQUAL, 250, 1, single);
        Result uncertainty = GradientBalancing.run(Problem.CLICKS, Method.UNCERTAINTY, 250, 1, single);
        Result mmoe = GradientBalancing.mmoe(LossWeighting.Kind.EQUAL, 250, 1, single);
        Result mmoeU = GradientBalancing.mmoe(LossWeighting.Kind.UNCERTAINTY, 250, 1, single);
        System.out.println("CLICKS single-task " + Arrays.toString(single) + " | " + equal + " | " + uncertainty
                + " | " + mmoe + " | " + mmoeU);
        assertTrue(equal.metrics()[0] > 0.55 && mmoe.metrics()[0] > 0.55, "click AUC beats chance");
    }
}
