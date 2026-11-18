package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class DevelopmentalInterpTest {

    @Test
    void llcIsTrackedOverTraining() {
        DevelopmentalInterp.Study s = DevelopmentalInterp.run(1200, 300, 300, 1);
        for (DevelopmentalInterp.Checkpoint c : s.curve()) {
            System.out.printf("LLC step %5d train %.3f test %.3f LLC %.2f Hessian trace %.2f top eig %.3f norm %.2f%n",
                    c.step(), c.trainAccuracy(), c.testAccuracy(), c.llc(), c.hessianTrace(), c.topEigenvalue(),
                    c.weightNorm());
        }
        System.out.printf("LLC stability over SGLD step sizes %s; Spearman with test accuracy: LLC %.2f norm %.2f;"
                + " diverged %s%n", Arrays.toString(s.stability()), s.rankLlc(), s.rankNorm(), s.diverged());
        DevelopmentalInterp.Checkpoint last = s.curve().get(s.curve().size() - 1);
        assertTrue(last.trainAccuracy() > 0.9, "the training set is memorised");
        assertTrue(Double.isFinite(last.llc()), "the LLC estimate is finite");
    }
}
