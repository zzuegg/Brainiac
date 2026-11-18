package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experiments.control.ClassicalControl.PrimitiveScore;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class MovementPrimitiveTest {

    @Test
    void primitivesGeneraliseToNewGoals() {
        StringBuilder sb = new StringBuilder();
        PrimitiveScore one = null;
        for (int demos : new int[] {1, 5, 20}) {
            PrimitiveScore s = ClassicalControl.primitives(demos, 400, 3);
            one = one == null ? s : one;
            sb.append(String.format("%d demos: goal error DMP %.4f / BC %.3f, path error %.3f / %.3f, jerk %.0f / %.0f "
                    + "(ideal %.0f); ", demos, s.dmpGoalError(), s.bcGoalError(), s.dmpPathError(), s.bcPathError(),
                    s.dmpJerk(), s.bcJerk(), s.idealJerk()));
        }
        double[] refine = ClassicalControl.refinement(30, 10, 3);
        System.out.printf("2-D reach to new goals: %s| reward-weighted refinement to a via-point: error %.3f -> %.4f "
                + "(goal error %.4f)%n", sb, refine[0], refine[1], refine[2]);
        assertTrue(one.dmpGoalError() < 0.02 && one.dmpGoalError() < one.bcGoalError(), "DMP reaches new goals");
        assertTrue(refine[1] < 0.3 * refine[0], "refinement reaches the via-point");
    }
}
