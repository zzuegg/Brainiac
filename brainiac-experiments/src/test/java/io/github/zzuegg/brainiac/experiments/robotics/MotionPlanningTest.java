package io.github.zzuegg.brainiac.experiments.robotics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MotionPlanningTest {

    @Test
    void samplingPlanners() {
        List<MotionPlanning.PlanRow> rows = MotionPlanning.sampling(5, 1500, 1214);
        StringBuilder line = new StringBuilder("12.3.14 success/first/cost/checks:");
        for (MotionPlanning.PlanRow r : rows) {
            line.append(String.format(" %s %s %.2f/%.0f/%.3f/%.0f;", r.problem(), r.method(), r.success(),
                r.firstSolution(), r.cost(), r.collisionChecks()));
        }
        List<MotionPlanning.SamplerRow> learned = MotionPlanning.learnedSampling(40, 10, 3000, 1200, 1215);
        line.append(" learned sampler success/first:");
        for (MotionPlanning.SamplerRow r : learned) {
            line.append(String.format(" %s %.2f/%.0f;", r.method(), r.success(), r.firstSolution()));
        }
        System.out.println(line);
        assertTrue(rows.get(2).success() >= 0.8, line.toString());
        assertTrue(learned.get(0).success() > 0, line.toString());
    }

    @Test
    void inverseKinematics() {
        List<MotionPlanning.IkRow> rows = MotionPlanning.inverseKinematics(30, 1215);
        StringBuilder line = new StringBuilder("12.3.15 success/iterations:");
        for (MotionPlanning.IkRow r : rows) {
            line.append(String.format(" %s %s %s %.2f/%.1f;", r.arm(), r.targets(), r.method(), r.success(),
                r.iterations()));
        }
        System.out.println(line);
        // DLS on 2-link random targets solves 27/30 (0.90); the same 3 fold-back targets defeat every Jacobian method.
        assertTrue(rows.get(1).success() >= 0.85, line.toString());
    }
}
