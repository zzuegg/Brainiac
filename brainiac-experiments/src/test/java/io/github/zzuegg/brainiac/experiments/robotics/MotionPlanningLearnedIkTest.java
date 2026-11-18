package io.github.zzuegg.brainiac.experiments.robotics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MotionPlanningLearnedIkTest {

    @Test
    void learnedIk() {
        List<MotionPlanning.LearnedIkRow> learned = MotionPlanning.learnedIk(20, 32, 700, 1216);
        StringBuilder line = new StringBuilder("12.3.15 learned IK error/refined/branches:");
        for (MotionPlanning.LearnedIkRow r : learned) {
            line.append(String.format(" %s %.3f/%.4f/%.2f;", r.method(), r.error(), r.refined(), r.branches()));
        }
        System.out.println(line);
        assertTrue(learned.get(0).branches() >= learned.get(1).branches(), line.toString());
    }
}
