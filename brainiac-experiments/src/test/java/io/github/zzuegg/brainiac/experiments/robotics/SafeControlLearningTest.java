package io.github.zzuegg.brainiac.experiments.robotics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SafeControlLearningTest {

    private static String format(String head, List<SafeControl.Row> rows) {
        StringBuilder line = new StringBuilder(head);
        for (SafeControl.Row r : rows) {
            line.append(String.format(" %s %.2f/%.2f/%.2f/%.0f/%.3f;", r.method(), r.success(), r.violations(),
                r.meanReturn(), r.trainingViolations(), r.interventions()));
        }
        return line.toString();
    }

    @Test
    void safeLearning() {
        List<SafeControl.Row> rows = SafeControl.learners(3000, 10, 2f, 1217);
        String line = format("12.3.16 RL success/violations/return/train-violations:", rows);
        System.out.println(line);
        assertTrue(rows.get(1).trainingViolations() <= rows.get(0).trainingViolations(), line);
    }
}
