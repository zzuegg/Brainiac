package io.github.zzuegg.brainiac.experiments.robotics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SafeControlTest {

    private static String format(String head, List<SafeControl.Row> rows) {
        StringBuilder line = new StringBuilder(head);
        for (SafeControl.Row r : rows) {
            line.append(String.format(" %s %.2f/%.2f/%.2f/%.0f/%.3f;", r.method(), r.success(), r.violations(),
                r.meanReturn(), r.trainingViolations(), r.interventions()));
        }
        return line.toString();
    }

    @Test
    void barrierFilteredController() {
        List<SafeControl.Row> rows = SafeControl.controllers(20, 1216);
        String line = format("12.3.16 success/violations/return/train-violations/interventions:", rows);
        System.out.println(line);
        assertTrue(rows.get(0).violations() > 0, line);
        assertTrue(rows.get(1).violations() < rows.get(0).violations(), line);
    }
}
