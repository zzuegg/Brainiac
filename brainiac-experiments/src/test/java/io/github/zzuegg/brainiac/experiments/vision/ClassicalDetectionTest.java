package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ClassicalDetectionTest {

    @Test
    void hogAndHaar() {
        List<ClassicalDetection.Row> rows = ClassicalDetection.run(40, 20, 1231);
        StringBuilder line = new StringBuilder("12.3.1-2 AP/FP-per-image/ms-per-image/features-per-window:");
        for (ClassicalDetection.Row r : rows) {
            line.append(String.format(" %s %.3f/%.2f/%.1f/%.1f;", r.method(), r.ap(), r.falsePositives(),
                r.millisPerImage(), r.featuresPerWindow()));
        }
        System.out.println(line);
        assertTrue(rows.get(2).ap() > 0.3, line.toString());
        assertTrue(rows.get(4).featuresPerWindow() < rows.get(3).featuresPerWindow(), line.toString());
    }
}
