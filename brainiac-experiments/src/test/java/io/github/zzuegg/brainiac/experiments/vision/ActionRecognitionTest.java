package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ActionRecognitionTest {

    @Test
    void actions() {
        List<ActionRecognition.Row> rows = ActionRecognition.run(240, 80, 150, 111);
        StringBuilder line = new StringBuilder("11.5.13 accuracy/direction accuracy (MACs):");
        for (ActionRecognition.Row r : rows) {
            line.append(String.format(" %s %.3f/%.3f (%d);", r.method(), r.accuracy(), r.direction(), r.macs()));
        }
        System.out.println(line);
        double best = rows.stream().mapToDouble(ActionRecognition.Row::accuracy).max().orElse(0);
        assertTrue(best > 0.3, line.toString());
    }
}
