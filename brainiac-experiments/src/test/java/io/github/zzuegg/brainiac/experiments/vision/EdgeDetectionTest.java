package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class EdgeDetectionTest {

    @Test
    void edges() {
        List<EdgeDetection.Row> rows = EdgeDetection.run(80, 20, 150, 0.2, 81);
        StringBuilder line = new StringBuilder("11.5.15 ODS/OIS/AP:");
        for (EdgeDetection.Row r : rows) {
            line.append(String.format(" %s %.3f/%.3f/%.3f;", r.method(), r.scores().ods(), r.scores().ois(),
                    r.scores().ap()));
        }
        System.out.println(line);
        assertTrue(rows.get(2).scores().ods() > 0.3, line.toString());
    }
}
