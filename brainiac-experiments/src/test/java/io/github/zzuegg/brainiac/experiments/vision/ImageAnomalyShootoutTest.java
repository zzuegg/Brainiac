package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ImageAnomalyShootoutTest {

    @Test
    void shootout() {
        List<ImageAnomalyShootout.Row> rows = ImageAnomalyShootout.run(40, 12, 3, 160, 71);
        StringBuilder line = new StringBuilder("11.5.11-12 image/pixel AUROC/PRO/misplaced AUROC:");
        String setting = "";
        for (ImageAnomalyShootout.Row r : rows) {
            if (!r.setting().equals(setting)) {
                setting = r.setting();
                line.append(" | ").append(setting).append(':');
            }
            line.append(String.format(" %s %.3f/%.3f/%.3f/%.3f;", r.method(), r.image(), r.pixel(), r.pro(),
                    r.misplaced()));
        }
        System.out.println(line);
        assertTrue(rows.get(0).image() > 0.6, line.toString());
    }
}
