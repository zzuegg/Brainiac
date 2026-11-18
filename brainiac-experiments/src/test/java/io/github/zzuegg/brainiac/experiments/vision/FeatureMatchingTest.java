package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class FeatureMatchingTest {

    @Test
    void homographies() {
        List<FeatureMatching.Row> rows = FeatureMatching.run(30, 15, 150, 101);
        StringBuilder line = new StringBuilder("11.5.14 accuracy/precision/inliers:");
        String cond = "";
        for (FeatureMatching.Row r : rows) {
            if (!r.condition().equals(cond)) {
                cond = r.condition();
                line.append(" | ").append(cond).append(':');
            }
            line.append(String.format(" %s %.2f/%.2f/%.2f;", r.method(), r.accuracy(), r.precision(), r.inliers()));
        }
        System.out.println(line);
        assertTrue(rows.get(0).accuracy() > 0.4, line.toString());
    }
}
