package io.github.zzuegg.brainiac.experiments.edge;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class KeywordBaselinesTest {

    @Test
    void unconstrainedBaselines() {
        List<EdgeDeployment.KeywordRow> rows = EdgeDeployment.keywordBaselines(150, 3, 5);
        for (EdgeDeployment.KeywordRow r : rows) {
            System.out.printf("9.6.17 %-13s %-34s %7d B, fits small %-5s, accuracy %.3f%n", r.budget(), r.family(),
                    r.weightBytes(), r.fits(), r.floatAccuracy());
        }
        double dtw = rows.get(rows.size() - 1).floatAccuracy();
        assertTrue(dtw > 1.0 / EdgeDeployment.KWS_CLASSES + 0.2, "DTW templates work: " + dtw);
    }
}
