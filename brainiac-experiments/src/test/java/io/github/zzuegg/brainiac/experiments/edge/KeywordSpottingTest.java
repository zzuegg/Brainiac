package io.github.zzuegg.brainiac.experiments.edge;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.edge.KeywordSpotters;
import java.util.List;
import org.junit.jupiter.api.Test;

class KeywordSpottingTest {

    @Test
    void dsCnnKeywordSpotterFitsSmallBudget() {
        List<EdgeDeployment.KeywordRow> rows = EdgeDeployment.keywords(List.of(KeywordSpotters.BUDGETS.get(0)),
                KeywordSpotters.Family.values(), 80, 5);
        for (EdgeDeployment.KeywordRow r : rows) {
            System.out.printf("9.6.17 %-13s %-34s %7d B, %8d MACs, fits %-5s peak %6d B, float %.3f, int8 %.3f%n",
                    r.budget(), r.family(), r.weightBytes(), r.macs(), r.fits(), r.peakBytes(), r.floatAccuracy(),
                    r.int8Accuracy());
        }
        EdgeDeployment.KeywordRow ds = rows.stream().filter(r -> r.family().equals("DS_CNN")).findFirst()
                .orElseThrow();
        assertTrue(ds.fits(), ds.toString());
        assertTrue(ds.floatAccuracy() > 1.0 / EdgeDeployment.KWS_CLASSES + 0.2, ds.toString());
    }
}
