package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** The low-label rows of {@link SeriesClassification} (TS2Vec against SimCLR and the rest). */
class SelfSupervisedSeriesTest {

    @Test
    void ts2vecAtTenPercentLabels() {
        Map<String, Double> r = SeriesClassification.lowLabels("shapelets", 200, 20, 40, 2);
        r.forEach((k, v) -> System.out.printf("series classification 10%% labels %s: acc %.3f%n", k, v));
        assertTrue(r.get("TS2Vec probe") > 0.5, r.toString());
        assertTrue(r.get("TS2Vec probe") > r.get("raw + ridge"), r.toString());
    }
}
