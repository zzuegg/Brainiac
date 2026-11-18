package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class AnomalyBenchmarkTest {

    private static void print(String task, Map<String, double[]> r) {
        r.forEach((k, v) -> System.out.printf("anomaly %s %s: AUROC %.3f (%.0f ms)%n", task, k, v[0], v[1]));
    }

    @Test
    void tabularDetectors() {
        Map<String, double[]> r = AnomalyBenchmark.tabular(0, 1);
        print("tabular", r);
        Map<String, double[]> noisy = AnomalyBenchmark.tabular(20, 1);
        print("tabular+noise", noisy);
        for (String m : new String[] {"LOF", "one-class SVM", "Deep SVDD", "Deep SAD (1% labels)"}) {
            assertTrue(r.get(m)[0] > 0.75, m + " " + r.get(m)[0]);
        }
        // ECOD scores features independently: the outliers sit inside each marginal's range
        assertTrue(r.get("ECOD")[0] > 0.6, r.toString());
        assertTrue(r.get("Deep SAD (1% labels)")[0] >= r.get("Deep SVDD")[0], r.toString());
        assertTrue(noisy.get("Deep SAD (1% labels)")[0] > noisy.get("one-class SVM")[0], r.toString());
    }

    @Test
    void lofWinsOnLocalDensity() {
        Map<String, double[]> r = AnomalyBenchmark.localDensity(2);
        print("local density", r);
        assertTrue(r.get("LOF")[0] > r.get("isolation forest")[0], r.toString());
        assertTrue(r.get("LOF")[0] > r.get("kNN distance")[0] - 0.02);
    }

    @Test
    void oneClassImages() {
        Map<String, double[]> r = AnomalyBenchmark.images(200, 3);
        print("images", r);
        assertTrue(r.get("Deep SVDD")[0] > 0.6, r.toString());
    }

    @Test
    void pointAdjustmentFlattersRandomScores() {
        Map<String, AnomalyBenchmark.SeriesScore> r = AnomalyBenchmark.series(2, 150);
        r.forEach((k, v) -> System.out.printf("anomaly series %s: %s%n", k, v));
        AnomalyBenchmark.SeriesScore random = r.get("random");
        assertTrue(random.adjustedF1() > 2 * random.pointF1(), r.toString());
        assertTrue(r.get("matrix profile (vs clean prefix)").pointF1() > random.pointF1(), r.toString());
        assertTrue(r.get("discord hit rate").pointF1() >= 0.5, r.toString());
    }
}
