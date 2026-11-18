package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SeriesClassificationTest {

    @Test
    void rocketAgainstDtwAndTrainedNets() {
        Map<String, double[]> r = SeriesClassification.compare("CBF", 60, 200, 40, 1);
        r.forEach((k, v) -> System.out.printf("series classification CBF %s: acc %.3f (%.0f ms)%n", k, v[0], v[1]));
        assertTrue(r.get("ROCKET")[0] >= r.get("DTW 1-NN")[0] - 0.05, r.get("ROCKET")[0] + " vs DTW");
        assertTrue(r.get("ROCKET")[0] > 0.85, "ROCKET " + r.get("ROCKET")[0]);
        assertTrue(r.get("MiniRocket")[0] > 0.8, "MiniRocket " + r.get("MiniRocket")[0]);
        assertTrue(r.get("ROCKET")[1] < r.get("TCN")[1], "ROCKET should fit faster than the TCN trains");
        assertTrue(r.get("scattering + ridge")[0] > 0.6, "scattering " + r.get("scattering + ridge")[0]);
    }

    @Test
    void scatteringAtSmallSizes() {
        Map<Integer, Map<String, Double>> r = SeriesClassification.scatteringCurve("frequencies", new int[] {8}, 3);
        r.forEach((n, row) -> row.forEach((k, v) -> System.out.printf(
                "series classification n=%d %s: acc %.3f%n", n, k, v)));
        assertTrue(r.get(8).get("scattering + ridge") > r.get(8).get("raw + ridge"), r.toString());
        Map<String, Double> img = SeriesClassification.imagesLowData(64, 100);
        img.forEach((k, v) -> System.out.printf("shape images 64 %s: acc %.3f%n", k, v));
        assertTrue(img.get("2-D scattering + ridge") > img.get("CNN"), img.toString());
    }
}
