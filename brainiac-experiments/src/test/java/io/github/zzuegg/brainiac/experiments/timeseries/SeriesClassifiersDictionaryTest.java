package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SeriesClassifiersDictionaryTest {

    @Test
    void dictionaryMethodsWinWherePatternCountsMatter() {
        Map<String, double[]> r = SeriesClassifiers.dictionary(SeriesClassifiers.split("dictionary", 90, 120, 128, 1),
                1);
        r.forEach((k, v) -> System.out.printf("dictionary task %s: acc %.3f (%.0f ms)%n", k, v[0], v[1]));
        assertTrue(r.get("BOSS")[0] > r.get("DTW 1-NN")[0] + 0.05, "BOSS vs DTW " + r.get("BOSS")[0]);
        assertTrue(r.get("WEASEL 2.0")[0] > r.get("DTW 1-NN")[0] + 0.1, "WEASEL 2.0 " + r.get("WEASEL 2.0")[0]);
        assertTrue(r.get("WEASEL")[0] > 0.45, "WEASEL above chance (1/3) " + r.get("WEASEL")[0]);
    }

    @Test
    void hydraAgainstRocket() {
        Map<String, double[]> r = SeriesClassifiers.convolution(SeriesClassifiers.split("dictionary", 90, 120, 128, 2),
                2);
        r.forEach((k, v) -> System.out.printf("convolution dictionary task %s: acc %.3f (%.0f ms)%n", k, v[0], v[1]));
        assertTrue(r.get("Hydra")[0] > 0.7, "Hydra " + r.get("Hydra")[0]);
        assertTrue(r.get("MultiRocket-Hydra")[0] >= r.get("ROCKET")[0] - 0.05, "MR-Hydra "
                + r.get("MultiRocket-Hydra")[0]);
    }
}
