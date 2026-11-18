package io.github.zzuegg.brainiac.experiments.statistics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class FairCalibrationTest {

    private static String fmt(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.3f/%.3f/%.3f; ", k, v[0], v[1], v[2])));
        return sb.toString();
    }

    @Test
    void multicalibrationClosesTheWorstGroupGap() {
        Map<String, double[]> r = FairCalibration.run(4000, 4000, 20000, 3);
        System.out.printf("subgroup calibration {ECE/worst-group ECE/AUROC}: %s%n", fmt(r));
        assertTrue(r.get("MULTICALIBRATE")[1] < 0.6 * r.get("temperature")[1], fmt(r));
        assertTrue(r.get("MULTICALIBRATE")[2] > r.get("temperature")[2] - 0.01, fmt(r));
        assertTrue(r.get("MULTIACCURACY tree")[1] < r.get("temperature")[1], fmt(r));
    }
}
