package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class OfflineRlTd3BcTest {

    @Test
    void td3BcAgainstOfflineBaselinesOnExpertPendulumData() {
        Map<String, OfflineRlComparison.OfflineRow> r = OfflineRlComparison.pendulum(
                OfflineRlComparison.PendulumData.EXPERT, 5, 300, 1);
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> b.append(String.format("%s %.2f (%.1fs), ", k, v.normalised(), v.seconds())));
        System.out.printf("W10 4.1 Pendulum expert data, 300 steps, normalised return: %s%n", b);
        assertTrue(r.get("data").normalised() > 0.8, b.toString());
        for (OfflineRlComparison.OfflineRow row : r.values()) {
            assertTrue(Double.isFinite(row.normalised()), b.toString());
        }
    }
}
