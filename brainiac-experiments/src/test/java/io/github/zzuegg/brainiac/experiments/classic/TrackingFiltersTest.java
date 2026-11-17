package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TrackingFiltersTest {

    private static String line(Map<String, double[]> m) {
        StringBuilder sb = new StringBuilder();
        m.forEach((k, v) -> sb.append(String.format("%s rmse %.3f nll %.3f; ", k, v[0], v[1])));
        return sb.toString();
    }

    @Test
    void kalmanFilterTracksAConstantVelocityTarget() {
        Map<String, double[]> r = TrackingFilters.linearTracking(1);
        System.out.printf("constant-velocity tracking: %s%n", line(r));
        assertTrue(r.get("Kalman filter")[0] < r.get("raw observations")[0] * 0.7, line(r));
        assertTrue(r.get("RTS smoother")[0] < r.get("Kalman filter")[0], line(r));
        assertTrue(r.get("Kalman filter")[1] < r.get("GRU (causal)")[1] + 0.05, line(r));
    }

    @Test
    void particleFilterHandlesTheAmbiguousPendulum() {
        Map<String, double[]> r = TrackingFilters.pendulum(400, 2);
        System.out.printf("pendulum observed through sin(theta): %s%n", line(r));
        assertTrue(r.get("particle filter")[0] < r.get("EKF")[0], line(r));
        assertTrue(r.get("particle filter")[1] < r.get("EKF")[1], line(r));
    }
}
