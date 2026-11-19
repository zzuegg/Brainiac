package io.github.zzuegg.brainiac.experiments.sensing;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.stream.Collectors;

class AdaptiveFiltersTest {

    static String line(Map<String, double[]> m, String fmt) {
        return m.entrySet().stream().map(e -> e.getKey() + " " + java.util.Arrays.stream(e.getValue())
                .mapToObj(v -> String.format(fmt, v)).collect(Collectors.joining("/")))
                .collect(Collectors.joining("; "));
    }

    @Test
    void systemIdentification() {
        Map<String, double[]> r = AdaptiveFilters.systemIdentification(16, 4000, 0.01, 3);
        System.out.printf("13.4.13 system identification N=16 {samples to -20 dB, final dB, after-change dB, ms}: %s%n",
                line(r, "%.1f"));
        String c = "AR(1) 0.9 / ";
        assertTrue(r.get(c + "RLS")[0] < r.get(c + "NLMS")[0] || Double.isNaN(r.get(c + "NLMS")[0]), line(r, "%.1f"));
        assertTrue(r.get(c + "RLS")[1] < r.get(c + "LMS")[1], line(r, "%.1f"));
    }

    @Test
    void echoCancellation() {
        Map<String, double[]> r = AdaptiveFilters.echoCancellation(64, 16000, 5);
        System.out.printf("13.4.13 echo cancellation 64 taps {ERLE single talk, ERLE double talk, misalignment dB,"
                + " double-talk flagged, single-talk flagged}: %s%n", line(r, "%.2f"));
        assertTrue(r.get("NLMS + Geigel")[1] > r.get("NLMS")[1], line(r, "%.2f"));
        assertTrue(r.get("Kalman FIR (adaptive noise)")[1] > r.get("NLMS")[1] + 10, line(r, "%.2f"));
    }
}
