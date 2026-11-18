package io.github.zzuegg.brainiac.experiments.context;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.Map;
import org.junit.jupiter.api.Test;

class PositionBiasSweepTest {

    @Test
    void calibrationFlattensThePositionCurve() {
        Map<String, PositionBiasSweep.Curve> r = PositionBiasSweep.run(60, new long[] {1, 2, 3});
        r.forEach((k, v) -> System.out.printf("8.4.11 %-20s mean %.2f, by position %s, middle-ends gap %+.2f%n", k,
                Arrays.stream(v.accuracy()).average().orElse(0),
                Arrays.toString(Arrays.stream(v.accuracy()).mapToObj(x -> String.format("%.2f", x)).toArray()),
                v.gap()));
        double spreadRaw = spread(r.get("uncalibrated").accuracy());
        double spreadCal = spread(r.get("found-in-the-middle").accuracy());
        assertTrue(spreadCal < spreadRaw, r.toString());
    }

    private static double spread(double[] a) {
        return Arrays.stream(a).max().orElse(0) - Arrays.stream(a).min().orElse(0);
    }
}
