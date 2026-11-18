package io.github.zzuegg.brainiac.experiments.robotics;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SystemIdComparisonTest {

    @Test
    void identification() {
        List<SystemIdComparison.Row> rows = SystemIdComparison.run(1000, 1217);
        StringBuilder line = new StringBuilder("12.3.17 fit%/one-step/pole-error/order:");
        for (SystemIdComparison.Row r : rows) {
            line.append(String.format(" %s %s %s %.1f/%.4f/%.3f/%d;", r.system(), r.noise(), r.method(), r.fit(),
                r.oneStep(), r.poleError(), r.order()));
        }
        System.out.println(line);
        double best = Double.NEGATIVE_INFINITY;
        for (SystemIdComparison.Row r : rows) {
            if (r.system().equals("spring chain") && r.noise().equals("white")
                && (r.method().equals("OE") || r.method().equals("N4SID"))) {
                best = Math.max(best, r.fit());
            }
        }
        assertTrue(best > 70, line.toString());
    }
}
