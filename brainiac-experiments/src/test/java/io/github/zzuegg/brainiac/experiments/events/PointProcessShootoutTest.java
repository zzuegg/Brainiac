package io.github.zzuegg.brainiac.experiments.events;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.EventStreams;
import org.junit.jupiter.api.Test;

import java.util.Map;

class PointProcessShootoutTest {

    static String line(String title, Map<String, double[]> r) {
        StringBuilder s = new StringBuilder(title).append(':');
        r.forEach((k, v) -> {
            s.append(' ').append(k);
            for (double x : v) {
                s.append(String.format(" %.3f", x));
            }
            s.append(';');
        });
        return s.toString();
    }

    static final PointProcessShootout.Budget BUDGET = new PointProcessShootout.Budget(2, 40, 20, 30, 40, 20);

    @Test
    void hawkesNearTheTruthOnExponentialStreams() {
        Map<String, double[]> r = PointProcessShootout.compare(EventStreams.Kind.EXPONENTIAL, BUDGET, 1);
        String line = line("TPP exponential LL/event, type acc, KS, ms", r);
        System.out.println(line);
        double truth = r.get("true process")[0];
        double hawkes = r.get("Hawkes exponential (EM)")[0];
        assertTrue(hawkes > r.get("Poisson")[0], line);
        assertTrue(truth - hawkes < 0.1, line);
        assertTrue(r.get("log-normal mixture (intensity-free)")[0] > r.get("Poisson")[0] - 0.2, line);
    }
}
