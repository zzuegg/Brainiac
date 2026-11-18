package io.github.zzuegg.brainiac.experiments.events;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.EventStreams;
import org.junit.jupiter.api.Test;

import java.util.Map;

class PointProcessShootoutInhibitoryTest {

    @Test
    void inhibitionIsOutsideTheLinearHawkesFamily() {
        Map<String, double[]> r = PointProcessShootout.compare(EventStreams.Kind.INHIBITORY,
                PointProcessShootoutTest.BUDGET, 2);
        String line = PointProcessShootoutTest.line("TPP inhibitory LL/event, type acc, KS, ms", r);
        System.out.println(line);
        double truth = r.get("true process")[0];
        assertTrue(r.get("neural Hawkes (decaying cell)")[0] > r.get("Poisson")[0], line);
        assertTrue(truth >= r.get("Hawkes spline (EMS)")[0] - 0.05, line);
    }
}
