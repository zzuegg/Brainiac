package io.github.zzuegg.brainiac.experiments.events;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.EventStreams;
import org.junit.jupiter.api.Test;

import java.util.Map;

class PointProcessShootoutDelayedTest {

    @Test
    void delayedKernelNeedsAFlexibleKernel() {
        Map<String, double[]> r = PointProcessShootout.compare(EventStreams.Kind.DELAYED,
                PointProcessShootoutTest.BUDGET, 2);
        String line = PointProcessShootoutTest.line("TPP delayed LL/event, type acc, KS, ms", r);
        System.out.println(line);
        assertTrue(r.get("Hawkes spline (EMS)")[0] > r.get("Hawkes exponential (EM)")[0], line);
        assertTrue(r.get("Hawkes spline (EMS)")[2] < r.get("Hawkes exponential (EM)")[2], line);
    }
}
