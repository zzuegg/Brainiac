package io.github.zzuegg.brainiac.experimental.events;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.EventStreams;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

class IntensityFreeTppTest {

    @Test
    void learnsAFiniteLikelihoodAndSamplesWithinTheHorizon() {
        TemporalPointProcess truth = EventStreams.process(EventStreams.Kind.EXPONENTIAL, 2, 1);
        List<EventSequence> train = EventStreams.sample(truth, 32, 20, 2);
        List<EventSequence> test = EventStreams.sample(truth, 16, 20, 3);
        IntensityFreeTpp m = new IntensityFreeTpp(2, 16, 4, 1);
        double before = m.evaluate(test)[0];
        m.train(train, 40, 16, 1e-2f, 1);
        double[] after = m.evaluate(test);
        EventSequence s = m.sample(20, 200, new SplittableRandom(4));
        System.out.printf("IFTPP LL/event %.3f -> %.3f (truth %.3f), type acc %.3f, next-time MAE %.3f, sample %d events%n",
                before, after[0], truth.logLikelihoodPerEvent(test), after[1], after[2], s.size());
        assertTrue(Double.isFinite(after[0]) && after[0] > before, "LL " + before + " -> " + after[0]);
        for (double t : s.times()) {
            assertTrue(t > 0 && t < 20);
        }
    }
}
