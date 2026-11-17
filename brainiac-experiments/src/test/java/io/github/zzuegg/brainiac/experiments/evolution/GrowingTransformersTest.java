package io.github.zzuegg.brainiac.experiments.evolution;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class GrowingTransformersTest {

    @Test
    void growingFromTinyBeatsStayingTiny() {
        GrowingTransformers.Outcome o = GrowingTransformers.run(600, 1);
        System.out.printf("grown %.3f (%d params) vs fixed %.3f (%d params), growths %s%n", o.grownLoss(),
                o.grownParameters(), o.staticLoss(), o.staticParameters(), o.growths());
        assertTrue(!o.growths().isEmpty() && o.grownParameters() > o.staticParameters());
        assertTrue(o.grownLoss() < o.staticLoss(), o.grownLoss() + " vs " + o.staticLoss());
    }
}
