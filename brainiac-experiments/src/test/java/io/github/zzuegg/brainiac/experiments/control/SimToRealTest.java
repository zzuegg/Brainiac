package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experiments.control.ClassicalControl.SimToReal;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class SimToRealTest {

    @Test
    void rapidMotorAdaptationOnRandomisedCartPole() {
        SimToReal r = ClassicalControl.simToReal(2000, 8, 30, 3);
        System.out.printf("randomised cart-pole (unknown motor polarity and gain), held-out return: nominal %.1f, "
                + "domain-randomised %.1f, RMA adapted %.1f, RMA oracle latent %.1f; last 50 steps after a change: "
                + "nominal %.1f, randomised %.1f, RMA %.1f%n", r.nominal(), r.randomised(), r.rmaAdapted(),
                r.rmaOracle(), r.nominalAfterChange(), r.randomisedAfterChange(), r.rmaAfterChange());
        assertTrue(r.rmaAdapted() > r.randomised() && r.rmaAdapted() > r.nominal(), "adaptation beats one policy");
        assertTrue(r.rmaOracle() >= r.rmaAdapted(), "the privileged latent bounds the estimate");
        assertTrue(r.rmaAfterChange() > r.randomisedAfterChange(), "RMA re-identifies after a change");
    }
}
