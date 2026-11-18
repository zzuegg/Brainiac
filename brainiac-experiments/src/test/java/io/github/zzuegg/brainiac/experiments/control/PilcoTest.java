package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experiments.control.ClassicalControl.DataEfficiency;

import org.junit.jupiter.api.Test;

import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertTrue;

class PilcoTest {

    @Test
    void pilcoLearnsToBalanceInAFewEpisodes() {
        DataEfficiency d = ClassicalControl.pilco(5, 50, 40, 2);
        int last = d.curve().length - 1;
        System.out.printf("cart-pole balance, test cost (lower is better): zero force %.1f, PILCO %s after %s real "
                + "steps, true-model LQR %.2f%n", d.zeroPolicy(),
                Arrays.toString(Arrays.stream(d.curve()).map(v -> Math.round(v * 10) / 10.0).toArray()),
                Arrays.toString(d.steps()), d.lqr());
        assertTrue(d.curve()[last] < 0.6 * d.zeroPolicy(), "PILCO controls the pole");
    }
}
