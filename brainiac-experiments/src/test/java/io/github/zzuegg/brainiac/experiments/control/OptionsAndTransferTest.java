package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class OptionsAndTransferTest {

    @Test
    void optionCriticReadaptsAfterTheGoalMoves() {
        Map<String, OptionsAndTransfer.Outcome> r = OptionsAndTransfer.run(30_000, 30, 3);
        OptionsAndTransfer.Outcome oc = r.get("option-critic (4 options)");
        OptionsAndTransfer.Outcome q = r.get("Q-learning (core)");
        OptionsAndTransfer.Outcome a2c = r.get("A2C (core)");
        System.out.printf("W6 3.12 four rooms after goal switch, first-30-episode length / greedy length: option-critic"
                + " %.1f/%.1f (option duration %.1f) vs Q-learning %.1f/%.1f, A2C %.1f/%.1f%n", oc.adaptLength(),
                oc.finalLength(), oc.optionDuration(), q.adaptLength(), q.finalLength(), a2c.adaptLength(),
                a2c.finalLength());
        assertTrue(oc.finalLength() < 60, r.toString());
        assertTrue(oc.optionDuration() > 1.5, r.toString());
    }
}
