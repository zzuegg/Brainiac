package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class Mamba2LayerTest {

    @Test
    void chunkedSsdEqualsRecurrence() {
        Mamba2Layer m = new Mamba2Layer(6, 2, 4, 4, 3, Mamba2Layer.Variant.REAL, new SplittableRandom(1));
        LinearRecurrenceTest.assertModesAgree(m.parameters(), m::forward, m::recurrent,
                LinearRecurrenceTest.input(2, 2, 8, 6), 1e-4f);
    }

    @Test
    void rotaryStateRopeTrickEqualsRotatingRecurrence() {
        Mamba2Layer m = new Mamba2Layer(6, 2, 4, 4, 3, Mamba2Layer.Variant.ROTARY, new SplittableRandom(3));
        LinearRecurrenceTest.assertModesAgree(m.parameters(), m::forward, m::recurrent,
                LinearRecurrenceTest.input(4, 2, 8, 6), 1e-4f);
    }

    @Test
    void mamba2LearnsRecall() {
        double acc = MixerHarness.train(new Mamba2Layer(16, 4, 8, 8, 4, Mamba2Layer.Variant.REAL,
                new SplittableRandom(5)), MixerHarness.recall(4, 4, 2), 16, 300, 0.01f, 6);
        assertTrue(acc > 0.7, "accuracy " + acc);
    }

    @Test
    void rotaryStatesSolveParityWhereRealDecaysCannot() {
        MixerHarness.Task task = MixerHarness.parity(8);
        double rotary = MixerHarness.train(new Mamba2Layer(16, 2, 8, 4, 4, Mamba2Layer.Variant.ROTARY,
                new SplittableRandom(7)), task, 16, 300, 0.02f, 8);
        double real = MixerHarness.train(new Mamba2Layer(16, 2, 8, 4, 4, Mamba2Layer.Variant.REAL,
                new SplittableRandom(7)), task, 16, 300, 0.02f, 8);
        System.out.println("parity rotary " + rotary + " real " + real);
        assertTrue(rotary > 0.95, "rotary " + rotary);
        assertTrue(real < 0.8, "real " + real);
    }
}
