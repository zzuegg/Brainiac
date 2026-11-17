package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import org.junit.jupiter.api.Test;

class InterpExperimentsTest {

    @Test
    void sparseAutoencoderRecoversSuperposedFeatures() {
        SuperpositionRecovery.Outcome o = SuperpositionRecovery.run(12, 6, 3000, 1);
        System.out.println(o);
        assertTrue(o.recovered() >= 9, o.toString());
        assertTrue(o.explainedVariance() > 0.8, o.toString());
    }

    @Test
    void unlearningRemovesOneClassAndKeepsTheRest() {
        ClassForgetting.Outcome o = ClassForgetting.run(2);
        System.out.println(o);
        assertTrue(o.forgetBefore() > 0.9, o.toString());
        assertTrue(o.forgetAfter() < 0.1, o.toString());
        assertTrue(o.unseenForgottenClassAfter() < 0.15, o.toString());
        assertTrue(o.retainAfter() > o.retainBefore() - 0.05, o.toString());
    }

    @Test
    void brainMovieFilmsEveryEpoch() {
        BrainMovies.Outcome o = BrainMovies.run(15, 1);
        System.out.println(o);
        assertEquals(16, o.frames());
        assertEquals("GIF89a", new String(Arrays.copyOf(o.gif(), 6), StandardCharsets.US_ASCII));
        assertTrue(o.neuronsAfter() > o.neuronsBefore(), o.toString());
        String dump = System.getenv("AB47_GIF");
        if (dump != null) {
            try {
                java.nio.file.Files.write(java.nio.file.Path.of(dump), o.gif());
            } catch (java.io.IOException e) {
                throw new java.io.UncheckedIOException(e);
            }
        }
    }
}
