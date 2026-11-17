package io.github.zzuegg.brainiac.experiments.sequence;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.sequence.Seq2SeqShootout.Architecture;

import org.junit.jupiter.api.Test;

class DateNormalisationTest {

    @Test
    void attentionNormalisesDates() {
        StringBuilder line = new StringBuilder("date normalisation exact match:");
        double[] acc = new double[Architecture.values().length];
        for (Architecture a : Architecture.values()) {
            acc[a.ordinal()] = Seq2SeqShootout.dates(a, 250, 2);
            line.append(String.format(" %s %.3f;", a, acc[a.ordinal()]));
        }
        System.out.println(line);
        assertTrue(acc[Architecture.BAHDANAU.ordinal()] > 0.9, "Bahdanau attention learns the date formats");
        assertTrue(acc[Architecture.BAHDANAU.ordinal()] > acc[Architecture.SEQ2SEQ.ordinal()] + 0.3,
                "attention beats the fixed context");
    }
}
