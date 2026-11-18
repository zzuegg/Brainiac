package io.github.zzuegg.brainiac.experiments.sequence;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.sequence.Seq2SeqShootout.Architecture;

import org.junit.jupiter.api.Test;

class Seq2SeqReversalTest {

    @Test
    void attentionReversesLongerStrings() {
        StringBuilder line = new StringBuilder("reversal exact match at length 6/8/10/12 (trained on 4-8):");
        double[][] acc = new double[Architecture.values().length][];
        for (Architecture a : Architecture.values()) {
            acc[a.ordinal()] = Seq2SeqShootout.reversal(a, 300, 1, 6, 8, 10, 12);
            line.append(String.format(" %s %.2f/%.2f/%.2f/%.2f;", a, acc[a.ordinal()][0], acc[a.ordinal()][1],
                    acc[a.ordinal()][2], acc[a.ordinal()][3]));
        }
        System.out.println(line);
        double[] none = acc[Architecture.SEQ2SEQ.ordinal()];
        double[] bahdanau = acc[Architecture.BAHDANAU.ordinal()];
        double[] luong = acc[Architecture.LUONG.ordinal()];
        assertTrue(bahdanau[1] > none[1] + 0.3 && luong[1] > none[1] + 0.3, "attention beats the fixed context");
        assertTrue(luong[2] > acc[Architecture.TRANSFORMER.ordinal()][2], "recurrent attention extrapolates further");
    }
}
