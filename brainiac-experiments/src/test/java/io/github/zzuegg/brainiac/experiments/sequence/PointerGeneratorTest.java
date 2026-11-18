package io.github.zzuegg.brainiac.experiments.sequence;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class PointerGeneratorTest {

    @Test
    void pointerGeneratorCopiesUnseenTokens() {
        double[][] copy = Seq2SeqShootout.summaries(true, 300, 3, 8, 12);
        double[][] base = Seq2SeqShootout.summaries(false, 300, 3, 8, 12);
        System.out.printf("copy-heavy summaries with unseen tokens, exact/content at length 8 and 12: "
                + "pointer-generator %.3f/%.3f, %.3f/%.3f; Bahdanau seq2seq %.3f/%.3f, %.3f/%.3f%n", copy[0][0],
                copy[0][1], copy[1][0], copy[1][1], base[0][0], base[0][1], base[1][0], base[1][1]);
        assertTrue(copy[0][1] > base[0][1] + 0.1, "copying reproduces tokens the generator never saw");
    }
}
