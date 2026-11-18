package io.github.zzuegg.brainiac.experiments.sequence;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.sequence.Seq2SeqShootout.Architecture;

import org.junit.jupiter.api.Test;

class PointerSortingTest {

    @Test
    void pointerNetworkSortsLongerListsThanSeen() {
        double[][] pointer = Seq2SeqShootout.sortWithPointer(300, 1, 8, 10);
        double[][] bahdanau = Seq2SeqShootout.sortWithTokens(Architecture.BAHDANAU, 300, 1, 8, 10);
        double[][] transformer = Seq2SeqShootout.sortWithTokens(Architecture.TRANSFORMER, 300, 1, 8, 10);
        System.out.printf("sorting exact/position accuracy at 8 and 10 numbers (trained on 5-8): pointer %.3f/%.3f, "
                + "%.3f/%.3f; Bahdanau seq2seq %.3f/%.3f, %.3f/%.3f; transformer %.3f/%.3f, %.3f/%.3f%n",
                pointer[0][0], pointer[0][1], pointer[1][0], pointer[1][1], bahdanau[0][0], bahdanau[0][1],
                bahdanau[1][0], bahdanau[1][1], transformer[0][0], transformer[0][1], transformer[1][0],
                transformer[1][1]);
        assertTrue(pointer[0][0] > bahdanau[0][0], "pointing sorts better than generating at length 8");
        assertTrue(pointer[1][1] > 0.2, "pointing places numbers right at the unseen length 10 (chance 0.1)");
    }
}
