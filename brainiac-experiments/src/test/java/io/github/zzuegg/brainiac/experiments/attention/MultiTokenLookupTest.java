package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MultiTokenLookupTest {

    @Test
    void convolvedLogitsFindTheBlockWithBothLetters() {
        List<MultiTokenLookup.Row> rows = MultiTokenLookup.run(List.of("MTA 2x7", "MHA"), 3, 500, 150, 1);
        System.out.print(MultiTokenLookup.table(rows));
        MultiTokenLookup.Row mta = rows.get(0);
        MultiTokenLookup.Row mha = rows.get(1);
        System.out.printf("block lookup (3 blocks): MTA %.3f vs MHA %.3f (chance 0.333); toy LM val loss MTA %.4f vs MHA "
                + "%.4f%n", mta.lookup(), mha.lookup(), mta.validationLoss(), mha.validationLoss());
        assertTrue(mta.lookup() > mha.lookup() + 0.1);
        assertTrue(mta.validationLoss() < mha.validationLoss());
    }
}
