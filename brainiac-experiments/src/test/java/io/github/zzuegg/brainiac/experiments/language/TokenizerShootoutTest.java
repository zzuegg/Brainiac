package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class TokenizerShootoutTest {

    @Test
    void lengthMaxAndLeastTokensUseFewerTokens() {
        Map<String, Map<String, TokenizerShootout.Row>> r = TokenizerShootout.vocabularies(64, 1);
        r.forEach((corpus, rows) -> rows.forEach((k, v) -> System.out.printf(
                "8.4.12/8.4.13 %-12s %-32s %.3f tokens/char, %.3f bits/byte, boundary F1 %.2f%n", corpus, k,
                v.tokensPerChar(), v.bitsPerByte(), v.boundaryF1())));
        for (Map<String, TokenizerShootout.Row> rows : r.values()) {
            assertTrue(rows.get("Length-MAX (longest match)").tokensPerChar()
                    < rows.get("BPE (merge order)").tokensPerChar(), rows.toString());
            assertTrue(rows.get("BPE vocab, least tokens").tokensPerChar()
                    <= rows.get("BPE (merge order)").tokensPerChar() + 1e-9, rows.toString());
        }
    }

    @Test
    void stochasticSplittingHelpsCharacterLevelQuestions() {
        Map<String, Map<String, TokenizerShootout.TaskRow>> r = TokenizerShootout.subwordUnderstanding(300, 200,
                new long[] {1, 2, 3});
        r.forEach((task, rows) -> rows.forEach((k, v) -> System.out.printf(
                "8.4.14 %-12s %-12s accuracy %.3f, test loss %.3f%n", task, k, v.accuracy(), v.loss())));
        for (Map<String, TokenizerShootout.TaskRow> rows : r.values()) {
            assertTrue(rows.get("characters").accuracy() >= rows.get("BPE").accuracy() - 0.02, rows.toString());
        }
    }
}
