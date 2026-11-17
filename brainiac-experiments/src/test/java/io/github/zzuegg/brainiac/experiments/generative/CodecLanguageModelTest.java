package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.experimental.data.CodebookPatterns;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CodecLanguageModelTest {

    @Test
    void patternsRoundTrip() {
        int[][] codes = {{1, 2, 3}, {4, 5, 6}};
        int[][] delayed = CodebookPatterns.delay(codes, 9);
        assertArrayEquals(new int[] {1, 2, 3, 9}, delayed[0]);
        assertArrayEquals(new int[] {9, 4, 5, 6}, delayed[1]);
        assertArrayEquals(codes, CodebookPatterns.undelay(delayed));
        assertArrayEquals(new int[] {1, 14, 2, 15, 3, 16}, CodebookPatterns.flatten(codes, 10));
        assertArrayEquals(codes, CodebookPatterns.unflatten(CodebookPatterns.flatten(codes, 10), 2, 10));
    }

    @Test
    void delayPatternMatchesFlatteningInFewerSteps() {
        CodecLanguageModel m = CodecLanguageModel.of(3, 8, 1);
        double codecSnr = m.trainCodec(400);
        m.trainLanguageModels(250);
        double[] nll = m.nll(7);
        double[] c = m.continuation(4, 8);
        System.out.printf("codec LM (3 RVQ stages, codec SNR %.1f dB): NLL per code delay %.3f vs flattened %.3f; "
                + "continuation code accuracy %.3f / SNR %.1f dB in %.0f calls vs flattened %.3f / %.1f dB in %.0f "
                + "calls%n", codecSnr, nll[0], nll[1], c[0], c[1], c[2], c[3], c[4], c[5]);
        assertTrue(nll[0] < Math.log(16) * 0.8, "the delay model learns the code streams");
        assertTrue(nll[0] < nll[1] + 0.3, "delay is close to flattening in likelihood");
        assertTrue(c[2] <= c[5] / 2, "delay needs half the calls or fewer");
    }
}
