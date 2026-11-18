package io.github.zzuegg.brainiac.experiments.document;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class GlyphOcrTransformerTest {

    @Test
    void transformerCtcReadsLines() {
        GlyphOcr.Scores s = GlyphOcr.run(GlyphOcr.Method.TRANSFORMER_CTC, 200, 4, 10, 100, 1);
        GlyphOcrTest.print(s);
        assertTrue(s.cer() < 0.8, "the transformer CTC recogniser learns to read");
    }
}
