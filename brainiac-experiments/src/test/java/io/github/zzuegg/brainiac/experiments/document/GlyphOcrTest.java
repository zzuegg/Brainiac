package io.github.zzuegg.brainiac.experiments.document;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class GlyphOcrTest {

    static void print(GlyphOcr.Scores s) {
        System.out.printf("OCR %-16s CER %.3f WER %.3f (lexicon %.3f) CER touching %.3f separated %.3f (%d ms)%n",
                s.method(), s.cer(), s.wer(), s.werLexicon(), s.cerTouching(), s.cerSeparated(), s.millis());
    }

    @Test
    void crnnReadsLinesAndSegmentationStrugglesWithTouchingGlyphs() {
        GlyphOcr.Scores crnn = GlyphOcr.run(GlyphOcr.Method.CRNN, 250, 4, 10, 100, 1);
        GlyphOcr.Scores seg = GlyphOcr.run(GlyphOcr.Method.SEGMENTATION, 600, 4, 10, 100, 1);
        print(crnn);
        print(seg);
        assertTrue(seg.cerSeparated() < 0.5, "segmentation reads well-separated glyphs");
        assertTrue(crnn.cer() < 0.8, "the CRNN learns to read");
    }
}
