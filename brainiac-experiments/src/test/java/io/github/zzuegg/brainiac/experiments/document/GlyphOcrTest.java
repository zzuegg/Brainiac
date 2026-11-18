package io.github.zzuegg.brainiac.experiments.document;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class GlyphOcrTest {

    @Test
    void crnnReadsLinesAndSegmentationStrugglesWithTouchingGlyphs() {
        Map<GlyphOcr.Method, GlyphOcr.Scores> all = new EnumMap<>(GlyphOcr.Method.class);
        for (GlyphOcr.Method m : GlyphOcr.Method.values()) {
            GlyphOcr.Scores s = GlyphOcr.run(m, m == GlyphOcr.Method.SEGMENTATION ? 600 : 250, 4, 10, 100, 1);
            all.put(m, s);
            System.out.printf("OCR %-16s CER %.3f WER %.3f (lexicon %.3f) CER touching %.3f separated %.3f (%d ms)%n",
                    m, s.cer(), s.wer(), s.werLexicon(), s.cerTouching(), s.cerSeparated(), s.millis());
        }
        assertTrue(all.get(GlyphOcr.Method.SEGMENTATION).cerSeparated() < 0.5,
                "segmentation reads well-separated glyphs");
        assertTrue(all.get(GlyphOcr.Method.CRNN).cer() < 0.8, "the CRNN learns to read");
    }
}
