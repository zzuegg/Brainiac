package io.github.zzuegg.brainiac.experiments.world;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class VideoTokenizerTest {

    @Test
    void tubeletTokensAgainstPerFrameTokens() {
        List<VideoGeneration.TokenizerRow> rows = VideoGeneration.tokenizers(120, 1);
        StringBuilder line = new StringBuilder("9.5.2/9.5.13 video tokenizers, tokens / PSNR dB / perplexity /"
                + " code use:");
        rows.forEach(r -> line.append(String.format(" %s %d/%.1f/%.0f/%.2f;", r.tokenizer(), r.tokens(), r.psnr(),
                r.perplexity(), r.utilisation())));
        System.out.println(line);
        assertEquals(48, rows.get(0).tokens(), line.toString());
        assertEquals(144, rows.get(2).tokens(), line.toString());
        for (VideoGeneration.TokenizerRow r : rows) {
            assertTrue(Double.isFinite(r.psnr()) && r.psnr() > 5, line.toString());
        }
    }
}
