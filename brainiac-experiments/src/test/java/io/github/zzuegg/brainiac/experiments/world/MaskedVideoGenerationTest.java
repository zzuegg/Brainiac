package io.github.zzuegg.brainiac.experiments.world;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MaskedVideoGenerationTest {

    @Test
    void multiTaskMaskGitOnTubeletTokens() {
        List<VideoGeneration.GenerationRow> rows = VideoGeneration.maskedGeneration(60, 50, 20, 6, 1);
        StringBuilder line = new StringBuilder("9.5.2 token video generation, calls / clipMMD / count err (ms):");
        rows.forEach(r -> line.append(String.format(" %s %d/%.3f/%.2f (%.0f);", r.model(), r.calls(),
                r.quality().clipMmd(), r.quality().countError(), r.millis())));
        System.out.println(line);
        for (VideoGeneration.GenerationRow r : rows) {
            assertTrue(Double.isFinite(r.quality().clipMmd()), line.toString());
        }
    }
}
