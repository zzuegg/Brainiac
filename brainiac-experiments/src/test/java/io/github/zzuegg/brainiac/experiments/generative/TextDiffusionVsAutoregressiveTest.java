package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TextDiffusionVsAutoregressiveTest {
    static final List<String> WORDS = List.of("banana", "bandit", "candle", "dancer", "garden", "handle", "random",
            "sandal", "tandem", "linden");

    @Test
    void bothModelsLearnWordsAndDiffusionTextCrystallizes() {
        TextDiffusionVsAutoregressive exp = TextDiffusionVsAutoregressive.of(WORDS, 32, 2, 1);
        double untrained = exp.validShare(exp.sampleDiffusion(20, 6, 0.5f, 3, null));
        exp.train(200, 32);
        List<List<String>> trace = new ArrayList<>();
        List<String> diffusion = exp.sampleDiffusion(20, 6, 0.5f, 3, trace);
        List<String> ar = exp.sampleAutoregressive(20, 0.5f, 3);
        double d = exp.validShare(diffusion);
        double a = exp.validShare(ar);
        assertTrue(d > 0.6 && d > untrained, "diffusion samples that are words: " + d + " " + diffusion);
        assertTrue(a > 0.6, "autoregressive samples that are words: " + a + " " + ar);

        // the trace crystallizes: one more letter per step, ending fully unmasked
        assertEquals(6, trace.size());
        for (int s = 0; s < trace.size(); s++) {
            String w = trace.get(s).get(0);
            long masks = w.chars().filter(c -> c == TextDiffusionVsAutoregressive.MASK_CHAR).count();
            assertEquals(5 - s, masks, "step " + s + ": " + w);
            if (s > 0) {
                String prev = trace.get(s - 1).get(0);
                for (int i = 0; i < w.length(); i++) {
                    if (prev.charAt(i) != TextDiffusionVsAutoregressive.MASK_CHAR) {
                        assertEquals(prev.charAt(i), w.charAt(i), "revealed letters stay");
                    }
                }
            }
        }
    }
}
