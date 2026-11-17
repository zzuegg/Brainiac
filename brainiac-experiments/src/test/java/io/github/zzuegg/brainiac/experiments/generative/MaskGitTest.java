package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MaskGitTest {

    @Test
    void tokenizerIsLosslessOnToyImages() {
        MaskGit m = MaskGit.of(32, 1);
        List<float[]> all = new ArrayList<>(ToyImages.all(2));
        for (float[] img : all) {
            assertArrayEquals(img, m.tokenizer().decode(m.tokenizer().encode(img)), 1e-4f);
        }
    }

    @Test
    void parallelDecodingTradesStepsForQuality() {
        MaskGit m = MaskGit.of(256, 1);
        m.trainMaskGit(700, 64);
        m.trainAutoregressive(700, 64);
        double[] mg4 = MaskGit.score(m.sampleMaskGit(128, 4, 4.5f, 5));
        double[] mg8 = MaskGit.score(m.sampleMaskGit(128, 8, 4.5f, 5));
        double[] mg16 = MaskGit.score(m.sampleMaskGit(128, 16, 4.5f, 5));
        double[] ar = MaskGit.score(m.sampleAutoregressive(128, 5));
        System.out.printf("MaskGIT vs autoregressive, toy-image tokens (distance/real share/distinct): MaskGIT 4 steps "
                + "%.3f/%.2f/%.0f, 8 steps %.3f/%.2f/%.0f, 16 steps %.3f/%.2f/%.0f; autoregressive 16 steps "
                + "%.3f/%.2f/%.0f%n", mg4[0], mg4[1], mg4[2], mg8[0], mg8[1], mg8[2], mg16[0], mg16[1], mg16[2],
                ar[0], ar[1], ar[2]);
        assertTrue(mg16[1] > 0.5 && ar[1] > 0.5, "both models learn the shapes");
        assertTrue(mg8[1] > 0.25 && mg8[3] == ToyImages.CLASSES, "8 parallel steps still make varied real shapes");
        assertTrue(mg8[0] < mg4[0], "more steps, better samples");
    }
}
