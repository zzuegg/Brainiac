package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ImageTranslationTest {

    @Test
    void pix2pixCommitsToRealShapes() {
        Map<String, Double> m = ImageTranslation.paired(200, 1);
        StringBuilder line = new StringBuilder("coarse 4x4 sketch -> shape:");
        m.forEach((k, v) -> line.append(String.format(" %s %.3f;", k, v)));
        System.out.println(line);
        assertTrue(m.get("pix2pix L1") < 0.3, "pix2pix follows the sketch");
        assertTrue(m.get("pix2pix real") >= m.get("L1 only real"), "pix2pix draws at least as many real shapes");
    }
}
