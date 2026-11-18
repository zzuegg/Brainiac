package io.github.zzuegg.brainiac.experiments.coding;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class LearnedCompressionTest {

    @Test
    void learnedCodecsTraceRateDistortionCurves() {
        List<LearnedCompression.RdPoint> rows = LearnedCompression.run(600, 100, new double[] {30, 300}, 300, 1);
        for (LearnedCompression.RdPoint r : rows) {
            System.out.printf("12.2.15 %-30s %-28s %.4f bits/pixel, PSNR %.2f dB%n", r.method(), r.setting(),
                    r.bitsPerPixel(), r.psnr());
        }
        for (LearnedCompression.RdPoint r : rows) {
            assertTrue(r.bitsPerPixel() > 0 && Double.isFinite(r.psnr()), r.toString());
        }
        double best = rows.stream().filter(r -> r.method().startsWith("learned")).mapToDouble(
                LearnedCompression.RdPoint::psnr).max().orElseThrow();
        assertTrue(best > 22, "a learned codec reaches a useful PSNR");
    }
}
