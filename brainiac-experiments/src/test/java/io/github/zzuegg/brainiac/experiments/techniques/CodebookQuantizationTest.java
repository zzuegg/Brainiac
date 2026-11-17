package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experiments.language.CountingTasks;

import java.util.Map;

import org.junit.jupiter.api.Test;

class CodebookQuantizationTest {

    private static HookedDecoder base;

    private static synchronized HookedDecoder base() {
        if (base == null) {
            base = CountingTasks.pretrained(150, 3);
        }
        return base;
    }

    private static String line(String title, Map<String, LowBitQuantization.Outcome> r) {
        StringBuilder s = new StringBuilder(title);
        r.forEach((k, v) -> s.append(String.format(" %s %.2e/%.3f (%.0f ms);", k, v.error(), v.perplexity(),
                v.millis())));
        return s.toString();
    }

    @Test
    void trellisAndAdditiveCodesBeatScalarGridsAtTwoBits() {
        Map<String, LowBitQuantization.Outcome> r = CodebookQuantization.weightOnly(base(), 2, 1);
        System.out.println(line("2-bit weights, decoder KL to float/perplexity:", r));
        double gptq = r.get("GPTQ per column").error();
        assertTrue(r.get("QTIP trellis").error() < gptq, "QTIP beats GPTQ at 2 bits");
        assertTrue(r.get("AQLM 2x4-bit codes / 4 weights (shared)").error() < gptq, "AQLM beats GPTQ at 2 bits");
    }

    @Test
    void trellisCodesHoldUpAtThreeBits() {
        Map<String, LowBitQuantization.Outcome> r = CodebookQuantization.weightOnly(base(), 3, 2);
        System.out.println(line("3-bit weights, decoder KL to float/perplexity:", r));
        assertTrue(r.get("QTIP trellis").error() < r.get("RTN per column").error());
        assertTrue(r.get("QTIP trellis").error() < r.get("HQQ groups of 16").error());
    }

    @Test
    void lowRankBranchAbsorbsOutliersAtW4A4() {
        Map<String, Double> layer = CodebookQuantization.w4a4OutlierLayer(1);
        System.out.printf("W4A4 outlier layer, relative output error: %s%n", layer);
        Map<String, LowBitQuantization.Outcome> dec = CodebookQuantization.w4a4(base(), 1);
        System.out.println(line("W4A4 decoder, KL to float/perplexity:", dec));
        assertTrue(layer.get("W4A4 SVDQuant rank 8") < layer.get("W4A4 SmoothQuant"));
        assertTrue(layer.get("W4A4 SVDQuant rank 8") < layer.get("W4A4 Hadamard"));
        assertTrue(dec.get("W4A4 SVDQuant rank 8").error() < dec.get("W4A4 SmoothQuant").error());
    }
}
