package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.MicroscalingFormat;
import io.github.zzuegg.brainiac.experiments.language.CountingTasks;

import java.util.Map;

import org.junit.jupiter.api.Test;

class LowBitQuantizationTest {

    private static HookedDecoder base;

    private static synchronized HookedDecoder base() {
        if (base == null) {
            base = CountingTasks.pretrained(150, 3);
        }
        return base;
    }

    private static String line(String title, Map<String, LowBitQuantization.Outcome> r, boolean time) {
        StringBuilder s = new StringBuilder(title);
        r.forEach((k, v) -> s.append(time ? String.format(" %s %.2e/%.3f (%.0f ms);", k, v.error(), v.perplexity(),
                v.millis()) : String.format(" %s %.2e/%.3f;", k, v.error(), v.perplexity())));
        return s.toString();
    }

    @Test
    void hadamardRotationRescuesW4A4FromOutliers() {
        Map<String, LowBitQuantization.Outcome> layer = LowBitQuantization.w4a4Layer(1);
        System.out.println(line("W4A4 outlier layer, relative output error/-:", layer, false));
        Map<String, LowBitQuantization.Outcome> dec = LowBitQuantization.w4a4Decoder(base(), 1);
        System.out.println(line("W4A4 decoder, KL to float/perplexity:", dec, false));
        double rot = layer.get("W4A4 Hadamard + RTN").error();
        assertTrue(rot < 0.5 * layer.get("W4A4 RTN").error());
        assertTrue(layer.get("W4A4 Hadamard + GPTQ").error() <= rot * 1.05);
        // the tiny decoder has no outlier channels: rotation only helps together with GPTQ there
        assertTrue(dec.get("W4A4 Hadamard + GPTQ").error() < dec.get("W4A4 RTN").error());
    }

    @Test
    void hqqBeatsRoundToNearestWithoutCalibration() {
        for (int bits : new int[] {3, 4}) {
            Map<String, LowBitQuantization.Outcome> r = LowBitQuantization.weightOnly(base(), bits, 2);
            System.out.println(line(bits + "-bit weights, decoder KL to float/perplexity:", r, true));
            assertTrue(r.get("HQQ groups of 16").error() < r.get("RTN asymmetric groups of 16").error());
        }
    }

    @Test
    void microscalingFormatsSitBetweenFp8AndPerColumnInt4() {
        Map<String, LowBitQuantization.Outcome> r = LowBitQuantization.formats(base(), 3);
        System.out.println(line("4-bit formats, decoder KL to float/perplexity:", r, false));
        double nv = r.get("NVFP4 (16 per E4M3 scale)").error();
        assertTrue(nv < r.get("int4 per column").error());
        assertTrue(r.get("MXFP4 (32 per E8M0 scale)").error() < 1.5 * r.get("core int4 groups of 32").error());
        assertTrue(nv < r.get("MXFP4 (32 per E8M0 scale)").error(), "smaller blocks and finer scales help");
        assertTrue(r.get("FP8 E4M3 per tensor").error() < nv);
        assertEquals(6f, MicroscalingFormat.roundE2m1(100f));
        assertEquals(1.5f, MicroscalingFormat.roundE2m1(1.4f));
        assertEquals(-4f, MicroscalingFormat.roundE2m1(-3.6f));
    }
}
