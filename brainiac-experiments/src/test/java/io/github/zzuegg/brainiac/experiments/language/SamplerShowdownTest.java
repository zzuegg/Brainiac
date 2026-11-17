package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.EntropyAdaptiveSampler;
import io.github.zzuegg.brainiac.experimental.llm.LogitMath;
import io.github.zzuegg.brainiac.experimental.llm.LogitsProcessors;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SamplerShowdownTest {

    @Test
    void topNSigmaAndEntropyAdaptiveStayAccurateAtHighTemperature() {
        Map<String, SamplerShowdown.Outcome> r = SamplerShowdown.run(PeftShootoutTest.base(), 3f, 48, 10, 1);
        StringBuilder line = new StringBuilder("Samplers at T=3 (accuracy/distinct):");
        r.forEach((k, v) -> line.append(String.format(" %s %.2f/%.2f;", k, v.accuracy(), v.distinct())));
        System.out.println(line);
        double topP = r.get("top-p 0.9").accuracy();
        double minP = r.get("min-p 0.1").accuracy();
        assertTrue(r.get("top-n-sigma 1").accuracy() >= Math.max(topP, minP));
        assertTrue(r.get("top-n-sigma 1").accuracy() > topP + 0.2);
        SamplerShowdown.Outcome adaptive = r.get("entropy-adaptive");
        assertTrue(adaptive.accuracy() > topP + 0.2);
        assertTrue(adaptive.distinct() > r.get("greedy").distinct() + 0.1, "it must keep the stride open");
    }

    @Test
    void topNSigmaIgnoresTemperatureAndXtcDropsTopChoices() {
        float[] l = {5, 4.5f, 1, 0, -1, -2};
        float[] hot = new float[l.length];
        for (int i = 0; i < l.length; i++) {
            hot[i] = l[i] / 4;
        }
        LogitsProcessors.topNSigma(1).process(new int[0], l);
        LogitsProcessors.topNSigma(1).process(new int[0], hot);
        for (int i = 0; i < l.length; i++) {
            assertEquals(l[i] == LogitMath.MASKED, hot[i] == LogitMath.MASKED);
        }
        float[] x = {3, 2.9f, 2.8f, -5, -5};
        LogitsProcessors.xtc(0.1, 1, 1).process(new int[0], x);
        assertEquals(2, LogitMath.argmax(x), "only the least likely top choice survives");
        double[] hv = EntropyAdaptiveSampler.entropyAndVarentropy(new float[] {0, 0, 0, 0});
        assertEquals(Math.log(4), hv[0], 1e-9);
        assertEquals(0, hv[1], 1e-9);
    }
}
