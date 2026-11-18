package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.timeseries.KernelSynth;
import io.github.zzuegg.brainiac.experimental.timeseries.ValueTokenizer;

import org.junit.jupiter.api.Test;

import java.util.Map;
import java.util.SplittableRandom;

/**
 * The Chronos-style zero-shot rows of {@link ProbabilisticForecasting}. Pretraining that beats
 * the per-dataset baselines takes thousands of steps (see the report), so at test budget this
 * checks the pipeline end to end and the tokenizer and KernelSynth blocks.
 */
class ZeroShotForecastingTest {

    @Test
    void tokenizerAndKernelSynth() {
        ValueTokenizer tok = new ValueTokenizer(128, -6, 6);
        float[] x = {0.5f, -1.2f, 2.0f, 0.0f};
        double scale = ValueTokenizer.scale(x);
        int[] ids = tok.encode(x, scale);
        for (int i = 0; i < x.length; i++) {
            assertTrue(Math.abs(tok.decode(ids[i], scale) - x[i]) <= 6.0 / 128 * scale + 1e-6, "round trip " + i);
        }
        KernelSynth synth = new KernelSynth(80, 4);
        SplittableRandom r = new SplittableRandom(1);
        double var = 0;
        for (int k = 0; k < 20; k++) {
            for (float v : synth.sample(80, r)) {
                assertTrue(Float.isFinite(v));
                var += v * v / 1600;
            }
        }
        float[][] group = synth.sampleGroup(80, 2, r);
        assertTrue(group.length == 3 && group[2].length == 80);
        System.out.printf("kernel synth: mean square %.3f%n", var);
        assertTrue(var > 0.05 && var < 100, "variance " + var);
    }

    @Test
    void zeroShotPipeline() {
        Map<String, ProbabilisticForecasting.Probabilistic> r = ProbabilisticForecasting.zeroShot(60, 40, 64, 1,
                "covariates");
        r.forEach((k, v) -> System.out.printf("zero-shot %s: mse %.4f crps %.4f coverage %.2f%n", k, v.mse(),
                v.crps(), v.coverage()));
        for (Map.Entry<String, ProbabilisticForecasting.Probabilistic> e : r.entrySet()) {
            assertTrue(Double.isFinite(e.getValue().crps()), e.getKey());
        }
        assertTrue(r.get("covariates: group attention + covariates zero-shot").coverage() > 0.3, r.toString());
    }
}
