package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.techniques.CnnNormalizationShootout.Norm;

import org.junit.jupiter.api.Test;

class CnnNormalizationShootoutTest {

    @Test
    void batchFreeNormsSurviveTinyAndNonIidBatches() {
        StringBuilder line = new StringBuilder("CNN norm accuracy at batch 2 / 8 / 64 / non-iid 8:");
        float[][] acc = new float[Norm.values().length][];
        for (Norm n : Norm.values()) {
            acc[n.ordinal()] = new float[] {CnnNormalizationShootout.run(n, 2, true, 150, 1),
                CnnNormalizationShootout.run(n, 8, true, 150, 1), CnnNormalizationShootout.run(n, 64, true, 60, 1),
                CnnNormalizationShootout.run(n, 8, false, 150, 1)};
            line.append(String.format(" %s %.2f/%.2f/%.2f/%.2f;", n, acc[n.ordinal()][0], acc[n.ordinal()][1],
                    acc[n.ordinal()][2], acc[n.ordinal()][3]));
        }
        System.out.println(line);
        float[] bn = acc[Norm.BATCH_NORM.ordinal()];
        float[] brn = acc[Norm.BATCH_RENORM.ordinal()];
        float[] frn = acc[Norm.FRN.ordinal()];
        float[] none = acc[Norm.NONE.ordinal()];
        assertTrue(brn[0] > bn[0], "batch renorm beats batch norm at batch 2");
        assertTrue(brn[3] > bn[3] - 0.05f, "batch renorm stays close to batch norm on non-iid batches");
        assertTrue(frn[1] > none[1] && frn[2] > none[2], "FRN beats no normaliser at batch 8 and 64");
    }

    @Test
    void weightNormTrainsAnMlp() {
        float[] plain = CnnNormalizationShootout.mlp(false, 16, 300, 2);
        float[] wn = CnnNormalizationShootout.mlp(true, 16, 300, 2);
        System.out.printf("MLP final loss/accuracy: dense %.3f/%.3f, weight norm %.3f/%.3f%n", plain[0], plain[1],
                wn[0], wn[1]);
        assertTrue(wn[0] < plain[0] * 1.2f, "weight norm trains at least about as fast");
    }
}
