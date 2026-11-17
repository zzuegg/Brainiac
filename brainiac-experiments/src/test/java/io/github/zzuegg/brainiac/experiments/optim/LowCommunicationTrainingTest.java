package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.train.StreamingDiLoCo;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Map;

import org.junit.jupiter.api.Test;

class LowCommunicationTrainingTest {

    @Test
    void streamingDiLoCoAndDeMoTrainWithFarFewerBytes() {
        Map<String, LowCommunicationTraining.Outcome> r = LowCommunicationTraining.run(300, 30, 1);
        StringBuilder line = new StringBuilder("Spiral, 4 workers, 300 steps, loss/accuracy/kB:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.2f/%d;", k, v.loss(), v.accuracy(), v.bytes() / 1000)));
        System.out.println(line);
        LowCommunicationTraining.Outcome dp = r.get("data parallel");
        LowCommunicationTraining.Outcome diloco = r.get("DiLoCo");
        LowCommunicationTraining.Outcome streaming = r.get("Streaming DiLoCo");
        LowCommunicationTraining.Outcome demo = r.get("DeMo");
        assertTrue(dp.accuracy() > 0.9, line.toString());
        assertTrue(streaming.bytes() * 5 < diloco.bytes(), "4-bit fragments send far less: " + line);
        assertTrue(streaming.accuracy() > diloco.accuracy() - 0.1, line.toString());
        assertTrue(demo.bytes() * 8 < dp.bytes(), line.toString());
        assertTrue(demo.accuracy() > 0.8, line.toString());
    }

    @Test
    void quantizationKeepsSignsAndScale() {
        Tensor t = Tensor.of(1f, -0.5f, 0.26f, 0f);
        assertArrayEquals(new float[] {1f, -0.42857146f, 0.2857143f, 0f}, StreamingDiLoCo.quantize(t, 4).toArray(),
                1e-6f);
        assertArrayEquals(new float[] {0.44f, -0.44f, 0.44f, 0f}, StreamingDiLoCo.quantize(t, 1).toArray(), 1e-6f);
    }
}
