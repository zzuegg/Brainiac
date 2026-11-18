package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.compress.DoReFaQuantizer;
import io.github.zzuegg.brainiac.experimental.compress.PactActivation;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

import org.junit.jupiter.api.Test;

class QuantizationAwareTest {

    @Test
    void pactClipsQuantisesAndLearnsItsLevel() {
        PactActivation pact = new PactActivation(2, 3f);
        Var x = Var.of(Tensor.of(-1f, 0.4f, 1.6f, 5f));
        Var y = pact.forward(x);
        assertArrayEquals(new float[] {0, 0, 2, 3}, y.value().toArray(), 1e-6f);
        y.sum().backward();
        assertTrue(pact.alpha().grad().item() == 1f, "only the clipped input pushes alpha");
        float[] w = DoReFaQuantizer.weight(Var.of(Tensor.of(-2f, -0.1f, 0.1f, 2f)), 1).value().toArray();
        assertArrayEquals(new float[] {-1, -1, 1, 1}, w, 1e-6f);
    }

    @Test
    void pactAndDorefaAgainstLsqAndFixedRange() {
        List<CompactRow> rows = CompactModels.quantizationAware(new int[] {2, 3, 4}, 80, 1);
        StringBuilder line = new StringBuilder("8.5.11 QAT W/A bits, digits acc:");
        for (CompactRow r : rows) {
            line.append(String.format(" %s %.3f;", r.method(), r.metric()));
        }
        System.out.println(line);
        for (CompactRow r : rows.subList(rows.size() - 4, rows.size())) {
            assertTrue(r.metric() > 0.3, "4-bit QAT trains: " + r);
        }
    }
}
