package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RecurrentMemoryTransformerTest {

    private static final int S = 4;
    private static final int F = 8;
    private static final int V = 4;

    /** Segment 1 shows S symbols, segment 2 asks to repeat them: only memory tokens can carry them over. */
    private static double copyAccuracy(int memoryTokens) {
        SplittableRandom random = new SplittableRandom(3);
        RecurrentMemoryTransformer rmt = new RecurrentMemoryTransformer(F, S, memoryTokens,
                TransformerBlock.of(2).causal(true), 1, random.split());
        Dense head = new Dense(F, V, Activation.IDENTITY, Init.XAVIER, random.split());
        List<Var> params = new ArrayList<>(rmt.parameters());
        params.addAll(head.parameters());
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom data = new SplittableRandom(9);
        int batch = 8;
        int train = 300;
        int test = 10;
        double correct = 0;
        for (int step = 0; step < train + test; step++) {
            float[] x = new float[batch * 2 * S * F];
            float[] y = new float[batch * S];
            for (int b = 0; b < batch; b++) {
                for (int i = 0; i < S; i++) {
                    int sym = data.nextInt(V);
                    x[(b * 2 * S + i) * F + sym] = 1;
                    x[(b * 2 * S + S + i) * F + V + i] = 1;   // "recall position i"
                    y[b * S + i] = sym;
                }
            }
            Var out = rmt.forward(Var.of(Tensor.of(Shape.of(batch, 2 * S, F), x))).slice(1, S, 2 * S);
            Var logits = head.forward(out.reshape(batch * S, F));
            if (step < train) {
                Functional.crossEntropy(logits, Tensor.of(y)).backward();
                opt.step(params);
            } else {
                float[] p = logits.value().argmax(1).toArray();
                for (int i = 0; i < p.length; i++) {
                    correct += p[i] == y[i] ? 1 : 0;
                }
            }
        }
        return correct / (test * batch * S);
    }

    @Test
    void memoryTokensCarryInformationAcrossSegments() {
        double with = copyAccuracy(4);
        double without = copyAccuracy(0);
        assertTrue(with > 0.8, "with memory " + with);
        assertTrue(without < 0.4, "without memory " + without);
    }

    @Test
    void truncatedBpttStillRuns() {
        RecurrentMemoryTransformer rmt = new RecurrentMemoryTransformer(F, S, 2, TransformerBlock.of(2), 1,
                new SplittableRandom(1)).bptt(1);
        Var out = rmt.forward(Var.of(Tensor.normal(Shape.of(2, 3 * S, F), 0, 1, new SplittableRandom(2))));
        assertEquals(Shape.of(2, 3 * S, F), out.shape());
        out.square().sum().backward();
        assertEquals(Shape.of(2, 2, F), rmt.lastMemory().shape());
    }
}
