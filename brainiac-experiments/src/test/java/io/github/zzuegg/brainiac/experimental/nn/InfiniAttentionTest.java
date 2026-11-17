package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertTrue;

class InfiniAttentionTest {

    private static final int K = 4;

    static int[] shuffled(SplittableRandom r, int k) {
        int[] p = new int[k];
        for (int i = 0; i < k; i++) {
            p[i] = i;
        }
        for (int i = k - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    /** Facts (key one-hot + value one-hot) for K steps, then K questions (key one-hot only). */
    static float[] episode(int[] perm, int[] order, int k) {
        float[] x = new float[2 * k * 2 * k];
        for (int i = 0; i < k; i++) {
            x[i * 2 * k + i] = 1;
            x[i * 2 * k + k + perm[i]] = 1;
            x[(k + i) * 2 * k + order[i]] = 1;
        }
        return x;
    }

    private static double accuracy(boolean memory, boolean delta) {
        SplittableRandom random = new SplittableRandom(11);
        InfiniAttention a = new InfiniAttention(2 * K, 1, 2 * K, K, random.split()).useMemory(memory).delta(delta);
        Dense head = new Dense(2 * K, K, Activation.IDENTITY, Init.XAVIER, random.split());
        List<Var> params = new ArrayList<>(a.parameters());
        params.addAll(head.parameters());
        Optimizer opt = Optimizer.adam(0.02f);
        SplittableRandom data = new SplittableRandom(2);
        double correct = 0;
        int train = 300;
        int test = 20;
        int batch = 4;
        for (int step = 0; step < train + test; step++) {
            float[] x = new float[batch * 4 * K * K];
            float[] y = new float[batch * K];
            for (int b = 0; b < batch; b++) {
                int[] perm = shuffled(data, K);
                int[] order = shuffled(data, K);
                System.arraycopy(episode(perm, order, K), 0, x, b * 4 * K * K, 4 * K * K);
                for (int i = 0; i < K; i++) {
                    y[b * K + i] = perm[order[i]];
                }
            }
            Var h = a.forward(Var.of(Tensor.of(Shape.of(batch, 2 * K, 2 * K), x))).slice(1, K, 2 * K);
            Var out = head.forward(h.reshape(batch * K, 2 * K));
            if (step < train) {
                Functional.crossEntropy(out, Tensor.of(y)).backward();
                opt.step(params);
            } else {
                float[] p = out.value().argmax(1).toArray();
                for (int i = 0; i < p.length; i++) {
                    correct += p[i] == y[i] ? 1 : 0;
                }
            }
        }
        return correct / (test * batch * K);
    }

    @Test
    void laterSegmentsRecallEarlierOnesThroughTheMemory() {
        double with = accuracy(true, false);
        double withDelta = accuracy(true, true);
        double without = accuracy(false, false);
        assertTrue(with > 0.85, "linear memory " + with);
        assertTrue(withDelta > 0.85, "delta memory " + withDelta);
        assertTrue(without < 0.45, "no memory " + without);
    }

    @Test
    void gradientsFlowThroughTheMemory() {
        InfiniAttention a = new InfiniAttention(4, 2, 2, 2, new SplittableRandom(1)).delta(true);
        Tensor x = Tensor.normal(Shape.of(1, 6, 4), 0, 1, new SplittableRandom(2));
        // the last segment's output depends on the first segment's input only through the memory
        GradCheck.Result r = GradCheck.check(xs -> a.forward(xs.get(0)).slice(1, 4, 6).square().sum(), List.of(x));
        assertTrue(r.passed(), r.toString());
    }
}
