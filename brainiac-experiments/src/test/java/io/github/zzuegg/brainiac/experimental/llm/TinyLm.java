package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/** Shared toy data and a cached tiny decoder trained on it. */
final class TinyLm {

    /** A tiny vocabulary of 12 symbols; sequences count up by a stride of 1 or 2 modulo 12. */
    static final int VOCAB = 12;

    private static HookedDecoder trained;

    private TinyLm() {
    }

    static int[][] counting(SplittableRandom r, int size, int length) {
        int[][] b = new int[size][length];
        for (int i = 0; i < size; i++) {
            int stride = 1 + r.nextInt(2);
            b[i][0] = r.nextInt(VOCAB);
            for (int t = 1; t < length; t++) {
                b[i][t] = (b[i][t - 1] + stride) % VOCAB;
            }
        }
        return b;
    }

    /** Trains {@code model} on counting sequences; returns the last loss. */
    static float train(HookedDecoder model, List<Var> params, Optimizer opt, int steps, int length, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            Var loss = model.loss(counting(r, 16, length), true);
            loss.backward();
            opt.step(params);
            model.zeroGrad();
            last = loss.value().item();
        }
        return last;
    }

    /** A 2-block decoder trained on counting sequences (shared; do not modify). */
    static synchronized HookedDecoder trained() {
        if (trained == null) {
            HookedDecoder m = HookedDecoder.create(HookedDecoder.Settings.tiny(VOCAB), 7);
            train(m, m.parameters(), Optimizer.adam(0.01f), 150, 12, 1);
            trained = m;
        }
        return trained;
    }

    static float meanLoss(HookedDecoder model, int length, long seed) {
        return model.loss(counting(new SplittableRandom(seed), 32, length), false).value().item();
    }
}
