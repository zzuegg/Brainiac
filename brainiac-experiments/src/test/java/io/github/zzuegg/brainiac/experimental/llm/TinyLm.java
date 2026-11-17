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

    private static HookedDecoder deep;

    /** A 4-block decoder trained with an exit loss at every block (shared; do not modify). */
    static synchronized HookedDecoder deeplySupervised() {
        if (deep == null) {
            HookedDecoder m = HookedDecoder.create(HookedDecoder.Settings.tiny(VOCAB).withLayers(4), 11);
            Optimizer opt = Optimizer.adam(0.01f);
            SplittableRandom r = new SplittableRandom(2);
            float[] w = {0.25f, 0.5f, 0.75f, 1f};
            for (int s = 0; s < 150; s++) {
                Var loss = EarlyExit.deepSupervisionLoss(m, counting(r, 16, 12), w);
                loss.backward();
                opt.step(m.parameters());
                m.zeroGrad();
            }
            deep = m;
        }
        return deep;
    }

    static int[] greedy(io.github.zzuegg.brainiac.llm.LanguageModel model, int[] prompt, int n) {
        return DecodingPipeline.of(model).generate(prompt, n, new SplittableRandom(0));
    }

    /** The share of tokens that continue the counting pattern set by the last two context tokens. */
    static double countingAccuracy(int[] prompt, int[] out) {
        int stride = Math.floorMod(prompt[prompt.length - 1] - prompt[prompt.length - 2], VOCAB);
        int prev = prompt[prompt.length - 1];
        int ok = 0;
        for (int t : out) {
            ok += t == (prev + stride) % VOCAB ? 1 : 0;
            prev = (prev + stride) % VOCAB;
        }
        return (double) ok / out.length;
    }

    static float meanLoss(HookedDecoder model, int length, long seed) {
        return model.loss(counting(new SplittableRandom(seed), 32, length), false).value().item();
    }
}
