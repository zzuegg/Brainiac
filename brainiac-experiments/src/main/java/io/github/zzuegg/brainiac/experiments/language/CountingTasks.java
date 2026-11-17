package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * The toy language of these experiments: sequences that count up modulo a small vocabulary
 * by a per-sequence stride ({@code 3 5 7 9 11 1 ...}). A model must read the stride from
 * the first two tokens and continue it, which a two-block decoder learns in about a hundred
 * steps. Pretraining uses some strides and fine-tuning another, so adaptation is measurable.
 */
public final class CountingTasks {

    private CountingTasks() {
    }

    /**
     * @param random the source of starts and strides
     * @param batch the sequences
     * @param length the tokens per sequence
     * @param vocab the modulus
     * @param strides the strides drawn from
     * @return {@code [batch][length]}
     */
    public static int[][] sequences(SplittableRandom random, int batch, int length, int vocab, int... strides) {
        int[][] b = new int[batch][length];
        for (int i = 0; i < batch; i++) {
            int stride = strides[random.nextInt(strides.length)];
            b[i][0] = random.nextInt(vocab);
            for (int t = 1; t < length; t++) {
                b[i][t] = (b[i][t - 1] + stride) % vocab;
            }
        }
        return b;
    }

    /**
     * Trains parameters of a decoder on counting sequences with next-token cross-entropy.
     *
     * @param model the decoder
     * @param params the values to train
     * @param optimizer the optimizer
     * @param steps the steps
     * @param length tokens per sequence
     * @param seed the data seed
     * @param strides the strides
     * @return the last loss
     */
    public static float train(HookedDecoder model, List<Var> params, Optimizer optimizer, int steps, int length,
            long seed, int... strides) {
        SplittableRandom r = new SplittableRandom(seed);
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            Var loss = model.loss(sequences(r, 16, length, model.vocabSize(), strides), true);
            loss.backward();
            optimizer.step(params);
            model.zeroGrad();
            for (Var p : params) {
                p.zeroGrad();
            }
            last = loss.value().item();
        }
        return last;
    }

    /**
     * @param model the decoder
     * @param length tokens per sequence
     * @param seed the data seed
     * @param strides the strides
     * @return the mean next-token loss on 32 fresh sequences
     */
    public static float evaluate(HookedDecoder model, int length, long seed, int... strides) {
        return model.loss(sequences(new SplittableRandom(seed), 32, length, model.vocabSize(), strides), false)
                .value().item();
    }

    /**
     * A two-block decoder over 12 tokens pretrained on strides 1 and 2.
     *
     * @param steps training steps, for example 150
     * @param seed the seed
     * @return the model
     */
    public static HookedDecoder pretrained(int steps, long seed) {
        HookedDecoder m = HookedDecoder.create(HookedDecoder.Settings.tiny(12), seed);
        train(m, m.parameters(), Optimizer.adam(0.01f), steps, 12, seed + 1, 1, 2);
        return m;
    }
}
