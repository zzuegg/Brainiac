package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SequenceLengthWarmup;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Long-context training on the counting language:
 * <ul>
 * <li>{@link #extend}: a decoder trained on short sequences is used at a longer length,
 *     either extrapolating its rotary positions or with position interpolation (positions
 *     scaled by {@code trained / target}, Chen et al. 2023,
 *     <a href="https://arxiv.org/abs/2306.15595">arXiv:2306.15595</a>), optionally followed
 *     by a short fine-tune at the target length, the recipe used to extend Llama's context;</li>
 * <li>{@link #train}: training from scratch with or without sequence-length warmup, reporting
 *     the final loss and the tokens processed.</li>
 * </ul>
 * The rotary base is small so positions beyond the trained range produce angles the model
 * never saw, which is the regime where interpolation matters.
 */
public final class LongContextTraining {

    /** The rotary base of these experiments. */
    public static final float ROPE_BASE = 16f;

    private LongContextTraining() {
    }

    /**
     * @param steps training steps
     * @param length the training length
     * @param seed the seed
     * @return a decoder trained on sequences of {@code length}
     */
    public static HookedDecoder shortContextModel(int steps, int length, long seed) {
        HookedDecoder m = HookedDecoder.create(new HookedDecoder.Settings(12, 32, 2, 4, 4, ROPE_BASE), seed);
        CountingTasks.train(m, m.parameters(), Optimizer.adam(0.01f), steps, length, seed + 1, 1, 2, 3);
        return m;
    }

    /**
     * @param lossBeforeTuning the loss at the target length before fine-tuning
     * @param lossAfterTuning the loss after
     */
    public record Extension(float lossBeforeTuning, float lossAfterTuning) {
    }

    /**
     * Uses a copy of {@code base} at a longer length.
     *
     * @param base the short-context model (not changed)
     * @param trainedLength the length it was trained at
     * @param targetLength the new length
     * @param interpolate whether to scale positions by {@code trained / target}
     * @param tuneSteps fine-tuning steps at the target length
     * @param seed the seed
     * @return the losses at the target length
     */
    public static Extension extend(HookedDecoder base, int trainedLength, int targetLength, boolean interpolate,
            int tuneSteps, long seed) {
        HookedDecoder m = base.copy();
        if (interpolate) {
            m.positionScale((float) trainedLength / targetLength);
        }
        float before = CountingTasks.evaluate(m, targetLength, seed, 1, 2, 3);
        CountingTasks.train(m, m.parameters(), Optimizer.adam(0.003f), tuneSteps, targetLength, seed + 1, 1, 2, 3);
        return new Extension(before, CountingTasks.evaluate(m, targetLength, seed, 1, 2, 3));
    }

    /**
     * @param finalLoss the loss at full length after training
     * @param tokens the tokens processed
     */
    public record Run(float finalLoss, long tokens) {
    }

    /**
     * Trains from scratch at full length, or with sequence-length warmup.
     *
     * @param steps the steps
     * @param length the full length
     * @param warmup the warmup schedule, or null for none
     * @param seed the seed
     * @return the final loss and the tokens processed
     */
    public static Run train(int steps, int length, SequenceLengthWarmup warmup, long seed) {
        HookedDecoder m = HookedDecoder.create(new HookedDecoder.Settings(12, 32, 2, 4, 4, ROPE_BASE), seed);
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        long tokens = 0;
        for (int s = 0; s < steps; s++) {
            int[][] batch = CountingTasks.sequences(r, 16, length, 12, 1, 2, 3);
            if (warmup != null) {
                batch = warmup.crop(batch, s);
            }
            tokens += (long) batch.length * batch[0].length;
            Var loss = m.loss(batch, true);
            loss.backward();
            opt.step(m.parameters());
            m.zeroGrad();
        }
        return new Run(CountingTasks.evaluate(m, length, seed + 7, 1, 2, 3), tokens);
    }
}
