package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.data.Batch;
import io.github.zzuegg.brainiac.experimental.nn.GrowableTransformer;
import io.github.zzuegg.brainiac.experimental.nn.GrowableTransformerBlock;
import io.github.zzuegg.brainiac.experimental.train.GrowingTrainer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * X17, growing transformers: a transformer that starts tiny (one block, one head, an FFN
 * of 4) and grows FFN width, heads and depth whenever its loss plateaus, against the same
 * tiny model trained for the same number of steps without growing.
 *
 * <p>The task is a small "fold-and-add" language: each sequence holds random tokens
 * {@code a_0..a_4}, a separator, then {@code (a_{4-i} + a_i) mod 8} for every {@code i},
 * so predicting the second half needs attention (finding two earlier tokens) and an FFN
 * (modular addition).
 */
public final class GrowingTransformers {

    private static final int VOCAB = 10;
    private static final int HALF = 5;
    private static final int LENGTH = 2 * HALF + 1;

    /**
     * The outcome.
     *
     * @param grownLoss the grown model's loss at the end (mean of the last 50 steps)
     * @param staticLoss the fixed tiny model's loss at the end
     * @param grownParameters the grown model's parameter count
     * @param staticParameters the fixed model's parameter count
     * @param growths what grew and when
     */
    public record Outcome(double grownLoss, double staticLoss, long grownParameters, long staticParameters,
            List<String> growths) {
    }

    private GrowingTransformers() {
    }

    /**
     * Draws a batch of the task.
     *
     * @param random the random source
     * @param size the batch size
     * @return token ids and next-token targets (only the second half counts)
     */
    public static Batch batch(SplittableRandom random, int size) {
        float[] x = new float[size * LENGTH];
        float[] y = new float[size * LENGTH];
        for (int b = 0; b < size; b++) {
            int[] seq = new int[LENGTH + 1];
            for (int i = 0; i < HALF; i++) {
                seq[i] = 1 + random.nextInt(VOCAB - 2);
            }
            seq[HALF] = VOCAB - 1;   // separator
            for (int i = 0; i < HALF; i++) {
                seq[HALF + 1 + i] = 1 + ((seq[HALF - 1 - i] + seq[i]) % (VOCAB - 2));
            }
            for (int t = 0; t < LENGTH; t++) {
                x[b * LENGTH + t] = seq[t];
                y[b * LENGTH + t] = t >= HALF ? seq[t + 1] : -1;
            }
        }
        return new Batch(Tensor.of(Shape.of(size, LENGTH), x), Tensor.of(Shape.of(size, LENGTH), y));
    }

    /**
     * Trains a growing and a fixed tiny transformer for the same number of steps.
     *
     * @param steps training steps
     * @param seed the seed
     * @return the comparison
     */
    public static Outcome run(int steps, long seed) {
        GrowableTransformerBlock.Config tiny = GrowableTransformerBlock.Config.of(8, 1, 4, 4, Activation.GELU);
        GrowableTransformer grown = GrowableTransformer.of(VOCAB, LENGTH, tiny, 1, new SplittableRandom(seed));
        GrowableTransformer fixed = GrowableTransformer.of(VOCAB, LENGTH, tiny, 1, new SplittableRandom(seed));
        GrowingTrainer g = GrowingTrainer.of(grown, r -> batch(r, 32)).plateau(20, 2, 0.03).limits(32, 4, 3, 3)
                .seed(seed);
        g.train(steps);
        GrowingTrainer f = GrowingTrainer.of(fixed, r -> batch(r, 32)).growing(false).seed(seed);
        f.train(steps);
        return new Outcome(tail(g.losses()), tail(f.losses()), grown.parameterCount(), fixed.parameterCount(),
                g.growths().stream().map(e -> e.step() + ":" + e.kind() + "@" + e.block()).toList());
    }

    private static double tail(List<Float> l) {
        return l.subList(l.size() - 50, l.size()).stream().mapToDouble(Float::doubleValue).average().orElse(0);
    }
}
