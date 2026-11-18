package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * DrQ and DrQ-v2: data-augmented RL from pixels (Kostrikov et al., ICLR 2021,
 * <a href="https://arxiv.org/abs/2004.13649">arXiv:2004.13649</a>; Yarats et al., ICLR 2022,
 * <a href="https://arxiv.org/abs/2107.09645">arXiv:2107.09645</a>). Image observations are
 * padded by a few pixels (replicating the border) and randomly cropped back to their size, a
 * random shift that regularises the critic without any auxiliary loss; DrQ also averages the Q
 * target over {@code K} and the Q estimate over {@code M} augmentations.
 *
 * <p>{@link #randomShift} is the augmentation as a transform of observation batches, so any
 * learner with an augmentation hook can use it: {@link SelfPredictive} (DQN, with K/M
 * averaging, {@link #dqn}), {@link ModernSac} ({@link #sac}: the DrQ-v2-style continuous row,
 * with n-step returns; SAC instead of DDPG with scheduled noise) and so on.
 */
public final class DrQ {

    private DrQ() {
    }

    /**
     * Random shifts of {@code [batch, channels * height * width]} images: pad by {@code pad}
     * pixels with the border values, crop a random window of the original size per sample.
     *
     * @param channels image channels
     * @param height image height
     * @param width image width
     * @param pad the most pixels shifted (4 in the papers for 84-pixel frames)
     * @param seed the seed of the shifts
     * @return the transform
     */
    public static UnaryOperator<Tensor> randomShift(int channels, int height, int width, int pad, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int plane = height * width;
        return x -> {
            int rows = x.shape().dim(0);
            float[] in = x.toArray();
            float[] out = new float[in.length];
            for (int r = 0; r < rows; r++) {
                int dy = random.nextInt(2 * pad + 1) - pad;
                int dx = random.nextInt(2 * pad + 1) - pad;
                int base = r * channels * plane;
                for (int c = 0; c < channels; c++) {
                    for (int y = 0; y < height; y++) {
                        int sy = Math.max(0, Math.min(height - 1, y + dy));
                        for (int xx = 0; xx < width; xx++) {
                            int sx = Math.max(0, Math.min(width - 1, xx + dx));
                            out[base + c * plane + y * width + xx] = in[base + c * plane + sy * width + sx];
                        }
                    }
                }
            }
            return Tensor.of(Shape.of(rows, channels * plane), out).to(x.device());
        };
    }

    /**
     * DrQ on a double DQN: the plain {@link SelfPredictive} with random shifts and K/M averaging.
     *
     * @param pixels the pixel environment (for its image size)
     * @param actions discrete actions
     * @param encoder the encoder factory
     * @param latent the embedding size
     * @param k target augmentations
     * @param m estimate augmentations
     * @param seed the seed
     * @return the learner
     */
    public static SelfPredictive dqn(PixelObservations pixels, int actions, RlNetworkFactory encoder, int latent, int k,
            int m, long seed) {
        int obs = pixels.channels() * pixels.height() * pixels.width();
        return SelfPredictive.of(obs, actions, encoder, latent, 64)
                .augment(randomShift(pixels.channels(), pixels.height(), pixels.width(), shiftFor(pixels), seed), k, m)
                .seed(seed);
    }

    /**
     * The continuous row: SAC with random shifts and 3-step returns (DrQ-v2 keeps the shifts and
     * n-step returns but drops K/M averaging).
     *
     * @param pixels the pixel environment
     * @param actionDims values per action
     * @param actor the actor factory
     * @param critic the critic factory
     * @param seed the seed
     * @return the learner
     */
    public static ModernSac sac(PixelObservations pixels, int actionDims, RlNetworkFactory actor,
            RlNetworkFactory critic, long seed) {
        int obs = pixels.channels() * pixels.height() * pixels.width();
        return ModernSac.of(obs, actionDims, actor, critic)
                .augment(randomShift(pixels.channels(), pixels.height(), pixels.width(), shiftFor(pixels), seed))
                .nStep(3).seed(seed);
    }

    private static int shiftFor(PixelObservations pixels) {
        return Math.max(1, Math.min(pixels.height(), pixels.width()) / 10);
    }
}
