package io.github.zzuegg.brainiac.experimental.world;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.UNet2d;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A frame-level autoregressive video diffusion model: a {@link UNet2d} that denoises the next
 * frame given the last {@code W} frames, each history frame carrying its own noise level. The
 * levels (sinusoidally embedded) condition the U-Net; the history frames and the noisy frame
 * are its input channels. Diffusion is rectified flow: {@code x_t = (1 - t) x_0 + t eps}, the
 * network predicts {@code eps - x_0}.
 *
 * <p>Two ways to train it ({@link #loss}):
 * <ul>
 *   <li>{@link Training#TEACHER_FORCING}: the history is the clean ground truth, as in
 *       ordinary autoregressive video diffusion; at inference the model then sees its own
 *       imperfect frames for the first time (exposure bias).
 *   <li>{@link Training#DIFFUSION_FORCING}: every history frame gets an independent random
 *       level and is dropped entirely (level 1, pure noise) with a given probability, as in
 *       the diffusion forcing transformer (Chen et al., 2024; Song et al., 2025); the model
 *       learns to treat history as uncertain, and the same model can be asked for the score
 *       with any subset of the history, which {@link HistoryGuidance} exploits.
 * </ul>
 *
 * <p>Missing history at the start of a clip is represented as dropped frames. {@link #velocity}
 * gives the raw prediction for any history levels; {@link #next} samples a frame with Euler
 * steps, optionally re-noising the history to a small level (diffusion forcing's stabiliser)
 * and optionally under a {@link Guide}; {@link #rollout} generates long videos.
 */
public final class FrameDenoiser {

    /** How the history is presented during training. */
    public enum Training {
        /** Clean history. */
        TEACHER_FORCING,
        /** Independently noised history, randomly dropped. */
        DIFFUSION_FORCING
    }

    /** Combines velocities of the same model under different histories, such as guidance. */
    @FunctionalInterface
    public interface Guide {
        /**
         * The velocity used for one Euler step.
         *
         * @param model the model
         * @param history the history {@code [b, W, side, side]}
         * @param available how many of the most recent history frames exist
         * @param x the current noisy frames {@code [b, 1, side, side]}
         * @param level the current level
         * @param historyLevel the level the history is noised to
         * @param random the noise source
         * @return the velocity {@code [b * side * side]}
         */
        float[] velocity(FrameDenoiser model, Tensor history, int available, Tensor x, float level,
                float historyLevel, SplittableRandom random);
    }

    private static final int LEVEL_FEATURES = 8;
    private final int side;
    private final int window;
    private final UNet2d net;

    private FrameDenoiser(int side, int window, int base, long seed) {
        this.side = side;
        this.window = window;
        this.net = new UNet2d(window + 1, 1, base, 2, true, (window + 1) * LEVEL_FEATURES, new SplittableRandom(seed));
    }

    /**
     * Creates a model.
     *
     * @param side the frame side (a multiple of 4)
     * @param window history frames {@code W}
     * @param base the U-Net's first-stage channels
     * @param seed the seed
     * @return the model
     */
    public static FrameDenoiser of(int side, int window, int base, long seed) {
        return new FrameDenoiser(side, window, base, seed);
    }

    /** @return the history length */
    public int window() {
        return window;
    }

    /** @return the frame side */
    public int side() {
        return side;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return net.parameters();
    }

    /**
     * The raw velocity prediction.
     *
     * @param input history and current frame {@code [b, W + 1, side, side]}, already noised
     * @param levels their levels {@code [b, W + 1]}
     * @return {@code [b, 1, side, side]}
     */
    public Var apply(Var input, Tensor levels) {
        int b = input.shape().dim(0);
        Tensor emb = Functional.timestepEmbedding(levels.reshape(b * (window + 1)).mul(Tensor.scalar(100f)),
                LEVEL_FEATURES, 1000f).reshape(b, (window + 1) * LEVEL_FEATURES);
        return net.forward(input, Var.of(emb));
    }

    /**
     * Noises a history to per-frame levels; frames at level 1 become pure noise.
     *
     * @param history {@code [b, W, side, side]}
     * @param levels {@code [b * W]}
     * @param random the noise source
     * @return the noised history values
     */
    float[] noiseHistory(Tensor history, float[] levels, SplittableRandom random) {
        float[] h = history.toArray();
        int pix = side * side;
        for (int f = 0; f < levels.length; f++) {
            float l = levels[f];
            if (l == 0) {
                continue;
            }
            for (int p = 0; p < pix; p++) {
                int i = f * pix + p;
                h[i] = (1 - l) * h[i] + l * (float) random.nextGaussian();
            }
        }
        return h;
    }

    /**
     * The velocity for the current frame with the given history levels.
     *
     * @param history {@code [b, W, side, side]}
     * @param historyLevels {@code [b * W]} (1 drops a frame)
     * @param x the current noisy frame {@code [b, 1, side, side]}
     * @param level its level
     * @param random the noise for re-noising the history
     * @return the velocity, a graph-free {@link Var} {@code [b, 1, side, side]}
     */
    public Var velocity(Tensor history, float[] historyLevels, Tensor x, float level, SplittableRandom random) {
        int b = x.shape().dim(0);
        Tensor h = Tensor.of(history.shape(), noiseHistory(history, historyLevels, random));
        float[] lv = new float[b * (window + 1)];
        for (int i = 0; i < b; i++) {
            for (int f = 0; f < window; f++) {
                lv[i * (window + 1) + f] = historyLevels[i * window + f];
            }
            lv[i * (window + 1) + window] = level;
        }
        return apply(Var.of(Tensor.concat(1, h, x)), Tensor.of(Shape.of(b, window + 1), lv));
    }

    /**
     * Uniform history levels: {@code level} for the {@code available} most recent frames, 1 for
     * the others.
     *
     * @param b the batch
     * @param available the existing recent frames
     * @param level their level
     * @return {@code [b * W]}
     */
    public float[] historyLevels(int b, int available, float level) {
        float[] lv = new float[b * window];
        for (int i = 0; i < b; i++) {
            for (int f = 0; f < window; f++) {
                lv[i * window + f] = f >= window - available ? level : 1f;
            }
        }
        return lv;
    }

    /**
     * The training loss on random windows of clips.
     *
     * @param clips {@code [b, T, side, side]}, {@code T >= 2}
     * @param mode how the history is presented
     * @param dropHistory for diffusion forcing, the probability of dropping each history frame
     * @param random the random source
     * @return the mean squared velocity error of the predicted frames
     */
    public Var loss(Tensor clips, Training mode, float dropHistory, SplittableRandom random) {
        int b = clips.shape().dim(0);
        int t = clips.shape().dim(1);
        int pix = side * side;
        float[] c = clips.toArray();
        float[] input = new float[b * (window + 1) * pix];
        float[] levels = new float[b * (window + 1)];
        float[] target = new float[b * pix];
        for (int i = 0; i < b; i++) {
            int end = 1 + random.nextInt(t - 1);
            for (int f = 0; f <= window; f++) {
                int src = end - window + f;
                float level;
                if (f == window) {
                    level = random.nextFloat();
                } else if (src < 0) {
                    level = 1;
                } else if (mode == Training.TEACHER_FORCING) {
                    level = 0;
                } else {
                    level = random.nextFloat() < dropHistory ? 1 : random.nextFloat();
                }
                levels[i * (window + 1) + f] = level;
                for (int p = 0; p < pix; p++) {
                    float x0 = src < 0 ? 0 : c[(i * t + src) * pix + p];
                    float e = (float) random.nextGaussian();
                    input[(i * (window + 1) + f) * pix + p] = (1 - level) * x0 + level * e;
                    if (f == window) {
                        target[i * pix + p] = e - x0;
                    }
                }
            }
        }
        Var pred = apply(Var.of(Tensor.of(Shape.of(b, window + 1, side, side), input)),
                Tensor.of(Shape.of(b, window + 1), levels));
        return pred.sub(Tensor.of(Shape.of(b, 1, side, side), target)).square().mean();
    }

    /**
     * Samples the next frame.
     *
     * @param history {@code [b, W, side, side]}, most recent last
     * @param available how many of the most recent history frames exist
     * @param historyLevel the level the history is re-noised to (0 keeps it clean)
     * @param steps Euler steps
     * @param guide the velocity rule, or {@code null} for the plain conditional velocity
     * @param random the noise source
     * @return {@code [b, 1, side, side]}
     */
    public Tensor next(Tensor history, int available, float historyLevel, int steps, Guide guide,
            SplittableRandom random) {
        int b = history.shape().dim(0);
        Shape s = Shape.of(b, 1, side, side);
        float[] x = new float[b * side * side];
        for (int i = 0; i < x.length; i++) {
            x[i] = (float) random.nextGaussian();
        }
        float dt = 1f / steps;
        for (int k = 0; k < steps; k++) {
            float level = 1 - k * dt;
            Tensor xt = Tensor.of(s, x);
            float[] v;
            if (guide == null) {
                float[] hl = historyLevels(b, available, historyLevel);
                v = Var.noGrad(() -> velocity(history, hl, xt, level, random)).value().toArray();
            } else {
                v = guide.velocity(this, history, available, xt, level, historyLevel, random);
            }
            for (int i = 0; i < x.length; i++) {
                x[i] -= dt * v[i];
            }
        }
        return Tensor.of(s, x);
    }

    /**
     * Generates long videos frame by frame from their first frames.
     *
     * @param first {@code [b, k, side, side]} given frames
     * @param total the frames wanted (including the given ones)
     * @param historyLevel the level the history is re-noised to
     * @param steps Euler steps per frame
     * @param guide the velocity rule, or {@code null}
     * @param random the noise source
     * @return {@code [b, total, side, side]}
     */
    public Tensor rollout(Tensor first, int total, float historyLevel, int steps, Guide guide,
            SplittableRandom random) {
        Tensor video = first;
        int b = first.shape().dim(0);
        while (video.shape().dim(1) < total) {
            int have = video.shape().dim(1);
            int available = Math.min(window, have);
            Tensor history = history(video, available, b);
            video = Tensor.concat(1, video, next(history, available, historyLevel, steps, guide, random));
        }
        return video;
    }

    /** The last {@code W} frames of a video, zero-padded in front when it is shorter. */
    Tensor history(Tensor video, int available, int b) {
        int have = video.shape().dim(1);
        Tensor recent = video.slice(1, have - available, have);
        if (available == window) {
            return recent;
        }
        return Tensor.concat(1, Tensor.zeros(Shape.of(b, window - available, side, side)), recent);
    }
}
