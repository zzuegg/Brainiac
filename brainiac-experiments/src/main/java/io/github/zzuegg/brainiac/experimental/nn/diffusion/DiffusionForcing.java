package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.diffusion.NoiseSchedule;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Diffusion Forcing (Chen et al. 2024, <a href="https://arxiv.org/abs/2407.01392">arXiv:2407.01392</a>):
 * a sequence denoiser trained with an <em>independent</em> noise level per token, so one
 * model can denoise the next token while the past is clean, slightly noisy or anything in
 * between. Rolling out, the past is re-noised to a small level {@code k}: the model then
 * treats its own earlier outputs as uncertain and does not compound their errors, which
 * keeps long rollouts stable where teacher-forced autoregression drifts.
 *
 * <p>Here the causal model is a Markov window: an MLP over the last {@code W + 1} tokens and
 * their noise levels (sinusoidal embeddings) predicts every token's noise. With
 * {@link Mode#FULL_SEQUENCE} the same network is trained the conventional way, one shared
 * level for the whole window, and rolled out by replacement inpainting (the context is
 * re-noised to the current level at every step), the full-sequence diffusion baseline.
 */
public final class DiffusionForcing {

    /** How noise levels are drawn in training, and how the rollout conditions on the past. */
    public enum Mode {
        /** Independent level per token; rollout conditions on the past at a fixed level. */
        INDEPENDENT,
        /** One shared level per window; rollout by replacement inpainting. */
        FULL_SEQUENCE
    }

    private static final int EMBED = 8;

    private final int dim;
    private final int window;
    private final Mode mode;
    private final NoiseSchedule schedule;
    private final List<Dense> layers = new ArrayList<>();
    private float clip = 4f;

    private DiffusionForcing(int dim, int window, Mode mode, int steps, int width, int depth, long seed) {
        this.dim = dim;
        this.window = window;
        this.mode = mode;
        this.schedule = NoiseSchedule.cosine(steps);
        SplittableRandom r = new SplittableRandom(seed);
        int in = (window + 1) * (dim + EMBED);
        for (int l = 0; l < depth; l++) {
            layers.add(GenBlocks.linear(l == 0 ? in : width, width, r));
        }
        layers.add(GenBlocks.zeroLinear(width, (window + 1) * dim));
    }

    /**
     * @param dim the token size
     * @param window past tokens the model sees, {@code W}
     * @param mode the training and rollout mode
     * @param steps diffusion levels
     * @param width hidden width
     * @param depth hidden layers
     * @param seed the seed
     * @return the model
     */
    public static DiffusionForcing of(int dim, int window, Mode mode, int steps, int width, int depth, long seed) {
        return new DiffusionForcing(dim, window, mode, steps, width, depth, seed);
    }

    /**
     * @param clip the clean-estimate clip during sampling
     * @return this
     */
    public DiffusionForcing clipSample(float clip) {
        this.clip = clip;
        return this;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Dense d : layers) {
            p.addAll(d.parameters());
        }
        return p;
    }

    /** @return the mode */
    public Mode mode() {
        return mode;
    }

    private Var predict(float[] noisy, int[] levels, int batch) {
        int tokens = window + 1;
        float[] t = new float[levels.length];
        for (int i = 0; i < t.length; i++) {
            t[i] = (float) levels[i] / schedule.steps();
        }
        float[] emb = GenBlocks.timeEmbedding(t, EMBED).toArray();
        float[] in = new float[batch * tokens * (dim + EMBED)];
        for (int b = 0; b < batch; b++) {
            for (int k = 0; k < tokens; k++) {
                int o = (b * tokens + k) * (dim + EMBED);
                System.arraycopy(noisy, (b * tokens + k) * dim, in, o, dim);
                System.arraycopy(emb, (b * tokens + k) * EMBED, in, o + dim, EMBED);
            }
        }
        Var h = Var.of(Tensor.of(Shape.of(batch, tokens * (dim + EMBED)), in));
        for (int l = 0; l < layers.size(); l++) {
            h = layers.get(l).forward(h);
            if (l < layers.size() - 1) {
                h = GenBlocks.silu(h);
            }
        }
        return h;
    }

    /**
     * The denoising loss over windows of {@code W + 1} clean tokens.
     *
     * @param windows clean windows, {@code [batch][(W + 1) * dim]}
     * @param random the source of levels and noise
     * @return the loss, a scalar
     */
    public Var loss(float[][] windows, SplittableRandom random) {
        int batch = windows.length;
        int tokens = window + 1;
        float[] noisy = new float[batch * tokens * dim];
        float[] eps = new float[noisy.length];
        int[] levels = new int[batch * tokens];
        for (int b = 0; b < batch; b++) {
            int shared = random.nextInt(schedule.steps());
            for (int k = 0; k < tokens; k++) {
                int level = mode == Mode.INDEPENDENT ? random.nextInt(schedule.steps()) : shared;
                levels[b * tokens + k] = level;
                double ab = schedule.alphaBar(level);
                for (int c = 0; c < dim; c++) {
                    int i = (b * tokens + k) * dim + c;
                    eps[i] = (float) random.nextGaussian();
                    noisy[i] = (float) (Math.sqrt(ab) * windows[b][k * dim + c] + Math.sqrt(1 - ab) * eps[i]);
                }
            }
        }
        Var out = predict(noisy, levels, batch);
        return out.sub(Var.of(Tensor.of(Shape.of(batch, tokens * dim), eps))).square().mean();
    }

    /**
     * Generates the next token for every context.
     *
     * @param context the last {@code W} tokens, {@code [batch][W * dim]}
     * @param contextLevel the level the past is noised to in {@link Mode#INDEPENDENT} mode
     *     (0 conditions on it as clean)
     * @param samplingSteps DDIM steps
     * @param random the noise source
     * @return the next tokens, {@code [batch][dim]}
     */
    public float[][] next(float[][] context, int contextLevel, int samplingSteps, SplittableRandom random) {
        int batch = context.length;
        int tokens = window + 1;
        int last = window * dim;
        float[] x = new float[batch * tokens * dim];
        float[] fixedContext = new float[batch * last];
        double abc = schedule.alphaBar(contextLevel);
        for (int b = 0; b < batch; b++) {
            for (int i = 0; i < last; i++) {
                fixedContext[b * last + i] = mode == Mode.INDEPENDENT && contextLevel > 0
                        ? (float) (Math.sqrt(abc) * context[b][i] + Math.sqrt(1 - abc) * random.nextGaussian())
                        : context[b][i];
            }
            for (int c = 0; c < dim; c++) {
                x[b * tokens * dim + last + c] = (float) random.nextGaussian();
            }
        }
        int[] levels = new int[batch * tokens];
        int top = schedule.steps() - 1;
        for (int s = 0; s < samplingSteps; s++) {
            int t = (int) Math.round(top * (1 - (double) s / samplingSteps));
            int tNext = s + 1 == samplingSteps ? -1 : (int) Math.round(top * (1 - (double) (s + 1) / samplingSteps));
            double ab = schedule.alphaBar(t);
            for (int b = 0; b < batch; b++) {
                for (int k = 0; k < tokens; k++) {
                    levels[b * tokens + k] = k < window ? (mode == Mode.INDEPENDENT ? contextLevel : t) : t;
                }
                for (int i = 0; i < last; i++) {
                    x[b * tokens * dim + i] = mode == Mode.INDEPENDENT ? fixedContext[b * last + i]
                            : (float) (Math.sqrt(ab) * context[b][i] + Math.sqrt(1 - ab) * random.nextGaussian());
                }
            }
            float[] eps = predict(x, levels, batch).value().toArray();
            double abNext = tNext < 0 ? 1 : schedule.alphaBar(tNext);
            for (int b = 0; b < batch; b++) {
                for (int c = 0; c < dim; c++) {
                    int i = b * tokens * dim + last + c;
                    double x0 = (x[i] - Math.sqrt(1 - ab) * eps[i]) / Math.sqrt(ab);
                    x0 = Math.max(-clip, Math.min(clip, x0));
                    double e = (x[i] - Math.sqrt(ab) * x0) / Math.sqrt(1 - ab);
                    x[i] = (float) (Math.sqrt(abNext) * x0 + Math.sqrt(1 - abNext) * e);
                }
            }
        }
        float[][] out = new float[batch][dim];
        for (int b = 0; b < batch; b++) {
            System.arraycopy(x, b * tokens * dim + last, out[b], 0, dim);
        }
        return out;
    }
}
