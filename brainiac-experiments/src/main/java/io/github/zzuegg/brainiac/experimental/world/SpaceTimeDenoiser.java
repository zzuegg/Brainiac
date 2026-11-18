package io.github.zzuegg.brainiac.experimental.world;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A video diffusion denoiser with factorised space-time attention (Video Diffusion Models, Ho
 * et al., NeurIPS 2022, <a href="https://arxiv.org/abs/2204.03458">arXiv:2204.03458</a>). A clip
 * {@code [batch, frames, side, side]} is cut per frame into {@code patch x patch} patches (the
 * tokens of a DiT); every block then alternates <em>spatial</em> attention among the patches of
 * one frame and <em>temporal</em> attention among the same patch position across frames, the
 * latter with rotary (relative) position encoding over time. Attention cost is
 * {@code T P^2 + P T^2} instead of {@code (T P)^2} for full 3-D attention, and because the
 * spatial blocks see one frame at a time they are an image model: {@link #imageLoss} trains
 * them on single frames (where temporal attention over one frame is the identity) jointly with
 * videos.
 *
 * <p>Every frame carries its own noise level, embedded and added to its tokens. That makes
 * first-frame (or any prefix) conditioning a matter of giving the context frames level 0:
 * {@link #loss} trains with a random clean prefix, and {@link #sample} keeps the context frames
 * clean while the others are generated, which also extends videos in sliding windows. The
 * diffusion is rectified flow ({@code x_t = (1 - t) x_0 + t eps}, the network predicts the
 * velocity {@code eps - x_0}) sampled with Euler steps.
 *
 * <p>{@link Attention#FRAME_WISE} drops temporal attention (an image DDPM applied per frame,
 * the no-consistency baseline) and {@link Attention#FULL_3D} replaces both by one attention
 * over all {@code T P} tokens with learned space and time embeddings; {@code causal} masks
 * attention to earlier frames, for autoregressive use.
 */
public final class SpaceTimeDenoiser {

    /** The attention layout. */
    public enum Attention {
        /** Spatial attention per frame alternating with temporal attention per patch position. */
        FACTORISED,
        /** Spatial attention only: every frame is denoised on its own. */
        FRAME_WISE,
        /** One attention over every patch of every frame. */
        FULL_3D
    }

    private static final int LEVEL_FEATURES = 16;
    private final int side;
    private final int patch;
    private final int width;
    private final int frames;
    private final Attention mode;
    private final boolean causal;
    private final Var embed;
    private final Var embedBias;
    private final Var spacePos;
    private final Var timePos;
    private final Var level1;
    private final Var level1Bias;
    private final Var level2;
    private final Var out;
    private final Var outBias;
    private final List<TransformerBlock> spatial = new ArrayList<>();
    private final List<TransformerBlock> temporal = new ArrayList<>();
    private final List<TransformerBlock> full = new ArrayList<>();

    private SpaceTimeDenoiser(int side, int patch, int width, int depth, int heads, int frames, Attention mode,
            boolean causal, long seed) {
        if (side % patch != 0) {
            throw new IllegalArgumentException("side " + side + " is not a multiple of the patch " + patch);
        }
        this.side = side;
        this.patch = patch;
        this.width = width;
        this.frames = frames;
        this.mode = mode;
        this.causal = causal;
        SplittableRandom r = new SplittableRandom(seed);
        int pp = patch * patch;
        int tokens = (side / patch) * (side / patch);
        embed = Var.param(Tensor.normal(Shape.of(pp, width), 0, (float) (1 / Math.sqrt(pp)), r));
        embedBias = Var.param(Tensor.zeros(Shape.of(width)));
        spacePos = Var.param(Tensor.normal(Shape.of(tokens, width), 0, 0.1f, r));
        timePos = Var.param(Tensor.normal(Shape.of(frames, width), 0, 0.1f, r));
        level1 = Var.param(Tensor.normal(Shape.of(LEVEL_FEATURES, width), 0, (float) (1 / Math.sqrt(LEVEL_FEATURES)),
                r));
        level1Bias = Var.param(Tensor.zeros(Shape.of(width)));
        level2 = Var.param(Tensor.normal(Shape.of(width, width), 0, (float) (1 / Math.sqrt(width)), r));
        out = Var.param(Tensor.normal(Shape.of(width, pp), 0, (float) (0.1 / Math.sqrt(width)), r));
        outBias = Var.param(Tensor.zeros(Shape.of(pp)));
        TransformerBlock.Spec space = TransformerBlock.of(heads).ffnSize(2 * width);
        TransformerBlock.Spec time = TransformerBlock.of(heads).ffnSize(2 * width).rotary(100f);
        for (int l = 0; l < depth; l++) {
            switch (mode) {
                case FACTORISED -> {
                    spatial.add((TransformerBlock) space.create(Shape.of(1, tokens, width), r));
                    temporal.add((TransformerBlock) time.create(Shape.of(1, frames, width), r));
                }
                case FRAME_WISE -> spatial.add((TransformerBlock) space.create(Shape.of(1, tokens, width), r));
                case FULL_3D -> full.add((TransformerBlock) space.create(Shape.of(1, frames * tokens, width), r));
                default -> throw new IllegalArgumentException(mode.toString());
            }
        }
    }

    /**
     * Creates a denoiser.
     *
     * @param side the frame side
     * @param patch the patch side
     * @param width the token width
     * @param depth the number of blocks
     * @param heads attention heads
     * @param frames the longest clip
     * @param mode the attention layout
     * @param causal whether frames attend only to themselves and earlier frames
     * @param seed the seed
     * @return the denoiser
     */
    public static SpaceTimeDenoiser of(int side, int patch, int width, int depth, int heads, int frames,
            Attention mode, boolean causal, long seed) {
        return new SpaceTimeDenoiser(side, patch, width, depth, heads, frames, mode, causal, seed);
    }

    /** @return the attention layout */
    public Attention mode() {
        return mode;
    }

    /** @return the frame side */
    public int side() {
        return side;
    }

    /** @return the longest clip */
    public int frames() {
        return frames;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, embedBias, spacePos, level1, level1Bias, level2, out, outBias));
        if (mode == Attention.FULL_3D) {
            p.add(timePos);
        }
        for (List<TransformerBlock> blocks : List.of(spatial, temporal, full)) {
            for (TransformerBlock b : blocks) {
                p.addAll(b.parameters());
            }
        }
        return p;
    }

    private static Tensor causalBias(int frames, int tokensPerFrame) {
        int n = frames * tokensPerFrame;
        float[] m = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                m[i * n + j] = j / tokensPerFrame > i / tokensPerFrame ? -1e9f : 0;
            }
        }
        return Tensor.of(Shape.of(n, n), m);
    }

    /**
     * Predicts the flow velocity of every frame.
     *
     * @param x the noisy clips {@code [batch, T, side, side]}, {@code T <= frames}
     * @param levels the noise level of every frame in {@code [0, 1]}, {@code [batch, T]}
     * @return the velocity {@code eps - x_0}, {@code [batch, T, side, side]}
     */
    public Var apply(Var x, Tensor levels) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        int g = side / patch;
        int p = g * g;
        Var tok = Functional.patchify(x.reshape(b * t, 1, side, side), patch).matmul(embed).add(embedBias).add(spacePos);
        Tensor lf = Functional.timestepEmbedding(levels.reshape(b * t).mul(Tensor.scalar(100f)), LEVEL_FEATURES,
                1000f);
        Var lev = Activation.SILU.forward(Var.of(lf).matmul(level1).add(level1Bias)).matmul(level2);
        Var h = tok.reshape(b, t, p, width).add(lev.reshape(b, t, 1, width));
        switch (mode) {
            case FACTORISED, FRAME_WISE -> {
                Tensor tb = causal ? causalBias(t, 1) : null;
                for (int l = 0; l < spatial.size(); l++) {
                    h = spatial.get(l).forward(h.reshape(b * t, p, width)).reshape(b, t, p, width);
                    if (mode == Attention.FACTORISED) {
                        Var ht = h.permute(0, 2, 1, 3).reshape(b * p, t, width);
                        ht = temporal.get(l).forward(List.of(ht), tb, false);
                        h = ht.reshape(b, p, t, width).permute(0, 2, 1, 3);
                    }
                }
            }
            case FULL_3D -> {
                Tensor fb = causal ? causalBias(t, p) : null;
                h = h.add(timePos.slice(0, 0, t).reshape(1, t, 1, width)).reshape(b, t * p, width);
                for (TransformerBlock blk : full) {
                    h = blk.forward(List.of(h), fb, false);
                }
            }
            default -> throw new IllegalStateException(mode.toString());
        }
        Var o = h.reshape(b * t, p, width).matmul(out).add(outBias);
        return Functional.unpatchify(o, 1, side, side, patch).reshape(b, t, side, side);
    }

    /**
     * The rectified-flow loss on clean clips. With probability one half (when
     * {@code maxContext > 0}) a random prefix of up to {@code maxContext} frames is kept clean as
     * context and left out of the loss; the other frames share one noise level per clip.
     *
     * @param clips {@code [batch, T, side, side]}
     * @param maxContext the longest clean prefix
     * @param random the source of levels and noise
     * @return the mean squared velocity error over the generated frames
     */
    public Var loss(Tensor clips, int maxContext, SplittableRandom random) {
        int b = clips.shape().dim(0);
        int t = clips.shape().dim(1);
        int[] context = new int[b];
        float[] lv = new float[b * t];
        for (int i = 0; i < b; i++) {
            context[i] = maxContext > 0 && random.nextBoolean() ? 1 + random.nextInt(Math.min(maxContext, t - 1)) : 0;
            float level = random.nextFloat();
            for (int f = 0; f < t; f++) {
                lv[i * t + f] = f < context[i] ? 0 : level;
            }
        }
        return flowLoss(clips, lv, context, random);
    }

    /**
     * Joint image training: the loss on single frames (clips of one frame), which trains the
     * spatial layers as an image model.
     *
     * @param images {@code [batch, 1, side, side]}
     * @param random the source of levels and noise
     * @return the loss
     */
    public Var imageLoss(Tensor images, SplittableRandom random) {
        return loss(images, 0, random);
    }

    /**
     * The loss with given per-frame levels.
     *
     * @param clips {@code [batch, T, side, side]}
     * @param levels {@code [batch * T]} per-frame levels
     * @param context per clip, the number of leading frames left out of the loss
     * @param random the noise source
     * @return the loss
     */
    Var flowLoss(Tensor clips, float[] levels, int[] context, SplittableRandom random) {
        int b = clips.shape().dim(0);
        int t = clips.shape().dim(1);
        int pix = side * side;
        float[] x0 = clips.toArray();
        float[] xt = new float[x0.length];
        float[] target = new float[x0.length];
        float[] mask = new float[x0.length];
        double count = 0;
        for (int i = 0; i < b; i++) {
            for (int f = 0; f < t; f++) {
                float lv = levels[i * t + f];
                boolean scored = f >= context[i];
                for (int k = 0; k < pix; k++) {
                    int idx = (i * t + f) * pix + k;
                    float e = (float) random.nextGaussian();
                    xt[idx] = (1 - lv) * x0[idx] + lv * e;
                    target[idx] = e - x0[idx];
                    mask[idx] = scored ? 1 : 0;
                }
                count += scored ? pix : 0;
            }
        }
        Shape s = clips.shape();
        Var pred = apply(Var.of(Tensor.of(s, xt)), Tensor.of(Shape.of(b, t), levels));
        return pred.sub(Tensor.of(s, target)).square().mul(Tensor.of(s, mask)).sum().mul((float) (1 / Math.max(1,
                count)));
    }

    /**
     * Generates clips with Euler steps, keeping the context frames clean.
     *
     * @param n how many clips
     * @param length the clip length
     * @param context clean leading frames {@code [n, k, side, side]}, or {@code null}
     * @param steps Euler steps
     * @param random the noise source
     * @return {@code [n, length, side, side]}
     */
    public Tensor sample(int n, int length, Tensor context, int steps, SplittableRandom random) {
        int k = context == null ? 0 : context.shape().dim(1);
        int pix = side * side;
        float[] x = new float[n * length * pix];
        float[] ctx = context == null ? new float[0] : context.toArray();
        for (int i = 0; i < n; i++) {
            for (int f = 0; f < length; f++) {
                for (int p = 0; p < pix; p++) {
                    x[(i * length + f) * pix + p] = f < k ? ctx[(i * k + f) * pix + p] : (float) random.nextGaussian();
                }
            }
        }
        Shape s = Shape.of(n, length, side, side);
        float dt = 1f / steps;
        for (int step = 0; step < steps; step++) {
            float level = 1 - step * dt;
            float[] lv = new float[n * length];
            for (int i = 0; i < n; i++) {
                for (int f = 0; f < length; f++) {
                    lv[i * length + f] = f < k ? 0 : level;
                }
            }
            Tensor in = Tensor.of(s, x);
            float[] v = Var.noGrad(() -> apply(Var.of(in), Tensor.of(Shape.of(n, length), lv))).value().toArray();
            for (int i = 0; i < n; i++) {
                for (int f = k; f < length; f++) {
                    for (int p = 0; p < pix; p++) {
                        int idx = (i * length + f) * pix + p;
                        x[idx] -= dt * v[idx];
                    }
                }
            }
        }
        return Tensor.of(s, x);
    }

    /**
     * Extends clips beyond the training length in sliding windows: each new window keeps the
     * last {@code overlap} generated frames as clean context and generates the rest.
     *
     * @param n how many clips
     * @param total the frames to generate
     * @param window the window length (at most {@link #frames})
     * @param overlap the context frames carried into each next window
     * @param first clean first frames {@code [n, k, side, side]}, or {@code null}
     * @param steps Euler steps per window
     * @param random the noise source
     * @return {@code [n, total, side, side]}
     */
    public Tensor extend(int n, int total, int window, int overlap, Tensor first, int steps, SplittableRandom random) {
        Tensor clip = sample(n, window, first, steps, random);
        int have = window;
        while (have < total) {
            Tensor ctx = clip.slice(1, have - overlap, have);
            Tensor next = sample(n, window, ctx, steps, random);
            int add = Math.min(window - overlap, total - have);
            clip = Tensor.concat(1, clip, next.slice(1, overlap, overlap + add));
            have += add;
        }
        return clip;
    }

    @Override
    public String toString() {
        return "SpaceTimeDenoiser(" + mode + (causal ? ", causal" : "") + ", width " + width + ")";
    }
}
