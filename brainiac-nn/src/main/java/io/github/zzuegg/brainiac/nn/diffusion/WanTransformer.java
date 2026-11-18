package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A video diffusion transformer as Wan 2.1 and 2.2 use (Alibaba, 2025): the latent video is
 * cut into space-time patches, which pass through blocks of self-attention with 3-d rotary
 * positions (frame, row, column), cross-attention to the text tokens (and, for
 * image-to-video, to image tokens) and a feed-forward, each block modulated by the noise
 * level through a learned table plus a shared projection. It predicts the flow velocity
 * {@code noise - data} (see {@link FlowSampler}).
 *
 * <p>The layer takes latents {@code [batch, channels, frames, height, width]} and returns
 * the velocity in the same layout. The noise level is given as timesteps in
 * {@code [0, 1000]}, one per example, or one per patch for models that keep some frames
 * clean (Wan 2.2's text-and-image-to-video).
 *
 * <pre>{@code
 * WanTransformer dit = new WanTransformer(WanTransformer.Settings.of(16, 2, 32, 2, 64, 48), random);
 * Var v = dit.forward(Var.of(latents), timesteps, Var.of(text), null, true);
 * }</pre>
 */
public final class WanTransformer implements MultiInputLayer {

    /**
     * The shape of the transformer.
     *
     * @param inChannels the channels of the input latents (more than the output for
     *     image-to-video models that take the image's latents and a mask alongside)
     * @param outChannels the channels of the predicted velocity
     * @param patch the patch size in frames, rows and columns (Wan: 1, 2, 2)
     * @param heads the attention heads
     * @param headSize the features per head (Wan: 128)
     * @param layers the blocks
     * @param ffnSize the feed-forward width
     * @param textFeatures the width of the text encoder's states (umT5-XXL: 4096)
     * @param frequencies the sinusoids of the time embedding (256)
     * @param epsilon the norms' epsilon
     * @param crossAttentionNorm whether the input of cross-attention is layer-normed
     * @param imageFeatures the width of image tokens for image-to-video (CLIP: 1280), or 0
     * @param theta the rotary frequency base
     */
    public record Settings(int inChannels, int outChannels, int[] patch, int heads, int headSize, int layers,
            int ffnSize, int textFeatures, int frequencies, float epsilon, boolean crossAttentionNorm,
            int imageFeatures, float theta) {

        /** Checks the values. */
        public Settings {
            patch = patch.clone();
            if (patch.length != 3 || patch[0] < 1 || patch[1] < 1 || patch[2] < 1) {
                throw new IllegalArgumentException("the patch has a size in frames, rows and columns, got "
                        + Arrays.toString(patch));
            }
            if (inChannels < 1 || outChannels < 1 || heads < 1 || headSize < 6 || headSize % 2 != 0 || layers < 0
                    || ffnSize < 1 || textFeatures < 1 || frequencies < 2 || frequencies % 2 != 0 || !(epsilon > 0)
                    || imageFeatures < 0) {
                throw new IllegalArgumentException("invalid Wan transformer shape");
            }
        }

        /**
         * Settings with Wan's patch (1, 2, 2), 256 time frequencies, epsilon 1e-6, normed
         * cross-attention and no image tokens.
         *
         * @param channels the latent channels in and out (Wan 2.1: 16, Wan 2.2 5B: 48)
         * @param heads the attention heads
         * @param headSize the features per head
         * @param layers the blocks
         * @param ffnSize the feed-forward width
         * @param textFeatures the width of the text states
         * @return the settings
         */
        public static Settings of(int channels, int heads, int headSize, int layers, int ffnSize, int textFeatures) {
            return new Settings(channels, channels, new int[] {1, 2, 2}, heads, headSize, layers, ffnSize,
                    textFeatures, 256, 1e-6f, true, 0, 10000);
        }

        /** @return the model width, {@code heads * headSize} */
        public int features() {
            return heads * headSize;
        }

        /** @return the values in one input patch */
        public int patchIn() {
            return inChannels * patch[0] * patch[1] * patch[2];
        }

        /** @return the values in one output patch */
        public int patchOut() {
            return outChannels * patch[0] * patch[1] * patch[2];
        }

        /**
         * How the head features split among the rotary axes, as Wan does: rows and columns
         * {@code 2 * (headSize / 6)} each, frames the rest.
         *
         * @return {frames, rows, columns}
         */
        public int[] axes() {
            int hw = 2 * (headSize / 6);
            return new int[] {headSize - 2 * hw, hw, hw};
        }

        @Override
        public int[] patch() {
            return patch.clone();
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Settings s && inChannels == s.inChannels && outChannels == s.outChannels
                    && Arrays.equals(patch, s.patch) && heads == s.heads && headSize == s.headSize
                    && layers == s.layers && ffnSize == s.ffnSize && textFeatures == s.textFeatures
                    && frequencies == s.frequencies && epsilon == s.epsilon
                    && crossAttentionNorm == s.crossAttentionNorm && imageFeatures == s.imageFeatures
                    && theta == s.theta;
        }

        @Override
        public int hashCode() {
            return Objects.hash(inChannels, outChannels, Arrays.hashCode(patch), heads, headSize, layers, ffnSize,
                    textFeatures, frequencies, epsilon, crossAttentionNorm, imageFeatures, theta);
        }

        @Override
        public String toString() {
            return "Settings[inChannels=" + inChannels + ", outChannels=" + outChannels + ", patch="
                    + Arrays.toString(patch) + ", heads=" + heads + ", headSize=" + headSize + ", layers=" + layers
                    + ", ffnSize=" + ffnSize + ", textFeatures=" + textFeatures + ", frequencies=" + frequencies
                    + ", epsilon=" + epsilon + ", crossAttentionNorm=" + crossAttentionNorm + ", imageFeatures="
                    + imageFeatures + ", theta=" + theta + "]";
        }
    }

    private final Settings settings;
    private final Dense patchIn;
    private final TimestepEmbedding time;
    private final Dense timeProjection;
    private final Dense text1;
    private final Dense text2;
    private final LayerNorm imageNorm1;
    private final Dense image1;
    private final Dense image2;
    private final LayerNorm imageNorm2;
    private final List<Block> blocks = new ArrayList<>();
    private final Var outTable;
    private final LayerNorm outNorm;
    private final Dense out;

    /**
     * Creates a transformer with fresh weights.
     *
     * @param settings its shape
     * @param random the source of initial weights
     */
    public WanTransformer(Settings settings, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        int d = settings.features();
        patchIn = Parts.dense(settings.patchIn(), d, Activation.IDENTITY, random);
        time = new TimestepEmbedding(new TimestepEmbedding.Settings(settings.frequencies(), d, true, 0), random);
        timeProjection = Parts.dense(d, 6 * d, Activation.IDENTITY, random);
        text1 = Parts.dense(settings.textFeatures(), d, Activation.GELU_TANH, random);
        text2 = Parts.dense(d, d, Activation.IDENTITY, random);
        if (settings.imageFeatures() > 0) {
            int c = settings.imageFeatures();
            imageNorm1 = affineNorm(c, 1e-5f);
            image1 = Parts.dense(c, c, Activation.GELU, random);
            image2 = Parts.dense(c, d, Activation.IDENTITY, random);
            imageNorm2 = affineNorm(d, 1e-5f);
        } else {
            imageNorm1 = null;
            image1 = null;
            image2 = null;
            imageNorm2 = null;
        }
        for (int i = 0; i < settings.layers(); i++) {
            blocks.add(new Block(settings, random));
        }
        outTable = Var.param(Tensor.normal(Shape.of(2, d), 0, (float) (1 / Math.sqrt(d)), random));
        outNorm = plainNorm(d, settings.epsilon());
        out = Parts.dense(d, settings.patchOut(), Activation.IDENTITY, random);
    }

    private static LayerNorm plainNorm(int d, float eps) {
        return new LayerNorm(new int[] {d}, eps, false, false, false);
    }

    private static LayerNorm affineNorm(int d, float eps) {
        return new LayerNorm(new int[] {d}, eps, true, true, false);
    }

    /**
     * Predicts the flow velocity.
     *
     * @param latents {@code [batch, inChannels, frames, height, width]}, each side divisible
     *     by the patch
     * @param timesteps the noise levels times 1000: {@code [batch]}, or {@code [batch, patches]}
     *     for a level per patch; no gradient flows to them
     * @param context the text states {@code [batch, tokens, textFeatures]} (Wan pads them
     *     with zeros to 512 tokens)
     * @param image image tokens {@code [batch, imageTokens, imageFeatures]} for image-to-video
     *     models, otherwise {@code null}
     * @param training whether the model is training
     * @return the velocity {@code [batch, outChannels, frames, height, width]}
     */
    public Var forward(Var latents, Tensor timesteps, Var context, Var image, boolean training) {
        Shape s = latents.shape();
        int[] p = settings.patch;
        if (s.rank() != 5 || s.dim(1) != settings.inChannels() || s.dim(2) % p[0] != 0 || s.dim(3) % p[1] != 0
                || s.dim(4) % p[2] != 0) {
            throw new IllegalArgumentException(this + " expects latents [batch, " + settings.inChannels()
                    + ", frames, h, w] with sides divisible by " + Arrays.toString(p) + ", got " + s);
        }
        if ((settings.imageFeatures() > 0) != (image != null)) {
            throw new IllegalArgumentException(settings.imageFeatures() > 0 ? this + " needs image tokens"
                    : this + " takes no image tokens");
        }
        int batch = s.dim(0);
        int f = s.dim(2) / p[0];
        int h = s.dim(3) / p[1];
        int w = s.dim(4) / p[2];
        int n = f * h * w;
        int d = settings.features();
        DType type = dtype();
        Device device = latents.value().device();
        boolean perPatch = timesteps.shape().rank() == 2;
        if (timesteps.shape().dim(0) != batch || (perPatch && timesteps.shape().dim(1) != n)) {
            throw new IllegalArgumentException(this + " takes timesteps [" + batch + "] or [" + batch + ", " + n
                    + "], got " + timesteps.shape());
        }
        Var x = Parts.step(() -> patchIn.forward(patchify(latents, p).to(type)));
        // the time embedding (temb) and its six modulations per block, [batch, 1 or n, ...]
        int m = perPatch ? n : 1;
        Var temb = Parts.step(() -> time.forward(Var.of(timesteps.reshape(batch * m))).reshape(batch, m, d));
        Var modulation = Parts.step(() -> timeProjection.forward(Parts.silu(temb)).to(DType.FLOAT32)
                .reshape(batch, m, 6, d));
        Var ctx = Parts.step(() -> {
            Var t = text2.forward(text1.forward(context.to(type)));
            if (image == null) {
                return t;
            }
            Var i = imageNorm2.forward(image2.forward(image1.forward(imageNorm1.forward(image.to(type)))));
            return Var.concat(1, i, t);
        });
        int imageTokens = image == null ? 0 : image.shape().dim(1);
        Tensor[] rope = FluxTransformer.rotaryTables(positions(f, h, w), settings.axes(), settings.theta(), device);
        for (Block b : blocks) {
            Var in = x;
            x = Parts.step(() -> b.forward(in, modulation, ctx, imageTokens, rope));
            Parts.free(in);
        }
        Var hidden = x;
        Var result = Parts.step(() -> {
            // shift, scale from the table plus the time embedding itself
            Var mod = outTable.to(DType.FLOAT32).reshape(1, 1, 2, d).add(temb.to(DType.FLOAT32).reshape(batch, m, 1, d));
            Var shift = mod.slice(2, 0, 1).reshape(batch, m, d);
            Var scale = mod.slice(2, 1, 2).reshape(batch, m, d);
            Var y = outNorm.forward(hidden.to(DType.FLOAT32)).mul(scale.add(1f)).add(shift).to(type);
            return unpatchify(out.forward(y), settings.outChannels(), f, h, w, p);
        });
        Parts.free(hidden);
        Parts.free(temb);
        Parts.free(modulation);
        Parts.free(ctx);
        rope[0].release();
        rope[1].release();
        return result;
    }

    /**
     * Inputs {@code [latents, timesteps, context]}, plus {@code image} for image-to-video
     * models, as {@link #forward(Var, Tensor, Var, Var, boolean)}.
     */
    @Override
    public Var forward(List<Var> inputs, boolean training) {
        int count = settings.imageFeatures() > 0 ? 4 : 3;
        if (inputs.size() != count) {
            throw new IllegalArgumentException(this + " takes " + count + " inputs, got " + inputs.size());
        }
        return forward(inputs.get(0), inputs.get(1).value(), inputs.get(2), count == 4 ? inputs.get(3) : null,
                training);
    }

    /**
     * This model as a {@link Denoiser} for one prompt: the noise level in {@code [0, 1]} is the
     * time, scaled to Wan's timesteps.
     *
     * @param context the text states {@code [1, tokens, textFeatures]} (or one per example)
     * @param image image tokens for image-to-video models, otherwise {@code null}
     * @return the denoiser; it runs with gradients if they are enabled
     */
    public Denoiser denoiser(Tensor context, Tensor image) {
        return (noisy, sigmas) -> {
            int b = noisy.shape().dim(0);
            Tensor c = context.shape().dim(0) == b ? context
                    : context.expand(Shape.of(b, context.shape().dim(1), context.shape().dim(2)));
            Tensor i = image == null || image.shape().dim(0) == b ? image
                    : image.expand(Shape.of(b, image.shape().dim(1), image.shape().dim(2)));
            return forward(noisy, sigmas.to(DType.FLOAT32).mul(1000f), Var.of(c), i == null ? null : Var.of(i),
                    false);
        };
    }

    /**
     * Cuts {@code [b, c, frames, h, w]} into patches {@code [b, patches, c * pt * ph * pw]},
     * patches frame-major, each laid out channel, frame, row, column (as a strided 3-d
     * convolution's weight).
     */
    static Var patchify(Var x, int[] p) {
        Shape s = x.shape();
        int b = s.dim(0);
        int c = s.dim(1);
        int f = s.dim(2) / p[0];
        int h = s.dim(3) / p[1];
        int w = s.dim(4) / p[2];
        if (p[0] == 1) {
            // one frame per patch: a rank-7 view is enough
            return x.reshape(b, c, f, h, p[1], w, p[2]).permute(0, 2, 3, 5, 1, 4, 6)
                    .reshape(b, f * h * w, c * p[1] * p[2]);
        }
        return x.reshape(b, c, f, p[0], h, p[1], w, p[2]).permute(0, 2, 4, 6, 1, 3, 5, 7)
                .reshape(b, f * h * w, c * p[0] * p[1] * p[2]);
    }

    /** The inverse of the output projection's layout: patches laid out frame, row, column, channel. */
    static Var unpatchify(Var tokens, int c, int f, int h, int w, int[] p) {
        int b = tokens.shape().dim(0);
        if (p[0] == 1) {
            return tokens.reshape(b, f, h, w, p[1], p[2], c).permute(0, 6, 1, 2, 4, 3, 5)
                    .reshape(b, c, f, h * p[1], w * p[2]);
        }
        return tokens.reshape(b, f, h, w, p[0], p[1], p[2], c).permute(0, 7, 1, 4, 2, 5, 3, 6)
                .reshape(b, c, f * p[0], h * p[1], w * p[2]);
    }

    /** Position ids {@code [f * h * w][3]}: (frame, row, column) of each patch, frame-major. */
    static int[][] positions(int f, int h, int w) {
        int[][] ids = new int[f * h * w][3];
        int i = 0;
        for (int t = 0; t < f; t++) {
            for (int r = 0; r < h; r++) {
                for (int c = 0; c < w; c++) {
                    ids[i][0] = t;
                    ids[i][1] = r;
                    ids[i][2] = c;
                    i++;
                }
            }
        }
        return ids;
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        Shape s = inputs.get(0);
        return Shape.of(s.dim(0), settings.outChannels(), s.dim(2), s.dim(3), s.dim(4));
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), settings.outChannels(), input.dim(2), input.dim(3), input.dim(4));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Layer l : List.of(patchIn, time, timeProjection, text1, text2)) {
            p.addAll(l.parameters());
        }
        if (image1 != null) {
            for (Layer l : List.of(imageNorm1, image1, image2, imageNorm2)) {
                p.addAll(l.parameters());
            }
        }
        for (Block b : blocks) {
            p.addAll(b.parameters());
        }
        p.add(outTable);
        p.addAll(out.parameters());
        return p;
    }

    /**
     * The type the model computes in: that of its unquantized weights (biases, norms), since
     * quantized projections report the float32 they were packed from.
     *
     * @return the type
     */
    public DType dtype() {
        for (Var p : parameters()) {
            if (!p.value().isPacked()) {
                return p.value().dtype();
            }
        }
        return DType.FLOAT32;
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    /** @return the patch embedding, a strided 3-d convolution as a projection of patches */
    public Dense patchIn() {
        return patchIn;
    }

    /** @return the time embedding */
    public TimestepEmbedding timeEmbedding() {
        return time;
    }

    /** @return the projection of the SiLU'd time embedding to six modulations of each block */
    public Dense timeProjection() {
        return timeProjection;
    }

    /** @return the first text projection (tanh GELU) */
    public Dense text1() {
        return text1;
    }

    /** @return the second text projection */
    public Dense text2() {
        return text2;
    }

    /** @return the norm before the image-token projection, or {@code null} */
    public LayerNorm imageNorm1() {
        return imageNorm1;
    }

    /** @return the first image-token projection (GELU), or {@code null} */
    public Dense image1() {
        return image1;
    }

    /** @return the second image-token projection, or {@code null} */
    public Dense image2() {
        return image2;
    }

    /** @return the norm after the image-token projections, or {@code null} */
    public LayerNorm imageNorm2() {
        return imageNorm2;
    }

    /** @return the blocks */
    public List<Block> blocks() {
        return List.copyOf(blocks);
    }

    /** @return the output modulation table {@code [2, features]}: shift, scale */
    public Var outTable() {
        return outTable;
    }

    /** @return the projection back to patches, laid out frame, row, column, channel */
    public Dense out() {
        return out;
    }

    @Override
    public String toString() {
        return "WanTransformer(" + settings.features() + ", " + settings.layers() + " blocks)";
    }

    /**
     * The most query-key pairs (times heads) one attention dispatch takes in inference
     * ({@code -Dbrainiac.wan.attentionWork}); longer videos attend a slice of queries at a
     * time, each run on its own. Windows resets a GPU whose work runs for more than about
     * two seconds, and one fused attention over the 14,000 patches of an 832x480 clip can
     * take that long on a consumer card.
     */
    static final long ATTENTION_WORK = Long.getLong("brainiac.wan.attentionWork", 1L << 28);

    /** Attention of {@code [batch, heads, tokens, size]}, in slices of queries above {@code work} pairs. */
    static Var attend(Var q, Var k, Var v, long work) {
        int heads = q.shape().dim(1);
        int n = q.shape().dim(2);
        int m = k.shape().dim(2);
        if (Var.gradEnabled() || (long) heads * n * m <= work) {
            return FluxTransformer.attend(q, k, v);
        }
        int chunk = (int) Math.max(16, Math.min(n, work / ((long) heads * m)));
        List<Var> pieces = new ArrayList<>();
        for (int i = 0; i < n; i += chunk) {
            Var qi = q.slice(2, i, Math.min(n, i + chunk));
            pieces.add(Parts.step(() -> FluxTransformer.attend(qi, k, v)));
            Parts.free(qi);
        }
        return pieces.size() == 1 ? pieces.get(0) : Var.concat(2, pieces.toArray(Var[]::new));
    }

    /** An attention's projections and its RMS norms of queries and keys over all heads. */
    public static final class Attention {
        private final Dense q;
        private final Dense k;
        private final Dense v;
        private final Dense o;
        private final LayerNorm qNorm;
        private final LayerNorm kNorm;
        private final Dense imageK;
        private final Dense imageV;
        private final LayerNorm imageKNorm;

        Attention(Settings s, boolean image, SplittableRandom random) {
            int d = s.features();
            q = Parts.dense(d, d, Activation.IDENTITY, random);
            k = Parts.dense(d, d, Activation.IDENTITY, random);
            v = Parts.dense(d, d, Activation.IDENTITY, random);
            o = Parts.dense(d, d, Activation.IDENTITY, random);
            qNorm = new LayerNorm(new int[] {d}, s.epsilon(), true, false, true);
            kNorm = new LayerNorm(new int[] {d}, s.epsilon(), true, false, true);
            if (image) {
                imageK = Parts.dense(d, d, Activation.IDENTITY, random);
                imageV = Parts.dense(d, d, Activation.IDENTITY, random);
                imageKNorm = new LayerNorm(new int[] {d}, s.epsilon(), true, false, true);
            } else {
                imageK = null;
                imageV = null;
                imageKNorm = null;
            }
        }

        List<Layer> layers() {
            List<Layer> l = new ArrayList<>(List.of(q, k, v, o, qNorm, kNorm));
            if (imageK != null) {
                l.addAll(List.of(imageK, imageV, imageKNorm));
            }
            return l;
        }

        /** @return the query projection */
        public Dense q() {
            return q;
        }

        /** @return the key projection */
        public Dense k() {
            return k;
        }

        /** @return the value projection */
        public Dense v() {
            return v;
        }

        /** @return the output projection */
        public Dense o() {
            return o;
        }

        /** @return the RMS norm of the queries, over all heads */
        public LayerNorm qNorm() {
            return qNorm;
        }

        /** @return the RMS norm of the keys, over all heads */
        public LayerNorm kNorm() {
            return kNorm;
        }

        /** @return the key projection of image tokens, or {@code null} */
        public Dense imageK() {
            return imageK;
        }

        /** @return the value projection of image tokens, or {@code null} */
        public Dense imageV() {
            return imageV;
        }

        /** @return the RMS norm of the image keys, or {@code null} */
        public LayerNorm imageKNorm() {
            return imageKNorm;
        }
    }

    /**
     * A block: modulated self-attention with rotary positions, cross-attention to the
     * context, and a modulated feed-forward.
     */
    public static final class Block implements Layer {

        private final int heads;
        private final LayerNorm norm1;
        private final Attention self;
        private final LayerNorm crossNorm;
        private final Attention cross;
        private final LayerNorm norm2;
        private final Dense up;
        private final Dense down;
        private final Var table;

        Block(Settings s, SplittableRandom random) {
            int d = s.features();
            heads = s.heads();
            norm1 = plainNorm(d, s.epsilon());
            self = new Attention(s, false, random);
            crossNorm = s.crossAttentionNorm() ? affineNorm(d, s.epsilon()) : null;
            cross = new Attention(s, s.imageFeatures() > 0, random);
            norm2 = plainNorm(d, s.epsilon());
            up = Parts.dense(d, s.ffnSize(), Activation.GELU_TANH, random);
            down = Parts.dense(s.ffnSize(), d, Activation.IDENTITY, random);
            table = Var.param(Tensor.normal(Shape.of(6, d), 0, (float) (1 / Math.sqrt(d)), random));
        }

        /**
         * Runs the block.
         *
         * @param x {@code [batch, patches, features]}
         * @param modulation the projected time embedding {@code [batch, 1 or patches, 6, features]}
         *     in float32
         * @param context the projected context {@code [batch, tokens, features]}, image tokens first
         * @param imageTokens how many context tokens are image tokens
         * @param rope the rotary tables of the patches
         * @return the new sequence
         */
        public Var forward(Var x, Var modulation, Var context, int imageTokens, Tensor[] rope) {
            Shape ms = modulation.shape();
            int b = ms.dim(0);
            int m = ms.dim(1);
            int d = ms.dim(3);
            DType type = x.value().dtype();
            Var mod = modulation.add(table.to(DType.FLOAT32).reshape(1, 1, 6, d));
            Var[] c = new Var[6];
            for (int i = 0; i < 6; i++) {
                c[i] = mod.slice(2, i, i + 1).reshape(b, m, d);
            }
            // self-attention: shift, scale, gate
            Var n1 = norm1.forward(x.to(DType.FLOAT32)).mul(c[1].add(1f)).add(c[0]).to(type);
            Var q = FluxTransformer.rotate(split(self.qNorm.forward(self.q.forward(n1)), heads), rope);
            Var k = FluxTransformer.rotate(split(self.kNorm.forward(self.k.forward(n1)), heads), rope);
            Var v = split(self.v.forward(n1), heads);
            Var a = self.o.forward(FluxTransformer.join(attend(q, k, v, ATTENTION_WORK)));
            x = x.to(DType.FLOAT32).add(a.to(DType.FLOAT32).mul(c[2])).to(type);
            // cross-attention to text (and image) tokens, unmodulated
            Var n2 = crossNorm == null ? x : crossNorm.forward(x);
            Var cq = split(cross.qNorm.forward(cross.q.forward(n2)), heads);
            int tokens = context.shape().dim(1);
            Var text = imageTokens == 0 ? context : context.slice(1, imageTokens, tokens);
            Var ca = FluxTransformer.attend(cq, split(cross.kNorm.forward(cross.k.forward(text)), heads),
                    split(cross.v.forward(text), heads));
            if (imageTokens > 0) {
                Var img = context.slice(1, 0, imageTokens);
                ca = ca.add(FluxTransformer.attend(cq,
                        split(cross.imageKNorm.forward(cross.imageK.forward(img)), heads),
                        split(cross.imageV.forward(img), heads)));
            }
            x = x.add(cross.o.forward(FluxTransformer.join(ca)));
            // feed-forward: shift, scale, gate
            Var n3 = norm2.forward(x.to(DType.FLOAT32)).mul(c[4].add(1f)).add(c[3]).to(type);
            Var f = down.forward(up.forward(n3));
            return x.to(DType.FLOAT32).add(f.to(DType.FLOAT32).mul(c[5])).to(type);
        }

        /** {@code [batch, tokens, features]} as heads {@code [batch, heads, tokens, headSize]}. */
        private static Var split(Var x, int heads) {
            Shape s = x.shape();
            return x.reshape(s.dim(0), s.dim(1), heads, s.dim(2) / heads).permute(0, 2, 1, 3);
        }

        @Override
        public Var forward(Var input) {
            throw new UnsupportedOperationException("a Wan block runs inside a WanTransformer");
        }

        /** Self-attention, cross-attention norm, cross-attention, feed-forward, then the modulation table. */
        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (Layer l : self.layers()) {
                p.addAll(l.parameters());
            }
            if (crossNorm != null) {
                p.addAll(crossNorm.parameters());
            }
            for (Layer l : cross.layers()) {
                p.addAll(l.parameters());
            }
            p.addAll(up.parameters());
            p.addAll(down.parameters());
            p.add(table);
            return p;
        }

        /** @return the self-attention */
        public Attention selfAttention() {
            return self;
        }

        /** @return the layer norm (with scale and shift) before cross-attention, or {@code null} */
        public LayerNorm crossNorm() {
            return crossNorm;
        }

        /** @return the cross-attention */
        public Attention crossAttention() {
            return cross;
        }

        /** @return the feed-forward's first layer (tanh GELU) */
        public Dense up() {
            return up;
        }

        /** @return the feed-forward's second layer */
        public Dense down() {
            return down;
        }

        /** @return the modulation table {@code [6, features]}: shift, scale, gate for attention, then the feed-forward */
        public Var table() {
            return table;
        }
    }
}
