package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
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
 * A multimodal diffusion transformer (MMDiT) as Flux uses: image patches and text tokens
 * attend to each other in joint blocks that keep separate weights per stream, then in
 * single-stream blocks over the joined sequence, every block modulated (adaLN-Zero) by the
 * noise level, a pooled text embedding and, for guidance-distilled models, the guidance
 * scale. Positions enter as axial rotary embeddings, so any image size works. It predicts
 * the flow velocity {@code noise - data} (see {@link FlowSampler}).
 *
 * <p>The layer takes latents {@code [batch, channels, height, width]} and returns a tensor
 * of the same shape; it cuts them into {@code patch x patch} patches itself.
 *
 * <pre>{@code
 * FluxTransformer dit = new FluxTransformer(FluxTransformer.Settings.of(4, 4, 32, 2, 2, 64, 32, false), random);
 * Var v = dit.forward(Var.of(latents), sigmas, Var.of(text), Var.of(pooled), null, true);
 * }</pre>
 */
public final class FluxTransformer implements MultiInputLayer {

    /**
     * The shape of the transformer.
     *
     * @param latentChannels the channels of the latents
     * @param patch the patch side (Flux: 2)
     * @param heads the attention heads
     * @param headSize the features per head
     * @param jointLayers the joint (double-stream) blocks
     * @param singleLayers the single-stream blocks
     * @param contextFeatures the width of the text tokens
     * @param pooledFeatures the width of the pooled text embedding
     * @param guidance whether the model embeds a guidance scale (Flux dev, not schnell)
     * @param axes the head features each position axis rotates (text index, row, column);
     *     they sum to {@code headSize}, each even
     * @param theta the rotary frequency base
     * @param mlpRatio the feed-forward width over the model width
     */
    public record Settings(int latentChannels, int patch, int heads, int headSize, int jointLayers, int singleLayers,
            int contextFeatures, int pooledFeatures, boolean guidance, int[] axes, float theta, float mlpRatio) {

        /** Checks the values. */
        public Settings {
            axes = axes.clone();
            int sum = 0;
            for (int a : axes) {
                if (a < 2 || a % 2 != 0) {
                    throw new IllegalArgumentException("rotary axes must be even, got " + Arrays.toString(axes));
                }
                sum += a;
            }
            if (latentChannels < 1 || patch < 1 || heads < 1 || headSize < 2 || jointLayers < 0 || singleLayers < 0
                    || contextFeatures < 1 || pooledFeatures < 1 || sum != headSize || !(mlpRatio > 0)) {
                throw new IllegalArgumentException("invalid Flux transformer shape (rotary axes " + Arrays.toString(axes)
                        + " must sum to the head size " + headSize + ")");
            }
        }

        /**
         * Settings with Flux's patch of 2, rotary base and feed-forward ratio, and its split
         * of the head features into text index, row and column (1/8, 7/16, 7/16, rounded to
         * even sizes).
         *
         * @param latentChannels the channels of the latents
         * @param heads the attention heads
         * @param headSize the features per head, even, at least 8
         * @param jointLayers the joint blocks
         * @param singleLayers the single-stream blocks
         * @param contextFeatures the width of the text tokens
         * @param pooledFeatures the width of the pooled text embedding
         * @param guidance whether a guidance scale is embedded
         * @return the settings
         */
        public static Settings of(int latentChannels, int heads, int headSize, int jointLayers, int singleLayers,
                int contextFeatures, int pooledFeatures, boolean guidance) {
            int first = Math.max(2, headSize / 16 * 2);
            int rest = (headSize - first) / 4 * 2;
            return new Settings(latentChannels, 2, heads, headSize, jointLayers, singleLayers, contextFeatures,
                    pooledFeatures, guidance, new int[] {first, rest, headSize - first - rest}, 10000, 4);
        }

        /** @return the model width, {@code heads * headSize} */
        public int features() {
            return heads * headSize;
        }

        /** @return the width of the patches, {@code latentChannels * patch * patch} */
        public int patchFeatures() {
            return latentChannels * patch * patch;
        }

        @Override
        public int[] axes() {
            return axes.clone();
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Settings s && latentChannels == s.latentChannels && patch == s.patch
                    && heads == s.heads && headSize == s.headSize && jointLayers == s.jointLayers
                    && singleLayers == s.singleLayers && contextFeatures == s.contextFeatures
                    && pooledFeatures == s.pooledFeatures && guidance == s.guidance && Arrays.equals(axes, s.axes)
                    && theta == s.theta && mlpRatio == s.mlpRatio;
        }

        @Override
        public int hashCode() {
            return Objects.hash(latentChannels, patch, heads, headSize, jointLayers, singleLayers, contextFeatures,
                    pooledFeatures, guidance, Arrays.hashCode(axes), theta, mlpRatio);
        }

        @Override
        public String toString() {
            return "Settings[latentChannels=" + latentChannels + ", patch=" + patch + ", heads=" + heads
                    + ", headSize=" + headSize + ", jointLayers=" + jointLayers + ", singleLayers=" + singleLayers
                    + ", contextFeatures=" + contextFeatures + ", pooledFeatures=" + pooledFeatures + ", guidance="
                    + guidance + ", axes=" + Arrays.toString(axes) + ", theta=" + theta + ", mlpRatio=" + mlpRatio
                    + "]";
        }
    }

    /** The frequencies of the time and guidance embeddings. */
    private static final int FREQUENCIES = 256;

    private final Settings settings;
    private final Dense imageIn;
    private final Dense contextIn;
    private final TimestepEmbedding time;
    private final TimestepEmbedding guidanceEmbedding;
    private final Dense pooled1;
    private final Dense pooled2;
    private final List<JointBlock> joint = new ArrayList<>();
    private final List<SingleBlock> single = new ArrayList<>();
    private final Dense outModulation;
    private final LayerNorm outNorm;
    private final Dense out;

    /**
     * Creates a transformer with fresh weights.
     *
     * @param settings its shape
     * @param random the source of initial weights
     */
    public FluxTransformer(Settings settings, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        int d = settings.features();
        imageIn = Parts.dense(settings.patchFeatures(), d, Activation.IDENTITY, random);
        contextIn = Parts.dense(settings.contextFeatures(), d, Activation.IDENTITY, random);
        time = new TimestepEmbedding(TimestepEmbedding.Settings.of(FREQUENCIES, d), random);
        guidanceEmbedding = settings.guidance()
                ? new TimestepEmbedding(TimestepEmbedding.Settings.of(FREQUENCIES, d), random) : null;
        pooled1 = Parts.dense(settings.pooledFeatures(), d, Activation.SILU, random);
        pooled2 = Parts.dense(d, d, Activation.IDENTITY, random);
        for (int i = 0; i < settings.jointLayers(); i++) {
            joint.add(new JointBlock(settings, random));
        }
        for (int i = 0; i < settings.singleLayers(); i++) {
            single.add(new SingleBlock(settings, random));
        }
        outModulation = zeroModulation(d, 2);
        outNorm = plainNorm(d);
        out = Parts.dense(d, settings.patchFeatures(), Activation.IDENTITY, random);
    }

    /** A projection of the conditioning to {@code count} vectors, starting at zero (adaLN-Zero). */
    private static Dense zeroModulation(int d, int count) {
        return new Dense(d, count * d, Activation.IDENTITY, Tensor.zeros(Shape.of(d, count * d)),
                Tensor.zeros(Shape.of(count * d)));
    }

    private static LayerNorm plainNorm(int d) {
        return new LayerNorm(new int[] {d}, 1e-6f, false, false, false);
    }

    /**
     * Predicts the flow velocity.
     *
     * @param latents {@code [batch, channels, height, width]}, sides divisible by the patch
     * @param sigmas the noise levels in {@code [0, 1]}, {@code [batch]}; no gradient flows to them
     * @param context the text tokens {@code [batch, tokens, contextFeatures]}
     * @param pooled the pooled text embedding {@code [batch, pooledFeatures]}
     * @param guidance the guidance scale per example {@code [batch]}, for guidance-distilled
     *     models; {@code null} otherwise
     * @param training whether the model is training
     * @return the velocity, shaped like {@code latents}
     */
    public Var forward(Var latents, Tensor sigmas, Var context, Var pooled, Tensor guidance, boolean training) {
        Shape s = latents.shape();
        int p = settings.patch();
        if (s.rank() != 4 || s.dim(1) != settings.latentChannels() || s.dim(2) % p != 0 || s.dim(3) % p != 0) {
            throw new IllegalArgumentException(this + " expects latents [batch, " + settings.latentChannels()
                    + ", h, w] with sides divisible by " + p + ", got " + s);
        }
        if (settings.guidance() != (guidance != null)) {
            throw new IllegalArgumentException(settings.guidance() ? this + " needs a guidance scale"
                    : this + " takes no guidance scale");
        }
        int rows = s.dim(2) / p;
        int cols = s.dim(3) / p;
        int textTokens = context.shape().dim(1);
        DType type = imageIn.weight().value().dtype();
        Device device = latents.value().device();
        Var image = Parts.step(() -> imageIn.forward(Functional.patchify(latents, p).to(type)));
        Var text = Parts.step(() -> contextIn.forward(context.to(type)));
        Var temb = Parts.step(() -> {
            // as diffusers: scaled to timesteps in the model's type
            Var t = time.forward(Var.of(sigmas.to(type).mul(1000f)));
            if (guidanceEmbedding != null) {
                t = t.add(guidanceEmbedding.forward(Var.of(guidance.to(type).mul(1000f))));
            }
            return Parts.silu(t.add(pooled2.forward(pooled1.forward(pooled.to(type)))));
        });
        Tensor[] rope = rotaryTables(positions(textTokens, rows, cols), settings.axes(), settings.theta(), device);
        Var x = Var.concat(1, text, image);
        Parts.free(text);
        Parts.free(image);
        for (JointBlock b : joint) {
            Var in = x;
            x = Parts.step(() -> b.forward(in, textTokens, temb, rope));
            Parts.free(in);
        }
        for (SingleBlock b : single) {
            Var in = x;
            x = Parts.step(() -> b.forward(in, temb, rope));
            Parts.free(in);
        }
        Var h = x;
        Var result = Parts.step(() -> {
            Var img = h.slice(1, textTokens, textTokens + rows * cols);
            Var[] m = chunks(outModulation.forward(temb), 2);
            // AdaLayerNormContinuous: scale first, then shift
            Var y = out.forward(Functional.film(outNorm.forward(img), m[0], m[1]));
            return Functional.unpatchify(y, settings.latentChannels(), s.dim(2), s.dim(3), p);
        });
        Parts.free(h);
        Parts.free(temb);
        rope[0].release();
        rope[1].release();
        return result;
    }

    /**
     * Inputs {@code [latents, sigmas, context, pooled]}, plus {@code guidance} for
     * guidance-distilled models, as {@link #forward(Var, Tensor, Var, Var, Tensor, boolean)}.
     */
    @Override
    public Var forward(List<Var> inputs, boolean training) {
        int n = settings.guidance() ? 5 : 4;
        if (inputs.size() != n) {
            throw new IllegalArgumentException(this + " takes " + n + " inputs, got " + inputs.size());
        }
        return forward(inputs.get(0), inputs.get(1).value(), inputs.get(2), inputs.get(3),
                settings.guidance() ? inputs.get(4).value() : null, training);
    }

    /**
     * This model as a {@link Denoiser} for one prompt: the noise level is the time.
     *
     * @param context the text tokens {@code [1, tokens, contextFeatures]} (or one per example)
     * @param pooled the pooled text embedding {@code [1, pooledFeatures]} (or one per example)
     * @param guidance the guidance scale, ignored unless the model embeds one
     * @return the denoiser; it runs with gradients if they are enabled
     */
    public Denoiser denoiser(Tensor context, Tensor pooled, float guidance) {
        return (noisy, sigmas) -> {
            int b = noisy.shape().dim(0);
            Tensor c = context.shape().dim(0) == b ? context
                    : context.expand(Shape.of(b, context.shape().dim(1), context.shape().dim(2)));
            Tensor p = pooled.shape().dim(0) == b ? pooled : pooled.expand(Shape.of(b, pooled.shape().dim(1)));
            Tensor g = null;
            if (settings.guidance()) {
                float[] gs = new float[b];
                Arrays.fill(gs, guidance);
                g = Tensor.of(Shape.of(b), gs, sigmas.device());
            }
            return forward(noisy, sigmas, Var.of(c), Var.of(p), g, false);
        };
    }

    /** Position ids {@code [text + rows * cols][3]}: text tokens at 0, patches at (0, row, column). */
    static int[][] positions(int text, int rows, int cols) {
        int[][] ids = new int[text + rows * cols][3];
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                ids[text + r * cols + c][1] = r;
                ids[text + r * cols + c][2] = c;
            }
        }
        return ids;
    }

    /**
     * The cosines and sines of axial rotary embeddings, each {@code [tokens, sum of axes]}:
     * axis {@code a} of each token's position rotates its own run of {@code axes[a]}
     * features, in adjacent pairs, at frequencies {@code theta^(-2j / axes[a])}.
     *
     * @param positions each token's position, one value per axis
     * @param axes the features per axis, each even
     * @param theta the frequency base
     * @param device where to put the tables
     * @return {@code {cos, sin}} in float32, each value repeated for both features of its pair
     */
    public static Tensor[] rotaryTables(int[][] positions, int[] axes, double theta, Device device) {
        int n = positions.length;
        int d = 0;
        for (int a : axes) {
            d += a;
        }
        float[] cos = new float[n * d];
        float[] sin = new float[n * d];
        for (int t = 0; t < n; t++) {
            int at = 0;
            for (int a = 0; a < axes.length; a++) {
                for (int j = 0; j < axes[a] / 2; j++) {
                    double freq = 1.0 / Math.pow(theta, (2.0 * j) / axes[a]);
                    double angle = positions[t][a] * freq;
                    float c = (float) Math.cos(angle);
                    float sn = (float) Math.sin(angle);
                    cos[t * d + at + 2 * j] = c;
                    cos[t * d + at + 2 * j + 1] = c;
                    sin[t * d + at + 2 * j] = sn;
                    sin[t * d + at + 2 * j + 1] = sn;
                }
                at += axes[a];
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, d), cos, device), Tensor.of(Shape.of(n, d), sin, device)};
    }

    /**
     * Rotates adjacent feature pairs {@code (2j, 2j + 1)} by the angles of {@link #rotaryTables},
     * in float32.
     *
     * @param x {@code [..., tokens, features]}
     * @param rope {@code {cos, sin}}, each {@code [tokens, features]}
     * @return the rotated values, in the type of {@code x}
     */
    public static Var rotate(Var x, Tensor[] rope) {
        Shape s = x.shape();
        int d = s.dim(-1);
        int[] pairs = Arrays.copyOf(s.toArray(), s.rank() + 1);
        pairs[s.rank() - 1] = d / 2;
        pairs[s.rank()] = 2;
        Var f = x.to(DType.FLOAT32);
        Var p = f.reshape(pairs);
        Var turned = Var.concat(-1, p.slice(-1, 1, 2).neg(), p.slice(-1, 0, 1)).reshape(s.toArray());
        return f.mul(rope[0]).add(turned.mul(rope[1])).to(x.value().dtype());
    }

    /** Splits {@code [batch, count * d]} into {@code count} parts {@code [batch, d]}. */
    static Var[] chunks(Var v, int count) {
        int d = v.shape().dim(1) / count;
        Var[] parts = new Var[count];
        for (int i = 0; i < count; i++) {
            parts[i] = v.slice(1, i * d, (i + 1) * d);
        }
        return parts;
    }

    /** Projects to heads {@code [batch, heads, tokens, headSize]}, RMS-normed unless {@code norm} is null. */
    static Var heads(Dense projection, LayerNorm norm, Var x, int heads) {
        Shape s = x.shape();
        Var y = projection.forward(x);
        Var h = y.reshape(s.dim(0), s.dim(1), heads, y.shape().dim(2) / heads).permute(0, 2, 1, 3);
        return norm == null ? h : norm.forward(h);
    }

    /** Joins heads {@code [batch, heads, tokens, headSize]} back into {@code [batch, tokens, features]}. */
    static Var join(Var h) {
        Shape s = h.shape();
        return h.permute(0, 2, 1, 3).reshape(s.dim(0), s.dim(2), s.dim(1) * s.dim(3));
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.get(0);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Layer l : layers()) {
            p.addAll(l.parameters());
        }
        return p;
    }

    /**
     * The sublayers in parameter order: image and text input projections, time embedding,
     * guidance embedding (if any), the two pooled-text projections, the joint blocks, the
     * single blocks, the output modulation and the output projection.
     *
     * @return the sublayers
     */
    public List<Layer> layers() {
        List<Layer> l = new ArrayList<>(List.of(imageIn, contextIn, time));
        if (guidanceEmbedding != null) {
            l.add(guidanceEmbedding);
        }
        l.addAll(List.of(pooled1, pooled2));
        l.addAll(joint);
        l.addAll(single);
        l.addAll(List.of(outModulation, out));
        return l;
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
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

    /** @return the projection of the patches into the model width */
    public Dense imageIn() {
        return imageIn;
    }

    /** @return the projection of the text tokens into the model width */
    public Dense contextIn() {
        return contextIn;
    }

    /** @return the noise-level embedding */
    public TimestepEmbedding timeEmbedding() {
        return time;
    }

    /** @return the guidance embedding, or {@code null} */
    public TimestepEmbedding guidanceEmbedding() {
        return guidanceEmbedding;
    }

    /** @return the first projection of the pooled text embedding (with SiLU) */
    public Dense pooled1() {
        return pooled1;
    }

    /** @return the second projection of the pooled text embedding */
    public Dense pooled2() {
        return pooled2;
    }

    /** @return the joint blocks */
    public List<JointBlock> jointBlocks() {
        return List.copyOf(joint);
    }

    /** @return the single-stream blocks */
    public List<SingleBlock> singleBlocks() {
        return List.copyOf(single);
    }

    /** @return the output modulation, giving scale then shift */
    public Dense outModulation() {
        return outModulation;
    }

    /** @return the projection back to patches */
    public Dense out() {
        return out;
    }

    @Override
    public String toString() {
        return "FluxTransformer(" + settings.features() + ", " + settings.jointLayers() + "+"
                + settings.singleLayers() + " blocks)";
    }

    /**
     * A joint block: text and image tokens, each with its own modulation, projections and
     * feed-forward, attend to the joined sequence.
     */
    public static final class JointBlock implements Layer {

        private final int heads;
        private final Stream image;
        private final Stream text;

        JointBlock(Settings s, SplittableRandom random) {
            heads = s.heads();
            image = new Stream(s, random);
            text = new Stream(s, random);
        }

        /** One stream's weights. */
        public static final class Stream {
            private final Dense modulation;
            private final LayerNorm norm1;
            private final Dense q;
            private final Dense k;
            private final Dense v;
            private final LayerNorm qNorm;
            private final LayerNorm kNorm;
            private final Dense attentionOut;
            private final LayerNorm norm2;
            private final Dense up;
            private final Dense down;

            Stream(Settings s, SplittableRandom random) {
                int d = s.features();
                int hidden = (int) (d * s.mlpRatio());
                modulation = zeroModulation(d, 6);
                norm1 = plainNorm(d);
                q = Parts.dense(d, d, Activation.IDENTITY, random);
                k = Parts.dense(d, d, Activation.IDENTITY, random);
                v = Parts.dense(d, d, Activation.IDENTITY, random);
                qNorm = LayerNorm.rms(s.headSize());
                kNorm = LayerNorm.rms(s.headSize());
                attentionOut = Parts.dense(d, d, Activation.IDENTITY, random);
                norm2 = plainNorm(d);
                up = Parts.dense(d, hidden, Activation.GELU_TANH, random);
                down = Parts.dense(hidden, d, Activation.IDENTITY, random);
            }

            List<Layer> layers() {
                return List.of(modulation, q, k, v, qNorm, kNorm, attentionOut, up, down);
            }

            /** @return the modulation: shift, scale, gate for attention, then for the feed-forward */
            public Dense modulation() {
                return modulation;
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

            /** @return the RMS norm of each query head */
            public LayerNorm qNorm() {
                return qNorm;
            }

            /** @return the RMS norm of each key head */
            public LayerNorm kNorm() {
                return kNorm;
            }

            /** @return the attention output projection */
            public Dense attentionOut() {
                return attentionOut;
            }

            /** @return the feed-forward's first layer (tanh GELU) */
            public Dense up() {
                return up;
            }

            /** @return the feed-forward's second layer */
            public Dense down() {
                return down;
            }
        }

        /**
         * Runs the block on the joined sequence.
         *
         * @param x {@code [batch, textTokens + imageTokens, features]}, text first
         * @param textTokens how many tokens are text
         * @param temb the SiLU'd conditioning {@code [batch, features]}
         * @param rope the rotary tables of {@link #rotaryTables} for the joined sequence
         * @return the new joined sequence
         */
        public Var forward(Var x, int textTokens, Var temb, Tensor[] rope) {
            int n = x.shape().dim(1);
            Var txt = x.slice(1, 0, textTokens);
            Var img = x.slice(1, textTokens, n);
            Var[] mi = chunks(image.modulation.forward(temb), 6);
            Var[] mt = chunks(text.modulation.forward(temb), 6);
            Var ni = Functional.film(image.norm1.forward(img), mi[1], mi[0]);
            Var nt = Functional.film(text.norm1.forward(txt), mt[1], mt[0]);
            Var q = rotate(Var.concat(2, heads(text.q, text.qNorm, nt, heads), heads(image.q, image.qNorm, ni, heads)),
                    rope);
            Var k = rotate(Var.concat(2, heads(text.k, text.kNorm, nt, heads), heads(image.k, image.kNorm, ni, heads)),
                    rope);
            Var v = Var.concat(2, heads(text.v, null, nt, heads), heads(image.v, null, ni, heads));
            Var a = join(Functional.attention(q, k, v, false, (float) (1 / Math.sqrt(q.shape().dim(3)))));
            img = Functional.gate(img, image.attentionOut.forward(a.slice(1, textTokens, n)), mi[2]);
            txt = Functional.gate(txt, text.attentionOut.forward(a.slice(1, 0, textTokens)), mt[2]);
            img = Functional.gate(img, image.down.forward(image.up.forward(
                    Functional.film(image.norm2.forward(img), mi[4], mi[3]))), mi[5]);
            txt = Functional.gate(txt, text.down.forward(text.up.forward(
                    Functional.film(text.norm2.forward(txt), mt[4], mt[3]))), mt[5]);
            return Var.concat(1, txt, img);
        }

        /** @return the image stream */
        public Stream image() {
            return image;
        }

        /** @return the text stream */
        public Stream text() {
            return text;
        }

        @Override
        public Var forward(Var input) {
            throw new UnsupportedOperationException("a joint block runs inside a FluxTransformer");
        }

        /** Image stream then text stream, each modulation, q, k, v, q-norm, k-norm, out, up, down. */
        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (Stream st : List.of(image, text)) {
                for (Layer l : st.layers()) {
                    p.addAll(l.parameters());
                }
            }
            return p;
        }
    }

    /**
     * A single-stream block: the joined sequence goes through one attention and a parallel
     * feed-forward, whose outputs one projection merges.
     */
    public static final class SingleBlock implements Layer {

        private final int heads;
        private final Dense modulation;
        private final LayerNorm norm;
        private final Dense q;
        private final Dense k;
        private final Dense v;
        private final LayerNorm qNorm;
        private final LayerNorm kNorm;
        private final Dense mlp;
        private final Dense out;

        SingleBlock(Settings s, SplittableRandom random) {
            int d = s.features();
            int hidden = (int) (d * s.mlpRatio());
            heads = s.heads();
            modulation = zeroModulation(d, 3);
            norm = plainNorm(d);
            q = Parts.dense(d, d, Activation.IDENTITY, random);
            k = Parts.dense(d, d, Activation.IDENTITY, random);
            v = Parts.dense(d, d, Activation.IDENTITY, random);
            qNorm = LayerNorm.rms(s.headSize());
            kNorm = LayerNorm.rms(s.headSize());
            mlp = Parts.dense(d, hidden, Activation.GELU_TANH, random);
            out = Parts.dense(d + hidden, d, Activation.IDENTITY, random);
        }

        /**
         * Runs the block.
         *
         * @param x {@code [batch, tokens, features]}
         * @param temb the SiLU'd conditioning {@code [batch, features]}
         * @param rope the rotary tables for the sequence
         * @return the new sequence
         */
        public Var forward(Var x, Var temb, Tensor[] rope) {
            Var[] m = chunks(modulation.forward(temb), 3);
            Var n = Functional.film(norm.forward(x), m[1], m[0]);
            Var qh = rotate(heads(q, qNorm, n, heads), rope);
            Var kh = rotate(heads(k, kNorm, n, heads), rope);
            Var vh = heads(v, null, n, heads);
            Var a = join(Functional.attention(qh, kh, vh, false, (float) (1 / Math.sqrt(qh.shape().dim(3)))));
            return Functional.gate(x, out.forward(Var.concat(2, a, mlp.forward(n))), m[2]);
        }

        @Override
        public Var forward(Var input) {
            throw new UnsupportedOperationException("a single block runs inside a FluxTransformer");
        }

        /** Modulation, q, k, v, q-norm, k-norm, mlp, out. */
        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (Layer l : List.of(modulation, q, k, v, qNorm, kNorm, mlp, out)) {
                p.addAll(l.parameters());
            }
            return p;
        }

        /** @return the modulation: shift, scale, gate */
        public Dense modulation() {
            return modulation;
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

        /** @return the RMS norm of each query head */
        public LayerNorm qNorm() {
            return qNorm;
        }

        /** @return the RMS norm of each key head */
        public LayerNorm kNorm() {
            return kNorm;
        }

        /** @return the feed-forward's first layer (tanh GELU) */
        public Dense mlp() {
            return mlp;
        }

        /** @return the projection of attention and feed-forward together */
        public Dense out() {
            return out;
        }
    }
}
