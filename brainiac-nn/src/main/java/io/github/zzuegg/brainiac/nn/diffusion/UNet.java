package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.GroupNorm;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * The denoising UNet of latent diffusion ({@code UNet2DConditionModel} in diffusers): a
 * {@link TimestepEmbedding}, {@link ResnetBlock}s conditioned on it and
 * {@link SpatialTransformer}s that attend to a context, on a down path, a middle and an up
 * path joined by skip connections. With pretrained weights it is the UNet of Stable
 * Diffusion 1.x, 2.x and XL; with fresh weights it trains like any other layer, for example
 * with {@link Diffusion#loss}.
 *
 * <p>Its inputs are the noisy images {@code [batch, inChannels, h, w]} (h and w divisible by
 * {@code 2^(levels - 1)}), the steps {@code [batch]}, the context {@code [batch, tokens,
 * contextFeatures]} if {@link Settings#contextFeatures()} is positive, and Stable Diffusion
 * XL's extra conditioning {@code [batch, pooledFeatures + timeIds]} (a pooled text embedding
 * followed by its size numbers) if {@link Settings#addTimeFeatures()} is positive. The output
 * is shaped like the images.
 *
 * <pre>{@code
 * UNet unet = new UNet(UNet.Settings.of(3, 3, new int[] {32, 64}, 0), new SplittableRandom(1));
 * Diffusion d = new Diffusion(NoiseSchedule.cosine(1000), Diffusion.Prediction.NOISE);
 * Var loss = d.loss(unet, images, random);
 * }</pre>
 */
public final class UNet implements MultiInputLayer, Denoiser {

    /**
     * The shape of a UNet.
     *
     * @param inChannels image channels in
     * @param outChannels image channels out
     * @param channels the width of each resolution level
     * @param attention whether each level's blocks have transformers
     * @param layersPerBlock residual blocks per level on the down path (the up path has one more)
     * @param heads attention heads per level
     * @param depth transformer blocks per transformer, per level
     * @param contextFeatures the width of the context, or 0 for transformers without one
     * @param linearProjection whether transformers project with dense layers instead of 1x1 convolutions
     * @param groups group-norm groups
     * @param epsilon the residual blocks' group-norm epsilon
     * @param addTimeFeatures the width of each size number's embedding (Stable Diffusion XL's
     *     {@code text_time} conditioning), or 0 without
     * @param pooledFeatures the width of the pooled text embedding of that conditioning
     * @param timeIds how many size numbers follow it (6 for Stable Diffusion XL)
     * @param flipSinToCos whether the timestep embedding puts cosines first
     * @param freqShift the timestep embedding's frequency shift
     */
    public record Settings(int inChannels, int outChannels, int[] channels, boolean[] attention, int layersPerBlock,
            int[] heads, int[] depth, int contextFeatures, boolean linearProjection, int groups, float epsilon,
            int addTimeFeatures, int pooledFeatures, int timeIds, boolean flipSinToCos, int freqShift) {

        /** Checks the values. */
        public Settings {
            int levels = channels.length;
            if (levels < 1 || attention.length != levels || heads.length != levels || depth.length != levels) {
                throw new IllegalArgumentException("give attention, heads and depth for each of " + levels
                        + " levels");
            }
            if (inChannels < 1 || outChannels < 1 || layersPerBlock < 1) {
                throw new IllegalArgumentException("a UNet needs channels and at least one block per level");
            }
            channels = channels.clone();
            attention = attention.clone();
            heads = heads.clone();
            depth = depth.clone();
        }

        /**
         * A UNet with two residual blocks per level, transformers on every level but the
         * first, 64-wide attention heads (or one head on narrower levels) and as many
         * group-norm groups (up to 32) as divide every width.
         *
         * @param inChannels image channels in
         * @param outChannels image channels out
         * @param channels the width of each level
         * @param contextFeatures the width of the context, or 0 for none
         * @return the settings
         */
        public static Settings of(int inChannels, int outChannels, int[] channels, int contextFeatures) {
            int levels = channels.length;
            boolean[] attention = new boolean[levels];
            int[] heads = new int[levels];
            int[] depth = new int[levels];
            int groups = 32;
            for (int i = 0; i < levels; i++) {
                attention[i] = i > 0 || levels == 1;
                heads[i] = channels[i] % 64 == 0 ? channels[i] / 64 : 1;
                depth[i] = 1;
                while (channels[i] % groups != 0) {
                    groups /= 2;
                }
            }
            return new Settings(inChannels, outChannels, channels, attention, 2, heads, depth, contextFeatures,
                    true, groups, 1e-5f, 0, 0, 0, true, 0);
        }

        /** @return the width of the time embedding, four times the first level's */
        public int timeFeatures() {
            return channels[0] * 4;
        }

        /** @return the number of resolution levels */
        public int levels() {
            return channels.length;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Settings s && inChannels == s.inChannels && outChannels == s.outChannels
                    && Arrays.equals(channels, s.channels) && Arrays.equals(attention, s.attention)
                    && layersPerBlock == s.layersPerBlock && Arrays.equals(heads, s.heads)
                    && Arrays.equals(depth, s.depth) && contextFeatures == s.contextFeatures
                    && linearProjection == s.linearProjection && groups == s.groups
                    && Float.compare(epsilon, s.epsilon) == 0 && addTimeFeatures == s.addTimeFeatures
                    && pooledFeatures == s.pooledFeatures && timeIds == s.timeIds
                    && flipSinToCos == s.flipSinToCos && freqShift == s.freqShift;
        }

        @Override
        public int hashCode() {
            return Objects.hash(inChannels, outChannels, Arrays.hashCode(channels), layersPerBlock, contextFeatures);
        }

        @Override
        public String toString() {
            return "UNet.Settings[in " + inChannels + ", out " + outChannels + ", channels "
                    + Arrays.toString(channels) + ", attention " + Arrays.toString(attention) + ", heads "
                    + Arrays.toString(heads) + ", depth " + Arrays.toString(depth) + ", context " + contextFeatures
                    + "]";
        }
    }

    /**
     * The blocks of one level of the down or up path.
     *
     * @param resnets the residual blocks
     * @param transformers each residual block's transformer, or {@code null} entries without
     * @param resample the downsampling (stride 2) or upsampling (after nearest 2x) convolution,
     *     or {@code null} on the last level
     */
    public record Level(List<ResnetBlock> resnets, List<SpatialTransformer> transformers, Conv resample) {
    }

    private final Settings settings;
    private final Conv convIn;
    private final TimestepEmbedding timeEmbedding;
    private final Dense addLinear1;
    private final Dense addLinear2;
    private final List<Level> down = new ArrayList<>();
    private final ResnetBlock midResnet1;
    private final SpatialTransformer midTransformer;
    private final ResnetBlock midResnet2;
    private final List<Level> up = new ArrayList<>();
    private final GroupNorm normOut;
    private final Conv convOut;

    /**
     * Creates a UNet with fresh weights.
     *
     * @param settings its shape
     * @param random the source of initial weights
     */
    public UNet(Settings settings, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        Settings s = settings;
        int[] ch = s.channels();
        int levels = ch.length;
        int time = s.timeFeatures();
        convIn = Parts.conv3(s.inChannels(), ch[0], random);
        timeEmbedding = new TimestepEmbedding(new TimestepEmbedding.Settings(ch[0], time, s.flipSinToCos(),
                s.freqShift()), random);
        if (s.addTimeFeatures() > 0) {
            addLinear1 = Parts.dense(s.pooledFeatures() + s.timeIds() * s.addTimeFeatures(), time, Activation.SILU,
                    random);
            addLinear2 = Parts.dense(time, time, Activation.IDENTITY, random);
        } else {
            addLinear1 = null;
            addLinear2 = null;
        }
        int width = ch[0];
        for (int i = 0; i < levels; i++) {
            List<ResnetBlock> resnets = new ArrayList<>();
            List<SpatialTransformer> transformers = new ArrayList<>();
            for (int j = 0; j < s.layersPerBlock(); j++) {
                resnets.add(resnet(width, ch[i], random));
                transformers.add(transformer(i, random));
                width = ch[i];
            }
            down.add(new Level(resnets, transformers, i < levels - 1 ? Parts.conv(width, width, 3, 2, 1, 1, random)
                    : null));
        }
        midResnet1 = resnet(width, width, random);
        midTransformer = new SpatialTransformer(new SpatialTransformer.Settings(width, s.heads()[levels - 1],
                s.depth()[levels - 1], s.contextFeatures(), s.groups(), s.linearProjection()), random);
        midResnet2 = resnet(width, width, random);
        for (int i = 0; i < levels; i++) {
            int level = levels - 1 - i;
            List<ResnetBlock> resnets = new ArrayList<>();
            List<SpatialTransformer> transformers = new ArrayList<>();
            for (int j = 0; j <= s.layersPerBlock(); j++) {
                // the skip joined here: the level's own outputs, then the level below's last
                int skip = j < s.layersPerBlock() ? ch[level] : ch[Math.max(0, level - 1)];
                resnets.add(resnet(width + skip, ch[level], random));
                transformers.add(transformer(level, random));
                width = ch[level];
            }
            up.add(new Level(resnets, transformers, i < levels - 1 ? Parts.conv3(width, width, random) : null));
        }
        normOut = new GroupNorm(s.groups(), width, s.epsilon(), true);
        convOut = Parts.conv3(width, s.outChannels(), random);
    }

    private ResnetBlock resnet(int in, int out, SplittableRandom random) {
        return new ResnetBlock(new ResnetBlock.Settings(in, out, settings.timeFeatures(), settings.groups(),
                settings.epsilon(), 0), random);
    }

    private SpatialTransformer transformer(int level, SplittableRandom random) {
        return settings.attention()[level] ? new SpatialTransformer(new SpatialTransformer.Settings(
                settings.channels()[level], settings.heads()[level], settings.depth()[level],
                settings.contextFeatures(), settings.groups(), settings.linearProjection()), random) : null;
    }

    /**
     * Predicts the noise (or whatever it was trained for) in a batch of images.
     *
     * @param x the noisy images {@code [batch, inChannels, h, w]}
     * @param steps each image's step {@code [batch]}
     * @param context the context {@code [batch, tokens, contextFeatures]}, or {@code null} without
     * @param added the extra conditioning {@code [batch, pooledFeatures + timeIds]}, or {@code null} without
     * @param training whether the model is training
     * @return the prediction, shaped and typed like {@code x}
     */
    public Var forward(Var x, Var steps, Var context, Var added, boolean training) {
        Settings s = settings;
        if (x.shape().rank() != 4 || x.shape().dim(1) != s.inChannels()) {
            throw new IllegalArgumentException(this + " expects [batch, " + s.inChannels() + ", h, w], got "
                    + x.shape());
        }
        if ((context == null) != (s.contextFeatures() == 0) || (added == null) != (s.addTimeFeatures() == 0)) {
            throw new IllegalArgumentException(this + " needs " + (s.contextFeatures() > 0 ? "a context" : "no context")
                    + " and " + (s.addTimeFeatures() > 0 ? "size conditioning" : "no size conditioning"));
        }
        DType type = dtype();
        Var emb = Parts.step(() -> {
            Var e = timeEmbedding.forward(steps);
            if (addLinear1 != null) {
                int n = added.shape().dim(0);
                int p = s.pooledFeatures();
                Tensor ids = added.value().slice(1, p, p + s.timeIds());
                Tensor sizes = TimestepEmbedding.sinusoids(ids.reshape(n * s.timeIds()), s.addTimeFeatures(),
                        s.flipSinToCos(), s.freqShift()).reshape(n, s.timeIds() * s.addTimeFeatures());
                Var a = Var.concat(1, added.slice(1, 0, p).to(DType.FLOAT32), Var.of(sizes)).to(type);
                e = e.add(addLinear2.forward(addLinear1.forward(a)));
            }
            return e;
        });
        Var ctx = context == null ? null : context.to(type);
        Var h0 = x.to(type);
        Var h = Parts.step(() -> convIn.forward(h0));
        Parts.free(h0, x);
        Deque<Var> skips = new ArrayDeque<>();
        skips.push(h);
        for (Level level : down) {
            for (int j = 0; j < level.resnets().size(); j++) {
                h = block(level.resnets().get(j), level.transformers().get(j), h, emb, ctx, null, training);
                skips.push(h);
            }
            if (level.resample() != null) {
                Var in = h;
                h = Parts.step(() -> level.resample().forward(in));
                skips.push(h);
            }
        }
        Var downOut = h;   // also the first skip of the up path
        h = block(midResnet1, midTransformer, h, emb, ctx, null, training);
        Var mid = h;
        h = block(midResnet2, null, h, emb, ctx, null, training);
        Parts.free(mid);
        for (Level level : up) {
            for (int j = 0; j < level.resnets().size(); j++) {
                Var skip = skips.pop();
                Var in = h;
                h = block(level.resnets().get(j), level.transformers().get(j), h, emb, ctx, skip, training);
                if (in != downOut) {
                    Parts.free(in);
                }
                Parts.free(skip);
            }
            if (level.resample() != null) {
                Var in = h;
                h = Parts.step(() -> level.resample().forward(Parts.upsample2x(in)));
                Parts.free(in);
            }
        }
        Var last = h;
        Var out = Parts.step(() -> convOut.forward(Parts.silu(normOut.forward(last))).to(x.value().dtype()));
        Parts.free(last);
        Parts.free(emb);
        Parts.free(ctx, context);
        return out;
    }

    /** A residual block (on {@code h} joined with {@code skip} on the up path) and its transformer. */
    private static Var block(ResnetBlock resnet, SpatialTransformer transformer, Var h, Var emb, Var ctx, Var skip,
            boolean training) {
        Var r = Parts.step(() -> resnet.forward(List.of(skip == null ? h : Var.concat(1, h, skip), emb), training));
        if (transformer == null) {
            return r;
        }
        Var out = transformer.forward(ctx == null ? List.of(r) : List.of(r, ctx), training);
        Parts.free(r);
        return out;
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        int need = 2 + (settings.contextFeatures() > 0 ? 1 : 0) + (settings.addTimeFeatures() > 0 ? 1 : 0);
        if (inputs.size() != need) {
            throw new IllegalArgumentException(this + " takes " + need + " inputs (images, steps"
                    + (settings.contextFeatures() > 0 ? ", context" : "")
                    + (settings.addTimeFeatures() > 0 ? ", size conditioning" : "") + "), got " + inputs.size());
        }
        Var context = settings.contextFeatures() > 0 ? inputs.get(2) : null;
        Var added = settings.addTimeFeatures() > 0 ? inputs.get(need - 1) : null;
        return forward(inputs.get(0), inputs.get(1), context, added, training);
    }

    /**
     * Predicts for a model without a context, as a {@link Denoiser}.
     *
     * @param noisy the noisy images
     * @param time the steps {@code [batch]}
     * @return the prediction
     */
    @Override
    public Var apply(Var noisy, Tensor time) {
        return forward(noisy, Var.of(time), null, null, Var.gradEnabled());
    }

    /**
     * This model with a context, as a {@link Denoiser.Conditional}: the condition is the
     * context {@code [batch, tokens, contextFeatures]}.
     *
     * @return the conditional denoiser
     */
    public Denoiser.Conditional conditional() {
        return (noisy, time, condition) -> forward(noisy, Var.of(time), Var.of(condition), null,
                Var.gradEnabled());
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        Shape x = inputs.get(0);
        return Shape.of(settings.outChannels(), x.dim(1), x.dim(2));
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(settings.outChannels(), input.dim(1), input.dim(2));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(convIn.parameters());
        p.addAll(timeEmbedding.parameters());
        if (addLinear1 != null) {
            p.addAll(addLinear1.parameters());
            p.addAll(addLinear2.parameters());
        }
        for (Level l : down) {
            add(p, l);
        }
        p.addAll(midResnet1.parameters());
        p.addAll(midTransformer.parameters());
        p.addAll(midResnet2.parameters());
        for (Level l : up) {
            add(p, l);
        }
        p.addAll(normOut.parameters());
        p.addAll(convOut.parameters());
        return p;
    }

    private static void add(List<Var> p, Level l) {
        for (int j = 0; j < l.resnets().size(); j++) {
            p.addAll(l.resnets().get(j).parameters());
            if (l.transformers().get(j) != null) {
                p.addAll(l.transformers().get(j).parameters());
            }
        }
        if (l.resample() != null) {
            p.addAll(l.resample().parameters());
        }
    }

    /** @return the type the weights are held in */
    public DType dtype() {
        return convIn.weight().value().dtype();
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    /** @return the first convolution */
    public Conv convIn() {
        return convIn;
    }

    /** @return the timestep embedding */
    public TimestepEmbedding timeEmbedding() {
        return timeEmbedding;
    }

    /** @return the first layer of the size conditioning's embedding, with SiLU, or {@code null} */
    public Dense addLinear1() {
        return addLinear1;
    }

    /** @return the second layer of the size conditioning's embedding, or {@code null} */
    public Dense addLinear2() {
        return addLinear2;
    }

    /** @return the levels of the down path, highest resolution first */
    public List<Level> down() {
        return List.copyOf(down);
    }

    /** @return the middle's residual blocks and transformer: resnet, transformer, resnet */
    public List<MultiInputLayer> middle() {
        return List.of(midResnet1, midTransformer, midResnet2);
    }

    /** @return the levels of the up path, lowest resolution first */
    public List<Level> up() {
        return List.copyOf(up);
    }

    /** @return the output group norm */
    public GroupNorm normOut() {
        return normOut;
    }

    /** @return the output convolution */
    public Conv convOut() {
        return convOut;
    }

    @Override
    public String toString() {
        return "UNet(" + settings.inChannels() + " -> " + settings.outChannels() + ", channels "
                + Arrays.toString(settings.channels()) + ")";
    }
}
