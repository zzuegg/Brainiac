package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * The causal 3-d video autoencoder of Wan 2.1 (Wan-VAE): it compresses a video 4 times in
 * time and 8 times in each spatial side into latents of 16 channels, and back. Its
 * convolutions are causal in time (a frame sees only itself and earlier frames), so the
 * first frame is coded on its own, as an image, and every further 4 frames become one latent
 * frame: {@code 1 + 4k} frames make {@code 1 + k} latent frames.
 *
 * <p>Like the reference code it runs in chunks (one latent frame at a time) and carries the
 * last frames each causal convolution saw from chunk to chunk, so memory stays that of one
 * chunk however long the video is. It runs without gradients.
 *
 * <p>Videos here are {@code [1, 3, frames, height, width]} with values in {@code [-1, 1]};
 * latents are {@code [1, 16, latentFrames, height / 8, width / 8]}, normalized per channel by
 * the model's latent mean and standard deviation (as the transformer expects them).
 */
public final class WanVae implements Layer {

    /**
     * The shape of the autoencoder.
     *
     * @param base the channels of the first level (Wan 2.1: 96)
     * @param latentChannels the latent channels (16)
     * @param multipliers the channel multiplier per level ({1, 2, 4, 4})
     * @param resBlocks residual blocks per encoder level; the decoder has one more (2)
     * @param temporalDown per level transition, whether the encoder also halves the frames
     *     ({false, true, true}); the decoder doubles them in reverse
     * @param mean the latent mean per channel
     * @param std the latent standard deviation per channel
     */
    public record Settings(int base, int latentChannels, int[] multipliers, int resBlocks, boolean[] temporalDown,
            float[] mean, float[] std) {

        /** Checks the values. */
        public Settings {
            multipliers = multipliers.clone();
            temporalDown = temporalDown.clone();
            mean = mean.clone();
            std = std.clone();
            if (base < 1 || latentChannels < 1 || multipliers.length < 2 || resBlocks < 1
                    || temporalDown.length != multipliers.length - 1 || mean.length != latentChannels
                    || std.length != latentChannels) {
                throw new IllegalArgumentException("invalid Wan VAE shape");
            }
        }

        /** Wan 2.1's latent mean per channel. */
        public static final float[] WAN21_MEAN = {-0.7571f, -0.7089f, -0.9113f, 0.1075f, -0.1745f, 0.9653f, -0.1517f,
                1.5508f, 0.4134f, -0.0715f, 0.5517f, -0.3632f, -0.1922f, -0.9497f, 0.2503f, -0.2921f};
        /** Wan 2.1's latent standard deviation per channel. */
        public static final float[] WAN21_STD = {2.8184f, 1.4541f, 2.3275f, 2.6558f, 1.2196f, 1.7708f, 2.6052f,
                2.0743f, 3.2687f, 2.1526f, 2.8652f, 1.5579f, 1.6382f, 1.1253f, 2.8251f, 1.9160f};

        /** @return Wan 2.1's autoencoder */
        public static Settings wan21() {
            return new Settings(96, 16, new int[] {1, 2, 4, 4}, 2, new boolean[] {false, true, true}, WAN21_MEAN,
                    WAN21_STD);
        }

        /** @return how many times the frames shrink: 2 per temporal level */
        public int temporalFactor() {
            int f = 1;
            for (boolean t : temporalDown) {
                f *= t ? 2 : 1;
            }
            return f;
        }

        /** @return how many times each spatial side shrinks */
        public int spatialFactor() {
            return 1 << temporalDown.length;
        }

        @Override
        public int[] multipliers() {
            return multipliers.clone();
        }

        @Override
        public boolean[] temporalDown() {
            return temporalDown.clone();
        }

        @Override
        public float[] mean() {
            return mean.clone();
        }

        @Override
        public float[] std() {
            return std.clone();
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Settings s && base == s.base && latentChannels == s.latentChannels
                    && Arrays.equals(multipliers, s.multipliers) && resBlocks == s.resBlocks
                    && Arrays.equals(temporalDown, s.temporalDown) && Arrays.equals(mean, s.mean)
                    && Arrays.equals(std, s.std);
        }

        @Override
        public int hashCode() {
            return Objects.hash(base, latentChannels, Arrays.hashCode(multipliers), resBlocks,
                    Arrays.hashCode(temporalDown), Arrays.hashCode(mean), Arrays.hashCode(std));
        }

        @Override
        public String toString() {
            return "Settings[base=" + base + ", latentChannels=" + latentChannels + ", multipliers="
                    + Arrays.toString(multipliers) + ", resBlocks=" + resBlocks + ", temporalDown="
                    + Arrays.toString(temporalDown) + "]";
        }
    }

    private final Settings settings;
    private final Coder encoder;
    private final CausalConv quant;
    private final CausalConv postQuant;
    private final Coder decoder;

    /**
     * Creates an autoencoder with fresh weights.
     *
     * @param settings its shape
     * @param encoder whether to build the encoder (only the decoder is needed to generate)
     * @param random the source of initial weights
     */
    public WanVae(Settings settings, boolean encoder, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        int z = settings.latentChannels();
        this.encoder = encoder ? Coder.encoder(settings, random) : null;
        quant = encoder ? new CausalConv(2 * z, 2 * z, 1, 1, random) : null;
        postQuant = new CausalConv(z, z, 1, 1, random);
        decoder = Coder.decoder(settings, random);
    }

    // ------------------------------------------------------------------------------- coding

    /**
     * Encodes a video into normalized latents (the mean of the posterior).
     *
     * @param video {@code [1, 3, 1 + 4k frames, height, width]} in {@code [-1, 1]}, sides
     *     divisible by 8
     * @return {@code [1, latentChannels, 1 + k, height / 8, width / 8]} in float32
     */
    public Tensor encode(Tensor video) {
        if (encoder == null) {
            throw new IllegalStateException(this + " was built without its encoder");
        }
        Shape s = video.shape();
        int tf = settings.temporalFactor();
        int sf = settings.spatialFactor();
        if (s.rank() != 5 || s.dim(0) != 1 || s.dim(1) != 3 || (s.dim(2) - 1) % tf != 0 || s.dim(3) % sf != 0
                || s.dim(4) % sf != 0) {
            throw new IllegalArgumentException(this + " encodes videos [1, 3, 1 + " + tf + "k, h, w] with sides "
                    + "divisible by " + sf + ", got " + s);
        }
        return Var.noGrad(() -> {
            DType type = dtype();
            State state = new State();
            List<Tensor> chunks = new ArrayList<>();
            int frames = s.dim(2);
            try {
                for (int start = 0; start < frames; start = start == 0 ? 1 : start + tf) {
                    int end = start == 0 ? 1 : start + tf;
                    int from = start;
                    Tensor latent = state.run(() -> {
                        Var x = Var.of(frames(video.slice(2, from, end)).to(type));
                        Var h = encoder.forward(x, state);
                        Var stats = quant.forward(h, state);
                        int z = settings.latentChannels();
                        return normalize(stats.slice(1, 0, z).to(DType.FLOAT32));
                    });
                    chunks.add(latent);
                }
                Tensor all = chunks.size() == 1 ? chunks.get(0) : Tensor.concat(0, chunks.toArray(Tensor[]::new));
                Tensor out = video(all);
                Tensor.evaluate(List.of(out));
                return out;
            } finally {
                if (chunks.size() > 1) {
                    for (Tensor c : chunks) {
                        c.release();
                    }
                }
                state.close();
            }
        });
    }

    /**
     * Decodes normalized latents into a video.
     *
     * @param latents {@code [1, latentChannels, k, h, w]}
     * @return {@code [1, 3, 1 + 4(k - 1), 8h, 8w]} in {@code [-1, 1]}, float32
     */
    public Tensor decode(Tensor latents) {
        List<Tensor> parts = new ArrayList<>();
        decode(latents, parts::add);
        Tensor all = parts.size() == 1 ? parts.get(0) : Tensor.concat(0, parts.toArray(Tensor[]::new));
        Tensor out = video(all);
        Tensor.evaluate(List.of(out));
        if (parts.size() > 1) {
            for (Tensor p : parts) {
                p.release();
            }
        }
        return out;
    }

    /**
     * Decodes normalized latents one latent frame at a time, handing out the frames as they
     * are ready, so that a long video need not be held on the device at once.
     *
     * @param latents {@code [1, latentChannels, k, h, w]}
     * @param frames receives frames {@code [count, 3, 8h, 8w]} in {@code [-1, 1]}, float32: 1,
     *     then 4 per further latent frame; the receiver owns them
     */
    public void decode(Tensor latents, Consumer<Tensor> frames) {
        Shape s = latents.shape();
        if (s.rank() != 5 || s.dim(0) != 1 || s.dim(1) != settings.latentChannels()) {
            throw new IllegalArgumentException(this + " decodes latents [1, " + settings.latentChannels()
                    + ", frames, h, w], got " + s);
        }
        Var.noGrad(() -> {
            DType type = dtype();
            State state = new State();
            try {
                for (int i = 0; i < s.dim(2); i++) {
                    int at = i;
                    Tensor out = state.run(() -> {
                        Var z = Var.of(denormalize(frames(latents.slice(2, at, at + 1)).to(DType.FLOAT32)).to(type));
                        Var x = decoder.forward(postQuant.forward(z, state), state);
                        return x.to(DType.FLOAT32).clamp(-1f, 1f);
                    });
                    frames.accept(out);
                }
            } finally {
                state.close();
            }
            return null;
        });
    }

    /** {@code [1, c, t, h, w]} as frames {@code [t, c, h, w]}. */
    private static Tensor frames(Tensor video) {
        Shape s = video.shape();
        return video.reshape(s.dim(1), s.dim(2), s.dim(3), s.dim(4)).permute(1, 0, 2, 3);
    }

    /** Frames {@code [t, c, h, w]} as {@code [1, c, t, h, w]}. */
    private static Tensor video(Tensor frames) {
        Shape s = frames.shape();
        return frames.permute(1, 0, 2, 3).reshape(1, s.dim(1), s.dim(0), s.dim(2), s.dim(3));
    }

    private Var normalize(Var mu) {
        int z = settings.latentChannels();
        float[] inv = new float[z];
        for (int i = 0; i < z; i++) {
            inv[i] = 1 / settings.std[i];
        }
        Tensor mean = Tensor.of(Shape.of(1, z, 1, 1), settings.mean, mu.value().device());
        Tensor scale = Tensor.of(Shape.of(1, z, 1, 1), inv, mu.value().device());
        return mu.sub(mean).mul(scale);
    }

    private Tensor denormalize(Tensor z) {
        int c = settings.latentChannels();
        Tensor mean = Tensor.of(Shape.of(1, c, 1, 1), settings.mean, z.device());
        Tensor std = Tensor.of(Shape.of(1, c, 1, 1), settings.std, z.device());
        return z.mul(std).add(mean);
    }

    // ------------------------------------------------------------------------------- state

    /**
     * What the causal layers carry from one chunk to the next: the last input frames of each
     * convolution, and which temporal resamplers have seen their first chunk.
     */
    public static final class State implements AutoCloseable {

        State() {
        }

        private final Map<Object, Tensor> frames = new IdentityHashMap<>();
        private final Set<Object> started = Collections.newSetFromMap(new IdentityHashMap<>());
        private final List<Tensor> stale = new ArrayList<>();

        Tensor get(Object owner) {
            return frames.get(owner);
        }

        /** Stores a layer's frames; the old ones are freed once the chunk is computed. */
        void put(Object owner, Tensor value) {
            Tensor old = frames.put(owner, value);
            if (old != null && old != value) {
                stale.add(old);
            }
        }

        /** @return whether {@code owner} ran before, marking it as having run */
        boolean start(Object owner) {
            return !started.add(owner);
        }

        /**
         * Computes one chunk and frees everything it made except its result and the carried
         * frames, and the carried frames it replaced.
         */
        Tensor run(Supplier<Var> body) {
            Var[] out = new Var[1];
            List<Tensor> made = Tensor.created(() -> out[0] = body.get());
            List<Tensor> keep = new ArrayList<>(frames.values());
            keep.add(out[0].value());
            Tensor.evaluate(keep);
            Tensor.releaseCreated(made, keep);
            for (Tensor t : stale) {
                t.release();
            }
            stale.clear();
            return out[0].value();
        }

        /** Frees the carried frames. */
        @Override
        public void close() {
            for (Tensor t : frames.values()) {
                t.release();
            }
            for (Tensor t : stale) {
                t.release();
            }
            frames.clear();
            stale.clear();
        }
    }

    // ------------------------------------------------------------------------------- parts

    /**
     * A 3-d convolution, causal in time: {@code kt - 1} earlier frames (zeros before the
     * video starts) are put before each chunk, and it is computed as one 2-d convolution per
     * temporal tap over frames-as-batch, which keeps memory to that of 2-d convolutions.
     */
    public static final class CausalConv implements Layer {
        private final Var weight;
        private final Var bias;
        private final int kt;
        private final int pad;

        CausalConv(int in, int out, int kt, int k, SplittableRandom random) {
            this.kt = kt;
            this.pad = k / 2;
            int fan = in * kt * k * k;
            weight = Var.param(Init.HE.create(Shape.of(out, in, kt, k, k), fan, out * kt * k * k, random));
            bias = Var.param(Tensor.zeros(Shape.of(out)));
        }

        /**
         * Convolves frames {@code [t, in, h, w]} after the carried ones.
         *
         * @param x the chunk's frames
         * @param state the carried frames
         * @return {@code [t, out, h, w]}
         */
        Var forward(Var x, State state) {
            int t = x.shape().dim(0);
            Var in = x;
            if (kt > 1) {
                Tensor before = state.get(this);
                if (before == null) {
                    Shape s = x.shape();
                    before = Tensor.zeros(Shape.of(kt - 1, s.dim(1), s.dim(2), s.dim(3)), x.value().dtype(),
                            x.value().device());
                }
                in = Var.concat(0, Var.of(before), x);
                int all = in.shape().dim(0);
                state.put(this, in.value().slice(0, all - (kt - 1), all));
            }
            return taps(in, t, 1);
        }

        /** The sum over temporal taps of 2-d convolutions; output frame {@code i} reads input frames {@code stride * i ...}. */
        Var taps(Var in, int t, int stride) {
            Shape ws = weight.shape();
            int out = ws.dim(0);
            int c = ws.dim(1);
            int k = ws.dim(3);
            Var sum = null;
            for (int j = 0; j < kt; j++) {
                Var frames = stride == 1 ? in.slice(0, j, j + t) : every(in, j, t, stride);
                Var w = weight.slice(2, j, j + 1).reshape(out, c, k, ws.dim(4));
                Var y = Functional.conv(frames, w, j == 0 ? bias : null, new int[] {1, 1}, new int[] {pad, pad},
                        new int[] {pad, pad}, new int[] {1, 1}, 1);
                sum = sum == null ? y : sum.add(y);
            }
            return sum;
        }

        /** Frames {@code start, start + stride, ...}, {@code count} of them. */
        private static Var every(Var in, int start, int count, int stride) {
            Var[] picked = new Var[count];
            for (int i = 0; i < count; i++) {
                picked[i] = in.slice(0, start + i * stride, start + i * stride + 1);
            }
            return count == 1 ? picked[0] : Var.concat(0, picked);
        }

        @Override
        public Var forward(Var input) {
            throw new UnsupportedOperationException("a causal convolution runs inside a WanVae");
        }

        @Override
        public List<Var> parameters() {
            return List.of(weight, bias);
        }

        /** @return the weight {@code [out, in, kt, k, k]} */
        public Var weight() {
            return weight;
        }

        /** @return the bias {@code [out]} */
        public Var bias() {
            return bias;
        }
    }

    /**
     * Wan's RMS norm over channels: each position divided by the length of its channel
     * vector, times {@code sqrt(channels)} and a learned scale.
     */
    public static final class ChannelNorm implements Layer {
        private final Var gamma;

        ChannelNorm(int channels) {
            gamma = Var.param(Tensor.ones(Shape.of(channels)));
        }

        @Override
        public Var forward(Var x) {
            int c = x.shape().dim(1);
            DType type = x.value().dtype();
            Var f = x.to(DType.FLOAT32);
            Var length = f.square().sum(1, true).sqrt().maximum(1e-12f);
            Var scale = gamma.to(DType.FLOAT32).mul((float) Math.sqrt(c)).reshape(1, c, 1, 1);
            return f.div(length).mul(scale).to(type);
        }

        @Override
        public List<Var> parameters() {
            return List.of(gamma);
        }

        /** @return the scale per channel */
        public Var gamma() {
            return gamma;
        }
    }

    /** One element of an encoder's or decoder's chain. */
    public sealed interface Stage extends Layer permits ResBlock, AttentionBlock, Resample {

        /** Runs the stage on a chunk of frames {@code [t, c, h, w]}. */
        Var forward(Var x, State state);

        @Override
        default Var forward(Var input) {
            throw new UnsupportedOperationException("a stage runs inside a WanVae");
        }
    }

    /** A residual block: norm, SiLU, causal 3x3x3 convolution, twice, plus a 1x1x1 shortcut if the width changes. */
    public static final class ResBlock implements Stage {
        private final ChannelNorm norm1;
        private final CausalConv conv1;
        private final ChannelNorm norm2;
        private final CausalConv conv2;
        private final CausalConv shortcut;

        ResBlock(int in, int out, SplittableRandom random) {
            norm1 = new ChannelNorm(in);
            conv1 = new CausalConv(in, out, 3, 3, random);
            norm2 = new ChannelNorm(out);
            conv2 = new CausalConv(out, out, 3, 3, random);
            shortcut = in != out ? new CausalConv(in, out, 1, 1, random) : null;
        }

        @Override
        public Var forward(Var x, State state) {
            Var h = shortcut == null ? x : shortcut.forward(x, state);
            Var y = conv1.forward(Parts.silu(norm1.forward(x)), state);
            y = conv2.forward(Parts.silu(norm2.forward(y)), state);
            return y.add(h);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (Layer l : shortcut == null ? List.of(norm1, conv1, norm2, conv2)
                    : List.of(norm1, conv1, norm2, conv2, shortcut)) {
                p.addAll(l.parameters());
            }
            return p;
        }

        /** @return the first norm */
        public ChannelNorm norm1() {
            return norm1;
        }

        /** @return the first convolution */
        public CausalConv conv1() {
            return conv1;
        }

        /** @return the second norm */
        public ChannelNorm norm2() {
            return norm2;
        }

        /** @return the second convolution */
        public CausalConv conv2() {
            return conv2;
        }

        /** @return the shortcut projection, or {@code null} */
        public CausalConv shortcut() {
            return shortcut;
        }
    }

    /** Single-head self-attention within each frame, over its positions. */
    public static final class AttentionBlock implements Stage {
        private final ChannelNorm norm;
        private final Var qkv;
        private final Var qkvBias;
        private final Var proj;
        private final Var projBias;

        AttentionBlock(int c, SplittableRandom random) {
            norm = new ChannelNorm(c);
            qkv = Var.param(Init.XAVIER.create(Shape.of(c, 3 * c), c, 3 * c, random));
            qkvBias = Var.param(Tensor.zeros(Shape.of(3 * c)));
            proj = Var.param(Tensor.zeros(Shape.of(c, c)));
            projBias = Var.param(Tensor.zeros(Shape.of(c)));
        }

        @Override
        public Var forward(Var x, State state) {
            Shape s = x.shape();
            int t = s.dim(0);
            int c = s.dim(1);
            int n = s.dim(2) * s.dim(3);
            Var tokens = norm.forward(x).reshape(t, c, n).transpose();   // [t, n, c]
            Var all = tokens.matmul(qkv).add(qkvBias);
            Var q = all.slice(2, 0, c).reshape(t, 1, n, c);
            Var k = all.slice(2, c, 2 * c).reshape(t, 1, n, c);
            Var v = all.slice(2, 2 * c, 3 * c).reshape(t, 1, n, c);
            Var a = FluxTransformer.attend(q, k, v).reshape(t, n, c);
            Var y = a.matmul(proj).add(projBias).transpose().reshape(t, c, s.dim(2), s.dim(3));
            return y.add(x);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(norm.parameters());
            p.addAll(List.of(qkv, qkvBias, proj, projBias));
            return p;
        }

        /** @return the norm */
        public ChannelNorm norm() {
            return norm;
        }

        /** @return the joint query, key and value projection {@code [c, 3c]} */
        public Var qkv() {
            return qkv;
        }

        /** @return its bias {@code [3c]} */
        public Var qkvBias() {
            return qkvBias;
        }

        /** @return the output projection {@code [c, c]} */
        public Var proj() {
            return proj;
        }

        /** @return its bias {@code [c]} */
        public Var projBias() {
            return projBias;
        }
    }

    /** How a {@link Resample} changes the size. */
    public enum Mode {
        /** Halves height and width. */
        DOWN_2D,
        /** Halves height, width and (after the first chunk) the frames. */
        DOWN_3D,
        /** Doubles height and width, halving the channels. */
        UP_2D,
        /** Doubles height, width and (after the first chunk) the frames, halving the channels. */
        UP_3D
    }

    /** A change of resolution, spatial by a 2-d convolution and temporal by a causal 3x1x1 convolution. */
    public static final class Resample implements Stage {
        private final Mode mode;
        private final Var weight;
        private final Var bias;
        private final CausalConv time;

        Resample(int c, Mode mode, SplittableRandom random) {
            this.mode = mode;
            boolean up = mode == Mode.UP_2D || mode == Mode.UP_3D;
            int out = up ? c / 2 : c;
            weight = Var.param(Init.HE.create(Shape.of(out, c, 3, 3), c * 9, out * 9, random));
            bias = Var.param(Tensor.zeros(Shape.of(out)));
            time = switch (mode) {
                case UP_3D -> new CausalConv(c, 2 * c, 3, 1, random);
                case DOWN_3D -> new CausalConv(c, c, 3, 1, random);
                default -> null;
            };
        }

        @Override
        public Var forward(Var x, State state) {
            if (mode == Mode.UP_3D && state.start(this)) {
                // every chunk after the first: each frame becomes two
                Var y = time.forward(x, state);
                Shape s = y.shape();
                int c = s.dim(1) / 2;
                x = y.reshape(s.dim(0) * 2, c, s.dim(2), s.dim(3));
            }
            Var y;
            if (mode == Mode.UP_2D || mode == Mode.UP_3D) {
                y = Functional.conv(Parts.upsample2x(x), weight, bias, new int[] {1, 1}, new int[] {1, 1},
                        new int[] {1, 1}, new int[] {1, 1}, 1);
            } else {
                y = Functional.conv(x, weight, bias, new int[] {2, 2}, new int[] {0, 0}, new int[] {1, 1},
                        new int[] {1, 1}, 1);
            }
            if (mode == Mode.DOWN_3D) {
                int t = y.shape().dim(0);
                Tensor last = y.value().slice(0, t - 1, t);
                if (state.start(this)) {
                    // the frame before the chunk, then pairs of frames with a stride of 2
                    Var in = Var.concat(0, Var.of(state.get(this)), y);
                    state.put(this, last);
                    y = time.taps(in, t / 2, 2);
                } else {
                    state.put(this, last);
                }
            }
            return y;
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(weight, bias));
            if (time != null) {
                p.addAll(time.parameters());
            }
            return p;
        }

        /** @return the mode */
        public Mode mode() {
            return mode;
        }

        /** @return the spatial convolution's weight {@code [out, c, 3, 3]} */
        public Var weight() {
            return weight;
        }

        /** @return the spatial convolution's bias */
        public Var bias() {
            return bias;
        }

        /** @return the temporal convolution, or {@code null} for 2-d modes */
        public CausalConv time() {
            return time;
        }
    }

    /** An encoder or decoder: input convolution, a chain of stages, a middle, and a head. */
    public static final class Coder implements Layer {
        private final CausalConv in;
        private final List<Stage> stages;
        private final List<Stage> middle;
        private final ChannelNorm headNorm;
        private final CausalConv head;
        private final boolean encoder;

        private Coder(CausalConv in, List<Stage> stages, List<Stage> middle, ChannelNorm headNorm, CausalConv head,
                boolean encoder) {
            this.in = in;
            this.stages = stages;
            this.middle = middle;
            this.headNorm = headNorm;
            this.head = head;
            this.encoder = encoder;
        }

        static Coder encoder(Settings s, SplittableRandom random) {
            int[] m = s.multipliers();
            int[] dims = new int[m.length + 1];
            dims[0] = s.base();
            for (int i = 0; i < m.length; i++) {
                dims[i + 1] = s.base() * m[i];
            }
            List<Stage> stages = new ArrayList<>();
            for (int i = 0; i < m.length; i++) {
                int inDim = dims[i];
                for (int j = 0; j < s.resBlocks(); j++) {
                    stages.add(new ResBlock(inDim, dims[i + 1], random));
                    inDim = dims[i + 1];
                }
                if (i != m.length - 1) {
                    stages.add(new Resample(dims[i + 1], s.temporalDown[i] ? Mode.DOWN_3D : Mode.DOWN_2D, random));
                }
            }
            int top = dims[m.length];
            return new Coder(new CausalConv(3, dims[0], 3, 3, random), stages, middle(top, random),
                    new ChannelNorm(top), new CausalConv(top, 2 * s.latentChannels(), 3, 3, random), true);
        }

        static Coder decoder(Settings s, SplittableRandom random) {
            int[] m = s.multipliers();
            int levels = m.length;
            int[] dims = new int[levels + 1];
            dims[0] = s.base() * m[levels - 1];
            for (int i = 0; i < levels; i++) {
                dims[i + 1] = s.base() * m[levels - 1 - i];
            }
            List<Stage> stages = new ArrayList<>();
            for (int i = 0; i < levels; i++) {
                // after a resample the channels are halved
                int inDim = i == 0 ? dims[0] : dims[i] / 2;
                for (int j = 0; j < s.resBlocks() + 1; j++) {
                    stages.add(new ResBlock(inDim, dims[i + 1], random));
                    inDim = dims[i + 1];
                }
                if (i != levels - 1) {
                    boolean temporal = s.temporalDown[levels - 2 - i];
                    stages.add(new Resample(dims[i + 1], temporal ? Mode.UP_3D : Mode.UP_2D, random));
                }
            }
            int last = dims[levels];
            return new Coder(new CausalConv(s.latentChannels(), dims[0], 3, 3, random), stages,
                    middle(dims[0], random), new ChannelNorm(last), new CausalConv(last, 3, 3, 3, random), false);
        }

        private static List<Stage> middle(int c, SplittableRandom random) {
            return List.of(new ResBlock(c, c, random), new AttentionBlock(c, random), new ResBlock(c, c, random));
        }

        Var forward(Var x, State state) {
            x = in.forward(x, state);
            // the encoder narrows then thinks in the middle; the decoder thinks first, then widens
            for (Stage s : encoder ? stages : middle) {
                x = s.forward(x, state);
            }
            for (Stage s : encoder ? middle : stages) {
                x = s.forward(x, state);
            }
            return head.forward(Parts.silu(headNorm.forward(x)), state);
        }

        @Override
        public Var forward(Var input) {
            throw new UnsupportedOperationException("a coder runs inside a WanVae");
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(in.parameters());
            for (Stage s : stages) {
                p.addAll(s.parameters());
            }
            for (Stage s : middle) {
                p.addAll(s.parameters());
            }
            p.addAll(headNorm.parameters());
            p.addAll(head.parameters());
            return p;
        }

        /** @return whether this is the encoder */
        public boolean isEncoder() {
            return encoder;
        }

        /** @return the input convolution */
        public CausalConv in() {
            return in;
        }

        /** @return the chain of residual blocks and resamplers, in Wan's order */
        public List<Stage> stages() {
            return stages;
        }

        /** @return the middle: residual block, attention, residual block */
        public List<Stage> middle() {
            return middle;
        }

        /** @return the norm before the head */
        public ChannelNorm headNorm() {
            return headNorm;
        }

        /** @return the output convolution */
        public CausalConv head() {
            return head;
        }
    }

    // ------------------------------------------------------------------------------- layer

    @Override
    public Var forward(Var input) {
        return Var.of(decode(input.value()));
    }

    @Override
    public Shape outputShape(Shape input) {
        int t = (input.dim(2) - 1) * settings.temporalFactor() + 1;
        int f = settings.spatialFactor();
        return Shape.of(input.dim(0), 3, t, input.dim(3) * f, input.dim(4) * f);
    }

    /** Encoder (if built) and its latent projection, then the decoder's latent projection and the decoder. */
    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (encoder != null) {
            p.addAll(encoder.parameters());
            p.addAll(quant.parameters());
        }
        p.addAll(postQuant.parameters());
        p.addAll(decoder.parameters());
        return p;
    }

    /** @return the type the weights are held in */
    public DType dtype() {
        return postQuant.weight.value().dtype();
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    /** @return the encoder, or {@code null} */
    public Coder encoder() {
        return encoder;
    }

    /** @return the encoder's 1x1x1 projection of the latent statistics, or {@code null} */
    public CausalConv quant() {
        return quant;
    }

    /** @return the decoder's 1x1x1 projection of the latents */
    public CausalConv postQuant() {
        return postQuant;
    }

    /** @return the decoder */
    public Coder decoder() {
        return decoder;
    }

    @Override
    public String toString() {
        return "WanVae(" + settings.base() + ", " + settings.latentChannels() + " latent channels)";
    }
}
