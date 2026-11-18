package io.github.zzuegg.brainiac.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * The encoder of T5 (version 1.1): token embeddings, then pre-norm blocks of
 * self-attention with learned relative position biases and a gated feed-forward, with
 * RMS norms and no biases. With pretrained T5-XXL weights it is the text encoder of Flux and
 * Stable Diffusion 3; with fresh weights it trains like any other layer.
 *
 * <p>As a layer it maps token ids {@code [batch, tokens]} (as numbers) to hidden states
 * {@code [batch, tokens, features]}.
 */
public final class T5Encoder implements Layer {

    /**
     * The shape of the encoder.
     *
     * @param vocabulary the number of tokens
     * @param features the model width ({@code d_model})
     * @param layers the number of blocks
     * @param heads the attention heads
     * @param headSize the features per head ({@code d_kv})
     * @param ffnSize the feed-forward width ({@code d_ff})
     * @param buckets the relative position buckets (32)
     * @param maxDistance the distance beyond which positions share a bucket (128)
     * @param epsilon the RMS norms' epsilon
     * @param gated whether the feed-forward is gated (T5 1.1: GELU times a linear gate),
     *     otherwise a plain ReLU feed-forward (the original T5)
     * @param biasPerLayer whether every block learns its own relative position biases (umT5,
     *     the text encoder of Wan), otherwise the first block's serve all (T5)
     */
    public record Settings(int vocabulary, int features, int layers, int heads, int headSize, int ffnSize,
            int buckets, int maxDistance, float epsilon, boolean gated, boolean biasPerLayer) {

        /**
         * A T5 shape whose blocks share one set of position biases.
         *
         * @param vocabulary the number of tokens
         * @param features the model width
         * @param layers the blocks
         * @param heads the heads
         * @param headSize the features per head
         * @param ffnSize the feed-forward width
         * @param buckets the relative position buckets
         * @param maxDistance the distance beyond which positions share a bucket
         * @param epsilon the RMS norms' epsilon
         * @param gated whether the feed-forward is gated
         */
        public Settings(int vocabulary, int features, int layers, int heads, int headSize, int ffnSize,
                int buckets, int maxDistance, float epsilon, boolean gated) {
            this(vocabulary, features, layers, heads, headSize, ffnSize, buckets, maxDistance, epsilon, gated, false);
        }

        /** Checks the values. */
        public Settings {
            if (vocabulary < 1 || features < 1 || layers < 1 || heads < 1 || headSize < 1 || ffnSize < 1
                    || buckets < 2 || maxDistance < 1 || !(epsilon > 0)) {
                throw new IllegalArgumentException("invalid T5 encoder shape");
            }
        }

        /**
         * T5 1.1's defaults for position buckets, epsilon and the gated feed-forward.
         *
         * @param vocabulary the number of tokens
         * @param features the model width
         * @param layers the blocks
         * @param heads the heads
         * @param headSize the features per head
         * @param ffnSize the feed-forward width
         * @return the settings
         */
        public static Settings of(int vocabulary, int features, int layers, int heads, int headSize, int ffnSize) {
            return new Settings(vocabulary, features, layers, heads, headSize, ffnSize, 32, 128, 1e-6f, true);
        }

        /** @return the attention width, {@code heads * headSize} */
        public int inner() {
            return heads * headSize;
        }
    }

    private final Settings settings;
    private final Var embedding;
    private final Var relativeBias;
    private final List<Block> blocks = new ArrayList<>();
    private final LayerNorm finalNorm;

    /**
     * Creates an encoder with fresh weights.
     *
     * @param settings its shape
     * @param random the source of initial weights
     */
    public T5Encoder(Settings settings, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        int d = settings.features();
        embedding = Var.param(Tensor.normal(Shape.of(settings.vocabulary(), d), 0, 1f, random));
        relativeBias = settings.biasPerLayer() ? null
                : Var.param(Tensor.normal(Shape.of(settings.buckets(), settings.heads()), 0, 0.1f, random));
        for (int i = 0; i < settings.layers(); i++) {
            blocks.add(new Block(settings, random));
        }
        finalNorm = norm(settings);
    }

    private static LayerNorm norm(Settings s) {
        return new LayerNorm(new int[] {s.features()}, s.epsilon(), true, false, true);
    }

    private static Var matrix(int in, int out, SplittableRandom random) {
        return Var.param(Init.XAVIER.create(Shape.of(in, out), in, out, random));
    }

    /** One block's weights, all {@code [in, out]}. */
    public static final class Block {
        private final LayerNorm attentionNorm;
        private final Var q;
        private final Var k;
        private final Var v;
        private final Var o;
        private final LayerNorm ffnNorm;
        private final Var up;
        private final Var gate;
        private final Var down;
        private final Var relativeBias;

        Block(Settings s, SplittableRandom random) {
            int d = s.features();
            int inner = s.inner();
            relativeBias = s.biasPerLayer()
                    ? Var.param(Tensor.normal(Shape.of(s.buckets(), s.heads()), 0, 0.1f, random)) : null;
            attentionNorm = norm(s);
            // T5 does not scale its scores, so the queries start small
            q = Var.param(Tensor.normal(Shape.of(d, inner), 0, (float) (1 / Math.sqrt(d * (double) s.headSize())),
                    random));
            k = matrix(d, inner, random);
            v = matrix(d, inner, random);
            o = matrix(inner, d, random);
            ffnNorm = norm(s);
            up = matrix(d, s.ffnSize(), random);
            gate = s.gated() ? matrix(d, s.ffnSize(), random) : null;
            down = matrix(s.ffnSize(), d, random);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            if (relativeBias != null) {
                p.add(relativeBias);
            }
            p.addAll(attentionNorm.parameters());
            p.addAll(List.of(q, k, v, o));
            p.addAll(ffnNorm.parameters());
            p.add(up);
            if (gate != null) {
                p.add(gate);
            }
            p.add(down);
            return p;
        }

        /** @return this block's relative position biases {@code [buckets, heads]}, or {@code null} if shared */
        public Var relativeBias() {
            return relativeBias;
        }

        /** @return the norm before attention */
        public LayerNorm attentionNorm() {
            return attentionNorm;
        }

        /** @return the query projection */
        public Var q() {
            return q;
        }

        /** @return the key projection */
        public Var k() {
            return k;
        }

        /** @return the value projection */
        public Var v() {
            return v;
        }

        /** @return the attention output projection */
        public Var o() {
            return o;
        }

        /** @return the norm before the feed-forward */
        public LayerNorm ffnNorm() {
            return ffnNorm;
        }

        /** @return the feed-forward's activated input projection ({@code wi_0}, or {@code wi}) */
        public Var up() {
            return up;
        }

        /** @return the feed-forward's linear gate ({@code wi_1}), or {@code null} if not gated */
        public Var gate() {
            return gate;
        }

        /** @return the feed-forward's output projection ({@code wo}) */
        public Var down() {
            return down;
        }
    }

    /**
     * Encodes token ids.
     *
     * @param ids {@code [batch, tokens]} token ids as numbers; every token is attended to,
     *     padding too, as Flux runs T5
     * @param training whether the model is training
     * @return {@code [batch, tokens, features]} after the final norm
     */
    public Var encode(Var ids, boolean training) {
        Shape s = ids.shape();
        if (s.rank() != 2) {
            throw new IllegalArgumentException(this + " expects ids [batch, tokens], got " + s);
        }
        int batch = s.dim(0);
        int n = s.dim(1);
        Device device = ids.value().device();
        Tensor index = ids.value().to(DType.FLOAT32);
        Var x = Parts.step(() -> embedding.gather(0, index));
        Tensor buckets = Tensor.of(Shape.of(n * n), relativeBuckets(n, settings.buckets(), settings.maxDistance()),
                device);
        // [heads, n, n], shared by every block unless each has its own
        Var shared = relativeBias == null ? null : Parts.step(() -> biases(relativeBias, buckets, n));
        for (Block b : blocks) {
            Var in = x;
            Var bias = shared != null ? shared : Parts.step(() -> biases(b.relativeBias, buckets, n));
            x = Parts.step(() -> block(b, in, bias, batch, n));
            Parts.free(in);
            if (shared == null) {
                Parts.free(bias);
            }
        }
        Var bias = shared;
        Var h = x;
        Var out = Parts.step(() -> finalNorm.forward(h));
        Parts.free(h);
        Parts.free(bias);
        buckets.release();
        return out;
    }

    private Var biases(Var table, Tensor buckets, int n) {
        return table.gather(0, buckets).reshape(n, n, settings.heads()).permute(2, 0, 1).to(DType.FLOAT32);
    }

    private Var block(Block b, Var x, Var bias, int batch, int n) {
        int heads = settings.heads();
        Var a = b.attentionNorm.forward(x);
        Var q = a.matmul(b.q).reshape(batch, n, heads, settings.headSize()).permute(0, 2, 1, 3);
        Var k = a.matmul(b.k).reshape(batch, n, heads, settings.headSize()).permute(0, 2, 1, 3);
        Var v = a.matmul(b.v).reshape(batch, n, heads, settings.headSize()).permute(0, 2, 1, 3);
        Var attended;
        if (!bias.requiresGrad()) {
            attended = Functional.attention(q, k, v, bias.value().to(q.value().dtype()), 1f, 0f, null);
        } else {
            // the position biases learn too: scores in float32, as T5 computes its softmax
            Var scores = q.matmul(k.transpose()).to(DType.FLOAT32).add(bias);
            attended = Activation.SOFTMAX.forward(scores).to(v.value().dtype()).matmul(v);
        }
        Var y = attended.permute(0, 2, 1, 3).reshape(batch, n, settings.inner()).matmul(b.o);
        x = x.add(y);
        Var f = b.ffnNorm.forward(x);
        Var hidden = b.gate != null
                ? Activation.GELU_TANH.forward(f.matmul(b.up)).mul(f.matmul(b.gate))
                : f.matmul(b.up).relu();
        return x.add(hidden.matmul(b.down));
    }

    /**
     * T5's bidirectional relative position buckets: half the buckets for keys after the
     * query, half for keys before; within each half, distances below a quarter of the buckets
     * get their own bucket and larger ones share buckets on a log scale up to {@code maxDistance}.
     *
     * @param n the sequence length
     * @param buckets the number of buckets
     * @param maxDistance the distance at which the largest bucket starts
     * @return the bucket of each (query, key) pair, row-major {@code [n * n]}, as numbers
     */
    public static float[] relativeBuckets(int n, int buckets, int maxDistance) {
        int half = buckets / 2;
        int exact = half / 2;
        float[] out = new float[n * n];
        for (int qi = 0; qi < n; qi++) {
            for (int ki = 0; ki < n; ki++) {
                int rel = ki - qi;
                int bucket = rel > 0 ? half : 0;
                int d = Math.abs(rel);
                if (d < exact) {
                    bucket += d;
                } else {
                    // in float32, as transformers computes it
                    float log = (float) Math.log((float) d / exact);
                    float large = log / (float) Math.log((double) maxDistance / exact) * (half - exact);
                    int b = exact + (int) large;
                    bucket += Math.min(b, half - 1);
                }
                out[qi * n + ki] = bucket;
            }
        }
        return out;
    }

    @Override
    public Var forward(Var input) {
        return encode(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        return encode(input, training);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), settings.features());
    }

    /** Embedding, the shared relative position bias (if any), each block's parameters, then the final norm. */
    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embedding));
        if (relativeBias != null) {
            p.add(relativeBias);
        }
        for (Block b : blocks) {
            p.addAll(b.parameters());
        }
        p.addAll(finalNorm.parameters());
        return p;
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    /** @return the type the weights are held in */
    public DType dtype() {
        return embedding.value().dtype();
    }

    /** @return the token embeddings {@code [vocabulary, features]} */
    public Var embedding() {
        return embedding;
    }

    /** @return the relative position biases {@code [buckets, heads]} shared by all blocks, or {@code null} for umT5 */
    public Var relativeBias() {
        return relativeBias;
    }

    /** @return the blocks */
    public List<Block> blocks() {
        return List.copyOf(blocks);
    }

    /** @return the final RMS norm */
    public LayerNorm finalNorm() {
        return finalNorm;
    }

    @Override
    public String toString() {
        return "T5Encoder(" + settings.features() + ", " + settings.layers() + " layers)";
    }
}
