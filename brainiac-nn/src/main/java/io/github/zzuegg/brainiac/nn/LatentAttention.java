package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Multi-head latent attention (DeepSeek-V2, Liu et al., 2024): keys and values come from one
 * small latent per step, shared by every head, so a decoding cache keeps {@code kvRank + rope}
 * numbers per step instead of keys and values for every head. Queries come straight from the
 * input or through a low-rank step of their own; each query and key has a non-rotary part and
 * a rotary part, and the rotary key is also shared by every head.
 * <pre>
 * q        = x Wq                    (or RMSNorm(x Wq_a) Wq_b)   per head [nope | rope]
 * [c | kr] = x Wkv_a;  c = RMSNorm(c)
 * [k | v]  = c Wkv_b                 per head, the key's rope part being rotary(kr)
 * out      = attention(q, k, v) Wo
 * </pre>
 * The whole-sequence {@link #forward(Var)} (training) expands the latent as written. Decoding
 * with a {@link Cache} keeps the latents and rotary keys only, and moves {@code Wkv_b} into
 * the queries and the output instead ({@link Functional#latentAttention}), as DeepSeek serves
 * it; both give the same result.
 *
 * <p>Input and output are {@code [batch, length, features]}. Linear weights are {@code [in,
 * out]}, without biases.
 */
public final class LatentAttention implements Layer {

    /**
     * The layer's shape.
     *
     * @param heads the attention heads
     * @param qRank the queries' low rank, or 0 for queries straight from the input
     * @param kvRank the latent's size
     * @param nope each head's non-rotary query and key features
     * @param rope the rotary features, even
     * @param valueDim each head's value features
     * @param rotaryBase the rotary frequency base, such as 10000
     * @param interleaved whether rotary features {@code 2i} and {@code 2i + 1} turn together
     *     (DeepSeek's weights) rather than {@code i} and {@code i + rope / 2}
     * @param causal whether a step sees only the steps up to itself
     * @param eps the RMS norms' epsilon
     */
    public record Settings(int heads, int qRank, int kvRank, int nope, int rope, int valueDim, float rotaryBase,
            boolean interleaved, boolean causal, float eps) {
        /** Checks the values. */
        public Settings {
            if (heads < 1 || qRank < 0 || kvRank < 1 || nope < 1 || rope < 2 || rope % 2 != 0 || valueDim < 1
                    || !(rotaryBase > 0) || !(eps > 0)) {
                throw new IllegalArgumentException("latent attention with " + heads + " heads, ranks " + qRank + " and "
                        + kvRank + ", " + nope + " + " + rope + " query features (rope even), values of " + valueDim
                        + ", base " + rotaryBase + ", eps " + eps);
            }
        }

        /**
         * DeepSeek-V2-Lite's proportions for a width: direct queries, a latent of 512, heads
         * of 128 + 64, causal, interleaved rotary features.
         *
         * @param heads the heads
         * @return the settings
         */
        public static Settings deepSeek(int heads) {
            return new Settings(heads, 0, 512, 128, 64, 128, 10_000f, true, true, 1e-6f);
        }

        /** @return the query and key features of a head */
        public int queryDim() {
            return nope + rope;
        }
    }

    /**
     * A decoding cache: the latents {@code [batch, 1, steps, kvRank]} and rotary keys {@code
     * [batch, 1, steps, rope]} seen so far. Each {@link #forward(Var, Cache)} appends to it.
     */
    public static final class Cache {
        private Tensor latents;
        private Tensor rotary;

        /** An empty cache; {@link LatentAttention#newCache} makes one too. */
        public Cache() {
        }

        /** @return the steps cached */
        public int length() {
            return latents == null ? 0 : latents.shape().dim(2);
        }

        /** @return the latents {@code [batch, 1, steps, kvRank]}, or {@code null} while empty */
        public Tensor latents() {
            return latents;
        }

        /** @return the rotary keys {@code [batch, 1, steps, rope]}, or {@code null} while empty */
        public Tensor rotary() {
            return rotary;
        }
    }

    private final Settings settings;
    private final Var wq;
    private final Var wqA;
    private final Var qNorm;
    private final Var wqB;
    private final Var wkvA;
    private final Var kvNorm;
    private final Var wkvB;
    private final Var wo;

    /**
     * A layer from given weights.
     *
     * @param settings the shape
     * @param wq the query projection {@code [features, heads * (nope + rope)]}, or {@code
     *     null} with a query rank
     * @param wqA the queries' down projection {@code [features, qRank]}, or {@code null}
     * @param qNorm its RMS norm's weight {@code [qRank]}, or {@code null}
     * @param wqB the queries' up projection {@code [qRank, heads * (nope + rope)]}, or {@code null}
     * @param wkvA the latent and rotary key projection {@code [features, kvRank + rope]}
     * @param kvNorm the latent's RMS norm weight {@code [kvRank]}
     * @param wkvB the latent's up projection {@code [kvRank, heads * (nope + valueDim)]}
     * @param wo the output projection {@code [heads * valueDim, features]}
     */
    public LatentAttention(Settings settings, Var wq, Var wqA, Var qNorm, Var wqB, Var wkvA, Var kvNorm, Var wkvB,
            Var wo) {
        this.settings = settings;
        int d = wkvA.shape().dim(0);
        int qOut = settings.heads() * settings.queryDim();
        boolean low = settings.qRank() > 0;
        boolean ok = low ? wq == null && wqA != null && qNorm != null && wqB != null
                && wqA.shape().equals(Shape.of(d, settings.qRank())) && qNorm.shape().equals(Shape.of(settings.qRank()))
                && wqB.shape().equals(Shape.of(settings.qRank(), qOut))
                : wq != null && wqA == null && qNorm == null && wqB == null && wq.shape().equals(Shape.of(d, qOut));
        ok = ok && wkvA.shape().equals(Shape.of(d, settings.kvRank() + settings.rope()))
                && kvNorm.shape().equals(Shape.of(settings.kvRank()))
                && wkvB.shape().equals(Shape.of(settings.kvRank(), settings.heads() * (settings.nope() + settings.valueDim())))
                && wo.shape().equals(Shape.of(settings.heads() * settings.valueDim(), d));
        if (!ok) {
            throw new IllegalArgumentException("weights do not fit " + settings + " over " + d + " features");
        }
        this.wq = wq;
        this.wqA = wqA;
        this.qNorm = qNorm;
        this.wqB = wqB;
        this.wkvA = wkvA;
        this.kvNorm = kvNorm;
        this.wkvB = wkvB;
        this.wo = wo;
    }

    /**
     * A layer with random weights.
     *
     * @param settings the shape
     * @param features the input's features
     * @param random the random source
     */
    public LatentAttention(Settings settings, int features, SplittableRandom random) {
        this(settings,
                settings.qRank() > 0 ? null : dense(features, settings.heads() * settings.queryDim(), random),
                settings.qRank() > 0 ? dense(features, settings.qRank(), random) : null,
                settings.qRank() > 0 ? ones(settings.qRank()) : null,
                settings.qRank() > 0 ? dense(settings.qRank(), settings.heads() * settings.queryDim(), random) : null,
                dense(features, settings.kvRank() + settings.rope(), random), ones(settings.kvRank()),
                dense(settings.kvRank(), settings.heads() * (settings.nope() + settings.valueDim()), random),
                dense(settings.heads() * settings.valueDim(), features, random));
    }

    private static Var dense(int in, int out, SplittableRandom random) {
        return Var.param(Init.XAVIER.create(Shape.of(in, out), in, out, random));
    }

    private static Var ones(int n) {
        return Var.param(Tensor.ones(Shape.of(n)));
    }

    /**
     * Describes a layer over sequences {@code [length, features]}.
     *
     * @param settings the shape
     * @return the spec
     */
    public static LayerSpec of(Settings settings) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                check(input);
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                check(input);
                return new LatentAttention(settings, input.dim(1), random);
            }
        };
    }

    private static void check(Shape input) {
        if (input.rank() != 2) {
            throw new IllegalArgumentException("latent attention takes [length, features], got " + input);
        }
    }

    /** @return the shape */
    public Settings settings() {
        return settings;
    }

    /** @return the input's features */
    public int features() {
        return wkvA.shape().dim(0);
    }

    /** @return an empty decoding cache */
    public Cache newCache() {
        return new Cache();
    }

    /**
     * Runs the layer over whole sequences, the latent expanded to every head (training).
     *
     * @param input {@code [batch, length, features]}
     * @return {@code [batch, length, features]}
     */
    @Override
    public Var forward(Var input) {
        Settings s = settings;
        int batch = input.shape().dim(0);
        int len = input.shape().dim(1);
        Var[] q = queries(input, 0);
        Var[] kv = latent(input, 0);
        Var up = kv[0].reshape(batch, len, s.kvRank()).matmul(wkvB)
                .reshape(batch, len, s.heads(), s.nope() + s.valueDim()).permute(0, 2, 1, 3);
        Var k = Var.concat(-1, up.slice(-1, 0, s.nope()), kv[1].expand(Shape.of(batch, s.heads(), len, s.rope())));
        Var v = up.slice(-1, s.nope(), s.nope() + s.valueDim());
        Var a = Functional.attention(Var.concat(-1, q[0], q[1]), k, v, s.causal() && len > 1, scale());
        return output(a, batch, len);
    }

    /**
     * Runs the layer over new steps, attending to them and to every step in the cache, which
     * then holds them too: the absorbed form, over the latents themselves. Steps continue the
     * cache's positions; within the new steps, attention follows {@link Settings#causal}.
     *
     * @param input {@code [batch, length, features]}
     * @param cache the cache, which this extends
     * @return {@code [batch, length, features]}
     */
    public Var forward(Var input, Cache cache) {
        Settings s = settings;
        int batch = input.shape().dim(0);
        int len = input.shape().dim(1);
        int past = cache.length();
        Var[] q = queries(input, past);
        Var[] kv = latent(input, past);
        cache.latents = past == 0 ? kv[0].value() : Tensor.concat(2, cache.latents, kv[0].value());
        cache.rotary = past == 0 ? kv[1].value() : Tensor.concat(2, cache.rotary, kv[1].value());
        Tensor[] w = Functional.absorbLatentUp(wkvB.value(), s.heads(), s.nope(), s.valueDim());
        Var qLatent = q[0].matmul(Var.of(w[0]));
        int total = past + len;
        Tensor bias = s.causal() && len > 1
                ? Functional.causalMask(len, total, input.value().device()).to(input.value().dtype()) : null;
        Var out = Functional.latentAttention(qLatent, q[1], Var.of(cache.latents), Var.of(cache.rotary), bias,
                scale());
        return output(out.matmul(Var.of(w[1])), batch, len);
    }

    private float scale() {
        return (float) (1 / Math.sqrt(settings.queryDim()));
    }

    /** {@code {non-rotary, rotary}} queries {@code [batch, heads, length, *]}. */
    private Var[] queries(Var x, int offset) {
        Settings s = settings;
        int batch = x.shape().dim(0);
        int len = x.shape().dim(1);
        Var q = wq != null ? x.matmul(wq) : rmsNorm(x.matmul(wqA), qNorm).matmul(wqB);
        q = q.reshape(batch, len, s.heads(), s.queryDim()).permute(0, 2, 1, 3);
        return new Var[] {q.slice(-1, 0, s.nope()), rotate(q.slice(-1, s.nope(), s.queryDim()), offset)};
    }

    /** {@code {latents [batch, 1, length, kvRank], rotary keys [batch, 1, length, rope]}}. */
    private Var[] latent(Var x, int offset) {
        Settings s = settings;
        int batch = x.shape().dim(0);
        int len = x.shape().dim(1);
        Var kv = x.matmul(wkvA);
        Var c = rmsNorm(kv.slice(-1, 0, s.kvRank()), kvNorm).reshape(batch, 1, len, s.kvRank());
        Var r = rotate(kv.slice(-1, s.kvRank(), s.kvRank() + s.rope()).reshape(batch, 1, len, s.rope()), offset);
        return new Var[] {c, r};
    }

    private Var rotate(Var x, int offset) {
        return settings.interleaved() ? Functional.rotaryInterleaved(x, settings.rotaryBase(), offset)
                : Functional.rotary(x, settings.rotaryBase(), offset);
    }

    private Var rmsNorm(Var x, Var weight) {
        int last = x.shape().rank() - 1;
        return x.div(x.square().mean(last, true).add(settings.eps()).sqrt()).mul(weight);
    }

    private Var output(Var a, int batch, int len) {
        return a.permute(0, 2, 1, 3).reshape(batch, len, settings.heads() * settings.valueDim()).matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (wq != null) {
            p.add(wq);
        } else {
            p.add(wqA);
            p.add(qNorm);
            p.add(wqB);
        }
        p.add(wkvA);
        p.add(kvNorm);
        p.add(wkvB);
        p.add(wo);
        return p;
    }

    @Override
    public String toString() {
        Settings s = settings;
        return "LatentAttention[" + features() + " features, " + s.heads() + " heads, latent " + s.kvRank()
                + (s.qRank() > 0 ? ", query rank " + s.qRank() : "") + ", " + s.nope() + " + " + s.rope()
                + " query features, values of " + s.valueDim() + (s.causal() ? ", causal" : "") + "]";
    }

    static Config write(LatentAttention a) {
        Settings s = a.settings;
        return Config.builder().put("features", a.features()).put("heads", s.heads()).put("qRank", s.qRank())
                .put("kvRank", s.kvRank()).put("nope", s.nope()).put("rope", s.rope()).put("valueDim", s.valueDim())
                .put("rotaryBase", s.rotaryBase()).put("interleaved", s.interleaved()).put("causal", s.causal())
                .put("eps", s.eps()).build();
    }

    static LatentAttention read(Config c) {
        Settings s = new Settings(c.getInt("heads"), c.getInt("qRank"), c.getInt("kvRank"), c.getInt("nope"),
                c.getInt("rope"), c.getInt("valueDim"), c.getFloat("rotaryBase"), c.getBoolean("interleaved"),
                c.getBoolean("causal"), c.getFloat("eps"));
        return new LatentAttention(s, c.getInt("features"), new SplittableRandom());
    }
}
