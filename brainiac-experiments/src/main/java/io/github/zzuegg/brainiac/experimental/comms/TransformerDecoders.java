package io.github.zzuegg.brainiac.experimental.comms;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes;
import io.github.zzuegg.brainiac.experimental.data.ChannelCodes.LinearCode;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Transformer decoders for binary linear codes, both model-free and code-aware: the inputs are
 * the channel magnitudes {@code |y|} (n values) and the bipolar syndrome of the hard decisions
 * (n-k values), so the decoder is invariant to the transmitted codeword and trains on noisy
 * all-zero words; it predicts, per bit, whether the hard decision is flipped (the
 * multiplicative noise), and the decoded word flips where that probability exceeds one half.
 *
 * <ul>
 *   <li>{@link Kind#ECCT}: one sequence of n + (n-k) tokens (value times a learned per-position
 *       embedding) through self-attention masked by the code: two tokens may attend when they
 *       are connected in the Tanner graph or share a neighbour (Choukroun and Wolf, NeurIPS 2022,
 *       <a href="https://arxiv.org/abs/2203.14966">arXiv:2203.14966</a>).</li>
 *   <li>{@link Kind#CROSS_MPT}: magnitudes and syndrome stay two sequences that alternately
 *       update each other by masked cross-attention (magnitudes query the syndrome under
 *       {@code H^T}, the syndrome queries magnitudes under {@code H}), mimicking message passing
 *       with one shared attention and feed-forward block per layer (Park et al., ICLR 2025,
 *       <a href="https://arxiv.org/abs/2405.01033">arXiv:2405.01033</a>).</li>
 * </ul>
 *
 * <p>Attention is the core {@link Functional#attention} with the masks passed as additive
 * biases.
 */
public final class TransformerDecoders {

    /** The architecture. */
    public enum Kind {
        /** masked self-attention over magnitudes and syndrome. */
        ECCT,
        /** alternating masked cross-attention between the two sequences. */
        CROSS_MPT
    }

    private final Kind kind;
    private final LinearCode code;
    private final int d;
    private final int layers;
    private final int n;
    private final int m;
    private final Var embed;
    private final List<Var[]> blocks = new ArrayList<>();
    private final Var outW;
    private final Var outB;
    private final Tensor selfMask;
    private final Tensor magMask;
    private final Tensor synMask;

    /**
     * Creates an untrained decoder.
     *
     * @param kind the architecture
     * @param code the code
     * @param dim the model width
     * @param layers the number of layers
     * @param seed the seed
     */
    public TransformerDecoders(Kind kind, LinearCode code, int dim, int layers, long seed) {
        this.kind = kind;
        this.code = code;
        this.d = dim;
        this.layers = layers;
        this.n = code.n();
        this.m = code.h().length;
        SplittableRandom rnd = new SplittableRandom(seed);
        embed = Var.param(rand(Shape.of(n + m, d), 1.0, rnd));
        for (int l = 0; l < layers; l++) {
            double s = 1 / Math.sqrt(d);
            blocks.add(new Var[] {Var.param(rand(Shape.of(d, d), s, rnd)), Var.param(rand(Shape.of(d, d), s, rnd)),
                Var.param(rand(Shape.of(d, d), s, rnd)), Var.param(rand(Shape.of(d, d), s, rnd)),
                Var.param(rand(Shape.of(d, 2 * d), s, rnd)), Var.param(Tensor.zeros(Shape.of(2 * d))),
                Var.param(rand(Shape.of(2 * d, d), 1 / Math.sqrt(2 * d), rnd))});
        }
        outW = Var.param(rand(Shape.of(d, 1), 1 / Math.sqrt(d), rnd));
        outB = Var.param(Tensor.zeros(Shape.of(1)));
        int[][] h = code.h();
        // ECCT mask over [magnitudes | syndrome]
        float[] self = new float[(n + m) * (n + m)];
        java.util.Arrays.fill(self, -1e9f);
        for (int i = 0; i < n + m; i++) {
            self[i * (n + m) + i] = 0;
        }
        for (int c = 0; c < m; c++) {
            for (int v = 0; v < n; v++) {
                if (h[c][v] == 0) {
                    continue;
                }
                self[v * (n + m) + n + c] = 0;
                self[(n + c) * (n + m) + v] = 0;
                for (int v2 = 0; v2 < n; v2++) {
                    if (h[c][v2] != 0) {
                        self[v * (n + m) + v2] = 0;
                    }
                }
                for (int c2 = 0; c2 < m; c2++) {
                    if (h[c2][v] != 0) {
                        self[(n + c) * (n + m) + n + c2] = 0;
                    }
                }
            }
        }
        selfMask = Tensor.of(Shape.of(1, 1, n + m, n + m), self);
        float[] mag = new float[n * m];
        float[] syn = new float[m * n];
        for (int v = 0; v < n; v++) {
            for (int c = 0; c < m; c++) {
                mag[v * m + c] = h[c][v] != 0 ? 0 : -1e9f;
                syn[c * n + v] = h[c][v] != 0 ? 0 : -1e9f;
            }
        }
        magMask = Tensor.of(Shape.of(1, 1, n, m), mag);
        synMask = Tensor.of(Shape.of(1, 1, m, n), syn);
    }

    private static Tensor rand(Shape shape, double scale, SplittableRandom rnd) {
        float[] v = new float[(int) shape.size()];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (scale * ChannelCodes.gaussian(rnd));
        }
        return Tensor.of(shape, v);
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        p.add(embed);
        for (Var[] b : blocks) {
            p.addAll(List.of(b));
        }
        p.add(outW);
        p.add(outB);
        return p;
    }

    /** @return the number of trainable scalars */
    public long parameterCount() {
        long c = 0;
        for (Var v : parameters()) {
            c += v.value().size();
        }
        return c;
    }

    private static Var layerNorm(Var x) {
        Var mu = x.mean(-1, true);
        Var c = x.sub(mu);
        return c.div(c.square().mean(-1, true).add(1e-5f).sqrt());
    }

    private Var linear(Var x, Var w) {
        int[] s = x.shape().toArray();
        int rows = 1;
        for (int i = 0; i < s.length - 1; i++) {
            rows *= s[i];
        }
        int[] outShape = s.clone();
        outShape[s.length - 1] = w.shape().dim(1);
        return x.reshape(rows, s[s.length - 1]).matmul(w).reshape(outShape);
    }

    /** Pre-norm attention + feed-forward; queries from {@code q}, keys/values from {@code kv}. */
    private Var block(Var[] w, Var q, Var kv, Tensor mask) {
        int batch = q.shape().dim(0);
        int tq = q.shape().dim(1);
        int tk = kv.shape().dim(1);
        Var qn = layerNorm(q);
        Var kn = kv == q ? qn : layerNorm(kv);
        Var qq = linear(qn, w[0]).reshape(batch, 1, tq, d);
        Var kk = linear(kn, w[1]).reshape(batch, 1, tk, d);
        Var vv = linear(kn, w[2]).reshape(batch, 1, tk, d);
        Var att = Functional.attention(qq, kk, vv, mask, (float) (1 / Math.sqrt(d)), 0f, null).reshape(batch, tq, d);
        Var x = q.add(linear(att, w[3]));
        Var hdn = linear(layerNorm(x), w[4]).add(w[5]).relu();
        return x.add(linear(hdn, w[6]));
    }

    /**
     * The flip logits.
     *
     * @param y received BPSK samples {@code [batch][n]}
     * @return {@code [batch, n]} logits that each hard decision is wrong
     */
    public Var forward(double[][] y) {
        int batch = y.length;
        float[] mag = new float[batch * n];
        float[] syn = new float[batch * m];
        for (int b = 0; b < batch; b++) {
            int[] hard = new int[n];
            for (int v = 0; v < n; v++) {
                mag[b * n + v] = (float) Math.abs(y[b][v]);
                hard[v] = y[b][v] < 0 ? 1 : 0;
            }
            int[] s = code.syndrome(hard);
            for (int c = 0; c < m; c++) {
                syn[b * m + c] = 1 - 2 * s[c];
            }
        }
        Var magTokens = Var.of(Tensor.of(Shape.of(batch, n, 1), mag)).mul(embed.slice(0, 0, n));
        Var synTokens = Var.of(Tensor.of(Shape.of(batch, m, 1), syn)).mul(embed.slice(0, n, n + m));
        Var magOut;
        if (kind == Kind.ECCT) {
            Var x = Var.concat(1, magTokens, synTokens);
            for (Var[] w : blocks) {
                x = block(w, x, x, selfMask);
            }
            magOut = x.slice(1, 0, n);
        } else {
            Var a = magTokens;
            Var s = synTokens;
            for (Var[] w : blocks) {
                a = block(w, a, s, magMask);
                s = block(w, s, a, synMask);
            }
            magOut = a;
        }
        return linear(layerNorm(magOut), outW).reshape(batch, n).add(outB);
    }

    /**
     * Trains on noisy all-zero codewords at random {@code Eb/N0} in {@code [loDb, hiDb]}.
     *
     * @param steps gradient steps
     * @param batch words per step
     * @param loDb the lowest SNR
     * @param hiDb the highest SNR
     * @param learningRate the Adam step size
     * @param seed the seed
     * @return the final loss
     */
    public double train(int steps, int batch, double loDb, double hiDb, float learningRate, long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom rnd = new SplittableRandom(seed);
        double last = 0;
        for (int step = 0; step < steps; step++) {
            double[][] y = new double[batch][n];
            float[] target = new float[batch * n];
            for (int b = 0; b < batch; b++) {
                double sigma = ChannelCodes.sigma(loDb + (hiDb - loDb) * rnd.nextDouble(), code.rate());
                for (int v = 0; v < n; v++) {
                    y[b][v] = 1 + sigma * ChannelCodes.gaussian(rnd);
                    target[b * n + v] = y[b][v] < 0 ? 1 : 0;
                }
            }
            Var z = forward(y);
            Var t = Var.of(Tensor.of(Shape.of(batch, n), target));
            Var loss = z.softplus().sub(z.mul(t)).mean();
            loss.backward();
            last = loss.value().item();
            opt.step(parameters());
        }
        return last;
    }

    /**
     * Decodes received words.
     *
     * @param y received BPSK samples {@code [batch][n]}
     * @return decoded codeword bits
     */
    public int[][] decode(double[][] y) {
        float[] z = Var.noGrad(() -> forward(y)).value().toArray();
        int[][] out = new int[y.length][n];
        for (int b = 0; b < y.length; b++) {
            for (int v = 0; v < n; v++) {
                int hard = y[b][v] < 0 ? 1 : 0;
                out[b][v] = z[b * n + v] > 0 ? 1 - hard : hard;
            }
        }
        return out;
    }
}
