package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Supervised deep hashing: an MLP maps inputs to {@code bits} outputs whose signs are the code.
 *
 * <ul>
 *   <li>{@link Loss#DSH}: deep supervised hashing, a contrastive loss on real-valued outputs
 *       (similar pairs pulled together, dissimilar pushed beyond margin {@code 2 bits}) plus
 *       {@code alpha |  |b| - 1 |_1} pulling outputs toward {@code +-1};
 *   <li>{@link Loss#HASHNET}: outputs {@code tanh(beta z)} with {@code beta} raised over
 *       training (continuation, converging to sign), and a weighted pairwise logistic
 *       likelihood of similarity on {@code alpha <h_i, h_j>}, similar and dissimilar pairs
 *       weighted by the inverse of their share in the batch.
 * </ul>
 *
 * <p>{@link Codes} packs codes into {@code long} words and ranks by Hamming distance
 * ({@link Long#bitCount}).
 *
 * <p>Papers: Liu et al., "Deep Supervised Hashing for Fast Image Retrieval", CVPR 2016,
 * <a href="https://doi.org/10.1109/CVPR.2016.227">doi:10.1109/CVPR.2016.227</a>; Cao et al.,
 * "HashNet: Deep Learning to Hash by Continuation", ICCV 2017,
 * <a href="https://arxiv.org/abs/1702.00758">arXiv:1702.00758</a>.
 */
public final class DeepHashing {

    /** Training objectives. */
    public enum Loss {
        /** contrastive loss with a binarising regulariser */
        DSH,
        /** continuation with weighted pairwise likelihood */
        HASHNET
    }

    private final Loss loss;
    private final int bits;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private double beta = 1;

    /**
     * Creates a network.
     *
     * @param loss the objective
     * @param inputs the input width
     * @param hidden the hidden width
     * @param bits the code length
     * @param seed the seed
     */
    public DeepHashing(Loss loss, int inputs, int hidden, int bits, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.loss = loss;
        this.bits = bits;
        w1 = Var.param(Tensor.normal(Shape.of(inputs, hidden), 0, (float) Math.sqrt(2.0 / inputs), r));
        b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
        w2 = Var.param(Tensor.normal(Shape.of(hidden, bits), 0, (float) Math.sqrt(1.0 / hidden), r));
        b2 = Var.param(Tensor.zeros(Shape.of(bits)));
    }

    private Var raw(Var x) {
        return x.matmul(w1).add(b1).relu().matmul(w2).add(b2);
    }

    /**
     * Trains on labelled examples.
     *
     * @param x {@code [n][inputs]}
     * @param labels class ids
     * @param steps steps
     * @param batch batch size
     * @param lr the learning rate
     * @param seed the seed
     */
    public void train(float[][] x, int[] labels, int steps, int batch, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        List<Var> params = List.of(w1, b1, w2, b2);
        int d = x[0].length;
        for (int s = 0; s < steps; s++) {
            float[] xb = new float[batch * d];
            int[] yb = new int[batch];
            for (int i = 0; i < batch; i++) {
                int k = r.nextInt(x.length);
                System.arraycopy(x[k], 0, xb, i * d, d);
                yb[i] = labels[k];
            }
            float[] sim = new float[batch * batch];
            int similar = 0;
            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < batch; j++) {
                    sim[i * batch + j] = yb[i] == yb[j] ? 1 : 0;
                    similar += yb[i] == yb[j] ? 1 : 0;
                }
            }
            Tensor st = Tensor.of(Shape.of(batch, batch), sim);
            Var z = raw(Var.of(Tensor.of(Shape.of(batch, d), xb)));
            Var l;
            if (loss == Loss.DSH) {
                Var sq = z.square().sum(1, true);
                Var dist = sq.add(sq.transpose()).sub(z.matmul(z.transpose()).mul(2f));      // [b, b]
                Var pull = dist.mul(st);
                Var push = dist.mul(-1f).add(2f * bits).relu().mul(Var.of(st).mul(-1f).add(1f));
                Var reg = z.square().sqrt().sub(1f).square().add(1e-6f).sqrt().sum();
                l = pull.add(push).sum().div(2f * batch * batch).add(reg.mul(0.01f / batch));
            } else {
                beta = Math.sqrt(1 + 0.05 * s);
                Var h = z.mul((float) beta).tanh();
                float alpha = 10f / bits;
                Var inner = h.matmul(h.transpose()).mul(alpha);
                float total = batch * batch;
                float[] w = new float[sim.length];
                for (int i = 0; i < sim.length; i++) {
                    w[i] = sim[i] > 0 ? total / Math.max(1, similar) : total / Math.max(1, total - similar);
                }
                // log(1 + exp(a)) = relu(a) + log(1 + exp(-|a|))
                Var abs = inner.square().add(1e-8f).sqrt();
                Var softplus = inner.relu().add(abs.mul(-1f).exp().add(1f).log());
                l = softplus.sub(inner.mul(st)).mul(Tensor.of(Shape.of(batch, batch), w)).mean();
            }
            params.forEach(Var::zeroGrad);
            l.backward();
            opt.step(params);
        }
        params.forEach(Var::zeroGrad);
    }

    /**
     * The real-valued outputs before binarisation.
     *
     * @param x {@code [n][inputs]}
     * @return {@code [n][bits]}
     */
    public float[][] embed(float[][] x) {
        int d = x[0].length;
        float[] all = new float[x.length * d];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, all, i * d, d);
        }
        float[] z = Var.noGrad(() -> raw(Var.of(Tensor.of(Shape.of(x.length, d), all)))).value().toArray();
        float[][] rows = new float[x.length][bits];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(z, i * bits, rows[i], 0, bits);
        }
        return rows;
    }

    /**
     * Encodes inputs.
     *
     * @param x {@code [n][inputs]}
     * @return the packed codes
     */
    public Codes encode(float[][] x) {
        return Codes.ofSigns(embed(x));
    }

    /** Packed binary codes. */
    public static final class Codes {
        private final long[][] words;
        private final int bits;

        private Codes(long[][] words, int bits) {
            this.words = words;
            this.bits = bits;
        }

        /**
         * @param values per item its real values; bit {@code k} is set when value {@code k > 0}
         * @return the codes
         */
        public static Codes ofSigns(float[][] values) {
            int bits = values[0].length;
            int n = (bits + 63) / 64;
            long[][] w = new long[values.length][n];
            for (int i = 0; i < values.length; i++) {
                for (int k = 0; k < bits; k++) {
                    if (values[i][k] > 0) {
                        w[i][k >> 6] |= 1L << (k & 63);
                    }
                }
            }
            return new Codes(w, bits);
        }

        /** @return the number of codes */
        public int size() {
            return words.length;
        }

        /** @return the code length */
        public int bits() {
            return bits;
        }

        /**
         * @param i an item
         * @param other another code set
         * @param j an item there
         * @return the Hamming distance
         */
        public int hamming(int i, Codes other, int j) {
            int d = 0;
            for (int k = 0; k < words[i].length; k++) {
                d += Long.bitCount(words[i][k] ^ other.words[j][k]);
            }
            return d;
        }

        /**
         * Ranks this set for a query by Hamming distance (counting sort, ties by index).
         *
         * @param query the query codes
         * @param q the query item
         * @return the item order
         */
        public int[] rank(Codes query, int q) {
            List<List<Integer>> buckets = new ArrayList<>();
            for (int b = 0; b <= bits; b++) {
                buckets.add(new ArrayList<>());
            }
            for (int i = 0; i < words.length; i++) {
                buckets.get(query.hamming(q, this, i)).add(i);
            }
            int[] out = new int[words.length];
            int k = 0;
            for (List<Integer> b : buckets) {
                for (int i : b) {
                    out[k++] = i;
                }
            }
            return out;
        }
    }
}
