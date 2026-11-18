package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GruCell;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Temporal link predictors for continuous-time edge streams. Every model reads the stream in
 * time order: {@link LinkModel#score} ranks candidate destinations for sources at a query time
 * using only the events observed before it, and {@link LinkModel#observe} then reveals a batch of
 * events. Learned models are fitted on a training prefix with random negatives, in time order, so
 * nothing from the future leaks into an embedding.
 * <ul>
 *   <li>{@link #edgeBank}: memorises past edges (Poursafaei et al., NeurIPS 2022,
 *       <a href="https://arxiv.org/abs/2207.10128">arXiv:2207.10128</a>), with unlimited memory or
 *       a time window;</li>
 *   <li>{@link #tgn}: a memory vector per node, updated by a GRU from the last message of each
 *       batch ({@code [s_self, s_other, cos(w dt + b)]}), with the update of the previous batch
 *       recomputed inside the current batch's graph so the GRU is trained; node embeddings by
 *       TGAT-style attention over the most recent neighbours with a learned functional time
 *       encoding (Rossi et al. 2020, <a href="https://arxiv.org/abs/2006.10637">arXiv:2006.10637</a>;
 *       Xu et al., ICLR 2020, <a href="https://arxiv.org/abs/2002.07962">arXiv:2002.07962</a>);</li>
 *   <li>{@link #graphMixer}: an MLP-Mixer over the fixed time encodings of the most recent links
 *       plus the mean of the recent neighbours' embeddings (Cong et al., ICLR 2023,
 *       <a href="https://arxiv.org/abs/2302.11636">arXiv:2302.11636</a>);</li>
 *   <li>{@link #hyperEvent}: each query event is described by the recent events of both endpoints,
 *       each encoded only relationally (does it touch the source, the destination, is it the query
 *       pair, how often its far end co-occurs with either endpoint in 1 and 2 hops, how long ago),
 *       and a self-attention layer classifies that set (after HyperEvent, Gao et al. 2025,
 *       <a href="https://arxiv.org/abs/2507.11836">arXiv:2507.11836</a>); no node identities, so
 *       it is inductive.</li>
 * </ul>
 */
public final class TemporalGraphModels {

    private TemporalGraphModels() {
    }

    /** A streaming link predictor. */
    public interface LinkModel {

        /** Forgets all observed events (learned weights are kept). */
        void reset();

        /**
         * Scores candidate destinations.
         *
         * @param sources query sources
         * @param candidates candidates per source
         * @param time the query time (after every observed event)
         * @return scores per source and candidate, higher is more likely
         */
        double[][] score(int[] sources, int[][] candidates, double time);

        /**
         * Reveals events.
         *
         * @param sources sources
         * @param destinations destinations
         * @param times times, non-decreasing and after earlier observations
         */
        void observe(int[] sources, int[] destinations, double[] times);

        /**
         * Trains on a stream prefix (in order) and leaves the model having observed all of it.
         *
         * @param sources sources
         * @param destinations destinations
         * @param times times
         * @param epochs passes
         * @param batch events per step
         * @param lr the learning rate
         * @param seed the negative-sampling seed
         */
        default void fit(int[] sources, int[] destinations, double[] times, int epochs, int batch, float lr,
                long seed) {
            reset();
            observe(sources, destinations, times);
        }
    }

    /** Per-node interaction history (both directions). */
    static final class History {
        final int nodes;
        final List<int[]> partners = new ArrayList<>();
        final List<double[]> times = new ArrayList<>();
        final int[] size;

        History(int nodes) {
            this.nodes = nodes;
            this.size = new int[nodes];
            for (int i = 0; i < nodes; i++) {
                partners.add(new int[16]);
                times.add(new double[16]);
            }
        }

        void clear() {
            Arrays.fill(size, 0);
        }

        void add(int u, int v, double t) {
            if (size[u] == partners.get(u).length) {
                partners.set(u, Arrays.copyOf(partners.get(u), size[u] * 2));
                times.set(u, Arrays.copyOf(times.get(u), size[u] * 2));
            }
            partners.get(u)[size[u]] = v;
            times.get(u)[size[u]] = t;
            size[u]++;
        }

        void addEvents(int[] s, int[] d, double[] t) {
            for (int i = 0; i < s.length; i++) {
                add(s[i], d[i], t[i]);
                add(d[i], s[i], t[i]);
            }
        }

        /** The {@code k}-th most recent partner of {@code u} (0 = latest), or -1. */
        int partner(int u, int k) {
            return k < size[u] ? partners.get(u)[size[u] - 1 - k] : -1;
        }

        double time(int u, int k) {
            return times.get(u)[size[u] - 1 - k];
        }

        int count(int u, int v, int window) {
            int c = 0;
            for (int k = 0; k < Math.min(window, size[u]); k++) {
                c += partner(u, k) == v ? 1 : 0;
            }
            return c;
        }
    }

    static Var matrix(SplittableRandom r, int in, int out) {
        float limit = (float) Math.sqrt(6.0 / (in + out));
        return Var.param(Tensor.uniform(Shape.of(in, out), -limit, limit, r));
    }

    /** Fixed cosine time features {@code cos(dt / 10^(4i / (dim - 1)))}, scaled by {@code scale}. */
    static float[] timeFeatures(double dt, int dim, double scale) {
        float[] f = new float[dim];
        for (int i = 0; i < dim; i++) {
            f[i] = (float) Math.cos(dt / scale / Math.pow(10, 3.0 * i / Math.max(1, dim - 1)));
        }
        return f;
    }

    private static Var bce(Var pos, Var neg) {
        // -log sigmoid(pos) - log(1 - sigmoid(neg))
        return pos.neg().softplus().mean().add(neg.softplus().mean());
    }

    /**
     * EdgeBank: 1 if the pair was seen (within {@code window} time units, if finite), else 0.
     *
     * @param window the memory window; {@code Double.POSITIVE_INFINITY} for unlimited
     * @return the model
     */
    public static LinkModel edgeBank(double window) {
        Map<Long, Double> last = new HashMap<>();
        return new LinkModel() {
            @Override
            public void reset() {
                last.clear();
            }

            @Override
            public double[][] score(int[] sources, int[][] candidates, double time) {
                double[][] s = new double[sources.length][];
                for (int i = 0; i < sources.length; i++) {
                    s[i] = new double[candidates[i].length];
                    for (int j = 0; j < s[i].length; j++) {
                        Double t = last.get(key(sources[i], candidates[i][j]));
                        s[i][j] = t != null && time - t <= window ? 1 : 0;
                    }
                }
                return s;
            }

            @Override
            public void observe(int[] sources, int[] destinations, double[] times) {
                for (int i = 0; i < sources.length; i++) {
                    last.put(key(sources[i], destinations[i]), times[i]);
                }
            }
        };
    }

    private static long key(int u, int v) {
        return (long) u << 32 | v;
    }

    /** Shared training loop for the learned models: chronological batches with random negatives. */
    private abstract static class Learned implements LinkModel {
        final int nodes;
        final History history;
        final List<Var> params = new ArrayList<>();

        Learned(int nodes) {
            this.nodes = nodes;
            this.history = new History(nodes);
        }

        /** Logits of pairs at a time, given everything observed. */
        abstract Var logits(int[] src, int[] dst, double time, boolean training);

        /** Called by {@link #observe} before the history grows, for models with a memory. */
        void commit(int[] s, int[] d, double[] t) {
        }

        @Override
        public void reset() {
            history.clear();
        }

        @Override
        public double[][] score(int[] sources, int[][] candidates, double time) {
            int total = 0;
            for (int[] c : candidates) {
                total += c.length;
            }
            int[] s = new int[total];
            int[] d = new int[total];
            int k = 0;
            for (int i = 0; i < sources.length; i++) {
                for (int c : candidates[i]) {
                    s[k] = sources[i];
                    d[k++] = c;
                }
            }
            float[] l = Var.noGrad(() -> logits(s, d, time, false).value().toArray());
            double[][] out = new double[sources.length][];
            k = 0;
            for (int i = 0; i < sources.length; i++) {
                out[i] = new double[candidates[i].length];
                for (int j = 0; j < out[i].length; j++) {
                    out[i][j] = l[k++];
                }
            }
            return out;
        }

        @Override
        public void observe(int[] sources, int[] destinations, double[] times) {
            commit(sources, destinations, times);
            history.addEvents(sources, destinations, times);
        }

        @Override
        public void fit(int[] sources, int[] destinations, double[] times, int epochs, int batch, float lr,
                long seed) {
            Optimizer opt = Optimizer.adam(lr);
            SplittableRandom r = new SplittableRandom(seed);
            for (int e = 0; e < epochs; e++) {
                reset();
                for (int start = 0; start < sources.length; start += batch) {
                    int end = Math.min(sources.length, start + batch);
                    int[] s = Arrays.copyOfRange(sources, start, end);
                    int[] d = Arrays.copyOfRange(destinations, start, end);
                    double[] t = Arrays.copyOfRange(times, start, end);
                    if (start > 0) {
                        int[] neg = new int[s.length];
                        for (int i = 0; i < neg.length; i++) {
                            neg[i] = r.nextInt(nodes);
                        }
                        int[] both = new int[2 * s.length];
                        int[] dd = new int[2 * s.length];
                        for (int i = 0; i < s.length; i++) {
                            both[i] = s[i];
                            both[s.length + i] = s[i];
                            dd[i] = d[i];
                            dd[s.length + i] = neg[i];
                        }
                        Var l = logits(both, dd, t[0], true);
                        Var loss = bce(l.slice(0, 0, s.length), l.slice(0, s.length, 2 * s.length));
                        loss.backward();
                        opt.step(params);
                    }
                    observe(s, d, t);
                }
            }
        }
    }

    /**
     * TGN with a GRU memory and one TGAT attention layer.
     *
     * @param nodes the node count
     * @param dim the memory and embedding width
     * @param neighbours recent neighbours attended to
     * @param seed the initialisation seed
     * @return the model
     */
    public static LinkModel tgn(int nodes, int dim, int neighbours, long seed) {
        return new Tgn(nodes, dim, neighbours, seed);
    }

    private static final class Tgn extends Learned {
        private final int dim;
        private final int k;
        private final int td = 8;
        private final Var omega;
        private final Var phase;
        private final GruCell gru;
        private final Var wq;
        private final Var wk;
        private final Var wv;
        private final Var wo;
        private final Var dec1;
        private final Var dec2;
        private Tensor memory;
        private final double[] lastUpdate;
        private int[] pendingNode = new int[0];
        private int[] pendingOther = new int[0];
        private double[] pendingTime = new double[0];
        private Var updated;

        Tgn(int nodes, int dim, int k, long seed) {
            super(nodes);
            SplittableRandom r = new SplittableRandom(seed);
            this.dim = dim;
            this.k = k;
            float[] w = new float[td];
            for (int i = 0; i < td; i++) {
                w[i] = (float) Math.pow(10, -3.0 * i / (td - 1));
            }
            omega = Var.param(Tensor.of(Shape.of(1, td), w));
            phase = Var.param(Tensor.zeros(Shape.of(1, td)));
            gru = new GruCell(2 * dim + td, dim, r);
            wq = matrix(r, dim + td, dim);
            wk = matrix(r, dim + td, dim);
            wv = matrix(r, dim + td, dim);
            wo = matrix(r, 2 * dim, dim);
            dec1 = matrix(r, 2 * dim, dim);
            dec2 = matrix(r, dim, 1);
            params.addAll(List.of(omega, phase, wq, wk, wv, wo, dec1, dec2));
            params.addAll(gru.parameters());
            memory = Tensor.zeros(Shape.of(nodes, dim));
            lastUpdate = new double[nodes];
        }

        private Var encodeTime(float[] dt, int rows) {
            return Var.of(Tensor.of(Shape.of(rows, 1), dt)).matmul(omega).add(phase).cos();
        }

        @Override
        public void reset() {
            super.reset();
            memory = Tensor.zeros(Shape.of(nodes, dim));
            Arrays.fill(lastUpdate, 0);
            pendingNode = new int[0];
            pendingOther = new int[0];
            pendingTime = new double[0];
            updated = null;
        }

        /** The memory after applying the pending messages (last message per node). */
        private Var updatedMemory() {
            Var mem = Var.of(memory);
            if (pendingNode.length == 0) {
                return mem;
            }
            int[] lastIndex = new int[nodes];
            Arrays.fill(lastIndex, -1);
            for (int i = 0; i < pendingNode.length; i++) {
                lastIndex[pendingNode[i]] = i;
            }
            float[] other = new float[nodes];
            float[] dt = new float[nodes];
            float[] mask = new float[nodes];
            for (int u = 0; u < nodes; u++) {
                int i = lastIndex[u];
                if (i >= 0) {
                    other[u] = pendingOther[i];
                    dt[u] = (float) (pendingTime[i] - lastUpdate[u]);
                    mask[u] = 1;
                }
            }
            Var msg = Var.concat(-1, mem, mem.gather(0, Tensor.of(Shape.of(nodes), other)), encodeTime(dt, nodes));
            Var next = gru.step(msg, mem);
            Tensor m = Tensor.of(Shape.of(nodes, 1), mask);
            return next.mul(m).add(mem.mul(Tensor.full(Shape.of(nodes, 1), 1f).sub(m)));
        }

        private Var embed(Var mem, double time) {
            float[] ids = new float[nodes * k];
            float[] dt = new float[nodes * k];
            float[] bias = new float[nodes * k];
            for (int u = 0; u < nodes; u++) {
                for (int j = 0; j < k; j++) {
                    int p = history.partner(u, j);
                    ids[u * k + j] = Math.max(0, p);
                    dt[u * k + j] = p < 0 ? 0 : (float) (time - history.time(u, j));
                    bias[u * k + j] = p < 0 ? -1e9f : 0;
                }
            }
            Var nb = mem.gather(0, Tensor.of(Shape.of(nodes * k), ids));
            Var kv = Var.concat(-1, nb, encodeTime(dt, nodes * k));
            Var q = Var.concat(-1, mem, encodeTime(new float[nodes], nodes)).matmul(wq).reshape(nodes, 1, dim);
            Var keys = kv.matmul(wk).reshape(nodes, k, dim);
            Var values = kv.matmul(wv).reshape(nodes, k, dim);
            // a self slot so isolated nodes still attend somewhere
            Var selfKey = q;
            Var allKeys = Var.concat(1, selfKey, keys);
            Var allValues = Var.concat(1, mem.matmul(wv.slice(0, 0, dim)).reshape(nodes, 1, dim), values);
            float[] fullBias = new float[nodes * (k + 1)];
            for (int u = 0; u < nodes; u++) {
                System.arraycopy(bias, u * k, fullBias, u * (k + 1) + 1, k);
            }
            Var s = q.matmul(allKeys.permute(0, 2, 1)).mul((float) (1 / Math.sqrt(dim)))
                    .add(Tensor.of(Shape.of(nodes, 1, k + 1), fullBias));
            Var a = s.sub(s.logsumexp(-1, true)).exp();
            Var att = a.matmul(allValues).reshape(nodes, dim);
            return Var.concat(-1, att, mem).matmul(wo).relu();
        }

        @Override
        Var logits(int[] src, int[] dst, double time, boolean training) {
            Var mem = updatedMemory();
            if (training) {
                updated = mem;
            }
            Var z = embed(mem, time);
            Var zs = z.gather(0, GraphIds.of(src));
            Var zd = z.gather(0, GraphIds.of(dst));
            return Var.concat(-1, zs, zd).matmul(dec1).relu().matmul(dec2).reshape(src.length);
        }

        @Override
        void commit(int[] s, int[] d, double[] t) {
            // the memory update of the previous batch is now final
            Var mem = updated != null ? updated : Var.noGrad(this::updatedMemory);
            memory = mem.value();
            updated = null;
            for (int i = 0; i < pendingNode.length; i++) {
                lastUpdate[pendingNode[i]] = pendingTime[i];
            }
            pendingNode = new int[2 * s.length];
            pendingOther = new int[2 * s.length];
            pendingTime = new double[2 * s.length];
            for (int i = 0; i < s.length; i++) {
                pendingNode[2 * i] = s[i];
                pendingOther[2 * i] = d[i];
                pendingNode[2 * i + 1] = d[i];
                pendingOther[2 * i + 1] = s[i];
                pendingTime[2 * i] = t[i];
                pendingTime[2 * i + 1] = t[i];
            }
        }

    }

    /**
     * GraphMixer: an MLP-Mixer link encoder over recent links' time encodings and a node encoder
     * averaging recent neighbours' learned embeddings.
     *
     * @param nodes the node count
     * @param dim the width
     * @param recent recent links per node
     * @param seed the initialisation seed
     * @return the model
     */
    public static LinkModel graphMixer(int nodes, int dim, int recent, long seed) {
        return new GraphMixer(nodes, dim, recent, seed);
    }

    private static final class GraphMixer extends Learned {
        private final int dim;
        private final int k;
        private final int td = 16;
        private final Var node;
        private final Var in;
        private final Var token1;
        private final Var token2;
        private final Var channel1;
        private final Var channel2;
        private final Var dec1;
        private final Var dec2;

        GraphMixer(int nodes, int dim, int k, long seed) {
            super(nodes);
            SplittableRandom r = new SplittableRandom(seed);
            this.dim = dim;
            this.k = k;
            node = Var.param(Tensor.uniform(Shape.of(nodes, dim), -0.3f, 0.3f, r));
            in = matrix(r, td + 1, dim);
            token1 = matrix(r, k, k);
            token2 = matrix(r, k, k);
            channel1 = matrix(r, dim, dim);
            channel2 = matrix(r, dim, dim);
            dec1 = matrix(r, 4 * dim, dim);
            dec2 = matrix(r, dim, 1);
            params.addAll(List.of(node, in, token1, token2, channel1, channel2, dec1, dec2));
        }

        private Var represent(double time) {
            float[] tok = new float[nodes * k * (td + 1)];
            float[] neigh = new float[nodes * nodes];
            for (int u = 0; u < nodes; u++) {
                int count = 0;
                for (int j = 0; j < k; j++) {
                    int p = history.partner(u, j);
                    if (p < 0) {
                        break;
                    }
                    float[] f = timeFeatures(time - history.time(u, j), td, 1);
                    System.arraycopy(f, 0, tok, (u * k + j) * (td + 1), td);
                    tok[(u * k + j) * (td + 1) + td] = 1;
                    neigh[u * nodes + p] += 1;
                    count++;
                }
                neigh[u * nodes + u] += 1;
                for (int v = 0; v < nodes; v++) {
                    neigh[u * nodes + v] /= count + 1;
                }
            }
            Var x = Var.of(Tensor.of(Shape.of(nodes, k, td + 1), tok)).matmul(in);     // [N, K, d]
            Var t = x.permute(0, 2, 1);                                                   // [N, d, K]
            x = x.add(t.matmul(token1).relu().matmul(token2).permute(0, 2, 1));
            x = x.add(x.matmul(channel1).relu().matmul(channel2));
            Var link = x.mean(1, false);
            Var nodeRep = Var.of(Tensor.of(Shape.of(nodes, nodes), neigh)).matmul(node);
            return Var.concat(-1, link, nodeRep);
        }

        @Override
        Var logits(int[] src, int[] dst, double time, boolean training) {
            Var h = represent(time);
            Var hs = h.gather(0, GraphIds.of(src));
            Var hd = h.gather(0, GraphIds.of(dst));
            return Var.concat(-1, hs, hd).matmul(dec1).relu().matmul(dec2).reshape(src.length);
        }
    }

    /**
     * HyperEvent-style relational event encoding with self-attention.
     *
     * @param nodes the node count (only to size the history; no node parameters)
     * @param dim the width
     * @param recent recent events taken from each endpoint
     * @param seed the initialisation seed
     * @return the model
     */
    public static LinkModel hyperEvent(int nodes, int dim, int recent, long seed) {
        return new HyperEvent(nodes, dim, recent, seed);
    }

    private static final class HyperEvent extends Learned {
        private static final int TD = 8;
        static final int FEATURES = 9 + TD;
        private final int dim;
        private final int k;
        private final Var in;
        private final Var inB;
        private final Var wq;
        private final Var wk;
        private final Var wv;
        private final Var dec1;
        private final Var dec2;

        HyperEvent(int nodes, int dim, int k, long seed) {
            super(nodes);
            SplittableRandom r = new SplittableRandom(seed);
            this.dim = dim;
            this.k = k;
            in = matrix(r, FEATURES, dim);
            inB = Var.param(Tensor.zeros(Shape.of(dim)));
            wq = matrix(r, dim, dim);
            wk = matrix(r, dim, dim);
            wv = matrix(r, dim, dim);
            dec1 = matrix(r, 2 * dim, dim);
            dec2 = matrix(r, dim, 1);
            params.addAll(List.of(in, inB, wq, wk, wv, dec1, dec2));
        }

        private void features(int u, int v, double time, float[] out, int offset, float[] mask, int maskOffset) {
            int window = 4 * k;
            for (int side = 0; side < 2; side++) {
                int x = side == 0 ? u : v;
                int y = side == 0 ? v : u;
                for (int j = 0; j < k; j++) {
                    int p = history.partner(x, j);
                    int base = offset + (side * k + j) * FEATURES;
                    if (p < 0) {
                        continue;
                    }
                    mask[maskOffset + side * k + j] = 1;
                    out[base] = 1;
                    out[base + 1] = side == 0 ? 1 : 0;
                    out[base + 2] = p == y ? 1 : 0;
                    out[base + 3] = (float) Math.log1p(history.count(u, p, window));
                    out[base + 4] = (float) Math.log1p(history.count(v, p, window));
                    int twoHop = 0;
                    for (int q = 0; q < Math.min(k, history.size[p]); q++) {
                        int pp = history.partner(p, q);
                        twoHop += pp == y || history.count(y, pp, window) > 0 ? 1 : 0;
                    }
                    out[base + 5] = (float) Math.log1p(twoHop);
                    out[base + 6] = (float) j / k;
                    out[base + 7] = (float) Math.log1p(history.count(x, y, window));
                    out[base + 8] = (float) Math.log1p(time - history.time(x, j));
                    float[] f = timeFeatures(time - history.time(x, j), TD, 1);
                    System.arraycopy(f, 0, out, base + 9, TD);
                }
            }
        }

        @Override
        Var logits(int[] src, int[] dst, double time, boolean training) {
            int q = src.length;
            int t = 2 * k;
            float[] f = new float[q * t * FEATURES];
            float[] mask = new float[q * t];
            for (int i = 0; i < q; i++) {
                features(src[i], dst[i], time, f, i * t * FEATURES, mask, i * t);
            }
            float[] bias = new float[q * t];
            float[] weights = new float[q * t];
            for (int i = 0; i < q; i++) {
                float count = 0;
                for (int j = 0; j < t; j++) {
                    count += mask[i * t + j];
                }
                for (int j = 0; j < t; j++) {
                    bias[i * t + j] = mask[i * t + j] > 0 ? 0 : -1e9f;
                    weights[i * t + j] = count == 0 ? 0 : mask[i * t + j] / count;
                }
                if (count == 0) {
                    bias[i * t] = 0;
                }
            }
            Var x = Var.of(Tensor.of(Shape.of(q, t, FEATURES), f)).matmul(in).add(inB).relu();
            Var s = x.matmul(wq).matmul(x.matmul(wk).permute(0, 2, 1)).mul((float) (1 / Math.sqrt(dim)))
                    .add(Tensor.of(Shape.of(q, 1, t), bias));
            Var a = s.sub(s.logsumexp(-1, true)).exp();
            Var h = x.add(a.matmul(x.matmul(wv)));
            Var pooled = Var.of(Tensor.of(Shape.of(q, 1, t), weights)).matmul(h).reshape(q, dim);
            Var maxed = h.add(Tensor.of(Shape.of(q, t, 1), bias)).max(1, false).mul(Tensor.of(Shape.of(q, 1), any(mask, q, t)));
            return Var.concat(-1, pooled, maxed).matmul(dec1).relu().matmul(dec2).reshape(q);
        }

        private static float[] any(float[] mask, int q, int t) {
            float[] a = new float[q];
            for (int i = 0; i < q; i++) {
                for (int j = 0; j < t; j++) {
                    a[i] = Math.max(a[i], mask[i * t + j]);
                }
            }
            return a;
        }
    }

    /** Index tensors. */
    static final class GraphIds {
        private GraphIds() {
        }

        static Tensor of(int[] v) {
            return RelationalGcn.ids(v);
        }
    }
}
