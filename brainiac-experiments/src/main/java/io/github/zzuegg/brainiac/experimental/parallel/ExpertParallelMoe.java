package io.github.zzuegg.brainiac.experimental.parallel;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FineGrainedMoe;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Expert parallelism for a {@link FineGrainedMoe}: its routed experts are placed on {@code ranks}
 * devices (contiguous blocks of {@code experts / ranks}), tokens are spread evenly over the ranks
 * as in data parallelism, and each layer needs two all-to-alls, one to dispatch every
 * (token, expert) assignment to its expert's rank and one to bring the results back.
 *
 * <p>Static communication buffers need a fixed capacity per expert,
 * {@code C = ceil(capacityFactor * tokens * topK / experts)}. With {@code dropTokens} an expert
 * accepts assignments in token order until it is full and drops the rest: a dropped assignment
 * contributes nothing, so the token passes through the residual connection only (GShard,
 * Switch Transformer). Without dropping every assignment is processed and the busiest rank
 * sets the step time. The layer reports, for its last forward pass, the dropped fraction, the
 * all-to-all bytes actually needed and those of capacity-padded static buffers, and each
 * rank's load.
 *
 * <p>Routing is the wrapped layer's: softmax affinities from its router, top-k by affinity plus
 * a selection-only bias. Balancing is either the auxiliary loss ({@link #balanceWeight}) or
 * DeepSeek-V3's auxiliary-loss-free bias update ({@link #updateBalanceBias}), the same rule as
 * {@link FineGrainedMoe#updateBalanceBias}, kept here because the bias must follow the loads after
 * capacity limits.
 *
 * <p>Papers: Lepikhin et al., "GShard", <a href="https://arxiv.org/abs/2006.16668">arXiv:2006.16668</a>;
 * Fedus et al., "Switch Transformers", <a href="https://arxiv.org/abs/2101.03961">arXiv:2101.03961</a>.
 */
public final class ExpertParallelMoe implements Layer {

    private final FineGrainedMoe moe;
    private final int ranks;
    private final float capacityFactor;
    private final boolean dropTokens;
    private final int topK;
    private final float[] bias;
    private final Collectives traffic;
    private float balanceWeight;
    private Var lastBalance;
    private int[] lastLoad;
    private int[] lastRankLoad;
    private double lastDropped;
    private long lastBytes;
    private long lastPaddedBytes;
    private int lastCapacity;

    /**
     * Wraps a mixture-of-experts layer without shared experts.
     *
     * @param moe the layer whose router and experts are used
     * @param topK experts per token, as the layer was built with
     * @param ranks devices, dividing the number of experts
     * @param capacityFactor the capacity factor, such as 1.0 to 2.0
     * @param dropTokens whether assignments beyond capacity are dropped
     */
    public ExpertParallelMoe(FineGrainedMoe moe, int topK, int ranks, float capacityFactor, boolean dropTokens) {
        int e = moe.routedExperts().size();
        if (e % ranks != 0) {
            throw new IllegalArgumentException(e + " experts do not divide over " + ranks + " ranks");
        }
        this.moe = moe;
        this.topK = topK;
        this.ranks = ranks;
        this.capacityFactor = capacityFactor;
        this.dropTokens = dropTokens;
        this.bias = new float[e];
        this.traffic = new Collectives(ranks);
    }

    /**
     * @param weight the auxiliary balance-loss weight; 0 for bias balancing only
     * @return this layer
     */
    public ExpertParallelMoe balanceWeight(float weight) {
        this.balanceWeight = weight;
        return this;
    }

    /** @return the counters of all-to-all traffic so far */
    public Collectives traffic() {
        return traffic;
    }

    /**
     * @param expert an expert
     * @return the rank holding it
     */
    public int rankOf(int expert) {
        return expert / (moe.routedExperts().size() / ranks);
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        int features = s.dim(-1);
        int rows = Math.toIntExact(s.size() / features);
        int e = moe.routedExperts().size();
        Var x = input.reshape(rows, features);
        Var logits = x.matmul(moe.router());
        Var affinity = Activation.softmax(1).forward(logits);
        float[] a = affinity.value().toArray();
        int capacity = (int) Math.ceil(capacityFactor * rows * topK / (double) e);
        int[] load = new int[e];
        int[] accepted = new int[e];
        List<List<Integer>> members = new ArrayList<>();
        for (int i = 0; i < e; i++) {
            members.add(new ArrayList<>());
        }
        boolean[][] keep = new boolean[rows][topK];
        int[][] chosen = new int[rows][];
        int dropped = 0;
        long bytes = 0;
        int[] rankLoad = new int[ranks];
        for (int n = 0; n < rows; n++) {
            chosen[n] = topK(a, n * e, e, bias, topK);
            int src = (int) ((long) n * ranks / rows);
            for (int k = 0; k < topK; k++) {
                int ex = chosen[n][k];
                load[ex]++;
                if (dropTokens && accepted[ex] >= capacity) {
                    dropped++;
                    continue;
                }
                accepted[ex]++;
                keep[n][k] = true;
                members.get(ex).add(n);
                rankLoad[rankOf(ex)]++;
                if (rankOf(ex) != src) {
                    bytes += 2L * features * Float.BYTES;   // dispatch and combine
                    traffic.account(src, (long) features * Float.BYTES, 1);
                    traffic.account(rankOf(ex), (long) features * Float.BYTES, 1);
                }
            }
        }
        // the per-expert outputs, gathered back to (row, slot)
        int[] position = new int[rows * topK];
        Arrays.fill(position, -1);
        List<Var> outs = new ArrayList<>();
        int offset = 0;
        for (int i = 0; i < e; i++) {
            List<Integer> m = members.get(i);
            if (m.isEmpty()) {
                continue;
            }
            int[] idx = m.stream().mapToInt(Integer::intValue).toArray();
            outs.add(moe.routedExperts().get(i).forward(x.gather(0, ids(idx))));
            for (int j = 0; j < idx.length; j++) {
                int n = idx[j];
                for (int k = 0; k < topK; k++) {
                    if (chosen[n][k] == i && keep[n][k] && position[n * topK + k] < 0) {
                        position[n * topK + k] = offset + j;
                        break;
                    }
                }
            }
            offset += idx.length;
        }
        int out = outs.isEmpty() ? features : outs.getFirst().shape().dim(1);
        // a zero row for dropped assignments
        Var zero = Var.of(Tensor.zeros(Shape.of(1, out)));
        outs.add(zero);
        Var all = outs.size() == 1 ? zero : Var.concat(0, outs.toArray(Var[]::new));
        float[] gateMask = new float[rows * topK];
        int[] flatChoice = new int[rows * topK];
        for (int n = 0; n < rows; n++) {
            for (int k = 0; k < topK; k++) {
                flatChoice[n * topK + k] = n * e + chosen[n][k];
                if (position[n * topK + k] < 0) {
                    position[n * topK + k] = offset;
                } else {
                    gateMask[n * topK + k] = 1;
                }
            }
        }
        Var gates = affinity.reshape(rows * e).gather(0, ids(flatChoice))
                .mul(Tensor.of(Shape.of(rows * topK), gateMask)).reshape(rows, topK, 1);
        Var y = all.gather(0, ids(position)).reshape(rows, topK, out).mul(gates).sum(1, false);
        float[] f = new float[e];
        for (int i = 0; i < e; i++) {
            f[i] = e * load[i] / (float) (topK * rows);
        }
        lastBalance = affinity.mean(0, false).mul(Tensor.of(Shape.of(e), f)).sum();
        lastLoad = load;
        lastRankLoad = rankLoad;
        lastDropped = dropped / (double) (rows * topK);
        lastBytes = bytes;
        lastCapacity = capacity;
        // static buffers: every rank sends a capacity-sized slice per remote expert, both ways
        lastPaddedBytes = 2L * e * capacity * features * Float.BYTES * (ranks - 1) / ranks;
        int[] dims = s.toArray();
        dims[dims.length - 1] = out;
        return y.reshape(dims);
    }

    private static int[] topK(float[] a, int from, int e, float[] bias, int k) {
        int[] best = new int[k];
        boolean[] used = new boolean[e];
        for (int j = 0; j < k; j++) {
            int arg = -1;
            for (int i = 0; i < e; i++) {
                if (!used[i] && (arg < 0 || a[from + i] + bias[i] > a[from + arg] + bias[arg])) {
                    arg = i;
                }
            }
            used[arg] = true;
            best[j] = arg;
        }
        return best;
    }

    private static Tensor ids(int[] idx) {
        float[] f = new float[idx.length];
        for (int i = 0; i < idx.length; i++) {
            f[i] = idx[i];
        }
        return Tensor.of(Shape.of(idx.length), f);
    }

    @Override
    public Var auxiliaryLoss() {
        return balanceWeight == 0 || lastBalance == null ? null : lastBalance.mul(balanceWeight);
    }

    /**
     * The auxiliary-loss-free balancing step on the last pass's routed loads (before capacity).
     *
     * @param rate the bias step, such as 0.001
     */
    public void updateBalanceBias(float rate) {
        if (lastLoad == null) {
            return;
        }
        double mean = Arrays.stream(lastLoad).average().orElse(0);
        for (int i = 0; i < bias.length; i++) {
            bias[i] += lastLoad[i] < mean ? rate : lastLoad[i] > mean ? -rate : 0;
        }
    }

    /** @return the fraction of (token, expert) assignments dropped in the last pass */
    public double droppedFraction() {
        return lastDropped;
    }

    /** @return the all-to-all bytes the last pass needed (dispatch plus combine, remote only) */
    public long allToAllBytes() {
        return lastBytes;
    }

    /** @return the all-to-all bytes of capacity-padded static buffers in the last pass */
    public long paddedAllToAllBytes() {
        return lastPaddedBytes;
    }

    /** @return the capacity per expert in the last pass */
    public int capacity() {
        return lastCapacity;
    }

    /** @return the assignments each rank processed in the last pass */
    public int[] rankLoad() {
        return lastRankLoad == null ? new int[ranks] : lastRankLoad.clone();
    }

    /**
     * The slowest rank's work relative to a perfectly even split; with static buffers a rank
     * always pays for its full capacity.
     *
     * @return {@code max load / mean load} (or the padded load when dropping)
     */
    public double slowestRankFactor() {
        if (lastRankLoad == null) {
            return 1;
        }
        double mean = Arrays.stream(lastRankLoad).average().orElse(1);
        double max = dropTokens ? (double) lastCapacity * moe.routedExperts().size() / ranks
                : Arrays.stream(lastRankLoad).max().orElse(0);
        return max / Math.max(mean, 1e-9);
    }

    @Override
    public List<Var> parameters() {
        return moe.parameters();
    }

    @Override
    public int outputSize(int inputSize) {
        return moe.outputSize(inputSize);
    }
}
