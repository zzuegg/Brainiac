package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;

/**
 * Streaming DiLoCo (Douillard et al. 2025, <a href="https://arxiv.org/abs/2501.18512">arXiv:2501.18512</a>),
 * the bandwidth-lean refinement of {@link DiLoCo}, simulated in-process. Three changes:
 * <ul>
 *   <li><b>Fragments.</b> The parameters are split into {@code P} contiguous fragments and
 *       fragment {@code p} synchronizes every {@code H} steps at offset {@code p H / P}, so
 *       only one fragment is in flight at a time and peak bandwidth drops by {@code P}.</li>
 *   <li><b>Low-bit outer gradients.</b> Each worker's pseudo-gradient is quantized to
 *       {@code bits} bits (symmetric, per tensor) before the all-reduce; the paper finds 4
 *       bits lossless.</li>
 *   <li><b>Overlap.</b> The merged fragment arrives {@code tau} inner steps after it was
 *       sent, while the workers keep training; on arrival a worker blends it with its
 *       local copy, {@code alpha local + (1 - alpha) global}.</li>
 * </ul>
 * Bytes are counted as every worker sending and receiving its fragment at {@code bits}.
 */
public final class StreamingDiLoCo {

    private final List<Var> global;
    private final List<LocalWorker> workers;
    private final Optimizer outer;
    private final int[][] fragments;
    private final int bits;
    private final int overlap;
    private final float alpha;
    private final List<Pending> pending = new ArrayList<>();
    private long step;
    private long bytes;

    private record Pending(long due, int fragment, Tensor[] values) {
    }

    /**
     * @param global the global model's parameters
     * @param workers the workers, each starting from the global weights
     * @param outer the outer optimizer, for example {@link DiLoCo#defaultOuter()}
     * @param fragments how many fragments to stream, at most the number of parameters
     * @param bits bits per outer-gradient value, 1 to 32 (32 sends floats unquantized)
     * @param overlap inner steps the merged fragment takes to arrive, {@code tau}
     * @param alpha the local weight of the blend on arrival (0 replaces the local copy)
     */
    public StreamingDiLoCo(List<Var> global, List<LocalWorker> workers, Optimizer outer, int fragments, int bits,
            int overlap, float alpha) {
        if (fragments < 1 || fragments > global.size() || bits < 1 || bits > 32 || overlap < 0) {
            throw new IllegalArgumentException("bad fragments, bits or overlap");
        }
        this.global = List.copyOf(global);
        this.workers = List.copyOf(workers);
        this.outer = outer;
        this.fragments = new int[fragments][];
        for (int p = 0; p < fragments; p++) {
            int from = p * global.size() / fragments;
            int to = (p + 1) * global.size() / fragments;
            this.fragments[p] = new int[to - from];
            for (int i = from; i < to; i++) {
                this.fragments[p][i - from] = i;
            }
        }
        this.bits = bits;
        this.overlap = overlap;
        this.alpha = alpha;
        for (LocalWorker w : this.workers) {
            FederatedAveraging.broadcast(this.global, w.parameters());
        }
    }

    /** @return bytes sent and received by all workers so far */
    public long bytes() {
        return bytes;
    }

    /**
     * One inner step on every worker, followed by any fragment synchronization that is due.
     *
     * @param period the synchronization period of each fragment, {@code H}
     * @return the workers' mean inner loss
     */
    public float step(int period) {
        double loss = 0;
        for (LocalWorker w : workers) {
            loss += w.train(1);
        }
        step++;
        for (int i = pending.size() - 1; i >= 0; i--) {
            Pending p = pending.get(i);
            if (p.due() <= step) {
                arrive(p);
                pending.remove(i);
            }
        }
        for (int p = 0; p < fragments.length; p++) {
            long offset = (long) p * period / fragments.length;
            if (step >= period && (step - offset) % period == 0) {
                sync(p);
            }
        }
        return (float) (loss / workers.size());
    }

    private void sync(int p) {
        List<Var> vars = new ArrayList<>();
        Tensor[] merged = new Tensor[fragments[p].length];
        for (int j = 0; j < fragments[p].length; j++) {
            int i = fragments[p][j];
            Var g = global.get(i);
            Tensor sum = null;
            for (LocalWorker w : workers) {
                Tensor delta = quantize(g.value().sub(w.parameters().get(i).value()), bits);
                sum = sum == null ? delta : sum.add(delta);
                bytes += 2 * (g.value().size() * bits + 7) / 8;
            }
            g.setGrad(sum.div(workers.size()));
            vars.add(g);
        }
        outer.step(vars);
        for (int j = 0; j < merged.length; j++) {
            merged[j] = global.get(fragments[p][j]).value();
        }
        Pending arrival = new Pending(step + overlap, p, merged);
        if (overlap == 0) {
            arrive(arrival);
        } else {
            pending.add(arrival);
        }
    }

    private void arrive(Pending p) {
        for (LocalWorker w : workers) {
            for (int j = 0; j < fragments[p.fragment()].length; j++) {
                Var local = w.parameters().get(fragments[p.fragment()][j]);
                Tensor blended = alpha == 0 ? p.values()[j]
                        : local.value().mul(alpha).add(p.values()[j].mul(1 - alpha));
                local.assign(blended);
            }
        }
    }

    /**
     * Symmetric uniform quantization with one scale per tensor (the largest magnitude; for one
     * bit, the sign times the mean magnitude).
     *
     * @param t the values
     * @param bits bits per value; 32 returns {@code t} unchanged
     * @return the dequantized values
     */
    public static Tensor quantize(Tensor t, int bits) {
        if (bits >= 32) {
            return t;
        }
        float[] v = t.toArray();
        float max = 0;
        double meanAbs = 0;
        for (float x : v) {
            max = Math.max(max, Math.abs(x));
            meanAbs += Math.abs(x) / v.length;
        }
        if (max == 0) {
            return t;
        }
        int levels = bits == 1 ? 1 : (1 << (bits - 1)) - 1;
        float scale = max / levels;
        for (int i = 0; i < v.length; i++) {
            v[i] = bits == 1 ? (float) (Math.signum(v[i]) * meanAbs) : Math.round(v[i] / scale) * scale;
        }
        return Tensor.of(t.shape(), v);
    }
}
