package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;

/**
 * Ring attention (Liu et al. 2023, <a href="https://arxiv.org/abs/2310.01889">arXiv:2310.01889</a>):
 * exact attention over a sequence split into {@code P} shards, one per device. Device
 * {@code i} keeps its query block and passes key/value blocks around a ring; at ring step
 * {@code s} it holds block {@code (i - s) mod P} and folds it into a running
 * (online, flash-style) softmax: a running maximum {@code m}, normaliser {@code l} and
 * weighted sum {@code acc}, rescaled by {@code exp(m_old - m_new)} whenever the maximum
 * grows (Milakov and Gimelshein 2018, <a href="https://arxiv.org/abs/1805.02867">arXiv:1805.02867</a>).
 * No device ever holds more than one {@code [T/P, T/P]} block of scores, and the result is
 * the same as full attention, up to float rounding, including gradients: the running maximum is
 * a detached constant, and softmax does not change when a constant is subtracted.
 * With causal attention, blocks from later shards are skipped.
 *
 * <p>Here the devices run one after another in one autograd graph; {@link Stats} counts
 * the block products and transfers a real ring would make.
 */
public final class RingAttention {

    private RingAttention() {
    }

    /**
     * What one ring pass did.
     *
     * @param blocks the score blocks computed
     * @param skipped the blocks skipped by the causal mask
     * @param transfers the key/value block sends between neighbours
     * @param largestBlock the most scores one device held at once (per batch and head)
     */
    public record Stats(int blocks, int skipped, int transfers, int largestBlock) { }

    /**
     * Ring attention.
     *
     * @param q queries {@code [batch, heads, steps, dk]}
     * @param k keys {@code [batch, heads, steps, dk]}
     * @param v values {@code [batch, heads, steps, dv]}
     * @param causal whether position {@code i} sees only keys up to {@code i}
     * @param scale the score scale
     * @param shards the devices {@code P}, at most {@code steps}
     * @return {@code [batch, heads, steps, dv]}
     */
    public static Var attention(Var q, Var k, Var v, boolean causal, float scale, int shards) {
        return attention(q, k, v, causal, scale, shards, null);
    }

    /**
     * As {@link #attention(Var, Var, Var, boolean, float, int)}, also reporting the work.
     *
     * @param q queries
     * @param k keys
     * @param v values
     * @param causal whether attention is causal
     * @param scale the score scale
     * @param shards the devices
     * @param stats receives the counts in {@code stats[0]}, or {@code null}
     * @return the attention output
     */
    public static Var attention(Var q, Var k, Var v, boolean causal, float scale, int shards, Stats[] stats) {
        int steps = q.shape().dim(2);
        if (k.shape().dim(2) != steps || v.shape().dim(2) != steps) {
            throw new IllegalArgumentException("ring attention needs as many keys as queries");
        }
        if (shards < 1 || shards > steps) {
            throw new IllegalArgumentException("shards must be in [1, " + steps + "], got " + shards);
        }
        int[] bounds = new int[shards + 1];
        for (int p = 0; p <= shards; p++) {
            bounds[p] = (int) ((long) steps * p / shards);
        }
        Var[] kb = new Var[shards];
        Var[] vb = new Var[shards];
        for (int p = 0; p < shards; p++) {
            kb[p] = k.slice(2, bounds[p], bounds[p + 1]);
            vb[p] = v.slice(2, bounds[p], bounds[p + 1]);
        }
        int blocks = 0;
        int skipped = 0;
        int largest = 0;
        Var[] out = new Var[shards];
        for (int i = 0; i < shards; i++) {
            Var qi = q.slice(2, bounds[i], bounds[i + 1]);
            int rows = bounds[i + 1] - bounds[i];
            Var m = null;
            Var l = null;
            Var acc = null;
            for (int s = 0; s < shards; s++) {
                int j = Math.floorMod(i - s, shards);
                if (causal && j > i) {
                    skipped++;
                    continue;
                }
                int cols = bounds[j + 1] - bounds[j];
                largest = Math.max(largest, rows * cols);
                blocks++;
                Var scores = qi.matmul(kb[j].transpose()).mul(scale);
                if (causal && j == i) {
                    scores = scores.add(Functional.causalMask(rows, cols, q.value().device()));
                }
                Var bm = scores.max(3, true).detach();
                if (m == null) {
                    Var p = scores.sub(bm).exp();
                    m = bm;
                    l = p.sum(3, true);
                    acc = p.matmul(vb[j]);
                } else {
                    Var mNew = m.maximum(bm).detach();
                    Var alpha = m.sub(mNew).exp();
                    Var p = scores.sub(mNew).exp();
                    l = l.mul(alpha).add(p.sum(3, true));
                    acc = acc.mul(alpha).add(p.matmul(vb[j]));
                    m = mNew;
                }
            }
            out[i] = acc.div(l);
        }
        if (stats != null) {
            stats[0] = new Stats(blocks, skipped, shards * (shards - 1), largest);
        }
        return shards == 1 ? out[0] : Var.concat(2, out);
    }
}
