package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The top-k selection shared by the sparse attentions of this package (DeepSeek Sparse
 * Attention, compressed sparse attention, MoBA): pick, per query row, the best visible
 * entries of a score matrix and return a 0/1 keep mask. Selection is not differentiable;
 * gradients reach the scorer through a separate loss or a dense branch.
 */
final class SparseSelectOps {

    private SparseSelectOps() {
    }

    /**
     * Keeps the {@code k} highest-scoring visible entries of every row (all visible ones when
     * fewer than {@code k} are visible), plus the entries {@code forced} marks.
     *
     * @param scores row-major scores {@code [rows, cols]}
     * @param visible 1 where an entry may be chosen, {@code [rows, cols]} or {@code [rowsPerMask, cols]} repeated
     * @param forced 1 where an entry is always kept (and must be visible), or {@code null}
     * @param rows the rows
     * @param cols the columns
     * @param k the entries to keep per row, besides the forced ones
     * @return the 0/1 keep mask {@code [rows * cols]}
     */
    static float[] topKeep(float[] scores, float[] visible, float[] forced, int rows, int cols, int k) {
        float[] keep = new float[rows * cols];
        int maskRows = visible.length / cols;
        Integer[] order = new Integer[cols];
        for (int r = 0; r < rows; r++) {
            int mr = (r % maskRows) * cols;
            int n = 0;
            for (int c = 0; c < cols; c++) {
                if (visible[mr + c] > 0) {
                    if (forced != null && forced[mr + c] > 0) {
                        keep[r * cols + c] = 1;
                    } else {
                        order[n++] = c;
                    }
                }
            }
            int base = r * cols;
            java.util.Arrays.sort(order, 0, n, (a, b) -> Float.compare(scores[base + b], scores[base + a]));
            for (int i = 0; i < Math.min(k, n); i++) {
                keep[base + order[i]] = 1;
            }
        }
        return keep;
    }

    /** A causal visibility mask {@code [t, t]}: 1 where key {@code j <= i}. */
    static float[] causal(int t) {
        float[] m = new float[t * t];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j <= i; j++) {
                m[i * t + j] = 1;
            }
        }
        return m;
    }

    /** The mean number of kept entries per row. */
    static double meanKept(float[] keep, int rows) {
        double s = 0;
        for (float v : keep) {
            s += v;
        }
        return s / rows;
    }

    /**
     * {@code KL(target || softmax(scores))} over the kept entries of every row, averaged over
     * rows: the distillation loss that teaches an indexer to imitate attention.
     *
     * @param scores the indexer scores {@code [..., cols]}
     * @param target a constant distribution over the same entries, rows summing to 1
     * @param keep 1 where an entry takes part
     * @return the scalar loss
     */
    static Var distillation(Var scores, Tensor target, Tensor keep) {
        Var logp = MixerOps.softmax(scores.add(RefinementOps.additive(keep))).add(1e-9f).log();
        Tensor t = target.mul(keep);
        Tensor entropy = t.mul(t.add(1e-9f).log());
        int rows = Math.toIntExact(target.shape().size() / target.shape().dim(-1));
        return logp.mul(t).neg().add(entropy).sum().div(rows);
    }

    /** {@code [b, t, t]} 0/1 mask tensor from a flat array, broadcast over heads as {@code [b, 1, t, t]}. */
    static Tensor perBatch(float[] keep, int b, int rows, int cols) {
        return Tensor.of(Shape.of(b, 1, rows, cols), keep);
    }
}
