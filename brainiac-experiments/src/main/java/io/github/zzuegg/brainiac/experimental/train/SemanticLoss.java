package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * The semantic loss: how far a network's independent Bernoulli outputs are from
 * satisfying a logical constraint {@code alpha}, measured as
 * {@code -log WMC(alpha, p)}, the negative log of the weighted model count, the
 * probability that a sample of the outputs satisfies the constraint. It is zero when the
 * network puts all its mass on satisfying assignments, needs no labels (so it regularises
 * unlabelled data in semi-supervised learning) and is differentiable in {@code p}.
 *
 * <p>The weighted model count is computed exactly for the constraints compiled here:
 * <ul>
 *   <li>{@link #exactlyOne}: {@code sum_i p_i prod_{j != i} (1 - p_j)}, in log space;</li>
 *   <li>{@link #atMostK}: a dynamic program over how many variables are true (the
 *       circuit of a cardinality constraint);</li>
 *   <li>{@link #latinSquare}: the conjunction of exactly-one constraints over every row and
 *       column for each value (and every cell), summed as independent factors. The cell,
 *       row and column constraints share variables, so this sum is the semantic loss of
 *       each constraint, not of their exact conjunction (whose count is the permanent-like
 *       number of Latin squares).</li>
 * </ul>
 *
 * <p>Paper: Xu, Zhang, Friedman, Liang, Van den Broeck, "A Semantic Loss Function for
 * Deep Learning with Symbolic Knowledge" (ICML 2018),
 * <a href="https://arxiv.org/abs/1711.11157">arXiv:1711.11157</a>.
 */
public final class SemanticLoss {

    private static final float EPS = 1e-6f;

    private SemanticLoss() {
    }

    /**
     * The semantic loss of "exactly one of these is true", per row.
     *
     * @param p probabilities {@code [batch, n]}
     * @return {@code [batch]} values {@code -log sum_i p_i prod_{j != i} (1 - p_j)}
     */
    public static Var exactlyOne(Var p) {
        Var q = p.clamp(EPS, 1 - EPS);
        Var logNot = q.neg().add(1f).log();
        Var all = logNot.sum(-1, true);
        Var terms = q.log().sub(logNot).add(all);   // log p_i + sum_{j != i} log(1 - p_j)
        return terms.logsumexp(-1, false).neg();
    }

    /**
     * The semantic loss of "at most {@code k} of these are true", per row, by a dynamic
     * program over the count of true variables.
     *
     * @param p probabilities {@code [batch, n]}
     * @param k the most that may be true
     * @return {@code [batch]} values {@code -log P(count <= k)}
     */
    public static Var atMostK(Var p, int k) {
        int n = p.shape().dim(1);
        int b = p.shape().dim(0);
        Var q = p.clamp(EPS, 1 - EPS);
        List<Var> count = new ArrayList<>();   // count.get(j) = P(j true so far), [b, 1]
        count.add(Var.of(Tensor.full(Shape.of(b, 1), 1f)));
        for (int i = 0; i < n; i++) {
            Var pi = q.slice(1, i, i + 1);
            Var notPi = pi.neg().add(1f);
            List<Var> next = new ArrayList<>();
            for (int j = 0; j <= Math.min(k, i + 1); j++) {
                Var stay = j < count.size() ? count.get(j).mul(notPi) : null;
                Var up = j > 0 && j - 1 < count.size() ? count.get(j - 1).mul(pi) : null;
                next.add(stay == null ? up : up == null ? stay : stay.add(up));
            }
            count = next;
        }
        Var total = count.get(0);
        for (int j = 1; j < count.size(); j++) {
            total = total.add(count.get(j));
        }
        return total.add(EPS).log().neg().reshape(b);
    }

    /**
     * The semantic loss of an {@code n x n} Latin square's constraints: for every cell,
     * every (row, value) and every (column, value), exactly one variable is true. Cell
     * {@code (r, c)} taking value {@code v} is variable {@code (r * n + c) * n + v}.
     *
     * @param p probabilities {@code [batch, n * n * n]}, such as a per-cell softmax
     * @param n the order of the square
     * @param includeCells whether to add the per-cell constraints (redundant after a softmax)
     * @return the mean over the batch of the summed constraint losses
     */
    public static Var latinSquare(Var p, int n, boolean includeCells) {
        int b = p.shape().dim(0);
        List<int[]> groups = new ArrayList<>();
        for (int a = 0; a < n; a++) {
            for (int v = 0; v < n; v++) {
                int[] row = new int[n];
                int[] col = new int[n];
                for (int k = 0; k < n; k++) {
                    row[k] = (a * n + k) * n + v;
                    col[k] = (k * n + a) * n + v;
                }
                groups.add(row);
                groups.add(col);
            }
        }
        if (includeCells) {
            for (int cell = 0; cell < n * n; cell++) {
                int[] g = new int[n];
                for (int v = 0; v < n; v++) {
                    g[v] = cell * n + v;
                }
                groups.add(g);
            }
        }
        float[] idx = new float[groups.size() * n];
        for (int g = 0; g < groups.size(); g++) {
            for (int k = 0; k < n; k++) {
                idx[g * n + k] = groups.get(g)[k];
            }
        }
        Var gathered = p.gather(1, Tensor.of(Shape.of(groups.size() * n), idx))
                .reshape(b * groups.size(), n);
        return exactlyOne(gathered).reshape(b, groups.size()).sum(1, false).mean();
    }
}
