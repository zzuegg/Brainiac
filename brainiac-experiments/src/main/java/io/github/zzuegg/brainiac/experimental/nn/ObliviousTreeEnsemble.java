package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A layer of differentiable oblivious decision trees, the block of NODE (Popov, Morozov and
 * Babenko, "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data", 2019,
 * <a href="https://arxiv.org/abs/1909.06312">arXiv:1909.06312</a>). Each of {@code T} trees has
 * depth {@code D}, and an oblivious tree asks the same question at every node of a level. Level
 * {@code l} of tree {@code t}:
 * <ul>
 *   <li>picks a column softly: {@code f = x . sparsemax(F_tl)}, with {@link Sparsemax} (the paper
 *       uses its smoother cousin entmax), so the choice becomes exactly sparse;</li>
 *   <li>splits softly: {@code c = sigmoid((f - b_tl) / tau_tl)}, with a learnable threshold and
 *       temperature.</li>
 * </ul>
 * The {@code 2^D} leaf weights are products of {@code c} or {@code 1 - c} over the levels, and
 * the tree outputs their weighted sum of learned leaf responses. The layer averages its trees.
 * Use {@link #initialize} to set the thresholds from data quantiles, as the paper does.
 *
 * <p>Input {@code [batch, columns]}, output {@code [batch, outputs]}.
 */
public final class ObliviousTreeEnsemble implements Layer {

    private final int columns;
    private final int trees;
    private final int depth;
    private final int outputs;
    private final Var selection;
    private final Var thresholds;
    private final Var logTemperatures;
    private final Var responses;

    /**
     * Creates the layer.
     *
     * @param columns input columns
     * @param trees the number of trees {@code T}
     * @param depth the depth {@code D} of each tree
     * @param outputs the output width
     * @param random the initialization source
     */
    public ObliviousTreeEnsemble(int columns, int trees, int depth, int outputs, SplittableRandom random) {
        this.columns = columns;
        this.trees = trees;
        this.depth = depth;
        this.outputs = outputs;
        this.selection = TabularOps.normal(random, 1f, trees * depth, columns);
        this.thresholds = TabularOps.normal(random, 0.5f, trees * depth);
        this.logTemperatures = TabularOps.zeros(trees * depth);
        this.responses = TabularOps.normal(random, 1f, trees << depth, outputs);
    }

    /**
     * @param trees the number of trees
     * @param depth the tree depth
     * @param outputs the output width
     * @return a spec for {@code Model.sequential().add(...)}
     */
    public static LayerSpec spec(int trees, int depth, int outputs) {
        return ClassicOps.spec(s -> ClassicOps.withLast(s, outputs),
                (s, r) -> new ObliviousTreeEnsemble(s.dim(-1), trees, depth, outputs, r));
    }

    /**
     * Data-aware initialization: every threshold becomes the value of its softly selected
     * column on a random training row (a random quantile), and every temperature the spread of
     * that column, so splits start inside the data.
     *
     * @param x training rows {@code [rows, columns]}
     * @param random the random source
     */
    public void initialize(Tensor x, SplittableRandom random) {
        float[] f = Var.noGrad(() -> features(Var.of(x))).value().toArray();
        int rows = x.shape().dim(0);
        int n = trees * depth;
        float[] b = new float[n];
        float[] t = new float[n];
        for (int k = 0; k < n; k++) {
            b[k] = f[random.nextInt(rows) * n + k];
            double mean = 0;
            double sq = 0;
            for (int i = 0; i < rows; i++) {
                mean += f[i * n + k];
                sq += f[i * n + k] * f[i * n + k];
            }
            mean /= rows;
            double sd = Math.sqrt(Math.max(1e-6, sq / rows - mean * mean));
            t[k] = (float) Math.log(0.5 * sd);
        }
        thresholds.assign(Tensor.of(thresholds.shape(), b));
        logTemperatures.assign(Tensor.of(logTemperatures.shape(), t));
    }

    private Var features(Var x) {
        return x.matmul(Sparsemax.apply(selection).transpose());
    }

    /**
     * @return the column-selection weights {@code [trees * depth, columns]}, each row a sparse
     *     distribution over the columns
     */
    public float[] selectionWeights() {
        return Var.noGrad(() -> Sparsemax.apply(selection)).value().toArray();
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        TabularOps.requireRows(input, columns, this);
        int b = input.shape().dim(0);
        Var c = features(input).sub(thresholds).mul(logTemperatures.neg().exp()).sigmoid().reshape(b, trees, depth);
        Var p = null;
        for (int l = 0; l < depth; l++) {
            Var cl = c.slice(2, l, l + 1);
            Var notCl = cl.neg().add(1);
            p = p == null ? Var.concat(2, cl, notCl) : Var.concat(2, p.mul(cl), p.mul(notCl));
        }
        return p.reshape(b, trees << depth).matmul(responses).div(trees);
    }

    @Override
    public List<Var> parameters() {
        return List.of(selection, thresholds, logTemperatures, responses);
    }

    @Override
    public String toString() {
        return "ObliviousTreeEnsemble(" + trees + " trees of depth " + depth + ", " + outputs + " outputs)";
    }
}
