package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A neural additive model (Agarwal et al., "Neural Additive Models: Interpretable Machine
 * Learning with Neural Nets", 2021, <a href="https://arxiv.org/abs/2004.13912">arXiv:2004.13912</a>):
 * {@code y = b + sum_j g_j(x_j)}, where every column has its own small MLP {@code g_j}. The model
 * is a generalised additive model, so each learned shape function can be read off and plotted
 * ({@link #shape}); it gives up all interactions between columns. All columns' networks run as
 * one batched product. ReLU hidden units stand in for the paper's ExU units.
 *
 * <p>Input {@code [batch, columns]}, output {@code [batch, 1]}.
 */
public final class NeuralAdditiveModel implements Layer {

    private final int columns;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private final Var w3;
    private final Var bias;

    /**
     * Creates the model.
     *
     * @param columns input columns
     * @param hidden the hidden width of each column's two-layer network
     * @param random the initialization source
     */
    public NeuralAdditiveModel(int columns, int hidden, SplittableRandom random) {
        this.columns = columns;
        this.w1 = TabularOps.normal(random, 1.5f, columns, 1, hidden);
        this.b1 = TabularOps.normal(random, 1.5f, columns, 1, hidden);
        this.w2 = TabularOps.normal(random, (float) Math.sqrt(2.0 / hidden), columns, hidden, hidden);
        this.b2 = TabularOps.zeros(columns, 1, hidden);
        this.w3 = TabularOps.normal(random, (float) Math.sqrt(1.0 / hidden), columns, hidden, 1);
        this.bias = TabularOps.zeros(1);
    }

    /**
     * @param hidden the hidden width per column
     * @return a spec for {@code Model.sequential().add(...)}
     */
    public static LayerSpec spec(int hidden) {
        return ClassicOps.spec(s -> ClassicOps.withLast(s, 1), (s, r) -> new NeuralAdditiveModel(s.dim(-1), hidden, r));
    }

    /** Every column's contribution: {@code [batch, columns]} to {@code [columns, batch, 1]}. */
    private Var contributions(Var input) {
        int b = input.shape().dim(0);
        Var x = input.transpose().reshape(columns, b, 1);
        Var h = x.matmul(w1).add(b1).relu();
        h = h.matmul(w2).add(b2).relu();
        return h.matmul(w3);
    }

    /**
     * The learned shape function of one column, centred to mean zero over the given values.
     *
     * @param column the column
     * @param values the values to evaluate it at
     * @return {@code g_column(v) - mean}, one per value
     */
    public double[] shape(int column, double[] values) {
        int n = values.length;
        float[] x = new float[n * columns];
        for (int i = 0; i < n; i++) {
            x[i * columns + column] = (float) values[i];
        }
        float[] c = Var.noGrad(() -> contributions(Var.of(Tensor.of(Shape.of(n, columns), x)))).value().toArray();
        double[] out = new double[n];
        double mean = 0;
        for (int i = 0; i < n; i++) {
            out[i] = c[column * n + i];
            mean += out[i] / n;
        }
        for (int i = 0; i < n; i++) {
            out[i] -= mean;
        }
        return out;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        TabularOps.requireRows(input, columns, this);
        return contributions(input).sum(0, false).add(bias);
    }

    @Override
    public List<Var> parameters() {
        return List.of(w1, b1, w2, b2, w3, bias);
    }

    @Override
    public String toString() {
        return "NeuralAdditiveModel(" + columns + " columns)";
    }
}
