package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * TabM (Gorishniy, Kotelnikov and Babenko, "TabM: Advancing Tabular Deep Learning with
 * Parameter-Efficient Ensembling", 2024, <a href="https://arxiv.org/abs/2410.24210">arXiv:2410.24210</a>):
 * an MLP whose every layer is a {@link BatchEnsembleDense}, so one network holds {@code k}
 * weight-sharing members. Each member is trained on its own loss ({@link #members}) and the
 * prediction is the mean of the members ({@link #forward(Var, boolean)}): an ensemble at roughly
 * the cost of a single MLP. Optionally the columns are first embedded with
 * {@link FeatureTokenizer.Numeric#PLR} embeddings shared by all members.
 *
 * <p>Input {@code [batch, columns]}, output {@code [batch, outputs]} (member mean).
 */
public final class TabM implements Layer {

    private final int columns;
    private final int members;
    private final FeatureTokenizer embedding;
    private final List<BatchEnsembleDense> layers = new ArrayList<>();
    private final BatchEnsembleDense head;

    /**
     * Creates the model.
     *
     * @param columns input columns
     * @param width hidden width
     * @param depth hidden layers
     * @param members the ensemble size {@code k}
     * @param outputs the output width
     * @param embedding PLR (or linear) column embeddings without {@code [CLS]}, or {@code null} for raw columns
     * @param random the initialization source
     */
    public TabM(int columns, int width, int depth, int members, int outputs, FeatureTokenizer embedding,
            SplittableRandom random) {
        this.columns = columns;
        this.members = members;
        this.embedding = embedding;
        int in = embedding == null ? columns : embedding.tokens() * embedding.dim();
        for (int l = 0; l < depth; l++) {
            layers.add(new BatchEnsembleDense(l == 0 ? in : width, width, members, random));
        }
        head = new BatchEnsembleDense(depth == 0 ? in : width, outputs, members, random);
    }

    /**
     * @param width hidden width
     * @param depth hidden layers
     * @param members the ensemble size
     * @param outputs the output width
     * @return a spec of TabM on raw columns
     */
    public static LayerSpec spec(int width, int depth, int members, int outputs) {
        return ClassicOps.spec(s -> ClassicOps.withLast(s, outputs),
                (s, r) -> new TabM(s.dim(-1), width, depth, members, outputs, null, r));
    }

    /** @return the ensemble size */
    public int size() {
        return members;
    }

    /**
     * Every member's output, for training each member on its own loss.
     *
     * @param input {@code [batch, columns]}
     * @param training the training flag
     * @return {@code [batch, k, outputs]}
     */
    public Var members(Var input, boolean training) {
        TabularOps.requireRows(input, columns, this);
        int b = input.shape().dim(0);
        Var x = embedding == null ? input : embedding.forward(input, training)
                .reshape(b, embedding.tokens() * embedding.dim());
        int in = x.shape().dim(1);
        Var h = x.reshape(b, 1, in).expand(Shape.of(b, members, in));
        for (BatchEnsembleDense layer : layers) {
            h = layer.forward(h, training).relu();
        }
        return head.forward(h, training);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        return members(input, training).mean(1, false);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (embedding != null) {
            p.addAll(embedding.parameters());
        }
        for (BatchEnsembleDense layer : layers) {
            p.addAll(layer.parameters());
        }
        p.addAll(head.parameters());
        return p;
    }

    @Override
    public String toString() {
        return "TabM(" + columns + ", " + layers.size() + " layers, k=" + members + ")";
    }
}
