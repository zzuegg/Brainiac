package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Turns a table row into one token per column, the input stage of the FT-Transformer
 * (Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data", 2021,
 * <a href="https://arxiv.org/abs/2106.11959">arXiv:2106.11959</a>), with the numerical
 * embeddings of Gorishniy, Rubachev and Babenko, "On Embeddings for Numerical Features in
 * Tabular Deep Learning" (2022, <a href="https://arxiv.org/abs/2203.05556">arXiv:2203.05556</a>).
 * <ul>
 *   <li>{@link Numeric#LINEAR}: numerical column {@code j} becomes {@code x_j w_j + b_j};</li>
 *   <li>{@link Numeric#PLR}: periodic, linear, ReLU: {@code ReLU(W_j [sin(2 pi c_j x_j),
 *       cos(2 pi c_j x_j)] + b_j)} with trainable frequencies {@code c_j ~ N(0, sigma)}. It lets
 *       even a plain MLP fit high-frequency functions of single columns;</li>
 *   <li>categorical columns look up one embedding per category plus a per-column bias;</li>
 *   <li>an optional trainable {@code [CLS]} token is prepended, whose output a transformer reads.</li>
 * </ul>
 *
 * <p>Input {@code [batch, numeric + categorical]}: the numerical columns first, then the
 * categorical ones as whole-number ids. Output {@code [batch, tokens, dim]} with
 * {@code tokens = numeric + categorical (+ 1)}.
 */
public final class FeatureTokenizer implements Layer {

    /** How numerical columns are embedded. */
    public enum Numeric {
        /** A per-column affine map of the scalar. */
        LINEAR,
        /** Periodic features of trainable frequency, a per-column linear map and a ReLU. */
        PLR
    }

    private final int numeric;
    private final int[] cardinalities;
    private final int dim;
    private final Numeric kind;
    private final Var weight;
    private final Var bias;
    private final Var frequencies;
    private final Var table;
    private final Var categoricalBias;
    private final Var cls;
    private final float[] offsets;

    /**
     * Creates a tokenizer.
     *
     * @param numeric numerical columns
     * @param cardinalities the number of categories of each categorical column (may be empty)
     * @param dim the token width
     * @param kind the numerical embedding
     * @param frequencies the number {@code k} of PLR frequencies per column (ignored for LINEAR)
     * @param sigma the standard deviation of the initial PLR frequencies
     * @param cls whether to prepend a {@code [CLS]} token
     * @param random the initialization source
     */
    public FeatureTokenizer(int numeric, int[] cardinalities, int dim, Numeric kind, int frequencies, float sigma,
            boolean cls, SplittableRandom random) {
        this.numeric = numeric;
        this.cardinalities = cardinalities.clone();
        this.dim = dim;
        this.kind = kind;
        float s = (float) (1 / Math.sqrt(dim));
        if (kind == Numeric.PLR) {
            this.frequencies = TabularOps.normal(random, sigma, numeric, frequencies);
            this.weight = TabularOps.normal(random, (float) Math.sqrt(2.0 / (2 * frequencies)), numeric,
                    2 * frequencies, dim);
        } else {
            this.frequencies = null;
            this.weight = TabularOps.normal(random, s, numeric, dim);
        }
        this.bias = TabularOps.normal(random, s, numeric, dim);
        int total = 0;
        offsets = new float[cardinalities.length];
        for (int c = 0; c < cardinalities.length; c++) {
            offsets[c] = total;
            total += cardinalities[c];
        }
        this.table = total > 0 ? TabularOps.normal(random, s, total, dim) : null;
        this.categoricalBias = total > 0 ? TabularOps.normal(random, s, cardinalities.length, dim) : null;
        this.cls = cls ? TabularOps.normal(random, s, 1, 1, dim) : null;
    }

    /**
     * @param numeric numerical columns
     * @param dim the token width
     * @param random the initialization source
     * @return a linear tokenizer with a {@code [CLS]} token and no categorical columns
     */
    public static FeatureTokenizer linear(int numeric, int dim, SplittableRandom random) {
        return new FeatureTokenizer(numeric, new int[0], dim, Numeric.LINEAR, 0, 0, true, random);
    }

    /**
     * @param numeric numerical columns
     * @param dim the token width
     * @param frequencies PLR frequencies per column
     * @param sigma the initial frequency scale
     * @param cls whether to prepend a {@code [CLS]} token
     * @param random the initialization source
     * @return a PLR tokenizer with no categorical columns
     */
    public static FeatureTokenizer plr(int numeric, int dim, int frequencies, float sigma, boolean cls,
            SplittableRandom random) {
        return new FeatureTokenizer(numeric, new int[0], dim, Numeric.PLR, frequencies, sigma, cls, random);
    }

    /** @return tokens per row */
    public int tokens() {
        return numeric + cardinalities.length + (cls != null ? 1 : 0);
    }

    /** @return the token width */
    public int dim() {
        return dim;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        TabularOps.requireRows(input, numeric + cardinalities.length, this);
        int b = input.shape().dim(0);
        List<Var> parts = new ArrayList<>();
        if (cls != null) {
            parts.add(cls.expand(Shape.of(b, 1, dim)));
        }
        if (numeric > 0) {
            Var x = (cardinalities.length > 0 ? input.slice(1, 0, numeric) : input).reshape(b, numeric, 1);
            if (kind == Numeric.PLR) {
                Var v = x.mul(frequencies).mul((float) (2 * Math.PI));
                Var periodic = Var.concat(2, v.sin(), v.cos());
                parts.add(TabularOps.perFeature(periodic, weight).add(bias).relu());
            } else {
                parts.add(x.mul(weight).add(bias));
            }
        }
        if (cardinalities.length > 0) {
            int c = cardinalities.length;
            float[] ids = input.slice(1, numeric, numeric + c).value().toArray();
            for (int i = 0; i < ids.length; i++) {
                ids[i] = Math.round(ids[i]) + offsets[i % c];
            }
            parts.add(table.gather(0, Tensor.of(Shape.of(b * c), ids)).reshape(b, c, dim).add(categoricalBias));
        }
        return parts.size() == 1 ? parts.get(0) : Var.concat(1, parts.toArray(new Var[0]));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(weight, bias));
        if (frequencies != null) {
            p.add(frequencies);
        }
        if (table != null) {
            p.add(table);
            p.add(categoricalBias);
        }
        if (cls != null) {
            p.add(cls);
        }
        return p;
    }

    @Override
    public String toString() {
        return "FeatureTokenizer(" + numeric + " " + kind + ", " + cardinalities.length + " categorical, " + dim + ")";
    }
}
