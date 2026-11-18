package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * RealMLP (Holzmüller, Grinsztajn and Steinwart, "Better by Default: Strong Pre-Tuned MLPs and
 * Boosted Trees on Tabular Data", 2024, <a href="https://arxiv.org/abs/2407.04491">arXiv:2407.04491</a>):
 * a plain MLP made strong by a bag of tuned defaults, of which this class carries the main ones:
 * <ul>
 *   <li><em>robust scaling and smooth clipping</em>, fit on the training rows:
 *       {@code z = (x - median) / IQR}, then {@code z / sqrt(1 + (z/3)^2)}, which tames outliers
 *       and heavy tails without a hard cut;</li>
 *   <li><em>PBLD embeddings</em> (periodic, bias, linear, DenseNet-style): each column becomes
 *       {@code [z_j, W_j cos(2 pi w_j z_j + b_j) + c_j]};</li>
 *   <li>a learnable <em>diagonal scaling</em> of the embedded features (a soft feature
 *       selection, initialised to 1);</li>
 *   <li><em>parametric activations</em> {@code f(h) = h + alpha (silu(h) - h)} with one
 *       {@code alpha} per neuron, initialised to 1 (plain SiLU);</li>
 *   <li>the <em>coslog4</em> learning-rate schedule {@link #learningRate(int, int, float)}, to be
 *       applied with {@code Optimizer.learningRate(float)} every step.</li>
 * </ul>
 * Label smoothing, NTK parametrisation and per-layer learning-rate factors of the paper are left
 * out.
 *
 * <p>Input {@code [batch, columns]} of raw features, output {@code [batch, outputs]}.
 */
public final class RealMlp implements Layer {

    private final int columns;
    private final int dim;
    private final Tensor center;
    private final Tensor scale;
    private final Var frequencies;
    private final Var phases;
    private final Var embedWeight;
    private final Var embedBias;
    private final Var diagonal;
    private final List<Var> weights = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();
    private final List<Var> alphas = new ArrayList<>();

    /**
     * Creates the model and fits its scaler.
     *
     * @param train the training features {@code [rows, columns]} the scaler is fit on
     * @param width hidden width
     * @param depth hidden layers
     * @param outputs the output width
     * @param dim the per-column embedding width, at least 2
     * @param random the initialization source
     */
    public RealMlp(Tensor train, int width, int depth, int outputs, int dim, SplittableRandom random) {
        this.columns = train.shape().dim(1);
        this.dim = dim;
        int rows = train.shape().dim(0);
        float[] x = train.toArray();
        float[] c = new float[columns];
        float[] s = new float[columns];
        double[] col = new double[rows];
        for (int j = 0; j < columns; j++) {
            for (int i = 0; i < rows; i++) {
                col[i] = x[i * columns + j];
            }
            Arrays.sort(col);
            c[j] = (float) col[rows / 2];
            double iqr = col[(3 * rows) / 4] - col[rows / 4];
            s[j] = (float) (iqr > 1e-8 ? iqr / 1.349 : 1);
        }
        this.center = Tensor.of(Shape.of(1, columns), c);
        this.scale = Tensor.of(Shape.of(1, columns), s);
        int k = 2 * dim;
        this.frequencies = TabularOps.normal(random, 0.1f, columns, k);
        float[] ph = new float[columns * k];
        for (int i = 0; i < ph.length; i++) {
            ph[i] = (float) (Math.PI * (2 * random.nextDouble() - 1));
        }
        this.phases = Var.param(Tensor.of(Shape.of(columns, k), ph));
        this.embedWeight = TabularOps.normal(random, (float) (1 / Math.sqrt(k)), columns, k, dim - 1);
        this.embedBias = TabularOps.zeros(columns, dim - 1);
        int in = columns * dim;
        this.diagonal = TabularOps.full(1f, in);
        for (int l = 0; l <= depth; l++) {
            int a = l == 0 ? in : width;
            int b = l == depth ? outputs : width;
            weights.add(Var.param(Tensor.normal(Shape.of(a, b), 0, (float) Math.sqrt((l == depth ? 1.0 : 2.0) / a),
                    random)));
            biases.add(TabularOps.zeros(b));
            if (l < depth) {
                alphas.add(TabularOps.full(1f, b));
            }
        }
    }

    /**
     * The coslog4 schedule of RealMLP: {@code base * (1 - cos(2 pi log2(1 + 15 t))) / 2} with
     * {@code t = step / total}, four cosine cycles evenly spaced in log time that end at zero.
     *
     * @param step the current step, from 0
     * @param total the number of steps
     * @param base the peak learning rate
     * @return the learning rate for this step
     */
    public static float learningRate(int step, int total, float base) {
        double t = (step + 0.5) / total;
        double u = Math.log1p(15 * t) / Math.log(2);
        return (float) (base * 0.5 * (1 - Math.cos(2 * Math.PI * u)));
    }

    /**
     * The preprocessed (robust-scaled, smoothly clipped) columns.
     *
     * @param input raw features {@code [batch, columns]}
     * @return {@code [batch, columns]}
     */
    public Var preprocess(Var input) {
        TabularOps.requireRows(input, columns, this);
        Var z = input.sub(center).div(scale);
        return z.div(z.mul(1f / 3).square().add(1).sqrt());
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        int b = input.shape().dim(0);
        Var z = preprocess(input).reshape(b, columns, 1);
        Var periodic = z.mul(frequencies).mul((float) (2 * Math.PI)).add(phases).cos();
        Var emb = Var.concat(2, z, TabularOps.perFeature(periodic, embedWeight).add(embedBias));
        Var h = emb.reshape(b, columns * dim).mul(diagonal);
        for (int l = 0; l < weights.size(); l++) {
            h = h.matmul(weights.get(l)).add(biases.get(l));
            if (l < alphas.size()) {
                h = h.add(h.mul(h.sigmoid()).sub(h).mul(alphas.get(l)));
            }
        }
        return h;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(frequencies, phases, embedWeight, embedBias, diagonal));
        p.addAll(weights);
        p.addAll(biases);
        p.addAll(alphas);
        return p;
    }

    @Override
    public String toString() {
        return "RealMlp(" + columns + " columns, " + alphas.size() + " hidden layers)";
    }
}
