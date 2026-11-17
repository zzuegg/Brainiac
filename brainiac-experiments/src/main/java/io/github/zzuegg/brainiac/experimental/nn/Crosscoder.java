package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A sparse crosscoder for model diffing: one shared dictionary trained on the activations of
 * several models (typically a base model and its fine-tuned version) on the same inputs.
 * Each model {@code m} has its own encoder and decoder, the codes are shared:
 *
 * <pre>z = relu(sum_m (x_m - b_m) E_m + b),    x'_m = z D_m + b_m</pre>
 *
 * <p>and the sparsity penalty weighs each feature by the summed norms of its decoders,
 * {@code lambda sum_f z_f sum_m ||D_m[f]||}. A feature used by both models pays for both
 * decoders only if it really needs both, so features that exist in only one model end up
 * with a near-zero decoder in the other. The <em>relative decoder norm</em>
 * {@code ||D_ft[f]|| / (||D_base[f]|| + ||D_ft[f]||)} then sorts the dictionary into shared
 * features (about 0.5) and model-exclusive ones (near 0 or 1): the exclusive features of the
 * fine-tuned model show what fine-tuning added.
 *
 * <p>Lindsey et al. 2024, "Sparse Crosscoders for Cross-Layer Features and Model Diffing",
 * <a href="https://transformer-circuits.pub/2024/crosscoders/index.html">Transformer Circuits</a>.
 */
public final class Crosscoder {

    private final int[] dims;
    private final int features;
    private final float l1;
    private final Var[] encoders;
    private final Var[] decoders;
    private final Var[] decoderBiases;
    private final Var encoderBias;

    private Crosscoder(int[] dims, int features, float l1, SplittableRandom r) {
        this.dims = dims.clone();
        this.features = features;
        this.l1 = l1;
        int models = dims.length;
        this.encoders = new Var[models];
        this.decoders = new Var[models];
        this.decoderBiases = new Var[models];
        for (int m = 0; m < models; m++) {
            Tensor dec = Tensor.normal(Shape.of(features, dims[m]), 0, (float) (0.1 / Math.sqrt(dims[m])), r);
            decoders[m] = Var.param(dec);
            encoders[m] = Var.param(dec.transpose().mul(1f / models).reshape(dims[m], features));
            decoderBiases[m] = Var.param(Tensor.zeros(Shape.of(dims[m])));
        }
        this.encoderBias = Var.param(Tensor.zeros(Shape.of(features)));
    }

    /**
     * A crosscoder over models with the given activation widths.
     *
     * @param dims the activation width of each model
     * @param features the shared dictionary size
     * @param l1 the weight of the decoder-norm-weighted L1 penalty
     * @param seed the seed
     * @return the crosscoder
     */
    public static Crosscoder of(int[] dims, int features, float l1, long seed) {
        return new Crosscoder(dims, features, l1, new SplittableRandom(seed));
    }

    /**
     * The shared codes.
     *
     * @param x one batch per model, {@code [batch, dims[m]]}, the same inputs in the same order
     * @return {@code [batch, features]}
     */
    public Var encode(List<Var> x) {
        Var pre = encoderBias;
        for (int m = 0; m < dims.length; m++) {
            pre = x.get(m).sub(decoderBiases[m]).matmul(encoders[m]).add(pre);
        }
        return pre.relu();
    }

    /**
     * One model's reconstruction.
     *
     * @param z shared codes
     * @param model the model index
     * @return {@code [batch, dims[model]]}
     */
    public Var decode(Var z, int model) {
        return z.matmul(decoders[model]).add(decoderBiases[model]);
    }

    private Var decoderNorms() {
        Var total = null;
        for (Var d : decoders) {
            Var n = d.square().sum(1, false).add(1e-8f).sqrt();
            total = total == null ? n : total.add(n);
        }
        return total;
    }

    /**
     * The summed reconstruction errors plus the decoder-norm-weighted L1 penalty.
     *
     * @param x one batch per model
     * @return the scalar loss
     */
    public Var loss(List<Var> x) {
        Var z = encode(x);
        Var total = z.mul(decoderNorms()).sum(1, false).mean().mul(l1);
        for (int m = 0; m < dims.length; m++) {
            total = total.add(decode(z, m).sub(x.get(m)).square().sum(1, false).mean());
        }
        return total;
    }

    /** @return every trainable value */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int m = 0; m < dims.length; m++) {
            p.add(encoders[m]);
            p.add(decoders[m]);
            p.add(decoderBiases[m]);
        }
        p.add(encoderBias);
        return p;
    }

    /**
     * Trains on paired activations.
     *
     * @param data {@code [model][sample][dim]}, row {@code i} of every model from the same input
     * @param steps optimizer steps
     * @param batch samples per step
     * @param learningRate the Adam learning rate
     * @param seed the seed
     * @return the loss of every step
     */
    public List<Float> train(float[][][] data, int steps, int batch, float learningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        for (int m = 0; m < dims.length; m++) {
            float[] mean = new float[dims[m]];
            for (float[] row : data[m]) {
                for (int d = 0; d < dims[m]; d++) {
                    mean[d] += row[d] / data[m].length;
                }
            }
            decoderBiases[m].assign(Tensor.of(Shape.of(dims[m]), mean));
        }
        List<Float> losses = new ArrayList<>();
        for (int s = 0; s < steps; s++) {
            int[] idx = r.ints(batch, 0, data[0].length).toArray();
            List<Var> xb = new ArrayList<>();
            for (float[][] model : data) {
                xb.add(Var.of(Transcoder.rows(model, idx)));
            }
            Var loss = loss(xb);
            loss.backward();
            losses.add(loss.value().item());
            opt.step(parameters());
        }
        return losses;
    }

    /**
     * The shared codes of many samples.
     *
     * @param data {@code [model][sample][dim]}
     * @return {@code [sample][feature]}
     */
    public float[][] codes(float[][][] data) {
        List<Var> xs = new ArrayList<>();
        for (float[][] model : data) {
            xs.add(Var.of(Transcoder.rows(model)));
        }
        return SparseCoding.rows(encode(xs));
    }

    /**
     * Each feature's decoder norm in one model divided by its summed norms over all models.
     *
     * @param model the model index
     * @return one value per feature in {@code [0, 1]}; 1 means exclusive to that model
     */
    public double[] relativeDecoderNorm(int model) {
        double[][] norms = new double[dims.length][features];
        for (int m = 0; m < dims.length; m++) {
            float[] d = decoders[m].value().to(DType.FLOAT32).toArray();
            for (int f = 0; f < features; f++) {
                double s = 0;
                for (int j = 0; j < dims[m]; j++) {
                    s += d[f * dims[m] + j] * d[f * dims[m] + j];
                }
                norms[m][f] = Math.sqrt(s);
            }
        }
        double[] out = new double[features];
        for (int f = 0; f < features; f++) {
            double total = 0;
            for (int m = 0; m < dims.length; m++) {
                total += norms[m][f];
            }
            out[f] = total == 0 ? 0 : norms[model][f] / total;
        }
        return out;
    }

    /**
     * The share of the activations' variance explained, pooled over models.
     *
     * @param data {@code [model][sample][dim]}
     * @return {@code 1 - SSE / SST}
     */
    public double explainedVariance(float[][][] data) {
        List<Var> xs = new ArrayList<>();
        for (float[][] model : data) {
            xs.add(Var.of(Transcoder.rows(model)));
        }
        Var z = encode(xs);
        double sse = 0;
        double sst = 0;
        for (int m = 0; m < dims.length; m++) {
            float[] p = decode(z, m).value().to(DType.FLOAT32).toArray();
            double[] mean = new double[dims[m]];
            for (float[] row : data[m]) {
                for (int d = 0; d < dims[m]; d++) {
                    mean[d] += row[d] / data[m].length;
                }
            }
            for (int i = 0; i < data[m].length; i++) {
                for (int d = 0; d < dims[m]; d++) {
                    double v = data[m][i][d];
                    sse += (v - p[i * dims[m] + d]) * (v - p[i * dims[m] + d]);
                    sst += (v - mean[d]) * (v - mean[d]);
                }
            }
        }
        return 1 - sse / Math.max(sst, 1e-12);
    }

    /** @return the dictionary size */
    public int features() {
        return features;
    }

    @Override
    public String toString() {
        return "Crosscoder(" + dims.length + " models -> " + features + ", l1=" + l1 + ")";
    }
}
