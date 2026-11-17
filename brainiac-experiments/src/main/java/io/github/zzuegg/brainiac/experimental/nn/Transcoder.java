package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A transcoder: a sparse dictionary that maps an MLP's <em>input</em> to its <em>output</em>,
 * {@code y' = W_dec topk(relu(W_enc x + b_enc)) + b_dec (+ x W_skip)}. Where a sparse
 * autoencoder only describes what an activation contains, a transcoder's features describe
 * the computation: each one reads a direction of the MLP input (its encoder column) and
 * writes a direction of the output (its decoder row), so it can stand in for the MLP and be
 * read as a sparse, input-invariant circuit. The optional linear skip ("skip-transcoder")
 * takes over the linear part of the MLP, so features only have to explain the nonlinear
 * remainder, which improves the reconstruction at the same sparsity.
 *
 * <p>As a {@link Layer} it computes the replacement output, so it drops in where the MLP
 * was ({@link #spec} for model builders).
 *
 * <p>Dunefsky, Chlenski and Nanda 2024, "Transcoders Find Interpretable LLM Feature
 * Circuits", <a href="https://arxiv.org/abs/2406.11944">arXiv:2406.11944</a>; Paulo et al.
 * 2025, "Transcoders Beat Sparse Autoencoders for Interpretability"
 * (<a href="https://arxiv.org/abs/2501.18823">arXiv:2501.18823</a>) for the skip
 * connection.
 */
public final class Transcoder implements Layer {

    private final int inputs;
    private final int outputs;
    private final int features;
    private final int k;
    private final Var encoder;
    private final Var encoderBias;
    private final Var decoder;
    private final Var decoderBias;
    private final Var skip;

    private Transcoder(int inputs, int outputs, int features, int k, boolean withSkip, SplittableRandom r) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.features = features;
        this.k = k;
        float[] dec = new float[features * outputs];
        for (int f = 0; f < features; f++) {
            double n = 0;
            for (int d = 0; d < outputs; d++) {
                dec[f * outputs + d] = (float) r.nextGaussian();
                n += dec[f * outputs + d] * dec[f * outputs + d];
            }
            for (int d = 0; d < outputs; d++) {
                dec[f * outputs + d] /= (float) Math.sqrt(n);
            }
        }
        this.decoder = Var.param(Tensor.of(Shape.of(features, outputs), dec));
        this.encoder = Var.param(Tensor.normal(Shape.of(inputs, features), 0, (float) (1 / Math.sqrt(inputs)), r));
        this.encoderBias = Var.param(Tensor.zeros(Shape.of(features)));
        this.decoderBias = Var.param(Tensor.zeros(Shape.of(outputs)));
        this.skip = withSkip ? Var.param(Tensor.zeros(Shape.of(inputs, outputs))) : null;
    }

    /**
     * A transcoder.
     *
     * @param inputs the MLP input width
     * @param outputs the MLP output width
     * @param features the dictionary size
     * @param k active features per sample
     * @param withSkip whether to add a learned linear skip connection
     * @param seed the seed
     * @return the transcoder
     */
    public static Transcoder of(int inputs, int outputs, int features, int k, boolean withSkip, long seed) {
        return new Transcoder(inputs, outputs, features, k, withSkip, new SplittableRandom(seed));
    }

    /**
     * A builder spec: a transcoder that reads the incoming width and writes {@code outputs}.
     *
     * @param outputs the output width
     * @param features the dictionary size
     * @param k active features per sample
     * @param withSkip whether to add the linear skip
     * @return the spec
     */
    public static LayerSpec spec(int outputs, int features, int k, boolean withSkip) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = outputs;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new Transcoder(input.dim(-1), outputs, features, k, withSkip, random);
            }
        };
    }

    /**
     * The sparse codes of a batch of MLP inputs.
     *
     * @param x {@code [batch, inputs]}
     * @return {@code [batch, features]}
     */
    public Var encode(Var x) {
        return SparseCoding.topK(x.matmul(encoder).add(encoderBias).relu(), k);
    }

    /**
     * The predicted MLP output from codes and inputs.
     *
     * @param z codes {@code [batch, features]}
     * @param x the MLP inputs {@code [batch, inputs]}, used by the skip
     * @return {@code [batch, outputs]}
     */
    public Var decode(Var z, Var x) {
        Var y = z.matmul(decoder).add(decoderBias);
        return skip == null ? y : y.add(x.matmul(skip));
    }

    @Override
    public Var forward(Var x) {
        Shape s = x.shape();
        Var flat = x.reshape(Math.toIntExact(s.size() / inputs), inputs);
        int[] dims = s.toArray();
        dims[dims.length - 1] = outputs;
        return decode(encode(flat), flat).reshape(dims);
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public List<Var> parameters() {
        return skip == null ? List.of(encoder, encoderBias, decoder, decoderBias)
                : List.of(encoder, encoderBias, decoder, decoderBias, skip);
    }

    /**
     * The mean squared error of the predicted MLP output.
     *
     * @param x MLP inputs {@code [batch, inputs]}
     * @param y MLP outputs {@code [batch, outputs]}
     * @return the scalar loss
     */
    public Var loss(Var x, Var y) {
        return decode(encode(x), x).sub(y).square().sum(1, false).mean();
    }

    /**
     * Trains on recorded MLP input/output pairs with Adam. Features that stop firing are
     * re-initialised toward the worst-predicted samples.
     *
     * @param x MLP inputs, one row per sample
     * @param y the matching MLP outputs
     * @param steps optimizer steps
     * @param batch rows per step
     * @param learningRate the learning rate
     * @param seed the seed
     * @return the loss of every step
     */
    public List<Float> train(float[][] x, float[][] y, int steps, int batch, float learningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        float[] mean = new float[outputs];
        for (float[] row : y) {
            for (int d = 0; d < outputs; d++) {
                mean[d] += row[d] / y.length;
            }
        }
        decoderBias.assign(Tensor.of(Shape.of(outputs), mean));
        List<Float> losses = new ArrayList<>();
        long[] fired = new long[features];
        for (int step = 1; step <= steps; step++) {
            int[] idx = r.ints(batch, 0, x.length).toArray();
            Var xb = Var.of(rows(x, idx));
            Var yb = Var.of(rows(y, idx));
            Var loss = loss(xb, yb);
            loss.backward();
            losses.add(loss.value().item());
            opt.step(parameters());
            float[] z = encode(xb).value().to(DType.FLOAT32).toArray();
            for (int i = 0; i < z.length; i++) {
                fired[i % features] += z[i] > 0 ? 1 : 0;
            }
            if (step % 250 == 0 && step < steps * 3 / 4) {
                resample(fired, x, y, r);
                Arrays.fill(fired, 0);
            }
        }
        return losses;
    }

    private void resample(long[] fired, float[][] x, float[][] y, SplittableRandom r) {
        float[] enc = encoder.value().to(DType.FLOAT32).toArray();
        float[] dec = decoder.value().to(DType.FLOAT32).toArray();
        for (int f = 0; f < features; f++) {
            if (fired[f] > 0) {
                continue;
            }
            int i = r.nextInt(x.length);
            double nx = 0;
            for (float v : x[i]) {
                nx += v * v;
            }
            for (int d = 0; d < inputs; d++) {
                enc[d * features + f] = (float) (x[i][d] / Math.sqrt(nx + 1e-8));
            }
            float[] pred = decode(encode(Var.of(rows(x, new int[] {i}))), Var.of(rows(x, new int[] {i}))).value()
                    .to(DType.FLOAT32).toArray();
            double ne = 0;
            for (int d = 0; d < outputs; d++) {
                ne += (y[i][d] - pred[d]) * (y[i][d] - pred[d]);
            }
            for (int d = 0; d < outputs; d++) {
                dec[f * outputs + d] = (float) ((y[i][d] - pred[d]) / Math.sqrt(ne + 1e-8));
            }
        }
        encoder.assign(Tensor.of(encoder.shape(), enc));
        decoder.assign(Tensor.of(decoder.shape(), dec));
    }

    static Tensor rows(float[][] data, int[] idx) {
        int d = data[0].length;
        float[] out = new float[idx.length * d];
        for (int i = 0; i < idx.length; i++) {
            System.arraycopy(data[idx[i]], 0, out, i * d, d);
        }
        return Tensor.of(Shape.of(idx.length, d), out);
    }

    static Tensor rows(float[][] data) {
        int[] idx = new int[data.length];
        Arrays.setAll(idx, i -> i);
        return rows(data, idx);
    }

    /**
     * The codes of many samples.
     *
     * @param x MLP inputs, one row per sample
     * @return {@code [sample][feature]}
     */
    public float[][] codes(float[][] x) {
        return SparseCoding.rows(encode(Var.of(rows(x))));
    }

    /**
     * The share of the MLP output's variance the transcoder explains, {@code 1 - SSE/SST}.
     *
     * @param x MLP inputs
     * @param y MLP outputs
     * @return the explained variance
     */
    public double explainedVariance(float[][] x, float[][] y) {
        Var xv = Var.of(rows(x));
        float[] pred = decode(encode(xv), xv).value().to(DType.FLOAT32).toArray();
        return explainedVariance(y, pred);
    }

    /**
     * {@code 1 - SSE / SST} of a flat prediction against rows of targets.
     *
     * @param y the targets
     * @param pred the predictions, row-major
     * @return the explained variance
     */
    public static double explainedVariance(float[][] y, float[] pred) {
        int d = y[0].length;
        double[] mean = new double[d];
        for (float[] row : y) {
            for (int j = 0; j < d; j++) {
                mean[j] += row[j] / y.length;
            }
        }
        double sse = 0;
        double sst = 0;
        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < d; j++) {
                sse += (y[i][j] - pred[i * d + j]) * (y[i][j] - pred[i * d + j]);
                sst += (y[i][j] - mean[j]) * (y[i][j] - mean[j]);
            }
        }
        return 1 - sse / Math.max(sst, 1e-12);
    }

    /**
     * The input direction a feature reads (its encoder column).
     *
     * @param feature the feature
     * @return the direction
     */
    public float[] inputDirection(int feature) {
        float[] enc = encoder.value().to(DType.FLOAT32).toArray();
        float[] out = new float[inputs];
        for (int d = 0; d < inputs; d++) {
            out[d] = enc[d * features + feature];
        }
        return out;
    }

    /**
     * The output direction a feature writes (its decoder row).
     *
     * @param feature the feature
     * @return the direction
     */
    public float[] outputDirection(int feature) {
        float[] dec = decoder.value().to(DType.FLOAT32).toArray();
        return Arrays.copyOfRange(dec, feature * outputs, (feature + 1) * outputs);
    }

    /** @return the dictionary size */
    public int features() {
        return features;
    }

    /** @return whether the transcoder has a linear skip */
    public boolean hasSkip() {
        return skip != null;
    }

    @Override
    public String toString() {
        return "Transcoder(" + inputs + " -> " + features + " -> " + outputs + ", k=" + k
                + (skip == null ? "" : ", skip") + ")";
    }
}
