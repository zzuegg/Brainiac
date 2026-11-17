package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A cross-layer transcoder (CLT): one sparse dictionary per layer that reads that layer's
 * MLP input and writes to the MLP outputs of that layer <em>and every later one</em>:
 *
 * <pre>a_l = topk(relu(x_l E_l + b_l)),    y'_m = sum_{l &lt;= m} a_l D_{l -&gt; m} + c_m</pre>
 *
 * <p>A feature computed once can thus feed several later MLPs directly, which removes the
 * duplicated features that per-layer {@link Transcoder}s need when a computation spreads
 * over layers ("cross-layer superposition"), and gives shorter, cleaner circuits. A model
 * whose MLPs are replaced by the CLT is the <em>replacement model</em>; with the features'
 * activations frozen it is linear in them, which is what attribution graphs are built from
 * (see {@code experimental.llm.AttributionGraph}).
 *
 * <p>With {@code crossLayer = false} the decoders stop at their own layer, which makes the
 * same object a stack of per-layer transcoders (the baseline).
 *
 * <p>Ameisen et al. 2025, "Circuit Tracing: Revealing Computational Graphs in Language
 * Models", <a href="https://transformer-circuits.pub/2025/attribution-graphs/methods.html">Transformer
 * Circuits</a>; Lindsey et al. 2024, "Sparse Crosscoders"
 * (<a href="https://transformer-circuits.pub/2024/crosscoders/index.html">Transformer Circuits</a>).
 */
public final class CrossLayerTranscoder {

    private final int layers;
    private final int inputs;
    private final int outputs;
    private final int features;
    private final int k;
    private final boolean crossLayer;
    private final Var[] encoders;
    private final Var[] encoderBiases;
    private final Var[][] decoders;
    private final Var[] outputBiases;

    private CrossLayerTranscoder(int layers, int inputs, int outputs, int features, int k, boolean crossLayer,
            SplittableRandom r) {
        this.layers = layers;
        this.inputs = inputs;
        this.outputs = outputs;
        this.features = features;
        this.k = k;
        this.crossLayer = crossLayer;
        this.encoders = new Var[layers];
        this.encoderBiases = new Var[layers];
        this.decoders = new Var[layers][layers];
        this.outputBiases = new Var[layers];
        for (int l = 0; l < layers; l++) {
            encoders[l] = Var.param(Tensor.normal(Shape.of(inputs, features), 0, (float) (1 / Math.sqrt(inputs)), r));
            encoderBiases[l] = Var.param(Tensor.zeros(Shape.of(features)));
            outputBiases[l] = Var.param(Tensor.zeros(Shape.of(outputs)));
            for (int m = l; m < layers; m++) {
                if (crossLayer || m == l) {
                    float scale = (float) (1 / Math.sqrt(outputs * (crossLayer ? m - l + 1.0 : 1.0)));
                    decoders[l][m] = Var.param(Tensor.normal(Shape.of(features, outputs), 0, scale, r));
                }
            }
        }
    }

    /**
     * A cross-layer transcoder.
     *
     * @param layers the number of MLP layers
     * @param inputs the MLP input width
     * @param outputs the MLP output width
     * @param features features per layer
     * @param k active features per layer and sample
     * @param seed the seed
     * @return the transcoder
     */
    public static CrossLayerTranscoder of(int layers, int inputs, int outputs, int features, int k, long seed) {
        return new CrossLayerTranscoder(layers, inputs, outputs, features, k, true, new SplittableRandom(seed));
    }

    /**
     * Per-layer transcoders with the same interface: every decoder writes only to its own
     * layer.
     *
     * @param layers the number of MLP layers
     * @param inputs the MLP input width
     * @param outputs the MLP output width
     * @param features features per layer
     * @param k active features per layer and sample
     * @param seed the seed
     * @return the stack
     */
    public static CrossLayerTranscoder perLayer(int layers, int inputs, int outputs, int features, int k, long seed) {
        return new CrossLayerTranscoder(layers, inputs, outputs, features, k, false, new SplittableRandom(seed));
    }

    /**
     * The codes of one layer.
     *
     * @param layer the layer
     * @param x that layer's MLP inputs {@code [batch, inputs]}
     * @return {@code [batch, features]}
     */
    public Var encode(int layer, Var x) {
        return SparseCoding.topK(x.matmul(encoders[layer]).add(encoderBiases[layer]).relu(), k);
    }

    /**
     * The predicted MLP output of one layer from the codes of it and every earlier layer.
     *
     * @param layer the layer
     * @param codes the codes of layers {@code 0..layer} (later entries are ignored)
     * @return {@code [batch, outputs]}
     */
    public Var decode(int layer, List<Var> codes) {
        Var y = null;
        for (int l = 0; l <= layer; l++) {
            if (decoders[l][layer] != null) {
                Var part = codes.get(l).matmul(decoders[l][layer]);
                y = y == null ? part : y.add(part);
            }
        }
        return y.add(outputBiases[layer]);
    }

    /**
     * Predicts every layer's MLP output from every layer's MLP input.
     *
     * @param x MLP inputs per layer
     * @return predicted MLP outputs per layer
     */
    public List<Var> reconstruct(List<Var> x) {
        List<Var> codes = new ArrayList<>();
        for (int l = 0; l < layers; l++) {
            codes.add(encode(l, x.get(l)));
        }
        List<Var> out = new ArrayList<>();
        for (int l = 0; l < layers; l++) {
            out.add(decode(l, codes));
        }
        return out;
    }

    /**
     * The summed mean squared error over layers.
     *
     * @param x MLP inputs per layer
     * @param y MLP outputs per layer
     * @return the scalar loss
     */
    public Var loss(List<Var> x, List<Var> y) {
        List<Var> pred = reconstruct(x);
        Var total = null;
        for (int l = 0; l < layers; l++) {
            Var e = pred.get(l).sub(y.get(l)).square().sum(1, false).mean();
            total = total == null ? e : total.add(e);
        }
        return total;
    }

    /** @return every trainable value */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int l = 0; l < layers; l++) {
            p.add(encoders[l]);
            p.add(encoderBiases[l]);
            p.add(outputBiases[l]);
            for (Var d : decoders[l]) {
                if (d != null) {
                    p.add(d);
                }
            }
        }
        return p;
    }

    /**
     * Trains on recorded MLP inputs and outputs of every layer.
     *
     * @param x {@code [layer][sample][inputs]}
     * @param y {@code [layer][sample][outputs]}
     * @param steps optimizer steps
     * @param batch samples per step
     * @param learningRate the Adam learning rate
     * @param seed the seed
     * @return the loss of every step
     */
    public List<Float> train(float[][][] x, float[][][] y, int steps, int batch, float learningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        for (int l = 0; l < layers; l++) {
            float[] mean = new float[outputs];
            for (float[] row : y[l]) {
                for (int d = 0; d < outputs; d++) {
                    mean[d] += row[d] / y[l].length;
                }
            }
            outputBiases[l].assign(Tensor.of(Shape.of(outputs), mean));
        }
        List<Float> losses = new ArrayList<>();
        int samples = x[0].length;
        for (int step = 0; step < steps; step++) {
            int[] idx = r.ints(batch, 0, samples).toArray();
            List<Var> xb = new ArrayList<>();
            List<Var> yb = new ArrayList<>();
            for (int l = 0; l < layers; l++) {
                xb.add(Var.of(Transcoder.rows(x[l], idx)));
                yb.add(Var.of(Transcoder.rows(y[l], idx)));
            }
            Var loss = loss(xb, yb);
            loss.backward();
            losses.add(loss.value().item());
            opt.step(parameters());
        }
        return losses;
    }

    /**
     * The share of every layer's MLP-output variance explained, pooled over layers.
     *
     * @param x {@code [layer][sample][inputs]}
     * @param y {@code [layer][sample][outputs]}
     * @return {@code 1 - SSE / SST}
     */
    public double explainedVariance(float[][][] x, float[][][] y) {
        List<Var> xs = new ArrayList<>();
        for (float[][] layer : x) {
            xs.add(Var.of(Transcoder.rows(layer)));
        }
        List<Var> pred = reconstruct(xs);
        double sse = 0;
        double sst = 0;
        for (int l = 0; l < layers; l++) {
            float[] p = pred.get(l).value().to(DType.FLOAT32).toArray();
            double ev = Transcoder.explainedVariance(y[l], p);
            double st = 0;
            double[] mean = new double[outputs];
            for (float[] row : y[l]) {
                for (int d = 0; d < outputs; d++) {
                    mean[d] += row[d] / y[l].length;
                }
            }
            for (float[] row : y[l]) {
                for (int d = 0; d < outputs; d++) {
                    st += (row[d] - mean[d]) * (row[d] - mean[d]);
                }
            }
            sse += (1 - ev) * st;
            sst += st;
        }
        return 1 - sse / Math.max(sst, 1e-12);
    }

    /**
     * The input direction of a feature.
     *
     * @param layer its layer
     * @param feature the feature
     * @return its encoder column
     */
    public float[] encoderDirection(int layer, int feature) {
        float[] e = encoders[layer].value().to(DType.FLOAT32).toArray();
        float[] out = new float[inputs];
        for (int d = 0; d < inputs; d++) {
            out[d] = e[d * features + feature];
        }
        return out;
    }

    /**
     * @param layer the feature's layer
     * @return the feature's encoder bias
     */
    public float[] encoderBias(int layer) {
        return encoderBiases[layer].value().to(DType.FLOAT32).toArray();
    }

    /**
     * What a feature writes to a later layer's MLP output.
     *
     * @param from the feature's layer
     * @param to the output layer, {@code >= from}
     * @param feature the feature
     * @return its decoder row, zeros if the feature does not write there
     */
    public float[] decoderDirection(int from, int to, int feature) {
        if (to < from || decoders[from][to] == null) {
            return new float[outputs];
        }
        float[] d = decoders[from][to].value().to(DType.FLOAT32).toArray();
        return Arrays.copyOfRange(d, feature * outputs, (feature + 1) * outputs);
    }

    /**
     * @param layer the output layer
     * @return the output bias of that layer
     */
    public float[] outputBias(int layer) {
        return outputBiases[layer].value().to(DType.FLOAT32).toArray();
    }

    /** @return the number of layers */
    public int layers() {
        return layers;
    }

    /** @return features per layer */
    public int features() {
        return features;
    }

    /** @return whether decoders reach later layers */
    public boolean crossLayer() {
        return crossLayer;
    }

    @Override
    public String toString() {
        return (crossLayer ? "CrossLayerTranscoder(" : "PerLayerTranscoders(") + layers + " x " + features
                + ", k=" + k + ")";
    }
}
