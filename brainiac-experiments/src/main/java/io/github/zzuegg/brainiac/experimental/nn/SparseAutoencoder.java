package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A sparse autoencoder that decomposes a layer's activations into many sparsely active,
 * usually interpretable features: {@code z = sparsify(relu(W_enc (x - b_dec) + b_enc))},
 * {@code x' = W_dec z + b_dec}, with an overcomplete dictionary (more features than
 * dimensions). Two ways to make {@code z} sparse:
 *
 * <ul>
 *   <li>{@link Sparsity#TOP_K}: keep the {@code k} largest pre-activations per sample
 *       (Gao et al. 2024, <a href="https://arxiv.org/abs/2406.04093">arXiv:2406.04093</a>);
 *       sparsity is exact and no penalty needs tuning;</li>
 *   <li>{@link Sparsity#L1}: an L1 penalty on {@code z}, with decoder columns kept at
 *       unit norm so the penalty cannot be dodged by shrinking codes (Bricken et al. 2023,
 *       <a href="https://transformer-circuits.pub/2023/monosemantic-features/">Towards
 *       Monosemanticity</a>).</li>
 * </ul>
 *
 * <p>Training resamples dead features (ones that never fire) toward badly reconstructed
 * inputs. {@link #stats} and {@link #topExamples} make a small feature browser. As a
 * {@link Layer}, the autoencoder returns its reconstruction, so it can be spliced into a
 * model's forward pass with a hook ({@link LayerHooks#at}) to see how much the features
 * explain.
 */
public final class SparseAutoencoder implements Layer {

    /** How codes are made sparse. */
    public enum Sparsity {
        /** keep the k largest */
        TOP_K,
        /** L1 penalty */
        L1
    }

    /**
     * Statistics of one feature over a data set.
     *
     * @param feature the feature index
     * @param frequency the share of samples where it fires
     * @param meanActivation its mean value where it fires
     * @param maxActivation its largest value
     */
    public record FeatureStats(int feature, double frequency, double meanActivation, double maxActivation) {
    }

    private final int dimension;
    private final int features;
    private final Sparsity sparsity;
    private final int k;
    private final float l1;
    private final Var encoder;
    private final Var encoderBias;
    private final Var decoder;
    private final Var decoderBias;

    private SparseAutoencoder(int dimension, int features, Sparsity sparsity, int k, float l1, SplittableRandom r) {
        this.dimension = dimension;
        this.features = features;
        this.sparsity = sparsity;
        this.k = k;
        this.l1 = l1;
        float[] dec = new float[features * dimension];
        for (int f = 0; f < features; f++) {
            double norm = 0;
            for (int d = 0; d < dimension; d++) {
                dec[f * dimension + d] = (float) r.nextGaussian();
                norm += dec[f * dimension + d] * dec[f * dimension + d];
            }
            for (int d = 0; d < dimension; d++) {
                dec[f * dimension + d] /= (float) Math.sqrt(norm);
            }
        }
        float[] enc = new float[dimension * features];
        for (int d = 0; d < dimension; d++) {
            for (int f = 0; f < features; f++) {
                enc[d * features + f] = dec[f * dimension + d];   // tied at the start
            }
        }
        decoder = Var.param(Tensor.of(Shape.of(features, dimension), dec));
        encoder = Var.param(Tensor.of(Shape.of(dimension, features), enc));
        encoderBias = Var.param(Tensor.zeros(Shape.of(features)));
        decoderBias = Var.param(Tensor.zeros(Shape.of(dimension)));
    }

    /**
     * A top-k autoencoder.
     *
     * @param dimension the activation width
     * @param features the dictionary size
     * @param k active features per sample
     * @param seed the seed
     * @return the autoencoder
     */
    public static SparseAutoencoder topK(int dimension, int features, int k, long seed) {
        return new SparseAutoencoder(dimension, features, Sparsity.TOP_K, k, 0, new SplittableRandom(seed));
    }

    /**
     * An L1-penalised autoencoder.
     *
     * @param dimension the activation width
     * @param features the dictionary size
     * @param l1 the penalty weight
     * @param seed the seed
     * @return the autoencoder
     */
    public static SparseAutoencoder l1(int dimension, int features, float l1, long seed) {
        return new SparseAutoencoder(dimension, features, Sparsity.L1, 0, l1, new SplittableRandom(seed));
    }

    /**
     * The sparse codes of a batch.
     *
     * @param x activations {@code [batch, dimension]}
     * @return codes {@code [batch, features]}
     */
    public Var encode(Var x) {
        Var z = x.sub(decoderBias).matmul(encoder).add(encoderBias).relu();
        if (sparsity == Sparsity.TOP_K) {
            z = z.mul(topKMask(z.value()));
        }
        return z;
    }

    private Tensor topKMask(Tensor z) {
        int rows = z.shape().dim(0);
        float[] v = z.to(DType.FLOAT32).toArray();
        float[] mask = new float[v.length];
        Integer[] idx = new Integer[features];
        for (int r = 0; r < rows; r++) {
            int base = r * features;
            for (int f = 0; f < features; f++) {
                idx[f] = f;
            }
            Arrays.sort(idx, (a, b) -> Float.compare(v[base + b], v[base + a]));
            for (int i = 0; i < Math.min(k, features); i++) {
                if (v[base + idx[i]] > 0) {
                    mask[base + idx[i]] = 1;
                }
            }
        }
        return Tensor.of(z.shape(), mask, z.device());
    }

    /**
     * Reconstructs activations from codes.
     *
     * @param z codes {@code [batch, features]}
     * @return activations {@code [batch, dimension]}
     */
    public Var decode(Var z) {
        return z.matmul(decoder).add(decoderBias);
    }

    @Override
    public Var forward(Var x) {
        Shape s = x.shape();
        Var flat = x.reshape(Math.toIntExact(s.size() / dimension), dimension);
        return decode(encode(flat)).reshape(s.toArray());
    }

    @Override
    public List<Var> parameters() {
        return List.of(encoder, encoderBias, decoder, decoderBias);
    }

    /**
     * The training loss of a batch: mean squared reconstruction error, plus the L1 penalty
     * for {@link Sparsity#L1}.
     *
     * @param x activations {@code [batch, dimension]}
     * @return the scalar loss
     */
    public Var loss(Var x) {
        Var z = encode(x);
        Var mse = decode(z).sub(x).square().sum(1, false).mean();
        return sparsity == Sparsity.L1 ? mse.add(z.abs().sum(1, false).mean().mul(l1)) : mse;
    }

    /**
     * Trains on recorded activations.
     *
     * @param data activations, one row per sample
     * @param steps optimizer steps
     * @param batch rows per step
     * @param learningRate the Adam learning rate
     * @param seed the seed for batches and resampling
     * @return the loss of every step
     */
    public List<Float> train(float[][] data, int steps, int batch, float learningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        // centre the decoder bias on the data mean: features then explain deviations
        float[] mean = new float[dimension];
        for (float[] row : data) {
            for (int d = 0; d < dimension; d++) {
                mean[d] += row[d] / data.length;
            }
        }
        decoderBias.assign(Tensor.of(Shape.of(dimension), mean));
        List<Float> losses = new ArrayList<>();
        long[] fired = new long[features];
        for (int step = 1; step <= steps; step++) {
            float[] rows = new float[batch * dimension];
            for (int b = 0; b < batch; b++) {
                System.arraycopy(data[r.nextInt(data.length)], 0, rows, b * dimension, dimension);
            }
            Var x = Var.of(Tensor.of(Shape.of(batch, dimension), rows));
            Var loss = loss(x);
            loss.backward();
            losses.add(loss.value().item());
            opt.step(parameters());
            if (sparsity == Sparsity.L1) {
                normalizeDecoder();
            }
            float[] z = encode(x).value().to(DType.FLOAT32).toArray();
            for (int i = 0; i < z.length; i++) {
                if (z[i] > 0) {
                    fired[i % features]++;
                }
            }
            if (step % 250 == 0 && step < steps * 3 / 4) {
                resampleDead(fired, data, r);
                Arrays.fill(fired, 0);
            }
        }
        return losses;
    }

    private void normalizeDecoder() {
        float[] dec = decoder.value().to(DType.FLOAT32).toArray();
        for (int f = 0; f < features; f++) {
            double norm = 0;
            for (int d = 0; d < dimension; d++) {
                norm += dec[f * dimension + d] * dec[f * dimension + d];
            }
            float s = (float) (1 / Math.max(Math.sqrt(norm), 1e-8));
            for (int d = 0; d < dimension; d++) {
                dec[f * dimension + d] *= s;
            }
        }
        decoder.assign(Tensor.of(decoder.shape(), dec, decoder.value().device()));
    }

    /** Points dead features at the residuals of badly reconstructed samples. */
    private void resampleDead(long[] fired, float[][] data, SplittableRandom r) {
        List<Integer> dead = new ArrayList<>();
        for (int f = 0; f < features; f++) {
            if (fired[f] == 0) {
                dead.add(f);
            }
        }
        if (dead.isEmpty()) {
            return;
        }
        int n = Math.min(data.length, 256);
        float[] rows = new float[n * dimension];
        for (int i = 0; i < n; i++) {
            System.arraycopy(data[r.nextInt(data.length)], 0, rows, i * dimension, dimension);
        }
        Var x = Var.of(Tensor.of(Shape.of(n, dimension), rows));
        float[] residual = x.sub(decode(encode(x))).value().to(DType.FLOAT32).toArray();
        Integer[] order = new Integer[n];
        double[] err = new double[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
            for (int d = 0; d < dimension; d++) {
                err[i] += residual[i * dimension + d] * residual[i * dimension + d];
            }
        }
        Arrays.sort(order, Comparator.comparingDouble(i -> -err[i]));
        float[] enc = encoder.value().to(DType.FLOAT32).toArray();
        float[] dec = decoder.value().to(DType.FLOAT32).toArray();
        float[] eb = encoderBias.value().to(DType.FLOAT32).toArray();
        for (int j = 0; j < dead.size(); j++) {
            int f = dead.get(j);
            int i = order[j % n];
            double norm = Math.sqrt(err[i]) + 1e-8;
            for (int d = 0; d < dimension; d++) {
                float u = (float) (residual[i * dimension + d] / norm);
                dec[f * dimension + d] = u;
                enc[d * features + f] = u * 0.2f;
            }
            eb[f] = 0;
        }
        encoder.assign(Tensor.of(encoder.shape(), enc, encoder.value().device()));
        decoder.assign(Tensor.of(decoder.shape(), dec, decoder.value().device()));
        encoderBias.assign(Tensor.of(encoderBias.shape(), eb, encoderBias.value().device()));
    }

    /**
     * The codes of many samples.
     *
     * @param data activations, one row per sample
     * @return codes {@code [sample][feature]}
     */
    public float[][] codes(float[][] data) {
        float[] flat = new float[data.length * dimension];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, flat, i * dimension, dimension);
        }
        float[] z = encode(Var.of(Tensor.of(Shape.of(data.length, dimension), flat))).value().to(DType.FLOAT32)
                .toArray();
        float[][] out = new float[data.length][features];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(z, i * features, out[i], 0, features);
        }
        return out;
    }

    /**
     * Per-feature statistics.
     *
     * @param data activations, one row per sample
     * @return one entry per feature
     */
    public List<FeatureStats> stats(float[][] data) {
        float[][] z = codes(data);
        List<FeatureStats> out = new ArrayList<>();
        for (int f = 0; f < features; f++) {
            int count = 0;
            double sum = 0;
            double max = 0;
            for (float[] row : z) {
                if (row[f] > 0) {
                    count++;
                    sum += row[f];
                    max = Math.max(max, row[f]);
                }
            }
            out.add(new FeatureStats(f, count / (double) z.length, count == 0 ? 0 : sum / count, max));
        }
        return out;
    }

    /**
     * The samples that activate a feature most.
     *
     * @param feature the feature
     * @param data activations, one row per sample
     * @param count how many
     * @return sample indices, strongest first
     */
    public int[] topExamples(int feature, float[][] data, int count) {
        float[][] z = codes(data);
        return java.util.stream.IntStream.range(0, z.length).boxed()
                .sorted(Comparator.comparingDouble(i -> -z[i][feature])).limit(count)
                .mapToInt(Integer::intValue).toArray();
    }

    /**
     * The share of the data's variance the reconstruction explains.
     *
     * @param data activations, one row per sample
     * @return {@code 1 - SSE / SST}
     */
    public double explainedVariance(float[][] data) {
        float[] flat = new float[data.length * dimension];
        for (int i = 0; i < data.length; i++) {
            System.arraycopy(data[i], 0, flat, i * dimension, dimension);
        }
        float[] rec = forward(Var.of(Tensor.of(Shape.of(data.length, dimension), flat))).value()
                .to(DType.FLOAT32).toArray();
        double[] mean = new double[dimension];
        for (float[] row : data) {
            for (int d = 0; d < dimension; d++) {
                mean[d] += row[d] / data.length;
            }
        }
        double sse = 0;
        double sst = 0;
        for (int i = 0; i < data.length; i++) {
            for (int d = 0; d < dimension; d++) {
                double x = data[i][d];
                sse += (x - rec[i * dimension + d]) * (x - rec[i * dimension + d]);
                sst += (x - mean[d]) * (x - mean[d]);
            }
        }
        return 1 - sse / Math.max(sst, 1e-12);
    }

    /**
     * A feature's direction in activation space (its decoder row).
     *
     * @param feature the feature
     * @return the direction
     */
    public float[] direction(int feature) {
        float[] dec = decoder.value().to(DType.FLOAT32).toArray();
        return Arrays.copyOfRange(dec, feature * dimension, (feature + 1) * dimension);
    }

    /** @return the dictionary size */
    public int features() {
        return features;
    }

    @Override
    public String toString() {
        return "SparseAutoencoder(" + dimension + " -> " + features + ", " + sparsity
                + (sparsity == Sparsity.TOP_K ? " k=" + k : " l1=" + l1) + ")";
    }
}
