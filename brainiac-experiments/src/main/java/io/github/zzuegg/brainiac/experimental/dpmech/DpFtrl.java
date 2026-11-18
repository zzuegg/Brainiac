package io.github.zzuegg.brainiac.experimental.dpmech;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.train.RdpAccountant;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * DP-FTRL in its matrix-factorisation form: gradient descent whose per-step noise is the
 * correlated {@code C⁻¹ z} of a {@link CorrelatedNoise} mechanism instead of independent
 * Gaussians, with the per-example clipping of {@code DpSgd}. Data are processed in
 * shuffled epochs of fixed batches (no privacy amplification by sampling), so each example
 * participates once per epoch, {@code k = epochs} times, {@code steps per epoch} apart. The
 * whole run is one Gaussian mechanism with sensitivity {@code clip · sens_k(C)} and noise
 * {@code σ · clip · sens_k(C)}, so its {@code ε} is that of a single Gaussian with
 * multiplier {@code σ} ({@link RdpAccountant} at sampling rate 1); with
 * {@link CorrelatedNoise#independent} this is exactly DP-SGD without amplification
 * (Kairouz et al., ICML 2021, <a href="https://arxiv.org/abs/2103.00039">arXiv:2103.00039</a>;
 * Choquette-Choo et al., NeurIPS 2023, <a href="https://arxiv.org/abs/2306.08153">arXiv:2306.08153</a>).
 */
public final class DpFtrl {

    private final Model model;
    private final Loss loss;
    private CorrelatedNoise noise;
    private double clipNorm = 1;
    private double noiseMultiplier = 1;
    private double learningRate = 0.5;
    private int batchSize = 32;
    private double momentum;
    private long seed = 1;

    private DpFtrl(Model model, Loss loss) {
        this.model = Objects.requireNonNull(model);
        this.loss = Objects.requireNonNull(loss);
    }

    /**
     * DP-FTRL for a model.
     *
     * @param model the model, trained in place
     * @param loss the loss of one example
     * @return the settings
     */
    public static DpFtrl of(Model model, Loss loss) {
        return new DpFtrl(model, loss);
    }

    /**
     * @param n the noise mechanism, sized for the run's steps
     * @return this
     */
    public DpFtrl noise(CorrelatedNoise n) {
        this.noise = Objects.requireNonNull(n);
        return this;
    }

    /**
     * @param c the per-example gradient norm bound
     * @return this
     */
    public DpFtrl clipNorm(double c) {
        this.clipNorm = c;
        return this;
    }

    /**
     * @param sigma the noise multiplier
     * @return this
     */
    public DpFtrl noiseMultiplier(double sigma) {
        this.noiseMultiplier = sigma;
        return this;
    }

    /**
     * @param lr the step size
     * @return this
     */
    public DpFtrl learningRate(double lr) {
        this.learningRate = lr;
        return this;
    }

    /**
     * @param b the batch size
     * @return this
     */
    public DpFtrl batchSize(int b) {
        this.batchSize = b;
        return this;
    }

    /**
     * @param m heavy-ball momentum on the noisy gradient, 0 for none
     * @return this
     */
    public DpFtrl momentum(double m) {
        this.momentum = m;
        return this;
    }

    /**
     * @param s the seed of shuffling and noise
     * @return this
     */
    public DpFtrl seed(long s) {
        this.seed = s;
        return this;
    }

    /**
     * The steps of a run.
     *
     * @param examples the training set size
     * @param batch the batch size
     * @param epochs the epochs
     * @return {@code epochs · ⌊n / batch⌋}
     */
    public static int steps(int examples, int batch, int epochs) {
        return epochs * (examples / batch);
    }

    /**
     * Trains.
     *
     * @param data the private data
     * @param epochs epochs, the participations per example
     * @return this
     */
    public DpFtrl train(Dataset data, int epochs) {
        int perEpoch = data.size() / batchSize;
        int total = epochs * perEpoch;
        CorrelatedNoise mech = noise == null ? CorrelatedNoise.independent(total) : noise;
        if (mech.steps() < total) {
            throw new IllegalArgumentException("noise mechanism sized for " + mech.steps() + " < " + total + " steps");
        }
        double sens = mech.sensitivity(epochs, perEpoch);
        List<Var> params = model.parameters();
        int dim = 0;
        for (Var p : params) {
            dim += Math.toIntExact(p.value().size());
        }
        SplittableRandom r = new SplittableRandom(seed);
        CorrelatedNoise.Stream stream = mech.stream(dim, r.split());
        double[] velocity = new double[dim];
        for (int e = 0; e < epochs; e++) {
            int[] order = PrivateSelection.permutation(data.size(), r);
            for (int b = 0; b < perEpoch; b++) {
                double[] g = new double[dim];
                for (int i = b * batchSize; i < (b + 1) * batchSize; i++) {
                    addClipped(data, order[i], params, g);
                }
                double[] z = stream.next();
                double scale = noiseMultiplier * clipNorm * sens;
                for (int j = 0; j < dim; j++) {
                    double step = (g[j] + scale * z[j]) / batchSize;
                    velocity[j] = momentum * velocity[j] + step;
                }
                apply(params, velocity);
            }
        }
        return this;
    }

    private void addClipped(Dataset data, int i, List<Var> params, double[] g) {
        Tensor x = Tensor.of(Shape.of(1, data.featureCount()), data.featuresOf(i));
        Tensor y = Tensor.of(Shape.of(1, data.targetCount()), data.targetsOf(i));
        Var l = loss.compute(model.forward(Var.of(x), true), y);
        List<Var> grads = Var.gradients(l, params, false);
        float[][] parts = new float[params.size()][];
        double norm = 0;
        for (int p = 0; p < parts.length; p++) {
            parts[p] = grads.get(p).value().to(DType.FLOAT32).toArray();
            for (float v : parts[p]) {
                norm += v * (double) v;
            }
        }
        double s = Math.min(1, clipNorm / Math.max(Math.sqrt(norm), 1e-12));
        int o = 0;
        for (float[] part : parts) {
            for (float v : part) {
                g[o++] += v * s;
            }
        }
    }

    private void apply(List<Var> params, double[] step) {
        int o = 0;
        for (Var p : params) {
            float[] w = p.value().to(DType.FLOAT32).toArray();
            for (int j = 0; j < w.length; j++) {
                w[j] -= (float) (learningRate * step[o++]);
            }
            p.assign(Tensor.of(p.shape(), w, p.value().device()).to(p.value().dtype()));
        }
    }

    /**
     * The run's privacy.
     *
     * @param delta the failure probability
     * @return the {@code ε} of one Gaussian mechanism with the noise multiplier
     */
    public double epsilon(double delta) {
        return RdpAccountant.epsilon(1, noiseMultiplier, 1, delta);
    }
}
