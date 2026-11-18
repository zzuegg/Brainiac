package io.github.zzuegg.brainiac.experimental.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.train.DpSgd;
import io.github.zzuegg.brainiac.experimental.train.RdpAccountant;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.BiFunction;

/**
 * DP-SGD with alternative per-example clipping, sharing {@link DpSgd#clipPerExample} and
 * {@link DpSgd#addNoise} and accounted by the unchanged {@link RdpAccountant}:
 *
 * <ul>
 *   <li>{@link Strategy#FIXED}: the usual clip norm {@code C}.</li>
 *   <li>{@link Strategy#ADAPTIVE_QUANTILE}: the clip norm tracks a target quantile {@code γ} of
 *       the per-example gradient norms with the private geometric update
 *       {@code C ← C exp(-η (b̃ - γ))}, where {@code b̃} is the noisy share of unclipped examples
 *       (noise {@code σ_b}); the gradient noise is raised to
 *       {@code σ_Δ = (σ^-2 - (2σ_b)^-2)^(-1/2)} so the total privacy cost equals plain DP-SGD at
 *       {@code σ} (Andrew, Thakkar, McMahan and Ramaswamy, NeurIPS 2021,
 *       <a href="https://arxiv.org/abs/1905.03871">arXiv:1905.03871</a>).</li>
 *   <li>{@link Strategy#AUTOMATIC}: every per-example gradient is normalised,
 *       {@code g / (||g|| + γ)}, so the sensitivity is 1 and no clip norm exists (Bu, Wang, Zha
 *       and Karypis, NeurIPS 2023, <a href="https://arxiv.org/abs/2206.07136">arXiv:2206.07136</a>).</li>
 *   <li>{@link Strategy#AUG_MULTIPLICITY}: each sampled example's gradient is averaged over
 *       {@code K} augmentations <em>before</em> clipping, which lowers its variance at no privacy
 *       cost because the example still contributes one clipped vector (De, Berrada, Hayes, Smith
 *       and Balle, 2022, <a href="https://arxiv.org/abs/2204.13650">arXiv:2204.13650</a>).</li>
 * </ul>
 */
public final class ClippingStrategies {

    /** The clipping strategy. */
    public enum Strategy {
        /** a fixed clip norm */
        FIXED,
        /** a clip norm tracking a quantile of the gradient norms */
        ADAPTIVE_QUANTILE,
        /** per-example normalisation */
        AUTOMATIC,
        /** gradients averaged over augmentations before clipping */
        AUG_MULTIPLICITY
    }

    private final Model model;
    private final Loss loss;
    private final Strategy strategy;
    private double clipNorm = 1;
    private double noiseMultiplier = 1;
    private double learningRate = 0.1;
    private double samplingRate = 0.01;
    private double targetQuantile = 0.5;
    private double quantileRate = 0.2;
    private double countNoise = -1;
    private double stability = 0.01;
    private int augmentations = 4;
    private BiFunction<float[], SplittableRandom, float[]> augment = (x, r) -> x;
    private long seed = 1;
    private RdpAccountant accountant;
    private SplittableRandom random;

    private ClippingStrategies(Model model, Loss loss, Strategy strategy) {
        this.model = Objects.requireNonNull(model);
        this.loss = Objects.requireNonNull(loss);
        this.strategy = Objects.requireNonNull(strategy);
    }

    /**
     * A trainer.
     *
     * @param model the model, trained in place
     * @param loss the loss of one example
     * @param strategy the clipping strategy
     * @return the settings
     */
    public static ClippingStrategies of(Model model, Loss loss, Strategy strategy) {
        return new ClippingStrategies(model, loss, strategy);
    }

    /**
     * @param c the clip norm (the initial one for the adaptive strategy)
     * @return this
     */
    public ClippingStrategies clipNorm(double c) {
        this.clipNorm = c;
        return this;
    }

    /**
     * @param sigma the noise multiplier {@code σ} of the whole mechanism
     * @return this
     */
    public ClippingStrategies noiseMultiplier(double sigma) {
        this.noiseMultiplier = sigma;
        return this;
    }

    /**
     * @param lr the SGD step size
     * @return this
     */
    public ClippingStrategies learningRate(double lr) {
        this.learningRate = lr;
        return this;
    }

    /**
     * @param q the Poisson sampling rate
     * @return this
     */
    public ClippingStrategies samplingRate(double q) {
        this.samplingRate = q;
        return this;
    }

    /**
     * Adaptive clipping settings.
     *
     * @param quantile the target quantile {@code γ}
     * @param rate the geometric step {@code η}
     * @param sigmaB the count noise {@code σ_b}; negative for the paper's {@code q N / 20}
     * @return this
     */
    public ClippingStrategies adaptive(double quantile, double rate, double sigmaB) {
        this.targetQuantile = quantile;
        this.quantileRate = rate;
        this.countNoise = sigmaB;
        return this;
    }

    /**
     * @param gamma the stability constant {@code γ} of automatic clipping
     * @return this
     */
    public ClippingStrategies stability(double gamma) {
        this.stability = gamma;
        return this;
    }

    /**
     * Augmentation multiplicity settings.
     *
     * @param k the augmentations per example
     * @param augmentation draws an augmented copy of an input
     * @return this
     */
    public ClippingStrategies augmentation(int k, BiFunction<float[], SplittableRandom, float[]> augmentation) {
        this.augmentations = k;
        this.augment = Objects.requireNonNull(augmentation);
        return this;
    }

    /**
     * @param s the seed of sampling and noise
     * @return this
     */
    public ClippingStrategies seed(long s) {
        this.seed = s;
        return this;
    }

    /** @return the current clip norm */
    public double currentClipNorm() {
        return clipNorm;
    }

    private float[][] gradient(float[] x, float[] y, Dataset data, List<Var> params) {
        Tensor xt = Tensor.of(Shape.of(1, data.featureCount()), x);
        Tensor yt = Tensor.of(Shape.of(1, data.targetCount()), y);
        Var l = loss.compute(model.forward(Var.of(xt), true), yt);
        List<Var> grads = Var.gradients(l, params, false);
        float[][] g = new float[params.size()][];
        for (int p = 0; p < g.length; p++) {
            g[p] = grads.get(p).value().to(DType.FLOAT32).toArray();
        }
        return g;
    }

    /**
     * Trains for a number of steps.
     *
     * @param data the private data
     * @param steps the steps
     * @return this
     */
    public ClippingStrategies train(Dataset data, int steps) {
        for (int s = 0; s < steps; s++) {
            step(data);
        }
        return this;
    }

    /**
     * One step.
     *
     * @param data the private data
     * @return the number of examples sampled
     */
    public int step(Dataset data) {
        double expected = samplingRate * data.size();
        double sigmaB = countNoise >= 0 ? countNoise : expected / 20;
        double gradientSigma = noiseMultiplier;
        if (strategy == Strategy.ADAPTIVE_QUANTILE) {
            double inv = 1 / (noiseMultiplier * noiseMultiplier) - 1 / (4 * sigmaB * sigmaB);
            gradientSigma = inv > 0 ? 1 / Math.sqrt(inv) : noiseMultiplier * 10;
        }
        if (accountant == null) {
            accountant = RdpAccountant.of(samplingRate, noiseMultiplier);
            random = new SplittableRandom(seed);
        }
        List<Var> params = model.parameters();
        float[][] sum = new float[params.size()][];
        for (int p = 0; p < sum.length; p++) {
            sum[p] = new float[Math.toIntExact(params.get(p).value().size())];
        }
        int sampled = 0;
        int unclipped = 0;
        double bound = strategy == Strategy.AUTOMATIC ? 1 : clipNorm;
        for (int i = 0; i < data.size(); i++) {
            if (random.nextDouble() >= samplingRate) {
                continue;
            }
            sampled++;
            float[] x = data.featuresOf(i);
            float[] y = data.targetsOf(i);
            float[][] g;
            if (strategy == Strategy.AUG_MULTIPLICITY) {
                g = null;
                for (int a = 0; a < augmentations; a++) {
                    float[][] ga = gradient(augment.apply(x, random), y, data, params);
                    if (g == null) {
                        g = ga;
                    } else {
                        for (int p = 0; p < g.length; p++) {
                            for (int j = 0; j < g[p].length; j++) {
                                g[p][j] += ga[p][j];
                            }
                        }
                    }
                }
                for (float[] row : g) {
                    for (int j = 0; j < row.length; j++) {
                        row[j] /= augmentations;
                    }
                }
            } else {
                g = gradient(x, y, data, params);
            }
            if (strategy == Strategy.AUTOMATIC) {
                double norm = 0;
                for (float[] row : g) {
                    for (float v : row) {
                        norm += v * (double) v;
                    }
                }
                double scale = 1 / (Math.sqrt(norm) + stability);
                for (float[] row : g) {
                    for (int j = 0; j < row.length; j++) {
                        row[j] = (float) (row[j] * scale);
                    }
                }
            } else {
                double norm = DpSgd.clipPerExample(g, clipNorm);
                unclipped += norm <= clipNorm ? 1 : 0;
            }
            for (int p = 0; p < g.length; p++) {
                for (int j = 0; j < g[p].length; j++) {
                    sum[p][j] += g[p][j];
                }
            }
        }
        double[][] noisy = DpSgd.addNoise(sum, gradientSigma, bound, random);
        for (int p = 0; p < params.size(); p++) {
            Var param = params.get(p);
            float[] w = param.value().to(DType.FLOAT32).toArray();
            for (int j = 0; j < w.length; j++) {
                w[j] -= (float) (learningRate * noisy[p][j] / expected);
            }
            param.assign(Tensor.of(param.shape(), w, param.value().device()).to(param.value().dtype()));
        }
        if (strategy == Strategy.ADAPTIVE_QUANTILE) {
            double share = (unclipped + random.nextGaussian() * sigmaB) / expected;
            clipNorm *= Math.exp(-quantileRate * (share - targetQuantile));
        }
        accountant.step(1);
        return sampled;
    }

    /**
     * @param delta the failure probability
     * @return the {@code ε} spent so far
     */
    public double epsilon(double delta) {
        return accountant == null ? 0 : accountant.epsilon(delta);
    }
}
