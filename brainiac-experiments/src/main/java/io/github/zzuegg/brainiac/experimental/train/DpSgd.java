package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Differentially private SGD (Abadi et al. 2016, "Deep Learning with Differential Privacy",
 * <a href="https://arxiv.org/abs/1607.00133">arXiv:1607.00133</a>). Each step:
 *
 * <ol>
 *   <li>samples every training example independently with probability {@code q}
 *       (Poisson sampling, which the {@link RdpAccountant} analysis assumes);</li>
 *   <li>computes each sampled example's gradient on its own (a micro-batch of one, with
 *       {@link Var#gradients}) and scales it to global L2 norm at most {@code C};</li>
 *   <li>adds Gaussian noise with standard deviation {@code sigma * C} to the sum, divides by
 *       the expected batch size {@code q n} and takes a plain SGD step.</li>
 * </ol>
 *
 * <p>The accountant records every step, so {@link #epsilon} reports the budget spent.
 * Noise multiplier 0 gives clipped, non-private SGD.
 */
public final class DpSgd {

    private final Model model;
    private final Loss loss;
    private double clipNorm = 1;
    private double noiseMultiplier = 1;
    private double learningRate = 0.1;
    private double samplingRate = 0.01;
    private long seed = 1;
    private RdpAccountant accountant;
    private SplittableRandom random;

    private DpSgd(Model model, Loss loss) {
        this.model = Objects.requireNonNull(model);
        this.loss = Objects.requireNonNull(loss);
    }

    /**
     * DP-SGD for a model.
     *
     * @param model the model, trained in place
     * @param loss the loss of one example
     * @return the settings
     */
    public static DpSgd of(Model model, Loss loss) {
        return new DpSgd(model, loss);
    }

    /**
     * @param c the per-example gradient norm bound
     * @return this
     */
    public DpSgd clipNorm(double c) {
        this.clipNorm = c;
        return this;
    }

    /**
     * @param sigma the noise standard deviation over the clipping norm
     * @return this
     */
    public DpSgd noiseMultiplier(double sigma) {
        this.noiseMultiplier = sigma;
        return this;
    }

    /**
     * @param lr the SGD step size
     * @return this
     */
    public DpSgd learningRate(double lr) {
        this.learningRate = lr;
        return this;
    }

    /**
     * @param q the per-example sampling probability
     * @return this
     */
    public DpSgd samplingRate(double q) {
        this.samplingRate = q;
        return this;
    }

    /**
     * @param s the seed for sampling and noise
     * @return this
     */
    public DpSgd seed(long s) {
        this.seed = s;
        return this;
    }

    /**
     * Trains for a number of steps.
     *
     * @param data the private training data
     * @param steps the steps
     * @return this
     */
    public DpSgd train(Dataset data, int steps) {
        for (int s = 0; s < steps; s++) {
            step(data);
        }
        return this;
    }

    /**
     * One DP-SGD step.
     *
     * @param data the private training data
     * @return the number of examples sampled
     */
    public int step(Dataset data) {
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
        for (int i = 0; i < data.size(); i++) {
            if (random.nextDouble() >= samplingRate) {
                continue;
            }
            sampled++;
            Tensor x = Tensor.of(Shape.of(1, data.featureCount()), data.featuresOf(i));
            Tensor y = Tensor.of(Shape.of(1, data.targetCount()), data.targetsOf(i));
            Var l = loss.compute(model.forward(Var.of(x), true), y);
            List<Var> grads = Var.gradients(l, params, false);
            float[][] g = new float[params.size()][];
            double norm = 0;
            for (int p = 0; p < g.length; p++) {
                g[p] = grads.get(p).value().to(DType.FLOAT32).toArray();
                for (float v : g[p]) {
                    norm += v * (double) v;
                }
            }
            double scale = Math.min(1, clipNorm / Math.max(Math.sqrt(norm), 1e-12));
            for (int p = 0; p < g.length; p++) {
                for (int j = 0; j < g[p].length; j++) {
                    sum[p][j] += (float) (g[p][j] * scale);
                }
            }
        }
        double expected = samplingRate * data.size();
        for (int p = 0; p < params.size(); p++) {
            Var param = params.get(p);
            float[] w = param.value().to(DType.FLOAT32).toArray();
            for (int j = 0; j < w.length; j++) {
                double noisy = sum[p][j] + random.nextGaussian() * noiseMultiplier * clipNorm;
                w[j] -= (float) (learningRate * noisy / expected);
            }
            param.assign(Tensor.of(param.shape(), w, param.value().device()).to(param.value().dtype()));
        }
        accountant.step(1);
        return sampled;
    }

    /**
     * @param delta the failure probability
     * @return the {@code ε} spent so far; infinite without noise
     */
    public double epsilon(double delta) {
        return accountant == null ? 0 : accountant.epsilon(delta);
    }
}
