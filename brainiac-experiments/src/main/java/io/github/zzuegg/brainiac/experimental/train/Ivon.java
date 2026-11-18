package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * IVON, the improved variational online Newton method (Shen et al., ICML 2024,
 * <a href="https://arxiv.org/abs/2402.17641">arXiv:2402.17641</a>): mean-field Gaussian
 * variational learning at the cost of Adam.
 *
 * <p>The posterior over each weight is {@code N(m, 1 / (lambda (h + delta)))}, with
 * {@code lambda} the effective dataset size and {@code delta} the weight decay (a Gaussian
 * prior). Each step the gradient {@code g} is taken at a weight sample {@code m + sigma eps};
 * the reparameterisation trick gives a Hessian estimate {@code g eps / sigma} for free, and
 * {@code h} follows it by an exponential average with a Riemannian correction
 * {@code (1 - beta2)^2 / 2 (h - h_hat)^2 / (h + delta)} that keeps it positive. The mean moves
 * like Adam, {@code m -= lr (gbar + delta m) / (h + delta)}, but the "second moment" is a
 * posterior precision, so after training {@link #sample} draws weight samples for a predictive
 * ensemble and {@link #useMean} restores the mean.
 *
 * <p>Use it like any {@link Optimizer}: it leaves a fresh weight sample in the parameters after
 * every step, so the next forward and backward pass are taken at a sample. Call
 * {@link #useMean} before evaluating the mean network.
 */
public final class Ivon implements Optimizer {

    private float learningRate;
    private final float beta1;
    private final float beta2;
    private final float weightDecay;
    private final float effectiveSampleSize;
    private final float initialHessian;
    private final float clipRadius;
    private final SplittableRandom random;
    private final Map<Var, State> states = new IdentityHashMap<>();
    private int t;

    private static final class State {
        float[] mean;
        float[] hessian;
        float[] momentum;
        float[] noise;
    }

    /**
     * Creates IVON.
     *
     * @param learningRate the step size, for example 0.1 (larger than Adam's)
     * @param effectiveSampleSize {@code lambda}, usually the training-set size
     * @param weightDecay {@code delta}, the prior precision per example, for example 1e-4
     * @param initialHessian {@code h0}, for example 0.1 (sets the initial noise level)
     * @param beta1 the gradient momentum, for example 0.9
     * @param beta2 the Hessian averaging, for example 0.99999
     * @param seed the weight-noise seed
     */
    public Ivon(float learningRate, float effectiveSampleSize, float weightDecay, float initialHessian, float beta1,
            float beta2, long seed) {
        this(learningRate, effectiveSampleSize, weightDecay, initialHessian, beta1, beta2, Float.POSITIVE_INFINITY,
                seed);
    }

    /**
     * Creates IVON with the per-weight update clipped to {@code [-clipRadius, clipRadius]} (the
     * reference implementation's {@code clip_radius}), which keeps the first steps stable while
     * {@code h} is still far from the loss curvature.
     *
     * @param learningRate the step size
     * @param effectiveSampleSize {@code lambda}
     * @param weightDecay {@code delta}
     * @param initialHessian {@code h0}
     * @param beta1 the gradient momentum
     * @param beta2 the Hessian averaging
     * @param clipRadius the largest preconditioned step per weight
     * @param seed the weight-noise seed
     */
    public Ivon(float learningRate, float effectiveSampleSize, float weightDecay, float initialHessian, float beta1,
            float beta2, float clipRadius, long seed) {
        this.clipRadius = clipRadius;
        this.learningRate = learningRate;
        this.effectiveSampleSize = effectiveSampleSize;
        this.weightDecay = weightDecay;
        this.initialHessian = initialHessian;
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.random = new SplittableRandom(seed);
    }

    private State state(Var p) {
        return states.computeIfAbsent(p, k -> {
            State s = new State();
            s.mean = k.value().toArray();
            int n = s.mean.length;
            s.hessian = new float[n];
            java.util.Arrays.fill(s.hessian, initialHessian);
            s.momentum = new float[n];
            return s;
        });
    }

    private float sigma(State s, int i) {
        return (float) (1 / Math.sqrt(effectiveSampleSize * (s.hessian[i] + weightDecay)));
    }

    @Override
    public void step(List<Var> parameters) {
        t++;
        float bc = (float) (1 - Math.pow(beta1, t));
        for (Var p : parameters) {
            State s = state(p);
            Tensor gt = p.grad();
            if (gt == null) {
                continue;
            }
            float[] g = gt.toArray();
            int n = s.mean.length;
            for (int i = 0; i < n; i++) {
                float sg = sigma(s, i);
                s.momentum[i] = beta1 * s.momentum[i] + (1 - beta1) * g[i];
                if (s.noise != null) {
                    float hHat = g[i] * s.noise[i] / sg;
                    float h = s.hessian[i];
                    float d = h - hHat;
                    s.hessian[i] = beta2 * h + (1 - beta2) * hHat
                            + 0.5f * (1 - beta2) * (1 - beta2) * d * d / (h + weightDecay);
                }
                float u = (s.momentum[i] / bc + weightDecay * s.mean[i]) / (s.hessian[i] + weightDecay);
                s.mean[i] -= learningRate * Math.max(-clipRadius, Math.min(clipRadius, u));
            }
            p.zeroGrad();
            resample(p, s);
        }
    }

    private void resample(Var p, State s) {
        int n = s.mean.length;
        if (s.noise == null) {
            s.noise = new float[n];
        }
        float[] w = new float[n];
        for (int i = 0; i < n; i++) {
            s.noise[i] = (float) random.nextGaussian();
            w[i] = s.mean[i] + sigma(s, i) * s.noise[i];
        }
        p.assign(Tensor.of(p.value().shape(), w));
    }

    /**
     * Writes the posterior means into the parameters (for evaluating the mean network).
     *
     * @param parameters the parameters
     */
    public void useMean(List<Var> parameters) {
        for (Var p : parameters) {
            State s = states.get(p);
            if (s != null) {
                p.assign(Tensor.of(p.value().shape(), s.mean.clone()));
            }
        }
    }

    /**
     * Writes a fresh posterior sample into the parameters.
     *
     * @param parameters the parameters
     */
    public void sample(List<Var> parameters) {
        for (Var p : parameters) {
            State s = states.get(p);
            if (s != null) {
                resample(p, s);
            }
        }
    }

    /**
     * The posterior standard deviations of one parameter.
     *
     * @param parameter the parameter
     * @return {@code sigma} per weight, or an empty array before its first step
     */
    public float[] standardDeviation(Var parameter) {
        State s = states.get(parameter);
        if (s == null) {
            return new float[0];
        }
        float[] out = new float[s.mean.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = sigma(s, i);
        }
        return out;
    }

    /**
     * Weight samples for a predictive ensemble, as {@link WeightPosterior#average} takes them.
     *
     * @param parameters the parameters
     * @param n the number of samples
     * @return {@code n} lists of tensors
     */
    public List<List<Tensor>> samples(List<Var> parameters, int n) {
        List<List<Tensor>> out = new ArrayList<>();
        for (int k = 0; k < n; k++) {
            List<Tensor> one = new ArrayList<>();
            for (Var p : parameters) {
                State s = states.get(p);
                float[] w = new float[s.mean.length];
                for (int i = 0; i < w.length; i++) {
                    w[i] = (float) (s.mean[i] + sigma(s, i) * random.nextGaussian());
                }
                one.add(Tensor.of(p.value().shape(), w));
            }
            out.add(one);
        }
        return out;
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void learningRate(float learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public String toString() {
        return "IVON(lr=" + learningRate + ", ess=" + effectiveSampleSize + ")";
    }
}
