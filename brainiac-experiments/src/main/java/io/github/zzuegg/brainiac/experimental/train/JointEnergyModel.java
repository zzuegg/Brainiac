package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Joint energy-based model training (Grathwohl et al., "Your classifier is secretly an energy
 * based model", <a href="https://arxiv.org/abs/1912.03263">arXiv:1912.03263</a>): a classifier's
 * logits {@code f(x)} define both {@code p(y | x) = softmax(f(x))} and an unnormalised density
 * {@code p(x) ~ exp(logsumexp_y f(x))}, i.e. the energy {@code E(x) = -logsumexp_y f(x)}. Training
 * adds to the cross-entropy the contrastive maximum-likelihood gradient of {@code log p(x)}:
 * {@code E(x_data) - E(x_model)}, with model samples drawn by stochastic-gradient Langevin
 * dynamics from a persistent replay buffer (re-initialised to uniform noise with probability
 * 5%), plus a small penalty on the squared energies that keeps them bounded.
 *
 * <p>Any core {@link Model} that maps a flat input to class logits can be wrapped; its
 * parameters are trained by whatever optimizer the caller steps on the returned loss.
 */
public final class JointEnergyModel {
    private final Model classifier;
    private final int features;
    private final float[] buffer;
    private final int bufferSize;
    private final float low;
    private final float high;
    private int sgldSteps = 20;
    private float sgldStep = 1f;
    private float sgldNoise = 0.01f;
    private float energyWeight = 1f;
    private float energyPenalty = 0.1f;
    private final SplittableRandom random;

    private JointEnergyModel(Model classifier, int features, int bufferSize, float low, float high, long seed) {
        this.classifier = classifier;
        this.features = features;
        this.bufferSize = bufferSize;
        this.low = low;
        this.high = high;
        this.random = new SplittableRandom(seed);
        this.buffer = new float[bufferSize * features];
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = uniform();
        }
    }

    /**
     * Wraps a classifier.
     *
     * @param classifier maps {@code [n, features]} to {@code [n, classes]} logits
     * @param features the flat input size
     * @param bufferSize the persistent SGLD buffer size, such as 1000
     * @param low the lower bound of the uniform noise that seeds chains
     * @param high its upper bound
     * @param seed the seed of the buffer and of the chains
     * @return the trainer
     */
    public static JointEnergyModel of(Model classifier, int features, int bufferSize, float low, float high,
            long seed) {
        return new JointEnergyModel(classifier, features, bufferSize, low, high, seed);
    }

    /**
     * Sets the Langevin sampler.
     *
     * @param steps steps per training iteration, 20 in the paper
     * @param step the gradient step size, 1 in the paper
     * @param noise the noise standard deviation, 0.01 in the paper
     * @return this
     */
    public JointEnergyModel sgld(int steps, float step, float noise) {
        this.sgldSteps = steps;
        this.sgldStep = step;
        this.sgldNoise = noise;
        return this;
    }

    /**
     * Sets the weights of the generative terms.
     *
     * @param weight the weight of {@code E(x_data) - E(x_model)}, 1 in the paper
     * @param penalty the weight of the squared-energy penalty
     * @return this
     */
    public JointEnergyModel energyWeights(float weight, float penalty) {
        this.energyWeight = weight;
        this.energyPenalty = penalty;
        return this;
    }

    /** @return the wrapped classifier */
    public Model classifier() {
        return classifier;
    }

    private float uniform() {
        return (float) (low + (high - low) * random.nextDouble());
    }

    /** {@code -logsumexp f(x)} per row: {@code [n, 1]}. */
    private Var energy(Var x, boolean training) {
        Var logits = classifier.forward(x, training);
        Var m = Var.of(logits.value().max(1, true));
        return logits.sub(m).exp().sum(1, true).log().add(m).neg();
    }

    /**
     * The energy of each input.
     *
     * @param x {@code [n, features]}
     * @return {@code [n]} energies; lower is more data-like
     */
    public double[] energy(Tensor x) {
        float[] e = energy(Var.of(x), false).value().toArray();
        double[] out = new double[e.length];
        for (int i = 0; i < e.length; i++) {
            out[i] = e[i];
        }
        return out;
    }

    /**
     * Runs Langevin dynamics on the energy, keeping the chains inside the noise box.
     *
     * @param start the chains' starting points {@code [n, features]}
     * @param steps the number of steps
     * @param rng the noise source
     * @return the end points
     */
    public Tensor langevin(Tensor start, int steps, RandomGenerator rng) {
        Tensor x = start;
        for (int k = 0; k < steps; k++) {
            Var xv = Var.param(x);
            Tensor g = Var.gradients(energy(xv, false).sum(), List.of(xv), false).get(0).value();
            float[] v = x.sub(g.mul(sgldStep)).add(Tensor.normal(x.shape(), 0, sgldNoise, rng)).toArray();
            for (int i = 0; i < v.length; i++) {
                v[i] = Math.max(low, Math.min(high, v[i]));
            }
            x = Tensor.of(x.shape(), v);
        }
        return x;
    }

    /**
     * The persistent chains' current states, JEM's usual samples.
     *
     * @param n how many, at most the buffer size
     * @return the first {@code n} buffer entries {@code [n, features]}
     */
    public Tensor buffer(int n) {
        return Tensor.of(Shape.of(n, features), java.util.Arrays.copyOf(buffer, n * features));
    }

    /**
     * Draws fresh samples from uniform noise.
     *
     * @param n how many
     * @param steps Langevin steps
     * @param rng the noise source
     * @return {@code [n, features]}
     */
    public Tensor sample(int n, int steps, RandomGenerator rng) {
        float[] v = new float[n * features];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (low + (high - low) * rng.nextDouble());
        }
        return langevin(Tensor.of(Shape.of(n, features), v), steps, rng);
    }

    /**
     * The JEM loss on one batch: cross-entropy plus the contrastive energy term; advances the
     * replay buffer.
     *
     * @param x inputs {@code [n, features]}
     * @param labels the classes
     * @param rng the noise source
     * @return the scalar loss
     */
    public Var loss(Tensor x, int[] labels, RandomGenerator rng) {
        int n = labels.length;
        Var logits = classifier.forward(Var.of(x), true);
        int classes = logits.shape().dim(1);
        float[] onehot = new float[n * classes];
        for (int i = 0; i < n; i++) {
            onehot[i * classes + labels[i]] = 1f / n;
        }
        Var ce = Activation.LOG_SOFTMAX.forward(logits).mul(Tensor.of(Shape.of(n, classes), onehot)).sum().neg();
        if (energyWeight == 0) {
            return ce;
        }
        int[] slots = new int[n];
        float[] start = new float[n * features];
        for (int i = 0; i < n; i++) {
            slots[i] = random.nextInt(bufferSize);
            boolean fresh = random.nextDouble() < 0.05;
            for (int f = 0; f < features; f++) {
                start[i * features + f] = fresh ? uniform() : buffer[slots[i] * features + f];
            }
        }
        float[] neg = langevin(Tensor.of(Shape.of(n, features), start), sgldSteps, rng).toArray();
        for (int i = 0; i < n; i++) {
            System.arraycopy(neg, i * features, buffer, slots[i] * features, features);
        }
        Var ePos = energy(Var.of(x), true);
        Var eNeg = energy(Var.of(Tensor.of(Shape.of(n, features), neg)), true);
        Var contrastive = ePos.mean().sub(eNeg.mean());
        Var penalty = ePos.square().mean().add(eNeg.square().mean());
        return ce.add(contrastive.mul(energyWeight)).add(penalty.mul(energyPenalty));
    }
}
