package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SpikeEncoding;
import io.github.zzuegg.brainiac.experimental.nn.LifLayer;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A two-layer spiking network of {@code LifLayer}s trained with surrogate gradients and
 * backpropagation through time to classify noisy 4x4 stripe patterns (horizontal,
 * vertical, diagonal, anti-diagonal) from spike trains. The class is the output neuron
 * that spikes most. Inputs can be rate coded or latency coded, so the two encodings can
 * be compared, along with how sparse the network's activity is.
 */
public final class SpikingClassifier {
    /** Time steps per sample. */
    public static final int STEPS = 12;
    private final LifLayer hidden;
    private final LifLayer output;
    private final boolean latency;
    private final Optimizer optimizer = Optimizer.adam(5e-3f);
    private final SplittableRandom random;
    private double lastRate;

    private SpikingClassifier(boolean latency, long seed) {
        this.latency = latency;
        this.random = new SplittableRandom(seed);
        hidden = LifLayer.of(16, 48, 0.9f, 1f, seed);
        output = LifLayer.of(48, 4, 0.9f, 1f, seed + 1);
    }

    /**
     * Creates an untrained network.
     *
     * @param latency latency coding if true, rate coding otherwise
     * @param seed the seed
     * @return the network
     */
    public static SpikingClassifier of(boolean latency, long seed) {
        return new SpikingClassifier(latency, seed);
    }

    /**
     * A noisy stripe pattern.
     *
     * @param label 0..3
     * @param random the random source
     * @return 16 pixel intensities in [0, 1]
     */
    public static float[] pattern(int label, SplittableRandom random) {
        float[] p = new float[16];
        int k = random.nextInt(4);
        for (int i = 0; i < 4; i++) {
            int idx = switch (label) {
                case 0 -> k * 4 + i;
                case 1 -> i * 4 + k;
                case 2 -> i * 4 + i;
                default -> i * 4 + 3 - i;
            };
            p[idx] = 1;
        }
        for (int i = 0; i < 16; i++) {
            p[i] = (float) Math.max(0, Math.min(1, p[i] * 0.9 + 0.1 * random.nextDouble()));
        }
        return p;
    }

    private Tensor encode(float[][] x) {
        return latency ? SpikeEncoding.latency(x, STEPS) : SpikeEncoding.rate(x, STEPS, random);
    }

    /** Spike counts of the output neurons, {@code [batch, 4]}. */
    private Var counts(float[][] x) {
        Var h = hidden.forward(Var.of(encode(x)));
        lastRate = h.value().mean().item();
        return output.forward(h).sum(1, false);
    }

    /**
     * One BPTT step: cross-entropy on the output spike counts.
     *
     * @param batch the batch size
     * @return the loss
     */
    public float trainStep(int batch) {
        float[][] x = new float[batch][];
        float[] y = new float[batch];
        for (int i = 0; i < batch; i++) {
            y[i] = random.nextInt(4);
            x[i] = pattern((int) y[i], random);
        }
        Var loss = Functional.crossEntropy(counts(x), Tensor.of(Shape.of(batch), y));
        float value = loss.value().item();
        loss.backward();
        optimizer.step(List.of(hidden.parameters().get(0), hidden.parameters().get(1), output.parameters().get(0),
                output.parameters().get(1)));
        return value;
    }

    /**
     * Accuracy on fresh patterns.
     *
     * @param n how many
     * @param seed the seed
     * @return the share correct
     */
    public double accuracy(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = r.nextInt(4);
            x[i] = pattern(y[i], r);
        }
        float[] c = counts(x).value().toArray();
        int right = 0;
        for (int i = 0; i < n; i++) {
            int best = 0;
            for (int k = 1; k < 4; k++) {
                if (c[i * 4 + k] > c[i * 4 + best]) {
                    best = k;
                }
            }
            right += best == y[i] ? 1 : 0;
        }
        return (double) right / n;
    }

    /** @return the hidden layer's mean spike rate per neuron and step in the last pass */
    public double hiddenSpikeRate() {
        return lastRate;
    }
}
