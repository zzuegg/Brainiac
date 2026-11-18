package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.autograd.AdaptiveSurrogateSpike;
import io.github.zzuegg.brainiac.experimental.autograd.SurrogateSpike;
import io.github.zzuegg.brainiac.experimental.data.SpikeEncoding;
import io.github.zzuegg.brainiac.experimental.nn.LifLayer;
import io.github.zzuegg.brainiac.experimental.nn.QuantClipFloorActivation;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.nn.SnnConverter;
import io.github.zzuegg.brainiac.experimental.nn.TemporalEfficientLoss;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A two-layer spiking network of {@code LifLayer}s trained with surrogate gradients and
 * backpropagation through time to classify noisy 4x4 stripe patterns (horizontal,
 * vertical, diagonal, anti-diagonal) from spike trains. The class is the output neuron
 * that spikes most. Inputs can be rate coded or latency coded, so the two encodings can
 * be compared, along with how sparse the network's activity is.
 *
 * <p>Wave 8 (8.6.16, 8.6.17) adds static rows on the toy digits: ANN-to-SNN conversion
 * ({@link #conversion}: data-normalised ReLU vs QCFS, {@link SnnConverter}), direct LIF
 * training ({@link #direct}: loss on the mean output vs {@link TemporalEfficientLoss}, fixed
 * vs scheduled or adaptive surrogates, {@link AdaptiveSurrogateSpike}) and the surrogate
 * gradient's mismatch with finite differences ({@link #gradientCosine}).
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

    // ---------------------------------------------------------------------------------------
    // Wave 8 items 8.6.16 and 8.6.17: an MLP on the toy digits of ToyTasks, converted to
    // integrate-and-fire neurons (plain data-normalised ReLU vs QCFS) or trained directly as a
    // LIF network (loss on the mean output vs TET, fixed vs scheduled/adaptive surrogates).
    // The stripe classifier above is unchanged.
    // ---------------------------------------------------------------------------------------

    /** How a directly trained spiking MLP is optimised. */
    public enum SpikeLoss {
        /** cross-entropy of the time-averaged output */
        MEAN_OUTPUT,
        /** TET: per-step cross-entropy plus a small MSE regulariser */
        TET
    }

    /** The spike function's surrogate gradient. */
    public enum Surrogate {
        /** {@link SurrogateSpike} with slope 5, as {@link LifLayer} uses */
        FAST_SIGMOID,
        /** rectangular, fixed half-width 0.5 */
        RECTANGULAR,
        /** rectangular, half-width 1 to 0.1 over training */
        SCHEDULED,
        /** rectangular, half-width from a grid by A²SG's variation score */
        GRID,
        /** rectangular, half-width following the potential spread */
        ADAPTIVE
    }

    /**
     * ANN-to-SNN conversion results.
     *
     * @param steps the time steps {@code T}
     * @param ann the ReLU ANN's accuracy
     * @param qcfsAnn the QCFS ANN's accuracy
     * @param plain plain conversion (99.9th-percentile normalisation) accuracy per {@code T}
     * @param qcfs QCFS conversion accuracy per {@code T}
     * @param plainSpikes hidden spikes per example per {@code T}, plain
     * @param qcfsSpikes hidden spikes per example per {@code T}, QCFS
     */
    public record Conversion(int[] steps, double ann, double qcfsAnn, double[] plain, double[] qcfs,
            double[] plainSpikes, double[] qcfsSpikes) {
    }

    private static final int DIGIT_HIDDEN = 128;

    private static float[][] clamp01(float[][] x) {
        float[][] out = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new float[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] = Math.max(0, Math.min(1, x[i][j]));
            }
        }
        return out;
    }

    private static ToyTasks.Data digitsTrain(long seed) {
        return ToyTasks.digits(10, 100, 8, 0.08, seed, seed + 1);
    }

    private static ToyTasks.Data digitsTest(long seed) {
        return ToyTasks.digits(10, 50, 8, 0.08, seed, seed + 2);
    }

    /** Hidden layer {@code l} of the ANN: ReLU, or QCFS when {@code qcfs} is given. */
    private static Var annForward(SimpleMlp net, QuantClipFloorActivation[] qcfs, Var x) {
        Var h = x;
        for (int l = 0; l < net.depth(); l++) {
            Var z = net.linear(l, h);
            h = l == net.depth() - 1 ? z : qcfs == null ? z.relu() : qcfs[l].forward(z);
        }
        return h;
    }

    private static double annAccuracy(SimpleMlp net, QuantClipFloorActivation[] qcfs, ToyTasks.Data test) {
        Tensor logits = annForward(net, qcfs, Var.of(Tensor.of(clamp01(test.x())))).value();
        return logits.argmax(1).equalTo(test.ids()).mean().item();
    }

    private static void trainAnn(SimpleMlp net, QuantClipFloorActivation[] qcfs, ToyTasks.Data train, int steps,
            long seed) {
        List<Var> params = new ArrayList<>(net.parameters());
        if (qcfs != null) {
            for (QuantClipFloorActivation q : qcfs) {
                params.addAll(q.parameters());
            }
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = clamp01(train.x());
        for (int s = 0; s < steps; s++) {
            float[][] bx = new float[64][];
            float[] by = new float[64];
            for (int i = 0; i < 64; i++) {
                int j = r.nextInt(x.length);
                bx[i] = x[j];
                by[i] = train.y()[j];
            }
            Functional.crossEntropy(annForward(net, qcfs, Var.of(Tensor.of(bx))), Tensor.of(Shape.of(64), by))
                    .backward();
            opt.step(params);
        }
    }

    private static List<Tensor> weights(SimpleMlp net) {
        List<Tensor> w = new ArrayList<>();
        for (int l = 0; l < net.depth(); l++) {
            w.add(net.weight(l).value());
        }
        return w;
    }

    private static List<Tensor> biases(SimpleMlp net) {
        List<Tensor> b = new ArrayList<>();
        for (int l = 0; l < net.depth(); l++) {
            b.add(net.bias(l).value());
        }
        return b;
    }

    /**
     * Trains a ReLU MLP and a QCFS MLP ({@code L = levels}) on the digits and converts both to
     * integrate-and-fire networks.
     *
     * @param steps the time steps to read the SNN at, ascending
     * @param levels QCFS quantisation levels
     * @param hiddenLayers hidden layers of 128 units
     * @param trainSteps Adam steps for each ANN
     * @param seed the seed
     * @return accuracy and spike counts per {@code T}
     */
    public static Conversion conversion(int[] steps, int levels, int hiddenLayers, int trainSteps, long seed) {
        int[] sizes = new int[hiddenLayers + 2];
        java.util.Arrays.fill(sizes, DIGIT_HIDDEN);
        sizes[0] = 64;
        sizes[hiddenLayers + 1] = 10;
        ToyTasks.Data train = digitsTrain(seed);
        ToyTasks.Data test = digitsTest(seed);
        Tensor tx = Tensor.of(clamp01(test.x()));
        SimpleMlp relu = SimpleMlp.relu(seed, sizes);
        trainAnn(relu, null, train, trainSteps, seed + 1);
        double[] scales = SnnConverter.percentileScales(weights(relu), biases(relu), Tensor.of(clamp01(train.x())),
                99.9);
        SnnConverter.Result plain = SnnConverter.of(weights(relu), biases(relu), scales)
                .simulate(tx, test.y(), steps, false);
        SimpleMlp q = SimpleMlp.relu(seed, sizes);
        QuantClipFloorActivation[] qcfs = new QuantClipFloorActivation[hiddenLayers];
        double[] thresholds = new double[hiddenLayers];
        for (int l = 0; l < hiddenLayers; l++) {
            qcfs[l] = QuantClipFloorActivation.of(levels, 2f);
        }
        trainAnn(q, qcfs, train, trainSteps, seed + 1);
        for (int l = 0; l < hiddenLayers; l++) {
            thresholds[l] = qcfs[l].threshold();
        }
        SnnConverter.Result converted = SnnConverter.of(weights(q), biases(q), thresholds)
                .simulate(tx, test.y(), steps, true);
        return new Conversion(steps.clone(), annAccuracy(relu, null, test), annAccuracy(q, qcfs, test),
                plain.accuracy(), converted.accuracy(), plain.spikes(), converted.spikes());
    }

    /** A LIF MLP whose spike function is pluggable; the output layer is a non-spiking readout. */
    private static final class DirectSnn {
        private final SimpleMlp net;
        private final Surrogate surrogate;
        private final AdaptiveSurrogateSpike adaptive;

        DirectSnn(int hiddenLayers, Surrogate surrogate, long seed) {
            int[] sizes = new int[hiddenLayers + 2];
            sizes[0] = 64;
            for (int l = 1; l <= hiddenLayers; l++) {
                sizes[l] = DIGIT_HIDDEN;
            }
            sizes[hiddenLayers + 1] = 10;
            this.net = SimpleMlp.relu(seed, sizes);
            for (int l = 0; l < hiddenLayers; l++) {
                net.weight(l).assign(net.weight(l).value().mul(2f));
            }
            this.surrogate = surrogate;
            this.adaptive = switch (surrogate) {
                case FAST_SIGMOID -> null;
                case RECTANGULAR -> AdaptiveSurrogateSpike.fixed(0.5f, 0);
                case SCHEDULED -> AdaptiveSurrogateSpike.scheduled(1f, 0.1f, 0);
                case GRID -> AdaptiveSurrogateSpike.grid(1f, 0.1f, 0);
                case ADAPTIVE -> AdaptiveSurrogateSpike.adaptive(1f, 0.1f, 1f, 0);
            };
        }

        DirectSnn(SimpleMlp net, AdaptiveSurrogateSpike op) {
            this.net = net;
            this.surrogate = op == null ? Surrogate.FAST_SIGMOID : Surrogate.RECTANGULAR;
            this.adaptive = op;
        }

        private Var spike(Var v) {
            return surrogate == Surrogate.FAST_SIGMOID ? SurrogateSpike.spike(v, 5f) : adaptive.spike(v);
        }

        /** Per-step outputs {@code [batch, steps, 10]} for input spikes {@code [batch, steps, 64]}. */
        Var outputs(Tensor spikes) {
            int b = spikes.shape().dim(0);
            int steps = spikes.shape().dim(1);
            Var s = Var.of(spikes.reshape(b * steps, 64));
            for (int l = 0; l < net.depth() - 1; l++) {
                int units = net.sizes()[l + 1];
                Var current = net.linear(l, s).reshape(b, steps, units);
                Var u = Var.of(Tensor.zeros(Shape.of(b, units)));
                Var previous = null;
                List<Var> out = new ArrayList<>();
                for (int t = 0; t < steps; t++) {
                    u = u.mul(0.9f).add(current.slice(1, t, t + 1).reshape(b, units));
                    if (previous != null) {
                        u = u.sub(previous);
                    }
                    previous = spike(u.sub(1f));
                    out.add(previous.reshape(b, 1, units));
                }
                s = Var.concat(1, out.toArray(Var[]::new)).reshape(b * steps, units);
            }
            return net.linear(net.depth() - 1, s).reshape(b, steps, 10);
        }

        void adapt(double progress) {
            if (adaptive != null) {
                adaptive.adapt(progress);
            }
        }
    }

    private static Tensor rateCode(float[][] x, int steps, SplittableRandom r) {
        return SpikeEncoding.rate(x, steps, r);
    }

    private static double snnAccuracy(DirectSnn snn, ToyTasks.Data test, int steps, long seed) {
        Var o = snn.outputs(rateCode(clamp01(test.x()), steps, new SplittableRandom(seed)));
        return o.value().mean(1, false).argmax(1).equalTo(test.ids()).mean().item();
    }

    private static DirectSnn trainDirect(int steps, SpikeLoss loss, Surrogate surrogate, int hiddenLayers,
            int trainSteps, long seed) {
        ToyTasks.Data train = digitsTrain(seed);
        float[][] x = clamp01(train.x());
        DirectSnn snn = new DirectSnn(hiddenLayers, surrogate, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < trainSteps; s++) {
            float[][] bx = new float[64][];
            float[] by = new float[64];
            for (int i = 0; i < 64; i++) {
                int j = r.nextInt(x.length);
                bx[i] = x[j];
                by[i] = train.y()[j];
            }
            Var out = snn.outputs(rateCode(bx, steps, r));
            Tensor labels = Tensor.of(Shape.of(64), by);
            Var l = loss == SpikeLoss.TET ? TemporalEfficientLoss.loss(out, labels, 0.05f, 0f)
                    : TemporalEfficientLoss.meanOutput(out, labels);
            l.backward();
            opt.step(snn.net.parameters());
            snn.adapt((s + 1) / (double) trainSteps);
        }
        return snn;
    }

    /**
     * Trains a LIF MLP directly with BPTT on rate-coded digits.
     *
     * @param steps the time steps {@code T}
     * @param loss the loss
     * @param surrogate the surrogate gradient
     * @param hiddenLayers the number of spiking hidden layers
     * @param trainSteps Adam steps
     * @param seed the seed
     * @return test accuracy (time-averaged output)
     */
    public static double direct(int steps, SpikeLoss loss, Surrogate surrogate, int hiddenLayers, int trainSteps,
            long seed) {
        DirectSnn snn = trainDirect(steps, loss, surrogate, hiddenLayers, trainSteps, seed);
        return snnAccuracy(snn, digitsTest(seed), steps, seed + 9);
    }

    /**
     * The gradient mismatch of surrogate gradients: a LIF MLP is trained briefly with the fast
     * sigmoid; then, on one fixed batch and fixed input spikes (so the forward pass is the same
     * for all), the surrogate gradient of the mean-output loss with respect to the first
     * weights is projected on random unit directions and compared with central finite
     * differences of the true (spiking) loss along them.
     *
     * @param steps the time steps
     * @param hiddenLayers the spiking hidden layers
     * @param widths rectangular half-widths to score (besides the fast sigmoid)
     * @param directions random directions
     * @param epsilon the finite-difference step along a unit direction
     * @param seed the seed
     * @return the cosine for the fast sigmoid first, then one per width
     */
    public static double[] gradientCosine(int steps, int hiddenLayers, float[] widths, int directions,
            float epsilon, long seed) {
        DirectSnn trained = trainDirect(steps, SpikeLoss.MEAN_OUTPUT, Surrogate.FAST_SIGMOID, hiddenLayers, 100,
                seed);
        ToyTasks.Data train = digitsTrain(seed);
        float[][] bx = java.util.Arrays.copyOf(clamp01(train.x()), 256);
        float[] by = new float[256];
        for (int i = 0; i < 256; i++) {
            by[i] = train.y()[i];
        }
        Tensor labels = Tensor.of(Shape.of(256), by);
        Tensor spikes = rateCode(bx, steps, new SplittableRandom(seed + 3));
        Var w = trained.net.weight(0);
        Tensor w0 = w.value();
        int size = (int) w0.shape().size();
        SplittableRandom r = new SplittableRandom(seed + 4);
        float[][] u = new float[directions][size];
        double[] fd = new double[directions];
        for (int k = 0; k < directions; k++) {
            double norm = 0;
            for (int i = 0; i < size; i++) {
                u[k][i] = (float) r.nextGaussian();
                norm += u[k][i] * u[k][i];
            }
            for (int i = 0; i < size; i++) {
                u[k][i] /= (float) Math.sqrt(norm);
            }
            Tensor dir = Tensor.of(w0.shape(), u[k]);
            w.assign(w0.add(dir.mul(epsilon)));
            double plus = TemporalEfficientLoss.meanOutput(trained.outputs(spikes), labels).value().item();
            w.assign(w0.sub(dir.mul(epsilon)));
            double minus = TemporalEfficientLoss.meanOutput(trained.outputs(spikes), labels).value().item();
            fd[k] = (plus - minus) / (2 * epsilon);
        }
        w.assign(w0);
        double[] out = new double[widths.length + 1];
        for (int m = 0; m <= widths.length; m++) {
            DirectSnn probe = new DirectSnn(trained.net, m == 0 ? null : AdaptiveSurrogateSpike.fixed(widths[m - 1],
                    0));
            for (Var p : trained.net.parameters()) {
                p.zeroGrad();
            }
            TemporalEfficientLoss.meanOutput(probe.outputs(spikes), labels).backward();
            float[] g = w.grad().toArray();
            double dot = 0;
            double gg = 0;
            double ff = 0;
            for (int k = 0; k < directions; k++) {
                double proj = 0;
                for (int i = 0; i < size; i++) {
                    proj += g[i] * u[k][i];
                }
                dot += proj * fd[k];
                gg += proj * proj;
                ff += fd[k] * fd[k];
            }
            out[m] = dot / Math.max(1e-12, Math.sqrt(gg * ff));
        }
        return out;
    }
}
