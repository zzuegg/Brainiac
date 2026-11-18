package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SpikeEncoding;
import io.github.zzuegg.brainiac.experimental.nn.LifLayer;
import io.github.zzuegg.brainiac.experimental.train.SoftHebbClassifier;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Unsupervised digit learning with spike-timing-dependent plasticity, after Diehl and Cook,
 * "Unsupervised learning of digit recognition using spike-timing-dependent plasticity",
 * <a href="https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full">Front. Comput.
 * Neurosci. 2015</a>. No gradients and no labels during learning:
 *
 * <ul>
 *   <li>pixels become Poisson spike trains ({@link SpikeEncoding#rate});</li>
 *   <li>a layer of leaky integrate-and-fire excitatory neurons (the {@link LifLayer} dynamics:
 *       leak, integrate, fire and reset) receives them through plastic synapses;</li>
 *   <li>lateral inhibition makes them compete: the first neuron over threshold fires and resets
 *       all the others (the paper's inhibitory population, in its strong-inhibition limit);</li>
 *   <li>each spike raises the neuron's own threshold (homeostasis), so no neuron wins
 *       everything;</li>
 *   <li>on every postsynaptic spike the synapses move by the trace rule
 *       {@code dw = eta (x_pre - x_target)}: inputs that just spiked are strengthened, silent
 *       ones weakened; columns are then renormalized to a fixed total.</li>
 * </ul>
 * After learning, every neuron is assigned the class it responded to most on the training set
 * (the only use of labels), and a test digit is classified by the class whose neurons spike most.
 * Baselines: the SoftHebb learner with a trained read-out ({@link SoftHebbClassifier}) and a
 * two-layer {@link LifLayer} network trained end to end with surrogate gradients.
 */
public final class StdpDigits {
    private static final int STEPS = 40;
    private static final double LEAK = 0.9;
    private static final double THRESHOLD = 2;
    private static final double THETA_PLUS = 0.05;
    private static final double TRACE_DECAY = 0.8;
    private static final double TARGET = 0.25;
    private static final double RATE = 0.25;

    private final int inputs;
    private final int neurons;
    private final double[][] w;          // [neurons][inputs]
    private final double[] theta;
    private final double eta;
    private final double columnTotal;
    private final SplittableRandom random;
    private int[] assignment;

    private StdpDigits(int inputs, int neurons, double eta, long seed) {
        this.inputs = inputs;
        this.neurons = neurons;
        this.eta = eta;
        this.random = new SplittableRandom(seed);
        this.w = new double[neurons][inputs];
        this.theta = new double[neurons];
        this.columnTotal = 0.1 * inputs;
        for (double[] row : w) {
            for (int i = 0; i < inputs; i++) {
                row[i] = random.nextDouble() * 0.3;
            }
            normalize(row);
        }
    }

    /**
     * Creates an untrained network.
     *
     * @param inputs pixels
     * @param neurons excitatory neurons
     * @param eta the STDP rate
     * @param seed the seed
     * @return the network
     */
    public static StdpDigits of(int inputs, int neurons, double eta, long seed) {
        return new StdpDigits(inputs, neurons, eta, seed);
    }

    private void normalize(double[] row) {
        double s = 0;
        for (double v : row) {
            s += v;
        }
        for (int i = 0; i < row.length; i++) {
            row[i] *= columnTotal / Math.max(s, 1e-9);
        }
    }

    /**
     * Presents one digit for {@code STEPS} steps.
     *
     * @param x pixels in [0, 1]
     * @param learn whether STDP and homeostasis are on
     * @return spikes per neuron
     */
    private int[] present(float[] x, boolean learn) {
        int[] counts = new int[neurons];
        double rate = RATE;
        for (int attempt = 0; attempt < 4; attempt++) {
            float[] scaled = new float[inputs];
            for (int i = 0; i < inputs; i++) {
                scaled[i] = (float) Math.min(1, Math.max(0, x[i]) * rate);
            }
            float[] spikes = SpikeEncoding.rate(new float[][] {scaled}, STEPS, random).toArray();
            double[] v = new double[neurons];
            double[] trace = new double[inputs];
            int total = 0;
            for (int t = 0; t < STEPS; t++) {
                for (int i = 0; i < inputs; i++) {
                    trace[i] *= TRACE_DECAY;
                    if (spikes[t * inputs + i] > 0) {
                        trace[i] = 1;
                    }
                }
                int winner = -1;
                double best = 0;
                for (int j = 0; j < neurons; j++) {
                    double in = 0;
                    double[] row = w[j];
                    for (int i = 0; i < inputs; i++) {
                        if (spikes[t * inputs + i] > 0) {
                            in += row[i];
                        }
                    }
                    v[j] = v[j] * LEAK + in;
                    double margin = v[j] - THRESHOLD - theta[j];
                    if (margin > 0 && (winner < 0 || margin > best)) {
                        winner = j;
                        best = margin;
                    }
                }
                if (winner >= 0) {
                    counts[winner]++;
                    total++;
                    java.util.Arrays.fill(v, 0);          // lateral inhibition resets the layer
                    if (learn) {
                        theta[winner] += THETA_PLUS;
                        double[] row = w[winner];
                        for (int i = 0; i < inputs; i++) {
                            row[i] = Math.max(0, Math.min(1, row[i] + eta * (trace[i] - TARGET)));
                        }
                        normalize(row);
                    }
                }
            }
            if (total >= 3) {
                break;
            }
            rate += 0.25;                                 // too quiet: present it again, brighter
            java.util.Arrays.fill(counts, 0);
        }
        return counts;
    }

    /**
     * Learns from the digits without labels, then assigns each neuron a class.
     *
     * @param train the training digits
     * @param epochs passes over them
     */
    public void train(ToyTasks.Data train, int epochs) {
        for (int e = 0; e < epochs; e++) {
            for (float[] x : train.x()) {
                present(x, true);
            }
        }
        double[][] response = new double[neurons][train.classes()];
        int[] perClass = new int[train.classes()];
        for (int n = 0; n < train.size(); n++) {
            int[] c = present(train.x()[n], false);
            perClass[train.y()[n]]++;
            for (int j = 0; j < neurons; j++) {
                response[j][train.y()[n]] += c[j];
            }
        }
        assignment = new int[neurons];
        for (int j = 0; j < neurons; j++) {
            int best = 0;
            for (int k = 1; k < train.classes(); k++) {
                if (response[j][k] / Math.max(1, perClass[k]) > response[j][best] / Math.max(1, perClass[best])) {
                    best = k;
                }
            }
            assignment[j] = best;
        }
    }

    /**
     * Classifies by the mean spike count of each class's neurons.
     *
     * @param test the digits
     * @return the accuracy
     */
    public double accuracy(ToyTasks.Data test) {
        int classes = test.classes();
        int[] members = new int[classes];
        for (int a : assignment) {
            members[a]++;
        }
        int right = 0;
        for (int n = 0; n < test.size(); n++) {
            int[] c = present(test.x()[n], false);
            double[] votes = new double[classes];
            for (int j = 0; j < neurons; j++) {
                votes[assignment[j]] += c[j] / (double) members[assignment[j]];
            }
            int best = 0;
            for (int k = 1; k < classes; k++) {
                if (votes[k] > votes[best]) {
                    best = k;
                }
            }
            right += best == test.y()[n] ? 1 : 0;
        }
        return right / (double) test.size();
    }

    /**
     * The SoftHebb baseline: Hebbian soft-WTA features and a trained linear read-out.
     *
     * @param train training digits
     * @param test held-out digits
     * @param units feature units
     * @param epochs passes of minibatches of 32
     * @param seed the seed
     * @return the accuracy
     */
    public static double softHebb(ToyTasks.Data train, ToyTasks.Data test, int units, int epochs, long seed) {
        SoftHebbClassifier c = new SoftHebbClassifier(train.features(), units, train.classes(), 0.05f, 0.1f,
                Optimizer.adam(0.01f), seed);
        Tensor x = train.inputs();
        Tensor y = train.ids();
        for (int e = 0; e < epochs; e++) {
            for (int from = 0; from + 32 <= train.size(); from += 32) {
                c.trainBatch(x.slice(0, from, from + 32), y.slice(0, from, from + 32));
            }
        }
        return ToyTasks.accuracy(c.predict(test.inputs()), test.y());
    }

    /**
     * The surrogate-gradient baseline: two {@link LifLayer}s trained by backpropagation through
     * time on rate-coded digits, the class being the output neuron that spikes most.
     *
     * @param train training digits
     * @param test held-out digits
     * @param hidden hidden neurons
     * @param steps Adam steps on minibatches of 32
     * @param seed the seed
     * @return the accuracy
     */
    public static double surrogate(ToyTasks.Data train, ToyTasks.Data test, int hidden, int steps, long seed) {
        LifLayer h = LifLayer.of(train.features(), hidden, 0.9f, 1f, seed);
        LifLayer o = LifLayer.of(hidden, train.classes(), 0.9f, 1f, seed + 1);
        List<Var> params = new ArrayList<>(h.parameters());
        params.addAll(o.parameters());
        Optimizer opt = Optimizer.adam(0.005f);
        SplittableRandom r = new SplittableRandom(seed);
        int t = 12;
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(train.size() - 32);
            float[][] x = java.util.Arrays.copyOfRange(train.x(), from, from + 32);
            Var counts = o.forward(h.forward(Var.of(SpikeEncoding.rate(clip(x), t, r)))).sum(1, false);
            Var loss = Functional.crossEntropy(counts, train.ids().slice(0, from, from + 32));
            loss.backward();
            opt.step(params);
        }
        Var counts = o.forward(h.forward(Var.of(SpikeEncoding.rate(clip(test.x()), t, r)))).sum(1, false);
        return ToyTasks.accuracy(counts.value(), test.y());
    }

    private static float[][] clip(float[][] x) {
        float[][] out = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new float[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] = Math.min(1, Math.max(0, x[i][j]));
            }
        }
        return out;
    }
}
