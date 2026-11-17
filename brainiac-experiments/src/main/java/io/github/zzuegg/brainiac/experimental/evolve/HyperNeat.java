package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.ToDoubleFunction;

/**
 * HyperNEAT: instead of evolving a network's weights directly, evolves a small
 * compositional pattern producing network (CPPN, with {@link Neat}) that <em>paints</em>
 * them. Every neuron of a large fixed network, the <b>substrate</b>, has coordinates; the
 * weight between neurons at {@code (x1, y1)} and {@code (x2, y2)} is
 * {@code cppn(x1, y1, x2, y2)}. Regularities of the task (symmetry, locality, repetition)
 * become regularities of the CPPN, so huge networks can be found with a small genome, and
 * the same CPPN can be re-queried at a higher resolution.
 *
 * <p>Weights whose CPPN output is below a threshold in magnitude are left out; the rest are
 * rescaled to {@code [-maxWeight, maxWeight]}. A neuron's bias is the CPPN queried at
 * {@code (0, 0, x, y)}.
 *
 * <p>Stanley, D'Ambrosio and Gauci 2009, "A Hypercube-Based Encoding for Evolving
 * Large-Scale Neural Networks",
 * <a href="https://doi.org/10.1162/artl.2009.15.2.15202">Artificial Life 15(2)</a>.
 */
public final class HyperNeat {

    /**
     * The fixed geometry of the network the CPPN paints: layers of neurons with 2-d
     * coordinates, wired layer to layer.
     *
     * @param layers the coordinates of each layer's neurons, input layer first
     * @param hiddenActivation the activation of hidden layers
     * @param outputActivation the activation of the output layer
     */
    public record Substrate(List<double[][]> layers, NeatActivation hiddenActivation,
            NeatActivation outputActivation) {
        /** Copies the list and checks there are at least two layers. */
        public Substrate {
            layers = List.copyOf(layers);
            if (layers.size() < 2) {
                throw new IllegalArgumentException("a substrate needs an input and an output layer");
            }
            Objects.requireNonNull(hiddenActivation);
            Objects.requireNonNull(outputActivation);
        }

        /**
         * Places {@code n} neurons evenly on a line {@code y = const}, x from -1 to 1.
         *
         * @param n the number of neurons
         * @param y their y coordinate
         * @return their coordinates
         */
        public static double[][] line(int n, double y) {
            double[][] out = new double[n][];
            for (int i = 0; i < n; i++) {
                out[i] = new double[] {n == 1 ? 0 : -1 + 2.0 * i / (n - 1), y};
            }
            return out;
        }

        /**
         * Places neurons on a {@code w x h} grid covering {@code [-1, 1]^2}.
         *
         * @param w columns
         * @param h rows
         * @return their coordinates, row by row
         */
        public static double[][] grid(int w, int h) {
            double[][] out = new double[w * h][];
            for (int r = 0; r < h; r++) {
                for (int c = 0; c < w; c++) {
                    out[r * w + c] = new double[] {w == 1 ? 0 : -1 + 2.0 * c / (w - 1), h == 1 ? 0 : -1 + 2.0 * r / (h - 1)};
                }
            }
            return out;
        }
    }

    /** A layered network decoded from a CPPN on a substrate. */
    public static final class SubstrateNetwork {
        private final double[][][] weights;
        private final double[][] biases;
        private final Substrate substrate;

        SubstrateNetwork(double[][][] weights, double[][] biases, Substrate substrate) {
            this.weights = weights;
            this.biases = biases;
            this.substrate = substrate;
        }

        /**
         * The weight matrix between layer {@code l} and {@code l + 1}, {@code [from][to]}.
         *
         * @param l the source layer
         * @return a copy of the weights
         */
        public double[][] weights(int l) {
            double[][] w = weights[l];
            double[][] out = new double[w.length][];
            for (int i = 0; i < w.length; i++) {
                out[i] = w[i].clone();
            }
            return out;
        }

        /** @return the number of nonzero connections */
        public int connections() {
            int n = 0;
            for (double[][] w : weights) {
                for (double[] row : w) {
                    for (double x : row) {
                        if (x != 0) {
                            n++;
                        }
                    }
                }
            }
            return n;
        }

        /**
         * Computes the output layer.
         *
         * @param input one value per input neuron
         * @return one value per output neuron
         */
        public double[] activate(double... input) {
            double[] a = input;
            for (int l = 0; l < weights.length; l++) {
                double[][] w = weights[l];
                double[] next = biases[l].clone();
                for (int i = 0; i < a.length; i++) {
                    if (a[i] == 0) {
                        continue;
                    }
                    for (int j = 0; j < next.length; j++) {
                        next[j] += a[i] * w[i][j];
                    }
                }
                NeatActivation f = l == weights.length - 1 ? substrate.outputActivation() : substrate.hiddenActivation();
                for (int j = 0; j < next.length; j++) {
                    next[j] = f.applyAsDouble(next[j]);
                }
                a = next;
            }
            return a;
        }
    }

    private final Substrate substrate;
    private final Neat neat;
    private double threshold = 0.2;
    private double maxWeight = 3;

    private HyperNeat(Substrate substrate, Neat neat) {
        this.substrate = substrate;
        this.neat = neat;
    }

    /**
     * Sets up HyperNEAT for a substrate. The CPPN has 4 inputs {@code (x1, y1, x2, y2)} and
     * one output, and its hidden nodes draw from sine, Gaussian, sigmoid, tanh, absolute
     * value and identity.
     *
     * @param substrate the network geometry
     * @return the search; configure the underlying CPPN search with {@link #neat()}
     */
    public static HyperNeat of(Substrate substrate) {
        Neat neat = Neat.of(4, 1)
                .outputActivation(NeatActivation.TANH)
                .hiddenActivations(NeatActivation.SIN, NeatActivation.GAUSS, NeatActivation.SIGMOID,
                        NeatActivation.TANH, NeatActivation.ABS, NeatActivation.LINEAR)
                .structuralRates(0.1, 0.15)
                .population(100)
                .targetSpecies(8);
        return new HyperNeat(substrate, neat);
    }

    /** @return the NEAT search over CPPNs, to configure */
    public Neat neat() {
        return neat;
    }

    /**
     * Sets the expression threshold and weight range.
     *
     * @param minMagnitude CPPN outputs below this magnitude give no connection
     * @param max the largest weight
     * @return this
     */
    public HyperNeat weights(double minMagnitude, double max) {
        this.threshold = minMagnitude;
        this.maxWeight = max;
        return this;
    }

    /**
     * Paints the substrate's weights with a CPPN.
     *
     * @param cppn the pattern producer
     * @return the substrate network
     */
    public SubstrateNetwork decode(Neat.Network cppn) {
        List<double[][]> layers = substrate.layers();
        double[][][] w = new double[layers.size() - 1][][];
        double[][] b = new double[layers.size() - 1][];
        for (int l = 0; l < layers.size() - 1; l++) {
            double[][] from = layers.get(l);
            double[][] to = layers.get(l + 1);
            w[l] = new double[from.length][to.length];
            b[l] = new double[to.length];
            for (int i = 0; i < from.length; i++) {
                for (int j = 0; j < to.length; j++) {
                    w[l][i][j] = express(cppn.activate(from[i][0], from[i][1], to[j][0], to[j][1])[0]);
                }
            }
            for (int j = 0; j < to.length; j++) {
                b[l][j] = express(cppn.activate(0, 0, to[j][0], to[j][1])[0]);
            }
        }
        return new SubstrateNetwork(w, b, substrate);
    }

    private double express(double v) {
        double a = Math.abs(v);
        if (a < threshold) {
            return 0;
        }
        return Math.signum(v) * (a - threshold) / (1 - threshold) * maxWeight;
    }

    /**
     * Evolves CPPNs whose substrate networks score well.
     *
     * @param fitness scores a substrate network (higher is better, non-negative)
     * @param generations the maximum number of generations
     * @param target stop when reached
     * @return the best substrate network
     */
    public SubstrateNetwork run(ToDoubleFunction<SubstrateNetwork> fitness, int generations, double target) {
        Neat.Genome best = neat.run(cppn -> fitness.applyAsDouble(decode(cppn)), generations, target);
        return decode(best.network());
    }

    /** @return the generations so far */
    public List<Neat.Generation> history() {
        return new ArrayList<>(neat.history());
    }
}
