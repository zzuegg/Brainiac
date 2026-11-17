package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.DartsCell.Candidate;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * ENAS (efficient neural architecture search): every architecture of a search space is a
 * sub-graph of one <b>supernet</b> whose weights all architectures share, so candidates are
 * scored without training each from scratch. A <b>controller</b> policy samples
 * architectures; search alternates between
 *
 * <ol>
 *   <li>training the shared weights on the training loss of architectures sampled from the
 *       controller, and</li>
 *   <li>training the controller by REINFORCE, with each sampled architecture's validation
 *       accuracy under the shared weights as its reward and a moving-average baseline.</li>
 * </ol>
 *
 * <p>The search space is the {@code DartsCell} one: a cell of {@code nodes} intermediate
 * nodes, each edge choosing one {@link Candidate} operation; {@link SharedCell} is the
 * supernet layer and, with fresh weights, also a stand-alone cell for a fixed architecture.
 * Simplified from the paper: the controller is an independent softmax per edge rather than
 * an LSTM over the decisions.
 *
 * <p>Paper: Pham, Guan, Zoph, Le, Dean, "Efficient Neural Architecture Search via Parameter
 * Sharing" (2018), <a href="https://arxiv.org/abs/1802.03268">arXiv:1802.03268</a>.
 */
public final class WeightSharingNas {

    /**
     * A cell holding weights for every operation on every edge, running one chosen
     * architecture at a time ({@link #select}). Feature size in equals feature size out.
     */
    public static final class SharedCell implements Layer {

        private final int nodes;
        private final List<Candidate> candidates;
        private final List<List<Var[]>> weights = new ArrayList<>();
        private int[] architecture;

        /**
         * Creates the cell.
         *
         * @param features the feature size
         * @param nodes intermediate nodes; the cell has {@code nodes (nodes + 1) / 2} edges
         * @param candidates the operations every edge chooses from
         * @param random the initializer
         */
        public SharedCell(int features, int nodes, List<Candidate> candidates, SplittableRandom random) {
            this.nodes = nodes;
            this.candidates = List.copyOf(candidates);
            float scale = (float) Math.sqrt(1.0 / features);
            for (int e = 0; e < nodes * (nodes + 1) / 2; e++) {
                List<Var[]> ops = new ArrayList<>();
                for (Candidate c : candidates) {
                    ops.add(c == Candidate.ZERO || c == Candidate.IDENTITY ? null : new Var[] {
                        Var.param(Tensor.normal(Shape.of(features, features), 0f, scale, random)),
                        Var.param(Tensor.zeros(Shape.of(features)))});
                }
                weights.add(ops);
            }
            this.architecture = new int[weights.size()];
        }

        /**
         * A spec for a stand-alone cell of one architecture.
         *
         * @param nodes intermediate nodes
         * @param candidates the operations
         * @param architecture the operation index per edge
         * @return the spec
         */
        public static LayerSpec spec(int nodes, List<Candidate> candidates, int[] architecture) {
            return new LayerSpec() {
                @Override
                public Shape outputShape(Shape input) {
                    return input;
                }

                @Override
                public Layer create(Shape input, SplittableRandom random) {
                    SharedCell cell = new SharedCell(input.dim(-1), nodes, candidates, random);
                    cell.select(architecture);
                    return cell;
                }
            };
        }

        /** @return the number of edges */
        public int edges() {
            return weights.size();
        }

        /** @return the candidate operations */
        public List<Candidate> candidates() {
            return candidates;
        }

        /**
         * Chooses the architecture the next forward passes run.
         *
         * @param ops one candidate index per edge
         */
        public void select(int[] ops) {
            if (ops.length != weights.size()) {
                throw new IllegalArgumentException("need " + weights.size() + " operations, got " + ops.length);
            }
            this.architecture = ops.clone();
        }

        /** @return the selected architecture */
        public int[] architecture() {
            return architecture.clone();
        }

        @Override
        public Var forward(Var input) {
            List<Var> states = new ArrayList<>();
            states.add(input);
            int edge = 0;
            for (int i = 1; i <= nodes; i++) {
                Var sum = null;
                for (int j = 0; j < i; j++, edge++) {
                    Var out = op(edge, architecture[edge], states.get(j));
                    if (out != null) {
                        sum = sum == null ? out : sum.add(out);
                    }
                }
                states.add(sum == null ? input.mul(0f) : sum);
            }
            return states.getLast();
        }

        private Var op(int edge, int k, Var x) {
            Candidate c = candidates.get(k);
            if (c == Candidate.ZERO) {
                return null;
            }
            if (c == Candidate.IDENTITY) {
                return x;
            }
            Var[] p = weights.get(edge).get(k);
            Var lin = x.matmul(p[0]).add(p[1]);
            return switch (c) {
                case LINEAR -> lin;
                case RELU -> lin.relu();
                case TANH -> lin.tanh();
                case SQUARE -> lin.square();
                case GELU -> Activation.GELU.forward(lin);
                default -> throw new IllegalStateException();
            };
        }

        /** @return the weights of the selected architecture only */
        @Override
        public List<Var> parameters() {
            List<Var> out = new ArrayList<>();
            for (int e = 0; e < weights.size(); e++) {
                Var[] p = weights.get(e).get(architecture[e]);
                if (p != null) {
                    out.add(p[0]);
                    out.add(p[1]);
                }
            }
            return out;
        }

        /** @return every shared weight of the supernet */
        public List<Var> allParameters() {
            List<Var> out = new ArrayList<>();
            for (List<Var[]> ops : weights) {
                for (Var[] p : ops) {
                    if (p != null) {
                        out.add(p[0]);
                        out.add(p[1]);
                    }
                }
            }
            return out;
        }

        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder("SharedCell[");
            for (int e = 0; e < architecture.length; e++) {
                sb.append(e == 0 ? "" : ", ").append(candidates.get(architecture[e]));
            }
            return sb.append(']').toString();
        }
    }

    /** One training step of the shared weights with the cell's selected architecture. */
    @FunctionalInterface
    public interface WeightStep {
        /** Runs one step. */
        void step();
    }

    private final SharedCell cell;
    private final double[][] logits;
    private final SplittableRandom random;
    private final float controllerRate;
    private double baseline;
    private boolean baselineSet;
    private long weightSteps;
    private long evaluations;

    /**
     * Creates the search over a supernet cell.
     *
     * @param cell the supernet cell, inside the model being trained
     * @param controllerRate the controller's learning rate, such as 0.1
     * @param seed seeds the sampling
     */
    public WeightSharingNas(SharedCell cell, float controllerRate, long seed) {
        this.cell = cell;
        this.logits = new double[cell.edges()][cell.candidates().size()];
        this.random = new SplittableRandom(seed);
        this.controllerRate = controllerRate;
    }

    /**
     * Samples an architecture from the controller.
     *
     * @return one operation index per edge
     */
    public int[] sample() {
        int[] a = new int[logits.length];
        for (int e = 0; e < a.length; e++) {
            double[] p = probabilities(e);
            double u = random.nextDouble();
            int k = 0;
            while (k < p.length - 1 && (u -= p[k]) > 0) {
                k++;
            }
            a[e] = k;
        }
        return a;
    }

    private double[] probabilities(int edge) {
        double max = Arrays.stream(logits[edge]).max().orElse(0);
        double[] p = new double[logits[edge].length];
        double sum = 0;
        for (int k = 0; k < p.length; k++) {
            p[k] = Math.exp(logits[edge][k] - max);
            sum += p[k];
        }
        for (int k = 0; k < p.length; k++) {
            p[k] /= sum;
        }
        return p;
    }

    /**
     * Runs the search.
     *
     * @param epochs alternations of weight training and controller training
     * @param weightSteps shared-weight steps per epoch, each with a freshly sampled architecture
     * @param step one optimizer step of the shared weights (on a training batch) with the
     *     cell's selected architecture
     * @param controllerSamples architectures scored per controller update
     * @param reward the validation accuracy of the cell's selected architecture with the
     *     shared weights
     * @return the best of {@code 2 * controllerSamples} final samples, by reward
     */
    public int[] search(int epochs, int weightSteps, WeightStep step, int controllerSamples,
            ToDoubleFunction<SharedCell> reward) {
        for (int e = 0; e < epochs; e++) {
            for (int s = 0; s < weightSteps; s++) {
                cell.select(sample());
                step.step();
                this.weightSteps++;
            }
            for (int s = 0; s < controllerSamples; s++) {
                int[] a = sample();
                cell.select(a);
                double r = reward.applyAsDouble(cell);
                evaluations++;
                if (!baselineSet) {
                    baseline = r;
                    baselineSet = true;
                }
                double advantage = r - baseline;
                baseline = 0.9 * baseline + 0.1 * r;
                for (int edge = 0; edge < a.length; edge++) {
                    double[] p = probabilities(edge);
                    for (int k = 0; k < p.length; k++) {
                        logits[edge][k] += controllerRate * advantage * ((k == a[edge] ? 1 : 0) - p[k]);
                    }
                }
            }
        }
        int[] best = null;
        double bestReward = Double.NEGATIVE_INFINITY;
        for (int s = 0; s < 2 * controllerSamples; s++) {
            int[] a = sample();
            cell.select(a);
            double r = reward.applyAsDouble(cell);
            evaluations++;
            if (r > bestReward) {
                bestReward = r;
                best = a;
            }
        }
        cell.select(best);
        return best;
    }

    /** @return the controller's most likely architecture */
    public int[] mostLikely() {
        int[] a = new int[logits.length];
        for (int e = 0; e < a.length; e++) {
            for (int k = 1; k < logits[e].length; k++) {
                if (logits[e][k] > logits[e][a[e]]) {
                    a[e] = k;
                }
            }
        }
        return a;
    }

    /** @return shared-weight steps taken (the search's training cost) */
    public long weightSteps() {
        return weightSteps;
    }

    /** @return validation evaluations made by the controller */
    public long evaluations() {
        return evaluations;
    }

    /**
     * A convenience shared-weight step: the optimizer over every supernet weight plus extra
     * parameters (such as the stem and head), stepping only those with gradients.
     *
     * @param loss computes the training loss of a batch with the selected architecture
     * @param optimizer the optimizer
     * @param extra parameters outside the cell
     * @return the step
     */
    public WeightStep step(java.util.function.Supplier<Var> loss, Optimizer optimizer, List<Var> extra) {
        return () -> {
            loss.get().backward();
            List<Var> ps = new ArrayList<>(extra);
            ps.addAll(cell.parameters());
            optimizer.step(ps);
        };
    }
}
