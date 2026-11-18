package io.github.zzuegg.brainiac.experimental.planning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.PlanningMaps;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A learned cost-to-go heuristic for search on puzzles, trained without any solved examples.
 *
 * <ul>
 *   <li>{@link Mode#DAVI}: DeepCubeA's deep approximate value iteration (Agostinelli et al.,
 *       Nature Machine Intelligence 2019, precursor arXiv:1805.07470). States are scrambled
 *       backwards from the goal; the network regresses by MSE onto the Bellman target
 *       {@code min_s' (1 + h_target(s'))} (0 at the goal) of a periodically copied target
 *       network. Weighted A* with it expands few nodes, but MSE targets overestimate on part
 *       of the states, so A* loses optimality.</li>
 *   <li>{@link Mode#ASYMMETRIC_UNDERESTIMATE}: the empirically admissible variant
 *       (arXiv:2606.04860). The Bellman target is additionally capped by the scramble depth
 *       (an upper bound on the true cost), the loss weighs overestimates {@code kappa} times
 *       more than underestimates, and {@link #calibrate} subtracts a safety offset measured
 *       on validation scrambles with known costs, so the heuristic is admissible on the
 *       validation distribution.</li>
 * </ul>
 *
 * The network is a ReLU {@link SimpleMlp}; evaluation inside the search uses a copy of the
 * weights in plain arrays, so a node costs one small matrix-vector product.
 */
public final class LearnedHeuristic {

    /** The training rule. */
    public enum Mode {
        /** Approximate value iteration with an MSE loss. */
        DAVI,
        /** Depth-capped Bellman targets, an asymmetric loss and offset calibration. */
        ASYMMETRIC_UNDERESTIMATE
    }

    /** A puzzle with unit-cost moves and a single goal. */
    public interface Domain {

        /** @return the encoded size of a state */
        int features();

        /**
         * Encodes a state.
         *
         * @param state the state
         * @param out the destination
         * @param offset where to write
         */
        void encode(int[] state, float[] out, int offset);

        /**
         * @param state a state
         * @return its neighbours
         */
        List<int[]> successors(int[] state);

        /**
         * @param state a state
         * @return whether it is the goal
         */
        boolean goal(int[] state);

        /**
         * A state scrambled from the goal.
         *
         * @param depth the number of random moves
         * @param random the random source
         * @return the state
         */
        int[] scramble(int depth, SplittableRandom random);
    }

    /** @return the 8-puzzle, one-hot tile-per-cell encoding (81 features) */
    public static Domain eightPuzzle() {
        return new Domain() {
            @Override
            public int features() {
                return 81;
            }

            @Override
            public void encode(int[] s, float[] out, int offset) {
                for (int i = 0; i < 9; i++) {
                    out[offset + i * 9 + s[i]] = 1;
                }
            }

            @Override
            public List<int[]> successors(int[] s) {
                return PlanningMaps.tileMoves(s, 3);
            }

            @Override
            public boolean goal(int[] s) {
                for (int i = 0; i < 8; i++) {
                    if (s[i] != i + 1) {
                        return false;
                    }
                }
                return true;
            }

            @Override
            public int[] scramble(int depth, SplittableRandom random) {
                return PlanningMaps.scrambleTiles(3, depth, random);
            }
        };
    }

    /**
     * Lights Out on an {@code n x n} board; a state is one 0/1 entry per cell.
     *
     * @param n the side
     * @return the domain
     */
    public static Domain lightsOut(int n) {
        return new Domain() {
            @Override
            public int features() {
                return n * n;
            }

            @Override
            public void encode(int[] s, float[] out, int offset) {
                for (int i = 0; i < n * n; i++) {
                    out[offset + i] = s[i];
                }
            }

            @Override
            public List<int[]> successors(int[] s) {
                List<int[]> out = new ArrayList<>(n * n);
                int mask = mask(s);
                for (int c = 0; c < n * n; c++) {
                    out.add(unmask(PlanningMaps.pressLight(mask, n, c), n));
                }
                return out;
            }

            @Override
            public boolean goal(int[] s) {
                return mask(s) == 0;
            }

            @Override
            public int[] scramble(int depth, SplittableRandom random) {
                int m = 0;
                for (int k = 0; k < depth; k++) {
                    m = PlanningMaps.pressLight(m, n, random.nextInt(n * n));
                }
                return unmask(m, n);
            }
        };
    }

    /**
     * @param s a Lights Out state
     * @return its bit mask
     */
    public static int mask(int[] s) {
        int m = 0;
        for (int i = 0; i < s.length; i++) {
            m |= s[i] << i;
        }
        return m;
    }

    private static int[] unmask(int m, int n) {
        int[] s = new int[n * n];
        for (int i = 0; i < s.length; i++) {
            s[i] = (m >> i) & 1;
        }
        return s;
    }

    private final Domain domain;
    private final Mode mode;
    private final SimpleMlp net;
    private float[][][] fast;
    private float[][][] target;
    private double offset;

    /**
     * Creates an untrained heuristic.
     *
     * @param domain the puzzle
     * @param mode the training rule
     * @param hidden the hidden width
     * @param seed the seed
     */
    public LearnedHeuristic(Domain domain, Mode mode, int hidden, long seed) {
        this.domain = domain;
        this.mode = mode;
        this.net = SimpleMlp.relu(seed, domain.features(), hidden, hidden, 1);
        this.fast = snapshot(net);
        this.target = fast;
    }

    private static float[][][] snapshot(SimpleMlp m) {
        float[][][] w = new float[m.depth()][][];
        for (int l = 0; l < m.depth(); l++) {
            w[l] = new float[][] {m.weight(l).value().toArray(), m.bias(l).value().toArray()};
        }
        return w;
    }

    private double eval(float[][][] w, float[] x) {
        float[] h = x;
        int[] sizes = net.sizes();
        for (int l = 0; l < w.length; l++) {
            int in = sizes[l];
            int out = sizes[l + 1];
            float[] o = w[l][1].clone();
            for (int i = 0; i < in; i++) {
                float v = h[i];
                if (v == 0) {
                    continue;
                }
                int base = i * out;
                for (int j = 0; j < out; j++) {
                    o[j] += v * w[l][0][base + j];
                }
            }
            if (l < w.length - 1) {
                for (int j = 0; j < out; j++) {
                    o[j] = Math.max(0, o[j]);
                }
            }
            h = o;
        }
        return h[0];
    }

    private double raw(float[][][] w, int[] state) {
        if (domain.goal(state)) {
            return 0;
        }
        float[] x = new float[domain.features()];
        domain.encode(state, x, 0);
        return eval(w, x);
    }

    /**
     * Trains by approximate value iteration.
     *
     * @param steps gradient steps
     * @param batch states per step
     * @param maxDepth the deepest scramble
     * @param learningRate Adam's rate
     * @param targetEvery steps between target-network copies
     * @param kappa the overestimate weight (asymmetric mode)
     * @param seed the seed
     */
    public void train(int steps, int batch, int maxDepth, float learningRate, int targetEvery, double kappa,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(learningRate);
        int f = domain.features();
        for (int step = 0; step < steps; step++) {
            float[] x = new float[batch * f];
            float[] y = new float[batch];
            for (int i = 0; i < batch; i++) {
                int depth = 1 + r.nextInt(maxDepth);
                int[] s = domain.scramble(depth, r);
                domain.encode(s, x, i * f);
                double best = 0;
                if (!domain.goal(s)) {
                    best = Double.POSITIVE_INFINITY;
                    for (int[] t : domain.successors(s)) {
                        best = Math.min(best, 1 + Math.max(0, raw(target, t)));
                    }
                    if (mode == Mode.ASYMMETRIC_UNDERESTIMATE) {
                        best = Math.min(best, depth);
                    }
                }
                y[i] = (float) best;
            }
            Var pred = net.forward(Var.of(Tensor.of(Shape.of(batch, f), x)));
            Var err = pred.sub(Var.of(Tensor.of(Shape.of(batch, 1), y)));
            Var loss;
            if (mode == Mode.ASYMMETRIC_UNDERESTIMATE) {
                Var over = err.relu();
                Var under = err.neg().relu();
                loss = over.square().mul((float) kappa).add(under.square()).mean();
            } else {
                loss = err.square().mean();
            }
            loss.backward();
            opt.step(net.parameters());
            if ((step + 1) % targetEvery == 0) {
                fast = snapshot(net);
                target = fast;
            }
        }
        fast = snapshot(net);
        target = fast;
    }

    /**
     * Measures the largest overestimate {@code h(s) - h*(s)} on validation states with known
     * costs and stores it as the safety offset (asymmetric mode only; DAVI keeps 0).
     *
     * @param states validation states
     * @param costs their optimal costs
     * @return the offset
     */
    public double calibrate(List<int[]> states, int[] costs) {
        double worst = 0;
        for (int i = 0; i < states.size(); i++) {
            worst = Math.max(worst, raw(fast, states.get(i)) - costs[i]);
        }
        offset = mode == Mode.ASYMMETRIC_UNDERESTIMATE ? worst : 0;
        return worst;
    }

    /** @return the calibrated offset */
    public double offset() {
        return offset;
    }

    /**
     * The heuristic: the network's estimate minus the offset, floored at 0, and 0 at the goal.
     *
     * @param state a state
     * @return the estimate
     */
    public double estimate(int[] state) {
        return Math.max(0, raw(fast, state) - offset);
    }

    /** @return the training rule */
    public Mode mode() {
        return mode;
    }

    /** @return the puzzle */
    public Domain domain() {
        return domain;
    }
}
