package io.github.zzuegg.brainiac.experimental.rl;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Monte Carlo tree search with the PUCT rule of AlphaZero (Silver et al., 2017,
 * <a href="https://arxiv.org/abs/1712.01815">arXiv:1712.01815</a>) and MuZero (Schrittwieser
 * et al., 2019, <a href="https://arxiv.org/abs/1911.08265">arXiv:1911.08265</a>).
 *
 * <p>The search grows a tree from the root by repeatedly descending along
 * {@code argmax Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))}, expanding one leaf with
 * the {@link Model}'s prior and value, and backing the value up. The visit counts of the
 * root's children are the improved policy. It works over any state type: real game
 * positions (AlphaZero), or learned latent states (MuZero), with intermediate rewards and
 * discounting, for single-agent tasks or alternating two-player zero-sum games. Q-values are
 * min-max normalized over the tree, as in MuZero, so rewards of any scale work.
 *
 * @param <S> the state type
 */
public final class Mcts<S> {

    /**
     * What the search needs to know about a problem.
     *
     * @param <S> the state type
     */
    public interface Model<S> {
        /** @return the number of actions */
        int actions();

        /**
         * Judges a state.
         *
         * @param state the state
         * @return a prior over actions (0 for illegal ones) and the value for the player to move
         */
        Evaluation evaluate(S state);

        /**
         * Applies an action.
         *
         * @param state the state
         * @param action the action
         * @return the next state and the reward for the player who acted
         */
        Step<S> step(S state, int action);

        /**
         * @return whether players alternate (two-player zero-sum), so a child's value is
         *     negated for its parent
         */
        default boolean alternating() {
            return false;
        }
    }

    /**
     * A judged state.
     *
     * @param priors one prior per action, 0 for illegal actions
     * @param value the value for the player to move
     * @param terminal whether the state ends the episode (then {@code value} is ignored and taken as 0)
     */
    public record Evaluation(float[] priors, float value, boolean terminal) {
    }

    /**
     * The result of applying an action.
     *
     * @param state the next state
     * @param reward the reward for the player who acted
     * @param <S> the state type
     */
    public record Step<S>(S state, float reward) {
    }

    /**
     * The outcome of one search.
     *
     * @param policy the root's visit distribution
     * @param value the root's searched value
     * @param visits the root's visit count per action
     */
    public record Result(float[] policy, float value, int[] visits) {
        /** @return the most visited action */
        public int best() {
            int b = 0;
            for (int i = 1; i < visits.length; i++) {
                if (visits[i] > visits[b]) {
                    b = i;
                }
            }
            return b;
        }

        /**
         * Samples an action proportionally to {@code visits^(1/temperature)}.
         *
         * @param temperature 0 for the most visited
         * @param random the random source
         * @return the action
         */
        public int sample(double temperature, SplittableRandom random) {
            if (temperature <= 1e-6) {
                return best();
            }
            double[] w = new double[visits.length];
            double sum = 0;
            for (int i = 0; i < w.length; i++) {
                w[i] = Math.pow(visits[i], 1 / temperature);
                sum += w[i];
            }
            double u = random.nextDouble() * sum;
            for (int i = 0; i < w.length; i++) {
                u -= w[i];
                if (u <= 0 && visits[i] > 0) {
                    return i;
                }
            }
            return best();
        }
    }

    private static final class Node {
        final float prior;
        Object state;
        float reward;
        boolean expanded;
        boolean terminal;
        int visits;
        double valueSum;
        Node[] children;

        Node(float prior) {
            this.prior = prior;
        }

        double value() {
            return visits == 0 ? 0 : valueSum / visits;
        }
    }

    private final Model<S> model;
    private int simulations = 50;
    private float cPuct = 1.25f;
    private float discount = 1f;
    private float dirichletAlpha = 0.3f;
    private float noiseFraction;
    private double minQ;
    private double maxQ;

    private Mcts(Model<S> model) {
        this.model = model;
    }

    /**
     * @param model the problem
     * @param <S> the state type
     * @return a search with 50 simulations, c = 1.25, no root noise
     */
    public static <S> Mcts<S> of(Model<S> model) {
        return new Mcts<>(Objects.requireNonNull(model, "model"));
    }

    /** @param n simulations per search @return this search */
    public Mcts<S> simulations(int n) {
        this.simulations = Math.max(1, n);
        return this;
    }

    /** @param c the exploration constant @return this search */
    public Mcts<S> cPuct(float c) {
        this.cPuct = c;
        return this;
    }

    /** @param gamma the discount of future rewards @return this search */
    public Mcts<S> discount(float gamma) {
        this.discount = gamma;
        return this;
    }

    /**
     * Mixes Dirichlet noise into the root prior, for exploration during self-play.
     *
     * @param alpha the Dirichlet concentration
     * @param fraction how much noise, 0.25 in AlphaZero
     * @return this search
     */
    public Mcts<S> rootNoise(float alpha, float fraction) {
        this.dirichletAlpha = alpha;
        this.noiseFraction = fraction;
        return this;
    }

    /**
     * Searches from a state.
     *
     * @param root the state
     * @param random noise for the root prior
     * @return the visit distribution and value
     */
    public Result search(S root, SplittableRandom random) {
        minQ = Double.POSITIVE_INFINITY;
        maxQ = Double.NEGATIVE_INFINITY;
        Node rootNode = new Node(1);
        rootNode.state = root;
        expand(rootNode);
        if (noiseFraction > 0 && !rootNode.terminal) {
            addNoise(rootNode, random);
        }
        int n = model.actions();
        Node[] path = new Node[256];
        for (int sim = 0; sim < simulations && !rootNode.terminal; sim++) {
            Node node = rootNode;
            int depth = 0;
            path[depth++] = node;
            while (node.expanded && !node.terminal) {
                int a = select(node);
                Node child = node.children[a];
                if (child.state == null) {
                    Step<S> s = model.step(stateOf(node), a);
                    child.state = s.state();
                    child.reward = s.reward();
                }
                node = child;
                if (depth == path.length) {
                    path = java.util.Arrays.copyOf(path, depth * 2);
                }
                path[depth++] = node;
            }
            double value = node.terminal ? 0 : expand(node);
            for (int i = depth - 1; i >= 0; i--) {
                Node p = path[i];
                p.valueSum += value;
                p.visits++;
                if (i > 0) {
                    double q = p.reward + discount * (model.alternating() ? -p.value() : p.value());
                    minQ = Math.min(minQ, q);
                    maxQ = Math.max(maxQ, q);
                    value = p.reward + discount * (model.alternating() ? -value : value);
                }
            }
        }
        int[] visits = new int[n];
        float[] policy = new float[n];
        int total = 0;
        if (rootNode.children != null) {
            for (int a = 0; a < n; a++) {
                Node c = rootNode.children[a];
                visits[a] = c == null ? 0 : c.visits;
                total += visits[a];
            }
        }
        for (int a = 0; a < n; a++) {
            policy[a] = total == 0 ? 0 : visits[a] / (float) total;
        }
        return new Result(policy, (float) rootNode.value(), visits);
    }

    /** Expands a leaf and returns its value. */
    private double expand(Node node) {
        Evaluation e = model.evaluate(stateOf(node));
        node.expanded = true;
        node.terminal = e.terminal();
        if (e.terminal()) {
            return 0;
        }
        float[] p = e.priors();
        double sum = 0;
        for (float v : p) {
            sum += Math.max(0, v);
        }
        node.children = new Node[p.length];
        for (int a = 0; a < p.length; a++) {
            if (p[a] > 0 || sum == 0) {
                node.children[a] = new Node((float) (sum == 0 ? 1.0 / p.length : p[a] / sum));
            }
        }
        return e.value();
    }

    @SuppressWarnings("unchecked")
    private S stateOf(Node node) {
        return (S) node.state;
    }

    private void addNoise(Node root, SplittableRandom random) {
        int legal = 0;
        for (Node c : root.children) {
            if (c != null) {
                legal++;
            }
        }
        double[] g = new double[root.children.length];
        double sum = 0;
        for (int a = 0; a < g.length; a++) {
            if (root.children[a] != null) {
                g[a] = gamma(dirichletAlpha, random);
                sum += g[a];
            }
        }
        if (legal == 0 || sum == 0) {
            return;
        }
        for (int a = 0; a < g.length; a++) {
            Node c = root.children[a];
            if (c != null) {
                Node noisy = new Node((float) ((1 - noiseFraction) * c.prior + noiseFraction * g[a] / sum));
                root.children[a] = noisy;
            }
        }
    }

    /** Marsaglia-Tsang gamma sampling. */
    private static double gamma(double shape, SplittableRandom random) {
        if (shape < 1) {
            return gamma(shape + 1, random) * Math.pow(random.nextDouble(), 1 / shape);
        }
        double d = shape - 1.0 / 3;
        double c = 1 / Math.sqrt(9 * d);
        while (true) {
            double x = random.nextGaussian();
            double v = 1 + c * x;
            if (v <= 0) {
                continue;
            }
            v = v * v * v;
            double u = random.nextDouble();
            if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
                return d * v;
            }
        }
    }

    private int select(Node node) {
        double sqrt = Math.sqrt(Math.max(1, node.visits));
        int best = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < node.children.length; a++) {
            Node c = node.children[a];
            if (c == null) {
                continue;
            }
            double q = 0;
            if (c.visits > 0) {
                double raw = c.reward + discount * (model.alternating() ? -c.value() : c.value());
                q = maxQ > minQ ? (raw - minQ) / (maxQ - minQ) : raw;
            }
            double score = q + cPuct * c.prior * sqrt / (1 + c.visits);
            if (score > bestScore) {
                bestScore = score;
                best = a;
            }
        }
        return best;
    }
}
