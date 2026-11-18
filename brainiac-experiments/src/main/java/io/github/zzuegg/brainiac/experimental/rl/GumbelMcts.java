package io.github.zzuegg.brainiac.experimental.rl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Gumbel MuZero / Gumbel AlphaZero search (Danihelka et al., 2022,
 * <a href="https://openreview.net/forum?id=bERaNdoegnO">ICLR 2022</a>): a policy-improving tree
 * search that works with very few simulations. At the root it samples Gumbel noise
 * {@code g(a)}, keeps the top {@code m} actions by {@code g(a) + logits(a)} (sampling without
 * replacement), and spends the budget by <b>sequential halving</b>: every round visits each
 * remaining action equally, then drops the worse half by
 * {@code g(a) + logits(a) + sigma(q(a))}. The chosen action is the survivor. Inside the tree
 * it acts deterministically, picking the action whose visit share lags most behind the
 * improved policy {@code softmax(logits + sigma(completedQ))}, where unvisited actions get the
 * mixed value estimate {@code v_mix} and {@code sigma(q) = (cVisit + max N) cScale q}.
 * That improved policy is also the training target, and it improves on the prior even with
 * 2 simulations, where PUCT's visit counts do not.
 *
 * <p>It searches the same {@link Mcts.Model} as the PUCT {@link Mcts}, so the two are
 * interchangeable; values are min-max normalized over the tree, as in {@link Mcts}.
 *
 * @param <S> the state type
 */
public final class GumbelMcts<S> {

    /**
     * The outcome of one search.
     *
     * @param action the selected action
     * @param policy the improved policy {@code softmax(logits + sigma(completedQ))} at the root
     * @param value the root's searched value
     * @param visits the root's visit count per action
     */
    public record Result(int action, float[] policy, float value, int[] visits) {
    }

    private static final class Node {
        final float prior;
        Object state;
        float reward;
        boolean expanded;
        boolean terminal;
        float rawValue;
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

    private final Mcts.Model<S> model;
    private int simulations = 16;
    private int maxConsidered = 16;
    private float cVisit = 50;
    private float cScale = 1;
    private float discount = 1;
    private boolean gumbel = true;
    private double minQ;
    private double maxQ;

    private GumbelMcts(Mcts.Model<S> model) {
        this.model = model;
    }

    /**
     * @param model the problem
     * @param <S> the state type
     * @return a search with 16 simulations, at most 16 root actions, {@code cVisit = 50}, {@code cScale = 1}
     */
    public static <S> GumbelMcts<S> of(Mcts.Model<S> model) {
        return new GumbelMcts<>(Objects.requireNonNull(model, "model"));
    }

    /** @param n simulations per search @return this search */
    public GumbelMcts<S> simulations(int n) {
        this.simulations = Math.max(1, n);
        return this;
    }

    /** @param m the most root actions sampled @return this search */
    public GumbelMcts<S> maxConsidered(int m) {
        this.maxConsidered = Math.max(1, m);
        return this;
    }

    /**
     * @param visit {@code cVisit}
     * @param scale {@code cScale}
     * @return this search
     */
    public GumbelMcts<S> sigma(float visit, float scale) {
        this.cVisit = visit;
        this.cScale = scale;
        return this;
    }

    /** @param gamma the discount @return this search */
    public GumbelMcts<S> discount(float gamma) {
        this.discount = gamma;
        return this;
    }

    /** @param on add Gumbel noise at the root (off: deterministic top-m by prior) @return this search */
    public GumbelMcts<S> gumbelNoise(boolean on) {
        this.gumbel = on;
        return this;
    }

    /**
     * Searches from a state.
     *
     * @param root the state
     * @param random the source of the Gumbel noise
     * @return the chosen action, improved policy and statistics
     */
    public Result search(S root, SplittableRandom random) {
        minQ = Double.POSITIVE_INFINITY;
        maxQ = Double.NEGATIVE_INFINITY;
        int n = model.actions();
        Node rootNode = new Node(1);
        rootNode.state = root;
        expand(rootNode);
        if (rootNode.terminal) {
            return new Result(0, new float[n], 0, new int[n]);
        }
        double[] logits = new double[n];
        double[] g = new double[n];
        List<Integer> legal = new ArrayList<>();
        for (int a = 0; a < n; a++) {
            if (rootNode.children[a] != null) {
                legal.add(a);
                logits[a] = Math.log(Math.max(1e-12, rootNode.children[a].prior));
                g[a] = gumbel ? -Math.log(-Math.log(random.nextDouble(1e-12, 1))) : 0;
            }
        }
        List<Integer> considered = new ArrayList<>(legal);
        considered.sort((x, y) -> Double.compare(g[y] + logits[y], g[x] + logits[x]));
        int m = Math.min(Math.min(maxConsidered, simulations), considered.size());
        considered = new ArrayList<>(considered.subList(0, m));
        int phases = Math.max(1, (int) Math.ceil(Math.log(m) / Math.log(2)));
        int used = 0;
        for (int phase = 0; phase < phases && used < simulations; phase++) {
            int per = Math.max(1, simulations / (phases * considered.size()));
            for (int k = 0; k < per && used < simulations; k++) {
                for (int a : considered) {
                    if (used >= simulations) {
                        break;
                    }
                    simulate(rootNode, a);
                    used++;
                }
            }
            if (considered.size() > 1) {
                double[] score = scores(rootNode, considered, g, logits);
                List<Integer> order = new ArrayList<>(considered);
                order.sort((x, y) -> Double.compare(score[y], score[x]));
                considered = new ArrayList<>(order.subList(0, Math.max(1, (considered.size() + 1) / 2)));
            }
        }
        // spend what the rounding left over on the survivors
        while (used < simulations) {
            for (int a : considered) {
                if (used >= simulations) {
                    break;
                }
                simulate(rootNode, a);
                used++;
            }
        }
        double[] score = scores(rootNode, considered, g, logits);
        int best = considered.get(0);
        for (int a : considered) {
            if (score[a] > score[best]) {
                best = a;
            }
        }
        float[] policy = improvedPolicy(rootNode);
        int[] visits = new int[n];
        for (int a = 0; a < n; a++) {
            visits[a] = rootNode.children[a] == null ? 0 : rootNode.children[a].visits;
        }
        return new Result(best, policy, (float) rootNode.value(), visits);
    }

    private double[] scores(Node root, List<Integer> actions, double[] g, double[] logits) {
        double[] q = completedQ(root);
        double maxN = maxVisits(root);
        double[] s = new double[root.children.length];
        Arrays.fill(s, Double.NEGATIVE_INFINITY);
        for (int a : actions) {
            s[a] = g[a] + logits[a] + (cVisit + maxN) * cScale * q[a];
        }
        return s;
    }

    private static double maxVisits(Node node) {
        int max = 0;
        for (Node c : node.children) {
            if (c != null) {
                max = Math.max(max, c.visits);
            }
        }
        return max;
    }

    private double qOf(Node child) {
        return child.reward + discount * (model.alternating() ? -child.value() : child.value());
    }

    private double normalize(double q) {
        return maxQ > minQ ? (q - minQ) / (maxQ - minQ) : 0.5;
    }

    /** Normalized Q of visited children; unvisited ones get the mixed value. */
    private double[] completedQ(Node node) {
        int n = node.children.length;
        double[] q = new double[n];
        double sumN = 0;
        double sumP = 0;
        double sumPq = 0;
        for (Node c : node.children) {
            if (c != null && c.visits > 0) {
                sumN += c.visits;
                sumP += c.prior;
                sumPq += c.prior * qOf(c);
            }
        }
        double vMix = sumP > 0 ? (node.rawValue + sumN / sumP * sumPq) / (1 + sumN) : node.rawValue;
        for (int a = 0; a < n; a++) {
            Node c = node.children[a];
            q[a] = c == null ? 0 : normalize(c.visits > 0 ? qOf(c) : vMix);
        }
        return q;
    }

    private float[] improvedPolicy(Node node) {
        int n = node.children.length;
        double[] q = completedQ(node);
        double sigma = (cVisit + maxVisits(node)) * cScale;
        double[] z = new double[n];
        double max = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < n; a++) {
            Node c = node.children[a];
            z[a] = c == null ? Double.NEGATIVE_INFINITY : Math.log(Math.max(1e-12, c.prior)) + sigma * q[a];
            max = Math.max(max, z[a]);
        }
        float[] p = new float[n];
        double sum = 0;
        for (int a = 0; a < n; a++) {
            double e = node.children[a] == null ? 0 : Math.exp(z[a] - max);
            p[a] = (float) e;
            sum += e;
        }
        for (int a = 0; a < n; a++) {
            p[a] = (float) (p[a] / sum);
        }
        return p;
    }

    private int selectInterior(Node node) {
        float[] pi = improvedPolicy(node);
        double total = 0;
        for (Node c : node.children) {
            if (c != null) {
                total += c.visits;
            }
        }
        int best = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int a = 0; a < pi.length; a++) {
            Node c = node.children[a];
            if (c == null) {
                continue;
            }
            double score = pi[a] - c.visits / (1 + total);
            if (score > bestScore) {
                bestScore = score;
                best = a;
            }
        }
        return best;
    }

    private void simulate(Node root, int first) {
        List<Node> path = new ArrayList<>();
        path.add(root);
        Node node = root;
        int a = first;
        while (true) {
            Node child = node.children[a];
            if (child.state == null) {
                Mcts.Step<S> s = model.step(stateOf(node), a);
                child.state = s.state();
                child.reward = s.reward();
            }
            node = child;
            path.add(node);
            if (!node.expanded || node.terminal) {
                break;
            }
            a = selectInterior(node);
        }
        double value = node.terminal ? 0 : (node.expanded ? node.value() : expand(node));
        for (int i = path.size() - 1; i >= 0; i--) {
            Node p = path.get(i);
            p.valueSum += value;
            p.visits++;
            if (i > 0) {
                double q = qOf(p);
                minQ = Math.min(minQ, q);
                maxQ = Math.max(maxQ, q);
                value = p.reward + discount * (model.alternating() ? -value : value);
            }
        }
    }

    private double expand(Node node) {
        Mcts.Evaluation e = model.evaluate(stateOf(node));
        node.expanded = true;
        node.terminal = e.terminal();
        if (e.terminal()) {
            return 0;
        }
        node.rawValue = e.value();
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
}
