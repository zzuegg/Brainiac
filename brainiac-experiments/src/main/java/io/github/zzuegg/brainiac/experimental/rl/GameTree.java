package io.github.zzuegg.brainiac.experimental.rl;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * An {@link ExtensiveFormGame} expanded once into flat arrays: every history is a node
 * index (the root is 0) and every information set an index, so solvers such as
 * {@link CounterfactualRegret} and {@link Exploitability} traverse integers instead of
 * building strings. Kuhn poker has 55 nodes and 12 information sets, Leduc poker 9,451 nodes and 288.
 *
 * <p>A <em>policy</em> for the whole game is a {@code double[infoSets()][maxActions()]} table
 * of action probabilities, one row per information set of either player; illegal actions get
 * probability 0.
 */
public final class GameTree {

    private final ExtensiveFormGame game;
    private final int[] player;
    private final int[][] children;
    private final double[][] chance;
    private final double[] payoff;
    private final int[] infoSet;
    private final String[] keys;
    private final int[] infoSetPlayer;
    private final boolean[][] legal;
    private final float[][] features;
    private final int[][] infoSetNodes;
    private final int[] depth;

    private GameTree(ExtensiveFormGame game, Builder b) {
        this.game = game;
        int n = b.player.size();
        player = b.player.stream().mapToInt(Integer::intValue).toArray();
        children = b.children.toArray(new int[n][]);
        chance = b.chance.toArray(new double[n][]);
        payoff = b.payoff.stream().mapToDouble(Double::doubleValue).toArray();
        infoSet = b.infoSet.stream().mapToInt(Integer::intValue).toArray();
        depth = b.depth.stream().mapToInt(Integer::intValue).toArray();
        int m = b.keys.size();
        keys = b.keys.toArray(new String[0]);
        infoSetPlayer = b.infoSetPlayer.stream().mapToInt(Integer::intValue).toArray();
        legal = b.legal.toArray(new boolean[m][]);
        features = b.features.toArray(new float[m][]);
        int[] counts = new int[m];
        for (int i = 0; i < n; i++) {
            if (infoSet[i] >= 0) {
                counts[infoSet[i]]++;
            }
        }
        infoSetNodes = new int[m][];
        for (int s = 0; s < m; s++) {
            infoSetNodes[s] = new int[counts[s]];
        }
        Arrays.fill(counts, 0);
        for (int i = 0; i < n; i++) {
            if (infoSet[i] >= 0) {
                infoSetNodes[infoSet[i]][counts[infoSet[i]]++] = i;
            }
        }
    }

    private static final class Builder {
        final List<Integer> player = new ArrayList<>();
        final List<int[]> children = new ArrayList<>();
        final List<double[]> chance = new ArrayList<>();
        final List<Double> payoff = new ArrayList<>();
        final List<Integer> infoSet = new ArrayList<>();
        final List<Integer> depth = new ArrayList<>();
        final Map<String, Integer> index = new HashMap<>();
        final List<String> keys = new ArrayList<>();
        final List<Integer> infoSetPlayer = new ArrayList<>();
        final List<boolean[]> legal = new ArrayList<>();
        final List<float[]> features = new ArrayList<>();

        int add(ExtensiveFormGame.State s, int maxActions, int d) {
            int id = player.size();
            player.add(s.terminal() ? -2 : s.player());
            children.add(null);
            chance.add(null);
            depth.add(d);
            if (s.terminal()) {
                payoff.add(s.payoff());
                infoSet.add(-1);
                return id;
            }
            payoff.add(0.0);
            int p = s.player();
            int width = p == ExtensiveFormGame.CHANCE ? s.actionCount() : maxActions;
            if (p == ExtensiveFormGame.CHANCE) {
                infoSet.add(-1);
                double[] probs = new double[width];
                for (int a = 0; a < width; a++) {
                    probs[a] = s.legal(a) ? s.chanceProbability(a) : 0;
                }
                chance.set(id, probs);
            } else {
                String key = p + "|" + s.infoSet();
                Integer is = index.get(key);
                if (is == null) {
                    is = keys.size();
                    index.put(key, is);
                    keys.add(s.infoSet());
                    infoSetPlayer.add(p);
                    boolean[] l = new boolean[maxActions];
                    for (int a = 0; a < maxActions; a++) {
                        l[a] = s.legal(a);
                    }
                    legal.add(l);
                    features.add(s.features());
                }
                infoSet.add(is);
            }
            int[] kids = new int[width];
            for (int a = 0; a < width; a++) {
                kids[a] = s.legal(a) ? add(s.play(a), maxActions, d + 1) : -1;
            }
            children.set(id, kids);
            return id;
        }
    }

    /**
     * Expands a game.
     *
     * @param game the game
     * @return its tree
     */
    public static GameTree compile(ExtensiveFormGame game) {
        Builder b = new Builder();
        b.add(game.root(), game.maxActions(), 0);
        return new GameTree(game, b);
    }

    /** @return the game */
    public ExtensiveFormGame game() {
        return game;
    }

    /** @return the number of nodes */
    public int nodes() {
        return player.length;
    }

    /** @return the number of information sets of both players */
    public int infoSets() {
        return keys.length;
    }

    /** @return action indices per decision */
    public int maxActions() {
        return game.maxActions();
    }

    /**
     * @param node a node
     * @return whether it is terminal
     */
    public boolean terminal(int node) {
        return player[node] == -2;
    }

    /**
     * @param node a non-terminal node
     * @return the player to act, or {@link ExtensiveFormGame#CHANCE}
     */
    public int player(int node) {
        return player[node];
    }

    /**
     * @param node a non-terminal node
     * @return the child per action index, -1 where illegal (do not modify)
     */
    public int[] children(int node) {
        return children[node];
    }

    /**
     * @param node a chance node
     * @return outcome probabilities (do not modify)
     */
    public double[] chance(int node) {
        return chance[node];
    }

    /**
     * @param node a terminal node
     * @return the payoff to player 0
     */
    public double payoff(int node) {
        return payoff[node];
    }

    /**
     * @param node a player node
     * @return its information set
     */
    public int infoSet(int node) {
        return infoSet[node];
    }

    /**
     * @param node a node
     * @return its depth, 0 at the root
     */
    public int depth(int node) {
        return depth[node];
    }

    /**
     * @param set an information set
     * @return the player who acts there
     */
    public int infoSetPlayer(int set) {
        return infoSetPlayer[set];
    }

    /**
     * @param set an information set
     * @return its key, as {@link ExtensiveFormGame.State#infoSet} gives it
     */
    public String key(int set) {
        return keys[set];
    }

    /**
     * @param set an information set
     * @return which actions are legal there (do not modify)
     */
    public boolean[] legal(int set) {
        return legal[set];
    }

    /**
     * @param set an information set
     * @return its feature encoding (do not modify)
     */
    public float[] features(int set) {
        return features[set];
    }

    /**
     * @param set an information set
     * @return the nodes in it (do not modify)
     */
    public int[] nodesOf(int set) {
        return infoSetNodes[set];
    }

    /** @return the policy that plays every legal action equally often */
    public double[][] uniformPolicy() {
        double[][] p = new double[infoSets()][maxActions()];
        for (int s = 0; s < p.length; s++) {
            int count = 0;
            for (boolean l : legal[s]) {
                count += l ? 1 : 0;
            }
            for (int a = 0; a < maxActions(); a++) {
                p[s][a] = legal[s][a] ? 1.0 / count : 0;
            }
        }
        return p;
    }

    /**
     * The expected payoff to player 0 when player 0 plays {@code policy0} and player 1 plays
     * {@code policy1} (each is read only at its own player's information sets).
     *
     * @param policy0 player 0's policy table
     * @param policy1 player 1's policy table
     * @return the expected payoff to player 0
     */
    public double expectedValue(double[][] policy0, double[][] policy1) {
        return value(0, policy0, policy1);
    }

    /**
     * Counterfactual action values of both players under one profile, in one traversal: for
     * every information set {@code I} of player {@code p} and action {@code a},
     * {@code q[I][a] = sum over h in I of reachOthers(h) * value_p(h a)}, where
     * {@code reachOthers} is the probability that the opponent and chance reach {@code h}
     * and {@code value_p} is player {@code p}'s expected payoff. Dividing by
     * {@code reach[I]} (the summed {@code reachOthers}) gives the expected action values given
     * that {@code I} is reached. These are the quantities CFR, policy gradient and mirror
     * descent update with.
     *
     * @param policy the profile
     * @param q output {@code [infoSets][maxActions]}, overwritten
     * @param reach output {@code [infoSets]}, overwritten
     * @return the expected payoff to player 0
     */
    public double counterfactualValues(double[][] policy, double[][] q, double[] reach) {
        for (double[] row : q) {
            java.util.Arrays.fill(row, 0);
        }
        java.util.Arrays.fill(reach, 0);
        return counterfactual(0, policy, 1, 1, 1, q, reach);
    }

    private double counterfactual(int node, double[][] policy, double r0, double r1, double rc, double[][] q,
            double[] reach) {
        if (terminal(node)) {
            return payoff[node];
        }
        int[] kids = children[node];
        double v = 0;
        if (player[node] == ExtensiveFormGame.CHANCE) {
            for (int a = 0; a < kids.length; a++) {
                if (kids[a] >= 0) {
                    v += chance[node][a] * counterfactual(kids[a], policy, r0, r1, rc * chance[node][a], q, reach);
                }
            }
            return v;
        }
        int p = player[node];
        int s = infoSet[node];
        double[] sigma = policy[s];
        double others = (p == 0 ? r1 : r0) * rc;
        double sign = p == 0 ? 1 : -1;
        reach[s] += others;
        for (int a = 0; a < kids.length; a++) {
            if (kids[a] < 0) {
                continue;
            }
            double child = counterfactual(kids[a], policy, p == 0 ? r0 * sigma[a] : r0, p == 1 ? r1 * sigma[a] : r1,
                    rc, q, reach);
            q[s][a] += others * sign * child;
            v += sigma[a] * child;
        }
        return v;
    }

    private double value(int node, double[][] p0, double[][] p1) {
        if (terminal(node)) {
            return payoff[node];
        }
        int[] kids = children[node];
        double v = 0;
        if (player[node] == ExtensiveFormGame.CHANCE) {
            for (int a = 0; a < kids.length; a++) {
                if (kids[a] >= 0) {
                    v += chance[node][a] * value(kids[a], p0, p1);
                }
            }
            return v;
        }
        double[] sigma = (player[node] == 0 ? p0 : p1)[infoSet[node]];
        for (int a = 0; a < kids.length; a++) {
            if (kids[a] >= 0 && sigma[a] > 0) {
                v += sigma[a] * value(kids[a], p0, p1);
            }
        }
        return v;
    }

    /**
     * Samples one action from a policy row, among the legal actions.
     *
     * @param row action probabilities
     * @param set the information set (for its legal actions)
     * @param random the random source
     * @return the action index
     */
    public int sample(double[] row, int set, SplittableRandom random) {
        double u = random.nextDouble();
        int last = -1;
        for (int a = 0; a < row.length; a++) {
            if (!legal[set][a]) {
                continue;
            }
            last = a;
            u -= row[a];
            if (u <= 0) {
                return a;
            }
        }
        return last;
    }

    /**
     * Samples a chance outcome.
     *
     * @param node a chance node
     * @param random the random source
     * @return the outcome index
     */
    public int sampleChance(int node, SplittableRandom random) {
        double u = random.nextDouble();
        int last = -1;
        for (int a = 0; a < chance[node].length; a++) {
            if (chance[node][a] > 0) {
                last = a;
                u -= chance[node][a];
                if (u <= 0) {
                    return a;
                }
            }
        }
        return last;
    }
}
