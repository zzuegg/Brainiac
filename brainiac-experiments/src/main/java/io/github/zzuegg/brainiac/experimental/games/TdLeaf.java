package io.github.zzuegg.brainiac.experimental.games;

import io.github.zzuegg.brainiac.experimental.rl.TwoPlayerGame;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * TD-Leaf(lambda) (Baxter, Tridgell &amp; Weaver, ICML 1998,
 * <a href="https://arxiv.org/abs/cs/9901002">arXiv:cs/9901002</a>): an evaluation function trained
 * by temporal differences between the values of the leaves of the principal variations that a
 * {@link MinimaxSearch} finds during self-play, so the learned evaluation is the one deep search
 * actually uses. The evaluation is a small tanh MLP on position features with hand-written
 * gradients; {@link #lineFeatures} gives n-in-a-row features (both stone planes plus counts of
 * open lines with 1 to {@code k-1} stones per side), and {@link #handWritten} a classic weighted
 * open-line count as the baseline evaluation.
 */
public final class TdLeaf {

    private final Function<TwoPlayerGame, float[]> features;
    private final int inputs;
    private final int hidden;
    private final double[][] w1;
    private final double[] b1;
    private final double[] w2;
    private double b2;
    private final SplittableRandom random;
    private double lambda = 0.7;
    private double learningRate = 0.01;
    private double exploration = 0.1;

    private TdLeaf(Function<TwoPlayerGame, float[]> features, int inputs, int hidden, long seed) {
        this.features = features;
        this.inputs = inputs;
        this.hidden = hidden;
        this.random = new SplittableRandom(seed);
        w1 = new double[hidden][inputs];
        b1 = new double[hidden];
        w2 = new double[hidden];
        double s = 1 / Math.sqrt(inputs);
        for (int h = 0; h < hidden; h++) {
            for (int i = 0; i < inputs; i++) {
                w1[h][i] = random.nextGaussian() * s;
            }
            w2[h] = random.nextGaussian() * 0.1;
        }
    }

    /**
     * @param features position features from the side to move
     * @param inputs their length
     * @param hidden hidden units
     * @param seed the seed
     * @return an untrained evaluation
     */
    public static TdLeaf of(Function<TwoPlayerGame, float[]> features, int inputs, int hidden, long seed) {
        return new TdLeaf(features, inputs, hidden, seed);
    }

    /** @param l the trace decay lambda @return this learner */
    public TdLeaf lambda(double l) {
        this.lambda = l;
        return this;
    }

    /** @param lr the step size @return this learner */
    public TdLeaf learningRate(double lr) {
        this.learningRate = lr;
        return this;
    }

    /** @param eps the share of random self-play moves @return this learner */
    public TdLeaf exploration(double eps) {
        this.exploration = eps;
        return this;
    }

    private double[] hiddenOut(float[] x) {
        double[] z = new double[hidden];
        for (int h = 0; h < hidden; h++) {
            double s = b1[h];
            for (int i = 0; i < inputs; i++) {
                s += w1[h][i] * x[i];
            }
            z[h] = Math.tanh(s);
        }
        return z;
    }

    private double output(double[] z) {
        double s = b2;
        for (int h = 0; h < hidden; h++) {
            s += w2[h] * z[h];
        }
        return Math.tanh(s);
    }

    /**
     * @param position a position
     * @return its value for the side to move
     */
    public double evaluate(TwoPlayerGame position) {
        return output(hiddenOut(features.apply(position)));
    }

    /** @return the evaluation as a search evaluator */
    public MinimaxSearch.Evaluator evaluator() {
        return this::evaluate;
    }

    /** Gradient of the output (for the side to move) with respect to every parameter, flattened. */
    private double[] gradient(TwoPlayerGame position) {
        float[] x = features.apply(position);
        double[] z = hiddenOut(x);
        double y = output(z);
        double dy = 1 - y * y;
        double[] g = new double[hidden * inputs + 2 * hidden + 1];
        int k = 0;
        for (int h = 0; h < hidden; h++) {
            double dz = dy * w2[h] * (1 - z[h] * z[h]);
            for (int i = 0; i < inputs; i++) {
                g[k++] = dz * x[i];
            }
        }
        for (int h = 0; h < hidden; h++) {
            g[k++] = dy * w2[h] * (1 - z[h] * z[h]);
        }
        for (int h = 0; h < hidden; h++) {
            g[k++] = dy * z[h];
        }
        g[k] = dy;
        return g;
    }

    private void apply(double[] step) {
        int k = 0;
        for (int h = 0; h < hidden; h++) {
            for (int i = 0; i < inputs; i++) {
                w1[h][i] += step[k++];
            }
        }
        for (int h = 0; h < hidden; h++) {
            b1[h] += step[k++];
        }
        for (int h = 0; h < hidden; h++) {
            w2[h] += step[k++];
        }
        b2 += step[k];
    }

    /**
     * Trains by self-play: both sides move by an alpha-beta search with the current evaluation
     * (with a share of random moves); after each game the weights move along
     * {@code sum_t grad V(leaf_t) sum_(j >= t) lambda^(j - t) d_j}, all values from the first
     * player's view and the final temporal difference against the game result.
     *
     * @param start the opening position
     * @param games self-play games
     * @param depth search plies
     * @return this learner
     */
    public TdLeaf selfPlay(TwoPlayerGame start, int games, int depth) {
        MinimaxSearch search = MinimaxSearch.of(MinimaxSearch.Method.ALPHA_BETA_PLAIN, evaluator()).depth(depth);
        for (int game = 0; game < games; game++) {
            List<Double> values = new ArrayList<>();
            List<double[]> grads = new ArrayList<>();
            TwoPlayerGame g = start;
            while (!g.over()) {
                MinimaxSearch.Result r = search.search(g);
                TwoPlayerGame leaf = r.leaf();
                if (leaf.over()) {
                    int w = leaf.winner();
                    values.add((double) w * start.toMove());
                    grads.add(null);
                } else {
                    double sign = leaf.toMove() == start.toMove() ? 1 : -1;
                    values.add(sign * evaluate(leaf));
                    double[] gr = gradient(leaf);
                    for (int i = 0; i < gr.length; i++) {
                        gr[i] *= sign;
                    }
                    grads.add(gr);
                }
                int move = r.move();
                if (random.nextDouble() < exploration) {
                    move = TwoPlayerGame.Player.random(random.nextLong()).move(g);
                }
                g = g.play(move);
            }
            double z = g.winner() * start.toMove();
            int n = values.size();
            double[] d = new double[n];
            for (int t = 0; t < n; t++) {
                double next = t + 1 < n ? values.get(t + 1) : z;
                d[t] = next - values.get(t);
            }
            double[] step = null;
            double trace = 0;
            for (int t = n - 1; t >= 0; t--) {
                trace = d[t] + lambda * trace;
                double[] gr = grads.get(t);
                if (gr == null) {
                    continue;
                }
                if (step == null) {
                    step = new double[gr.length];
                }
                for (int i = 0; i < gr.length; i++) {
                    step[i] += learningRate * gr[i] * trace;
                }
            }
            if (step != null) {
                apply(step);
            }
        }
        return this;
    }

    // ------------------------------------------------------------------ n-in-a-row features

    private static List<int[]> windows(int rows, int cols, int k) {
        List<int[]> out = new ArrayList<>();
        int[][] dirs = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                for (int[] dir : dirs) {
                    int er = r + dir[0] * (k - 1);
                    int ec = c + dir[1] * (k - 1);
                    if (er < 0 || er >= rows || ec < 0 || ec >= cols) {
                        continue;
                    }
                    int[] w = new int[k];
                    for (int i = 0; i < k; i++) {
                        w[i] = (r + dir[0] * i) * cols + c + dir[1] * i;
                    }
                    out.add(w);
                }
            }
        }
        return out;
    }

    /* Open-line counts [mover 1..k-1, opponent 1..k-1] from the encoding's two planes. */
    private static double[] lineCounts(float[] planes, List<int[]> windows, int cells, int k) {
        double[] counts = new double[2 * (k - 1)];
        for (int[] w : windows) {
            int mine = 0;
            int theirs = 0;
            for (int c : w) {
                mine += planes[c] > 0.5f ? 1 : 0;
                theirs += planes[cells + c] > 0.5f ? 1 : 0;
            }
            if (theirs == 0 && mine > 0 && mine < k) {
                counts[mine - 1]++;
            } else if (mine == 0 && theirs > 0 && theirs < k) {
                counts[k - 1 + theirs - 1]++;
            }
        }
        return counts;
    }

    /**
     * Features for {@code NInARow} boards (whose encoding is the mover's plane then the
     * opponent's): both planes followed by the open-line counts scaled by the window count.
     *
     * @param rows board rows
     * @param cols board columns
     * @param k stones in a row to win
     * @return the feature function, of length {@link #lineFeatureCount}
     */
    public static Function<TwoPlayerGame, float[]> lineFeatures(int rows, int cols, int k) {
        List<int[]> windows = windows(rows, cols, k);
        int cells = rows * cols;
        return g -> {
            float[] planes = g.encode();
            double[] counts = lineCounts(planes, windows, cells, k);
            float[] x = new float[2 * cells + counts.length];
            System.arraycopy(planes, 0, x, 0, 2 * cells);
            for (int i = 0; i < counts.length; i++) {
                x[2 * cells + i] = (float) (counts[i] / windows.size() * 4);
            }
            return x;
        };
    }

    /**
     * @param rows board rows
     * @param cols board columns
     * @param k stones in a row to win
     * @return the length of {@link #lineFeatures}
     */
    public static int lineFeatureCount(int rows, int cols, int k) {
        return 2 * rows * cols + 2 * (k - 1);
    }

    /**
     * The classic hand-written evaluation: open lines weighted 1, 4, 16, ... by their stone count,
     * mover minus opponent, squashed by tanh.
     *
     * @param rows board rows
     * @param cols board columns
     * @param k stones in a row to win
     * @return the evaluator
     */
    public static MinimaxSearch.Evaluator handWritten(int rows, int cols, int k) {
        List<int[]> windows = windows(rows, cols, k);
        int cells = rows * cols;
        return g -> {
            double[] counts = lineCounts(g.encode(), windows, cells, k);
            double score = 0;
            double weight = 1;
            for (int i = 0; i < k - 1; i++) {
                score += weight * (counts[i] - counts[k - 1 + i]);
                weight *= 4;
            }
            return Math.tanh(score / 40);
        };
    }
}
