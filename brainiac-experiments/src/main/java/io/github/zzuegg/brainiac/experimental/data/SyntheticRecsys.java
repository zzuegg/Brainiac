package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Synthetic recommendation and tabular data with known structure:
 * <ul>
 *   <li>{@link #preferences}: users and items with hidden taste vectors, observed through
 *       noisy feature vectors; a user's relevant items are those with the highest hidden
 *       affinity (retrieval);</li>
 *   <li>{@link #clicks}: click-through data over categorical fields, one-hot encoded,
 *       where the click probability depends on a hidden pairwise interaction between
 *       two fields (what factorization machines model);</li>
 *   <li>{@link #tabular}: numeric features whose label depends on products of features
 *       and on a few informative columns among distractors (feature interactions and
 *       feature selection);</li>
 *   <li>{@link #sequences}: per-user item histories that follow planted item-transition chains
 *       inside drifting interest clusters, with timestamps, click/skip actions and item content
 *       vectors that place each chain on a smooth ring (sequential and generative
 *       recommendation, cold-start items);</li>
 *   <li>{@link #banditWorld}: a contextual bandit with known expected rewards and a known
 *       logging policy, which logs feedback with exact propensities (off-policy evaluation)
 *       and serves as an online simulator (contextual bandits).</li>
 * </ul>
 */
public final class SyntheticRecsys {

    private SyntheticRecsys() {
    }

    /**
     * Users, items and their relevant items.
     *
     * @param userFeatures {@code [users, features]}
     * @param itemFeatures {@code [items, features]}
     * @param relevant {@code [users][k]} each user's most relevant items, best first
     */
    public record Preferences(Tensor userFeatures, Tensor itemFeatures, int[][] relevant) {
    }

    /**
     * A labelled table.
     *
     * @param features {@code [rows, columns]}
     * @param labels {@code [rows, 1]}, 0 or 1
     */
    public record Table(Tensor features, Tensor labels) {
        /** @return the number of rows */
        public int rows() {
            return features.shape().dim(0);
        }

        /**
         * @param from the first row
         * @param to one past the last row
         * @return those rows, for train/test splits (tables drawn with one seed share their
         *     hidden structure, so split one table rather than drawing two)
         */
        public Table rows(int from, int to) {
            return new Table(features.slice(0, from, to), labels.slice(0, from, to));
        }
    }

    /**
     * @param users the number of users
     * @param items the number of items
     * @param latent the hidden taste dimension
     * @param features the observed feature dimension
     * @param noise the observation noise
     * @param k relevant items per user
     * @param seed the seed
     * @return the data
     */
    public static Preferences preferences(int users, int items, int latent, int features, double noise, int k,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] u = gaussian(users, latent, r);
        float[][] v = gaussian(items, latent, r);
        float[][] mixU = gaussian(latent, features, r);
        float[][] mixV = gaussian(latent, features, r);
        int[][] relevant = new int[users][];
        for (int i = 0; i < users; i++) {
            final float[] score = new float[items];
            for (int j = 0; j < items; j++) {
                score[j] = dot(u[i], v[j]);
            }
            relevant[i] = java.util.stream.IntStream.range(0, items).boxed()
                    .sorted((a, b) -> Float.compare(score[b], score[a])).limit(k).mapToInt(Integer::intValue)
                    .toArray();
        }
        return new Preferences(observe(u, mixU, noise, r), observe(v, mixV, noise, r), relevant);
    }

    /**
     * Click-through data over categorical fields.
     *
     * @param rows samples
     * @param fields categorical fields
     * @param categories categories per field
     * @param seed the seed
     * @return one-hot features ({@code fields * categories} columns) and clicks
     */
    public static Table clicks(int rows, int fields, int categories, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] pair = gaussian(categories, categories, r);
        float[] bias = new float[categories];
        for (int c = 0; c < categories; c++) {
            bias[c] = (float) (0.3 * r.nextGaussian());
        }
        int width = fields * categories;
        float[] x = new float[rows * width];
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            int[] cat = new int[fields];
            for (int f = 0; f < fields; f++) {
                cat[f] = r.nextInt(categories);
                x[i * width + f * categories + cat[f]] = 1;
            }
            double logit = 2 * pair[cat[0]][cat[1]] + bias[cat[fields - 1]];
            y[i] = r.nextDouble() < 1 / (1 + Math.exp(-logit)) ? 1 : 0;
        }
        return new Table(Tensor.of(Shape.of(rows, width), x), Tensor.of(Shape.of(rows, 1), y));
    }

    /**
     * Numeric table: the label is {@code x0 * x1 + x2 * x3 - 0.5 x4 > 0} (with 2% label
     * noise); every other column is a distractor.
     *
     * @param rows samples
     * @param columns columns, at least 5
     * @param seed the seed
     * @return the table
     */
    public static Table tabular(int rows, int columns, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[rows * columns];
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            for (int c = 0; c < columns; c++) {
                x[i * columns + c] = (float) r.nextGaussian();
            }
            int o = i * columns;
            double s = x[o] * x[o + 1] + x[o + 2] * x[o + 3] - 0.5 * x[o + 4];
            boolean label = s > 0;
            if (r.nextDouble() < 0.02) {
                label = !label;
            }
            y[i] = label ? 1 : 0;
        }
        return new Table(Tensor.of(Shape.of(rows, columns), x), Tensor.of(Shape.of(rows, 1), y));
    }

    /**
     * Per-user interaction sequences with a planted transition structure.
     *
     * @param items the number of items
     * @param histories {@code [users][length]} item ids in time order
     * @param times {@code [users][length]} increasing timestamps (seconds)
     * @param clicks {@code [users][length]} 1 for a click, 0 for a skip
     * @param content {@code [items, contentDim]} item content vectors
     * @param cluster each item's interest cluster
     * @param successor each item's planted next item on its chain
     */
    public record Sequences(int items, int[][] histories, long[][] times, int[][] clicks, Tensor content,
            int[] cluster, int[] successor) {

        /** @return the number of users */
        public int users() {
            return histories.length;
        }
    }

    /**
     * Item histories: every item belongs to one of {@code clusters} interest clusters, and the
     * items of a cluster form a ring (chain) whose order is also visible in the content vectors.
     * Each step follows the chain with probability {@code follow}; otherwise the user jumps to
     * a random item of the current cluster, and with probability {@code drift} first switches to
     * another cluster (drifting interests). A user clicks an item with a probability that grows
     * with their hidden affinity to its cluster and shrinks after long gaps between events.
     *
     * @param users the number of users
     * @param items the number of items
     * @param length events per user
     * @param clusters interest clusters
     * @param contentDim the content vector width, at least 2
     * @param follow the probability of following the chain
     * @param drift the probability of switching cluster on a jump
     * @param seed the seed
     * @return the sequences
     */
    public static Sequences sequences(int users, int items, int length, int clusters, int contentDim, double follow,
            double drift, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] perm = new int[items];
        for (int i = 0; i < items; i++) {
            perm[i] = i;
        }
        for (int i = items - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
        }
        int[] cluster = new int[items];
        int[][] members = new int[clusters][];
        int[] successor = new int[items];
        float[] content = new float[items * contentDim];
        for (int c = 0; c < clusters; c++) {
            int size = items / clusters + (c < items % clusters ? 1 : 0);
            members[c] = new int[size];
            float[] centre = new float[contentDim];
            float[] a = new float[contentDim];
            float[] b = new float[contentDim];
            for (int d = 0; d < contentDim; d++) {
                centre[d] = (float) (1.5 * r.nextGaussian());
                a[d] = (float) (r.nextGaussian() / Math.sqrt(contentDim));
                b[d] = (float) (r.nextGaussian() / Math.sqrt(contentDim));
            }
            for (int k = 0; k < size; k++) {
                int item = perm[c + k * clusters];
                members[c][k] = item;
                cluster[item] = c;
                double angle = 2 * Math.PI * k / size;
                for (int d = 0; d < contentDim; d++) {
                    content[item * contentDim + d] = (float) (centre[d] + 2 * (Math.cos(angle) * a[d]
                            + Math.sin(angle) * b[d]) + 0.05 * r.nextGaussian());
                }
            }
            for (int k = 0; k < size; k++) {
                successor[members[c][k]] = members[c][(k + 1) % size];
            }
        }
        int[][] hist = new int[users][length];
        long[][] times = new long[users][length];
        int[][] clicks = new int[users][length];
        for (int u = 0; u < users; u++) {
            double[] affinity = new double[clusters];
            for (int c = 0; c < clusters; c++) {
                affinity[c] = r.nextGaussian();
            }
            int c = r.nextInt(clusters);
            int cur = members[c][r.nextInt(members[c].length)];
            long t = r.nextInt(1000);
            for (int s = 0; s < length; s++) {
                if (s > 0) {
                    if (r.nextDouble() < follow) {
                        cur = successor[cur];
                    } else {
                        if (r.nextDouble() < drift) {
                            c = r.nextInt(clusters);
                        }
                        cur = members[c][r.nextInt(members[c].length)];
                    }
                }
                long gap = 1 + (long) (-600 * Math.log(1 - r.nextDouble()));
                t += gap;
                hist[u][s] = cur;
                times[u][s] = t;
                double logit = 1.5 * affinity[cluster[cur]] - (gap > 600 ? 1.0 : -0.5);
                clicks[u][s] = r.nextDouble() < 1 / (1 + Math.exp(-logit)) ? 1 : 0;
            }
        }
        return new Sequences(items, hist, times, clicks, Tensor.of(Shape.of(items, contentDim), content), cluster,
                successor);
    }

    /**
     * Logged bandit feedback.
     *
     * @param contexts {@code [rounds][dim]}
     * @param actions the logged action of each round
     * @param propensities the logging policy's probability of that action
     * @param rewards the observed 0/1 reward
     * @param expected {@code [rounds][actions]} the true expected reward of every action (for
     *     evaluation only)
     * @param logging {@code [rounds][actions]} the logging policy's full distribution
     */
    public record BanditLog(double[][] contexts, int[] actions, double[] propensities, double[] rewards,
            double[][] expected, double[][] logging) {

        /** @return the number of logged rounds */
        public int rounds() {
            return actions.length;
        }
    }

    /**
     * A contextual bandit with Bernoulli rewards: {@code P(r = 1 | x, a) = sigmoid(f_a(x))},
     * linear {@code f_a(x) = theta_a . x} or non-linear
     * {@code f_a(x) = 2 sin(theta_a . x) + (u_a . x)^2 - 1}. Its logging policy is a softmax over
     * a noisy copy of the true linear scores.
     */
    public static final class BanditWorld {
        private final double[][] theta;
        private final double[][] u;
        private final double[][] logWeights;
        private final boolean nonlinear;
        private final SplittableRandom random;

        private BanditWorld(int dim, int actions, boolean nonlinear, double logNoise, long seed) {
            this.random = new SplittableRandom(seed);
            this.nonlinear = nonlinear;
            theta = new double[actions][dim];
            u = new double[actions][dim];
            logWeights = new double[actions][dim];
            for (int a = 0; a < actions; a++) {
                for (int d = 0; d < dim; d++) {
                    theta[a][d] = random.nextGaussian() / Math.sqrt(dim) * 1.5;
                    u[a][d] = random.nextGaussian() / Math.sqrt(dim);
                    logWeights[a][d] = theta[a][d] + logNoise * random.nextGaussian() / Math.sqrt(dim);
                }
            }
        }

        /** @return the context width */
        public int dim() {
            return theta[0].length;
        }

        /** @return the number of actions */
        public int actions() {
            return theta.length;
        }

        /** @return a fresh standard-normal context */
        public double[] context() {
            double[] x = new double[dim()];
            for (int d = 0; d < x.length; d++) {
                x[d] = random.nextGaussian();
            }
            return x;
        }

        /**
         * @param x a context
         * @param a an action
         * @return the true probability of reward 1
         */
        public double expected(double[] x, int a) {
            double s = 0;
            double q = 0;
            for (int d = 0; d < x.length; d++) {
                s += theta[a][d] * x[d];
                q += u[a][d] * x[d];
            }
            double f = nonlinear ? 2 * Math.sin(s) + q * q - 1 : s;
            return 1 / (1 + Math.exp(-f));
        }

        /**
         * @param x a context
         * @param a an action
         * @return a sampled 0/1 reward
         */
        public double pull(double[] x, int a) {
            return random.nextDouble() < expected(x, a) ? 1 : 0;
        }

        /**
         * The logging policy: {@code softmax(inverseTemperature * w_a . x)}.
         *
         * @param x a context
         * @param inverseTemperature 0 for uniform logging, larger for a greedier logger
         * @return the action probabilities
         */
        public double[] loggingPolicy(double[] x, double inverseTemperature) {
            double[] p = new double[actions()];
            double max = Double.NEGATIVE_INFINITY;
            for (int a = 0; a < p.length; a++) {
                double s = 0;
                for (int d = 0; d < x.length; d++) {
                    s += logWeights[a][d] * x[d];
                }
                p[a] = inverseTemperature * s;
                max = Math.max(max, p[a]);
            }
            double z = 0;
            for (int a = 0; a < p.length; a++) {
                p[a] = Math.exp(p[a] - max);
                z += p[a];
            }
            for (int a = 0; a < p.length; a++) {
                p[a] /= z;
            }
            return p;
        }

        /**
         * Logs rounds of the logging policy.
         *
         * @param rounds how many
         * @param inverseTemperature see {@link #loggingPolicy}
         * @return the log
         */
        public BanditLog log(int rounds, double inverseTemperature) {
            int k = actions();
            double[][] xs = new double[rounds][];
            int[] acts = new int[rounds];
            double[] prop = new double[rounds];
            double[] rew = new double[rounds];
            double[][] exp = new double[rounds][k];
            double[][] logp = new double[rounds][];
            for (int i = 0; i < rounds; i++) {
                xs[i] = context();
                logp[i] = loggingPolicy(xs[i], inverseTemperature);
                double c = random.nextDouble();
                int a = 0;
                double acc = logp[i][0];
                while (c > acc && a < k - 1) {
                    acc += logp[i][++a];
                }
                acts[i] = a;
                prop[i] = logp[i][a];
                for (int b = 0; b < k; b++) {
                    exp[i][b] = expected(xs[i], b);
                }
                rew[i] = random.nextDouble() < exp[i][a] ? 1 : 0;
            }
            return new BanditLog(xs, acts, prop, rew, exp, logp);
        }
    }

    /**
     * A contextual bandit world.
     *
     * @param dim the context width
     * @param actions the number of actions
     * @param nonlinear whether rewards are non-linear in the context
     * @param logNoise how far the logging policy's scores are from the true ones
     * @param seed the seed
     * @return the world
     */
    public static BanditWorld banditWorld(int dim, int actions, boolean nonlinear, double logNoise, long seed) {
        return new BanditWorld(dim, actions, nonlinear, logNoise, seed);
    }

    private static float[][] gaussian(int n, int d, SplittableRandom r) {
        float[][] m = new float[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                m[i][j] = (float) r.nextGaussian();
            }
        }
        return m;
    }

    private static float dot(float[] a, float[] b) {
        float s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static Tensor observe(float[][] latent, float[][] mix, double noise, SplittableRandom r) {
        int n = latent.length;
        int f = mix[0].length;
        float[] out = new float[n * f];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < f; j++) {
                float s = 0;
                for (int k = 0; k < latent[i].length; k++) {
                    s += latent[i][k] * mix[k][j];
                }
                out[i * f + j] = (float) (s / Math.sqrt(latent[i].length) + noise * r.nextGaussian());
            }
        }
        return Tensor.of(Shape.of(n, f), out);
    }
}
