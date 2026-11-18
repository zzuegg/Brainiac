package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Synthetic logs of ranking, ads and recommendation systems with hidden ground truth, next to
 * {@link SyntheticRecsys} (whose generators are unchanged):
 * <ul>
 *   <li>{@link #multiTask}: impressions with two labels (say click and like) whose logits share a
 *       planted field interaction with correlation {@code rho} (negative: the tasks conflict);</li>
 *   <li>{@link #conversions}: impressions, clicks and conversions where the conversion
 *       probability is known for <em>every</em> impression, including unclicked ones (entire-space
 *       conversion modelling);</li>
 *   <li>{@link #rankingLog}: queries with graded relevance, rankings from a noisy logging ranker
 *       and clicks from a position-based or cascade click model (unbiased learning to rank);</li>
 *   <li>{@link #interactionClicks}: clicks driven by a planted interaction of order 2, 3 or 4
 *       among many categorical fields (feature-interaction networks);</li>
 *   <li>{@link #zipfIds}: a huge Zipf-distributed ID field whose IDs carry signal (compact
 *       embeddings);</li>
 *   <li>{@link #interactions}: user-item interactions with Zipf item popularity (sampling bias of
 *       in-batch negatives);</li>
 *   <li>{@link #topics}: items in latent topics and users with multi-topic tastes (diversity);</li>
 *   <li>{@link #auctions}: a day of ad impressions with values and competing bids that follow a
 *       daily traffic cycle (budget pacing).</li>
 * </ul>
 */
public final class ClickLogs {

    private ClickLogs() {
    }

    // ------------------------------------------------------------------ multi-task

    /**
     * Impressions with several binary labels.
     *
     * @param features {@code [rows, fields * categories]} one-hot
     * @param labels {@code [rows, tasks]} 0/1
     */
    public record MultiTaskTable(Tensor features, Tensor labels) {

        /** @return the number of rows */
        public int rows() {
            return features.shape().dim(0);
        }

        /**
         * @param from first row
         * @param to one past the last
         * @return those rows
         */
        public MultiTaskTable rows(int from, int to) {
            return new MultiTaskTable(features.slice(0, from, to), labels.slice(0, from, to));
        }
    }

    /**
     * Two tasks over {@code fields >= 3} one-hot fields:
     * {@code logit1 = 2 A[c0][c1] + b[c2]} and
     * {@code logit2 = 2 (rho A[c0][c1] + sqrt(1 - rho^2) B[c1][c2]) + b'[c0]}, so the tasks share
     * the hidden interaction {@code A} with correlation {@code rho}.
     *
     * @param rows impressions
     * @param fields categorical fields, at least 3
     * @param categories categories per field
     * @param rho the correlation of the two tasks' interaction logits, in {@code [-1, 1]}
     * @param seed the seed (the same seed gives the same {@code A} and {@code B} for every rho)
     * @return the table
     */
    public static MultiTaskTable multiTask(int rows, int fields, int categories, double rho, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] a = gaussian(categories, categories, r);
        double[][] b = gaussian(categories, categories, r);
        double[] bias1 = scaled(gaussian(1, categories, r)[0], 0.3);
        double[] bias2 = scaled(gaussian(1, categories, r)[0], 0.3);
        int width = fields * categories;
        float[] x = new float[rows * width];
        float[] y = new float[rows * 2];
        double other = Math.sqrt(Math.max(0, 1 - rho * rho));
        for (int i = 0; i < rows; i++) {
            int[] c = new int[fields];
            for (int f = 0; f < fields; f++) {
                c[f] = r.nextInt(categories);
                x[i * width + f * categories + c[f]] = 1;
            }
            double l1 = 2 * a[c[0]][c[1]] + bias1[c[2]];
            double l2 = 2 * (rho * a[c[0]][c[1]] + other * b[c[1]][c[2]]) + bias2[c[0]];
            y[2 * i] = r.nextDouble() < sigmoid(l1) ? 1 : 0;
            y[2 * i + 1] = r.nextDouble() < sigmoid(l2) ? 1 : 0;
        }
        return new MultiTaskTable(Tensor.of(Shape.of(rows, width), x), Tensor.of(Shape.of(rows, 2), y));
    }

    // ------------------------------------------------------------------ conversions

    /**
     * Impressions with clicks and conversions.
     *
     * @param features {@code [rows, fields * categories]} one-hot
     * @param click 0/1 per impression
     * @param convert 0/1 per impression: whether the user would convert after clicking (observed
     *     only for clicked impressions; the click-and-convert label is {@code click * convert})
     * @param trueCvr the true conversion probability of every impression
     * @param trueCtr the true click probability of every impression
     */
    public record ConversionLog(Tensor features, float[] click, float[] convert, double[] trueCvr, double[] trueCtr) {

        /** @return the number of impressions */
        public int rows() {
            return click.length;
        }

        /**
         * @param from first row
         * @param to one past the last
         * @return those rows
         */
        public ConversionLog rows(int from, int to) {
            return new ConversionLog(features.slice(0, from, to), java.util.Arrays.copyOfRange(click, from, to),
                    java.util.Arrays.copyOfRange(convert, from, to), java.util.Arrays.copyOfRange(trueCvr, from, to),
                    java.util.Arrays.copyOfRange(trueCtr, from, to));
        }
    }

    /**
     * Clicks follow {@code logit = 2 A[c0][c1] - 1}. Conversions follow
     * {@code logit = 2 C[c0][c2] + D[c1] - 1.2 A[c0][c1] - 1}: impressions that are very
     * clickable convert worse (curiosity clicks), so the clicked sub-population is a biased
     * sample of the conversion function and a click-space CVR model learns the wrong ranking.
     *
     * @param rows impressions
     * @param fields categorical fields, at least 3
     * @param categories categories per field
     * @param seed the seed
     * @return the log
     */
    public static ConversionLog conversions(int rows, int fields, int categories, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] a = gaussian(categories, categories, r);
        double[][] c = gaussian(categories, categories, r);
        double[] d = gaussian(1, categories, r)[0];
        int width = fields * categories;
        float[] x = new float[rows * width];
        float[] click = new float[rows];
        float[] convert = new float[rows];
        double[] cvr = new double[rows];
        double[] ctr = new double[rows];
        for (int i = 0; i < rows; i++) {
            int[] cat = new int[fields];
            for (int f = 0; f < fields; f++) {
                cat[f] = r.nextInt(categories);
                x[i * width + f * categories + cat[f]] = 1;
            }
            ctr[i] = sigmoid(2 * a[cat[0]][cat[1]] - 1);
            cvr[i] = sigmoid(2 * c[cat[0]][cat[2]] + d[cat[1]] - 1.2 * a[cat[0]][cat[1]] - 1);
            click[i] = r.nextDouble() < ctr[i] ? 1 : 0;
            convert[i] = r.nextDouble() < cvr[i] ? 1 : 0;
        }
        return new ConversionLog(Tensor.of(Shape.of(rows, width), x), click, convert, cvr, ctr);
    }

    // ------------------------------------------------------------------ learning to rank

    /** The click model that generates a ranking log. */
    public enum ClickModel {
        /** Position-based: {@code P(click) = P(examine rank k) P(attractive)}. */
        POSITION_BASED,
        /** Cascade: scan top-down, click an attractive result and stop with probability 0.6. */
        CASCADE
    }

    /**
     * A click log over ranked lists.
     *
     * @param features {@code [queries][docs][d]} document features
     * @param relevance {@code [queries][docs]} graded relevance 0..4
     * @param shown {@code [sessions][k]} the documents shown, top first (as indices into the query's docs)
     * @param query the query of each session
     * @param clicks {@code [sessions][k]} 0/1 clicks per shown position
     * @param examination the true examination probability per rank (position-based model)
     */
    public record RankingLog(double[][][] features, int[][] relevance, int[][] shown, int[] query, int[][] clicks,
            double[] examination) {

        /** @return the number of sessions */
        public int sessions() {
            return query.length;
        }
    }

    /**
     * Queries whose documents have relevance from a hidden function
     * {@code s = w . x + 0.5 x0 x1}, cut into grades 0..4 by quantiles; a logging ranker sorts by
     * {@code s + noise}; sessions show the top {@code k} and record clicks with attractiveness
     * {@code 0.05 + 0.95 (2^g - 1) / 15}.
     *
     * @param queries queries
     * @param docs documents per query
     * @param d feature dimension, at least 2
     * @param k results shown per session
     * @param sessions sessions per query
     * @param eta the position-bias exponent: examination {@code (1 / rank)^eta}
     * @param model the click model
     * @param loggingNoise the logging ranker's score noise
     * @param seed the seed
     * @return the log
     */
    public static RankingLog rankingLog(int queries, int docs, int d, int k, int sessions, double eta,
            ClickModel model, double loggingNoise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] w = new double[d];
        for (int j = 0; j < d; j++) {
            w[j] = r.nextGaussian();
        }
        double[][][] x = new double[queries][docs][d];
        int[][] grade = new int[queries][docs];
        double[] exam = new double[k];
        for (int p = 0; p < k; p++) {
            exam[p] = Math.pow(1.0 / (p + 1), eta);
        }
        int total = queries * sessions;
        int[][] shown = new int[total][];
        int[] query = new int[total];
        int[][] clicks = new int[total][k];
        double[] all = new double[queries * docs];
        double[][] score = new double[queries][docs];
        for (int q = 0; q < queries; q++) {
            for (int i = 0; i < docs; i++) {
                double s = 0;
                for (int j = 0; j < d; j++) {
                    x[q][i][j] = r.nextGaussian();
                    s += w[j] * x[q][i][j];
                }
                s += 0.5 * x[q][i][0] * x[q][i][1];
                score[q][i] = s;
                all[q * docs + i] = s;
            }
        }
        double[] sorted = all.clone();
        java.util.Arrays.sort(sorted);
        double[] cut = {sorted[(int) (0.5 * sorted.length)], sorted[(int) (0.75 * sorted.length)],
            sorted[(int) (0.9 * sorted.length)], sorted[(int) (0.97 * sorted.length)]};
        for (int q = 0; q < queries; q++) {
            for (int i = 0; i < docs; i++) {
                int g = 0;
                while (g < 4 && score[q][i] >= cut[g]) {
                    g++;
                }
                grade[q][i] = g;
            }
        }
        int s = 0;
        for (int q = 0; q < queries; q++) {
            double[] logged = new double[docs];
            for (int i = 0; i < docs; i++) {
                logged[i] = score[q][i] + loggingNoise * r.nextGaussian();
            }
            final double[] lg = logged;
            int[] top = java.util.stream.IntStream.range(0, docs).boxed()
                    .sorted((a, b) -> Double.compare(lg[b], lg[a])).limit(k).mapToInt(Integer::intValue).toArray();
            for (int rep = 0; rep < sessions; rep++) {
                shown[s] = top;
                query[s] = q;
                for (int p = 0; p < k; p++) {
                    double attract = 0.05 + 0.95 * (Math.pow(2, grade[q][top[p]]) - 1) / 15;
                    if (model == ClickModel.POSITION_BASED) {
                        clicks[s][p] = r.nextDouble() < exam[p] * attract ? 1 : 0;
                    } else if (r.nextDouble() < attract) {
                        clicks[s][p] = 1;
                        if (r.nextDouble() < 0.6) {
                            break;
                        }
                    }
                }
                s++;
            }
        }
        return new RankingLog(x, grade, shown, query, clicks, exam);
    }

    // ------------------------------------------------------------------ interaction order

    /**
     * Clicks over {@code fields} one-hot fields driven by one planted interaction of the given
     * order among the first {@code order} fields (a Gaussian effect per combination of their
     * categories, scale 2) plus small main effects on every field.
     *
     * @param rows impressions
     * @param fields categorical fields, at least {@code order}
     * @param categories categories per field
     * @param order the interaction order, 2 to 4
     * @param seed the seed
     * @return one-hot features and clicks
     */
    public static SyntheticRecsys.Table interactionClicks(int rows, int fields, int categories, int order,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int combos = (int) Math.pow(categories, order);
        double[] effect = new double[combos];
        for (int c = 0; c < combos; c++) {
            effect[c] = 2 * r.nextGaussian();
        }
        double[][] main = gaussian(fields, categories, r);
        int width = fields * categories;
        float[] x = new float[rows * width];
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            int key = 0;
            double logit = 0;
            for (int f = 0; f < fields; f++) {
                int c = r.nextInt(categories);
                x[i * width + f * categories + c] = 1;
                if (f < order) {
                    key = key * categories + c;
                }
                logit += 0.15 * main[f][c];
            }
            logit += effect[key];
            y[i] = r.nextDouble() < sigmoid(logit) ? 1 : 0;
        }
        return new SyntheticRecsys.Table(Tensor.of(Shape.of(rows, width), x), Tensor.of(Shape.of(rows, 1), y));
    }

    // ------------------------------------------------------------------ huge ID field

    /**
     * Clicks over one huge ID field and one small context field.
     *
     * @param id the ID of each row (Zipf distributed: ID 0 is the most frequent)
     * @param context a small categorical context per row
     * @param label 0/1 clicks
     * @param ids the ID vocabulary size
     * @param contexts the context vocabulary size
     */
    public record IdTable(int[] id, int[] context, float[] label, int ids, int contexts) {

        /** @return the number of rows */
        public int rows() {
            return id.length;
        }

        /**
         * @param from first row
         * @param to one past the last
         * @return those rows
         */
        public IdTable rows(int from, int to) {
            return new IdTable(java.util.Arrays.copyOfRange(id, from, to),
                    java.util.Arrays.copyOfRange(context, from, to), java.util.Arrays.copyOfRange(label, from, to), ids,
                    contexts);
        }
    }

    /**
     * Each ID has a hidden 4-dimensional taste vector and each context one too; the click logit is
     * {@code 1.5 u_id . v_ctx}. IDs are drawn from a Zipf law with the given exponent.
     *
     * @param rows rows
     * @param ids ID vocabulary size, such as 100,000
     * @param contexts contexts, such as 8
     * @param exponent the Zipf exponent, such as 1.1
     * @param seed the seed
     * @return the table
     */
    public static IdTable zipfIds(int rows, int ids, int contexts, double exponent, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] cdf = zipfCdf(ids, exponent);
        double[][] u = gaussian(ids, 4, r);
        double[][] v = gaussian(contexts, 4, r);
        int[] id = new int[rows];
        int[] ctx = new int[rows];
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            id[i] = sample(cdf, r);
            ctx[i] = r.nextInt(contexts);
            double s = 0;
            for (int j = 0; j < 4; j++) {
                s += u[id[i]][j] * v[ctx[i]][j];
            }
            y[i] = r.nextDouble() < sigmoid(0.75 * s) ? 1 : 0;
        }
        return new IdTable(id, ctx, y, ids, contexts);
    }

    // ------------------------------------------------------------------ popularity

    /**
     * Logged user-item interactions.
     *
     * @param user the user of each interaction
     * @param item the item of each interaction
     * @param users users
     * @param items items
     * @param popularity each item's popularity weight (normalised)
     * @param affinity {@code [users][items]} the true {@code P(item | user)}
     */
    public record Interactions(int[] user, int[] item, int users, int items, double[] popularity,
            double[][] affinity) {

        /** @return the number of interactions */
        public int size() {
            return user.length;
        }
    }

    /**
     * Interactions with {@code P(i | u) proportional to pop_i exp(u . v_i)}, where {@code pop} is
     * Zipf with the given exponent over a random item order and {@code u, v} are hidden Gaussian
     * tastes of dimension {@code latent}.
     *
     * @param users users
     * @param items items
     * @param latent taste dimension
     * @param exponent the Zipf exponent of item popularity
     * @param perUser interactions per user
     * @param seed the seed
     * @return the interactions
     */
    public static Interactions interactions(int users, int items, int latent, double exponent, int perUser,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] u = gaussian(users, latent, r);
        double[][] v = gaussian(items, latent, r);
        int[] perm = permutation(items, r);
        double[] pop = new double[items];
        double z = 0;
        for (int i = 0; i < items; i++) {
            pop[perm[i]] = Math.pow(i + 1, -exponent);
            z += pop[perm[i]];
        }
        for (int i = 0; i < items; i++) {
            pop[i] /= z;
        }
        double[][] p = new double[users][items];
        int[] us = new int[users * perUser];
        int[] is = new int[users * perUser];
        int k = 0;
        for (int a = 0; a < users; a++) {
            double total = 0;
            for (int i = 0; i < items; i++) {
                double s = 0;
                for (int j = 0; j < latent; j++) {
                    s += u[a][j] * v[i][j];
                }
                p[a][i] = pop[i] * Math.exp(s / Math.sqrt(latent) * 2);
                total += p[a][i];
            }
            double[] cdf = new double[items];
            double acc = 0;
            for (int i = 0; i < items; i++) {
                p[a][i] /= total;
                acc += p[a][i];
                cdf[i] = acc;
            }
            for (int t = 0; t < perUser; t++) {
                us[k] = a;
                is[k++] = sample(cdf, r);
            }
        }
        return new Interactions(us, is, users, items, pop, p);
    }

    // ------------------------------------------------------------------ topics

    /**
     * Topic-structured preferences.
     *
     * @param topic each item's topic
     * @param quality each item's base quality
     * @param taste {@code [users][topics]} each user's interest per topic (0 to 1)
     * @param embedding {@code [items][dim]} item content embeddings (clustered by topic)
     * @param userFeatures {@code [users][dim]} user features (the taste-weighted topic centres plus noise)
     */
    public record Topics(int[] topic, double[] quality, double[][] taste, double[][] embedding,
            double[][] userFeatures) {

        /**
         * The true relevance of an item to a user: {@code taste[u][topic] * quality}.
         *
         * @param user the user
         * @param item the item
         * @return the relevance in {@code [0, 1]}
         */
        public double relevance(int user, int item) {
            return taste[user][topic[item]] * quality[item];
        }
    }

    /**
     * Items spread over latent topics; each user likes two or three topics (one strongly), so a
     * relevance-sorted list tends to fill with the strongest topic.
     *
     * @param users users
     * @param items items
     * @param topics topics
     * @param dim embedding dimension
     * @param seed the seed
     * @return the data
     */
    public static Topics topics(int users, int items, int topics, int dim, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] centre = gaussian(topics, dim, r);
        int[] topic = new int[items];
        double[] quality = new double[items];
        double[][] emb = new double[items][dim];
        for (int i = 0; i < items; i++) {
            topic[i] = r.nextInt(topics);
            quality[i] = 0.5 + 0.5 * r.nextDouble();
            for (int j = 0; j < dim; j++) {
                emb[i][j] = centre[topic[i]][j] + 0.35 * r.nextGaussian();
            }
        }
        double[][] taste = new double[users][topics];
        double[][] uf = new double[users][dim];
        for (int u = 0; u < users; u++) {
            int liked = 2 + r.nextInt(2);
            int[] perm = permutation(topics, r);
            for (int k = 0; k < liked; k++) {
                taste[u][perm[k]] = k == 0 ? 1 : 0.55 + 0.25 * r.nextDouble();
            }
            for (int j = 0; j < dim; j++) {
                double s = 0;
                for (int t = 0; t < topics; t++) {
                    s += taste[u][t] * centre[t][j];
                }
                uf[u][j] = s + 0.3 * r.nextGaussian();
            }
        }
        return new Topics(topic, quality, taste, emb, uf);
    }

    // ------------------------------------------------------------------ auctions

    /**
     * A day of second-price auctions for one advertiser.
     *
     * @param value the advertiser's value of each impression ({@code pCTR * value per click})
     * @param competing the highest competing bid of each impression
     * @param hour the hour of day (0 to 23) of each impression
     */
    public record AuctionDay(double[] value, double[] competing, int[] hour) {

        /** @return the number of auctions */
        public int size() {
            return value.length;
        }
    }

    /**
     * Auctions spread over 24 hours: competition is fierce in the working day (lognormal competing
     * bids with a high median) and cheap in the evening, while traffic and values follow their own
     * cycle. Values are {@code pCTR * 1.0}, with {@code pCTR} the true click probability of an
     * {@link #interactionClicks}-style planted model.
     *
     * @param auctions auctions in the day
     * @param seed the seed
     * @return the day
     */
    public static AuctionDay auctions(int auctions, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] value = new double[auctions];
        double[] comp = new double[auctions];
        int[] hour = new int[auctions];
        double[][] effect = gaussian(8, 8, r);
        for (int i = 0; i < auctions; i++) {
            double pos = (i + r.nextDouble()) / auctions;
            int h = Math.min(23, (int) (pos * 24));
            hour[i] = h;
            double pctr = sigmoid(effect[r.nextInt(8)][r.nextInt(8)] - 1.5);
            value[i] = pctr;
            double busy = 0.5 + 0.5 * Math.cos(2 * Math.PI * (h - 13) / 24);
            double median = 0.05 + 0.3 * busy;
            comp[i] = median * Math.exp(0.5 * r.nextGaussian());
        }
        return new AuctionDay(value, comp, hour);
    }

    // ------------------------------------------------------------------ helpers

    private static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    private static double[][] gaussian(int n, int d, SplittableRandom r) {
        double[][] m = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                m[i][j] = r.nextGaussian();
            }
        }
        return m;
    }

    private static double[] scaled(double[] v, double s) {
        double[] out = v.clone();
        for (int i = 0; i < out.length; i++) {
            out[i] *= s;
        }
        return out;
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    private static double[] zipfCdf(int n, double exponent) {
        double[] cdf = new double[n];
        double acc = 0;
        for (int i = 0; i < n; i++) {
            acc += Math.pow(i + 1, -exponent);
            cdf[i] = acc;
        }
        for (int i = 0; i < n; i++) {
            cdf[i] /= acc;
        }
        return cdf;
    }

    private static int sample(double[] cdf, SplittableRandom r) {
        int k = java.util.Arrays.binarySearch(cdf, r.nextDouble());
        k = k < 0 ? -k - 1 : k;
        return Math.min(cdf.length - 1, k);
    }
}
