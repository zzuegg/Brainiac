package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Sampling-bias-corrected retrieval (Yi et al., RecSys 2019, "Sampling-Bias-Corrected Neural
 * Modeling for Large Corpus Item Recommendations"; Yang et al., WWW 2020, "Mixed Negative
 * Sampling"): a two-tower ID model trained with a softmax over in-batch negatives. In-batch
 * negatives are drawn in proportion to item popularity, so the plain in-batch softmax
 * over-penalises popular items. The fix is the <em>logQ correction</em>: subtract
 * {@code log q_j}, the sampling probability of each candidate, from its logit. {@code q_j} is
 * not known in a stream, so it is estimated by {@link FrequencyEstimator}, which tracks the
 * average number of steps between two appearances of an item (hashed into buckets) and reports
 * {@code 1 / gap}.
 * <ul>
 *   <li>{@link Mode#IN_BATCH}: plain in-batch softmax (the biased baseline);</li>
 *   <li>{@link Mode#LOG_Q}: in-batch softmax with streaming logQ correction;</li>
 *   <li>{@link Mode#MIXED}: in-batch plus uniformly sampled negatives, corrected with the
 *       mixture sampling probability;</li>
 *   <li>{@link Mode#FULL}: softmax over the whole catalogue (the unbiased oracle, feasible only
 *       for small catalogues).</li>
 * </ul>
 * Embeddings are L2-normalised and scores divided by a temperature. Accidental hits (a candidate
 * equal to a row's own positive in another column) are masked.
 */
public final class CorrectedInBatchSoftmax {

    /** The negative-sampling scheme. */
    public enum Mode {
        /** In-batch negatives, no correction. */
        IN_BATCH,
        /** In-batch negatives with logQ correction. */
        LOG_Q,
        /** In-batch plus uniform negatives with mixture logQ correction. */
        MIXED,
        /** Full softmax over every item. */
        FULL
    }

    /**
     * The streaming frequency estimator of Yi et al.: for each hash bucket, the step at which it
     * was last seen ({@code A}) and an exponential moving average of the gap between sightings
     * ({@code B}); the per-step sampling probability is {@code 1 / B}.
     */
    public static final class FrequencyEstimator {
        private final long[] last;
        private final double[] gap;
        private final int buckets;
        private final long seed;
        private final double alpha;

        /**
         * @param buckets hash buckets (collisions merge items)
         * @param alpha the moving-average rate, such as 0.05
         * @param seed the hash seed
         */
        public FrequencyEstimator(int buckets, double alpha, long seed) {
            this.buckets = buckets;
            this.alpha = alpha;
            this.seed = seed;
            this.last = new long[buckets];
            this.gap = new double[buckets];
            java.util.Arrays.fill(last, -1);
        }

        /**
         * Records that the item appeared at a step.
         *
         * @param item the item
         * @param step the global step (batch counter)
         */
        public void observe(int item, long step) {
            int h = CompactEmbedding.hash(item, seed, buckets);
            if (last[h] >= 0) {
                long d = step - last[h];
                if (d > 0) {
                    gap[h] = gap[h] == 0 ? d : (1 - alpha) * gap[h] + alpha * d;
                }
            }
            last[h] = step;
        }

        /**
         * @param item an item
         * @param floor the probability to report for an item seen at most once
         * @return the estimated probability that the item appears in a batch
         */
        public double probability(int item, double floor) {
            double g = gap[CompactEmbedding.hash(item, seed, buckets)];
            return g == 0 ? floor : Math.min(1, 1 / g);
        }
    }

    private final Var users;
    private final Var items;
    private final int itemCount;
    private final Mode mode;
    private final int uniform;
    private final FrequencyEstimator estimator;
    private final SplittableRandom random;
    private Optimizer optimizer = Optimizer.adam(0.01f);
    private float temperature = 0.1f;
    private long step;

    /**
     * Creates the model.
     *
     * @param userCount users
     * @param itemCount items
     * @param dim embedding width
     * @param mode the sampling scheme
     * @param uniformNegatives extra uniform negatives per batch ({@link Mode#MIXED} only)
     * @param seed the seed
     */
    public CorrectedInBatchSoftmax(int userCount, int itemCount, int dim, Mode mode, int uniformNegatives,
            long seed) {
        this.random = new SplittableRandom(seed);
        this.users = Var.param(Tensor.normal(Shape.of(userCount, dim), 0, 0.1f, random));
        this.items = Var.param(Tensor.normal(Shape.of(itemCount, dim), 0, 0.1f, random));
        this.itemCount = itemCount;
        this.mode = mode;
        this.uniform = mode == Mode.MIXED ? uniformNegatives : 0;
        this.estimator = new FrequencyEstimator(Math.max(16, itemCount * 2), 0.05, seed ^ 0x9E37L);
    }

    /** @param t the softmax temperature @return this model */
    public CorrectedInBatchSoftmax temperature(float t) {
        this.temperature = t;
        return this;
    }

    /** @param o the optimizer @return this model */
    public CorrectedInBatchSoftmax optimizer(Optimizer o) {
        this.optimizer = o;
        return this;
    }

    /** @return the streaming frequency estimator */
    public FrequencyEstimator estimator() {
        return estimator;
    }

    /** @return the number of training steps taken */
    public long steps() {
        return step;
    }

    /**
     * One training step on a batch of positive pairs.
     *
     * @param user user of each pair
     * @param item positive item of each pair
     * @return the loss
     */
    public double trainStep(int[] user, int[] item) {
        int b = user.length;
        for (int i : item) {
            estimator.observe(i, step);
        }
        step++;
        int[] candidates;
        if (mode == Mode.FULL) {
            candidates = new int[itemCount];
            for (int j = 0; j < itemCount; j++) {
                candidates[j] = j;
            }
        } else {
            candidates = new int[b + uniform];
            System.arraycopy(item, 0, candidates, 0, b);
            for (int j = 0; j < uniform; j++) {
                candidates[b + j] = random.nextInt(itemCount);
            }
        }
        int c = candidates.length;
        Var u = RecsysOps.normalize(users.gather(0, RecsysOps.ids(user)));
        Var v = RecsysOps.normalize(items.gather(0, RecsysOps.ids(candidates)));
        Var logits = u.matmul(v.transpose()).div(temperature);
        float[] offset = new float[b * c];
        float[] target = new float[b * c];
        double floor = 1.0 / Math.max(1, step);
        for (int j = 0; j < c; j++) {
            float corr = 0;
            if (mode == Mode.LOG_Q) {
                corr = (float) Math.log(estimator.probability(candidates[j], floor));
            } else if (mode == Mode.MIXED) {
                double q = estimator.probability(candidates[j], floor) + uniform / (double) itemCount;
                corr = (float) Math.log(q);
            }
            for (int i = 0; i < b; i++) {
                int col = mode == Mode.FULL ? item[i] : i;
                boolean hit = candidates[j] == item[i];
                offset[i * c + j] = -corr + (hit && j != col ? -1e4f : 0);
                if (j == col) {
                    target[i * c + j] = 1;
                }
            }
        }
        Var z = logits.add(Tensor.of(Shape.of(b, c), offset));
        Var logp = z.sub(z.logsumexp(1, true));
        Var loss = logp.mul(Tensor.of(Shape.of(b, c), target)).sum().neg().div(b);
        loss.backward();
        optimizer.step(List.of(users, items));
        return loss.value().item();
    }

    /**
     * @param user a user
     * @return the cosine score of every item
     */
    public float[] scores(int user) {
        Tensor u = RecsysOps.normalize(Var.of(users.value().slice(0, user, user + 1))).value();
        Tensor v = RecsysOps.normalize(Var.of(items.value())).value();
        return u.matmul(v.transpose()).toArray();
    }

    @Override
    public String toString() {
        return "CorrectedInBatchSoftmax(" + mode + ", " + itemCount + " items)";
    }
}
