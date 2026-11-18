package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.train.LearningRateSchedule;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Continual pretraining of a pretrained model on a new domain: the learning-rate schedule, replay,
 * subset selection and the stability gap.
 * <ul>
 *   <li>{@link #rewarmRedecay}: Ibrahim et al. ("Simple and Scalable Strategies to Continually
 *       Pre-train Large Language Models", TMLR 2024, <a href="https://arxiv.org/abs/2403.08763">arXiv:2403.08763</a>):
 *       warm the rate back up linearly, then decay it again by cosine over the new run;</li>
 *   <li>{@link #replaySource}: mix a few percent of the old data into every batch (Ibrahim et
 *       al. find ~5% replay plus re-warming matches retraining on the union);</li>
 *   <li>{@link #lowPerplexitySubset}: Guo et al. ("Efficient Continual Pre-training by
 *       Mitigating the Stability Gap", 2024, <a href="https://arxiv.org/abs/2406.14833">arXiv:2406.14833</a>):
 *       train several epochs on a properly sized high-quality subset (here the documents a
 *       reference model finds least perplexing) instead of one pass over everything;</li>
 *   <li>{@link GapTracker}: the stability gap: performance first gets worse at the start of
 *       continual pretraining and then recovers; it records the depth and the length of that
 *       dip in an evaluation loss.</li>
 * </ul>
 */
public final class ContinualPretrainer {

    private ContinualPretrainer() {
    }

    /**
     * Re-warming and re-decaying: linear warm-up to the base rate over {@code warmup} steps,
     * then cosine decay to {@code minRate} at step {@code total}.
     *
     * @param warmup warm-up steps
     * @param total steps of the continual run
     * @param minRate the final rate
     * @return the schedule (steps counted from 1)
     */
    public static LearningRateSchedule rewarmRedecay(int warmup, int total, float minRate) {
        return LearningRateSchedule.warmup(warmup, LearningRateSchedule.cosine(Math.max(1, total - warmup), minRate),
                true);
    }

    /**
     * Picks the source of the next example: old data with probability {@code replay}.
     *
     * @param replay the replay ratio, for example 0.05
     * @param random the random source
     * @return true for old data, false for new
     */
    public static boolean replaySource(double replay, RandomGenerator random) {
        return random.nextDouble() < replay;
    }

    /**
     * Old-data examples per batch for a replay ratio, rounding stochastically.
     *
     * @param batch the batch size
     * @param replay the replay ratio
     * @param random the random source
     * @return the number of old examples
     */
    public static int replayCount(int batch, double replay, RandomGenerator random) {
        double expected = batch * replay;
        int n = (int) Math.floor(expected);
        return n + (random.nextDouble() < expected - n ? 1 : 0);
    }

    /**
     * The documents with the lowest perplexity under a reference model.
     *
     * @param losses each document's mean loss under the reference
     * @param fraction the share to keep
     * @return the kept document indices, best first
     */
    public static int[] lowPerplexitySubset(double[] losses, double fraction) {
        Integer[] order = new Integer[losses.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(losses[a], losses[b]));
        int[] keep = new int[Math.max(1, (int) Math.round(fraction * losses.length))];
        for (int i = 0; i < keep.length; i++) {
            keep[i] = order[i];
        }
        return keep;
    }

    /** Records an evaluation loss during continual pretraining and measures the stability gap. */
    public static final class GapTracker {
        private final List<double[]> points = new ArrayList<>();

        /** Creates an empty tracker. */
        public GapTracker() {
        }

        /**
         * Records a measurement.
         *
         * @param step the training step
         * @param loss the evaluation loss
         */
        public void record(long step, double loss) {
            points.add(new double[] {step, loss});
        }

        /** @return the loss at the first record */
        public double start() {
            return points.get(0)[1];
        }

        /** @return the largest rise above the starting loss (0 when it never rises) */
        public double depth() {
            double s = start();
            double worst = s;
            for (double[] p : points) {
                worst = Math.max(worst, p[1]);
            }
            return worst - s;
        }

        /**
         * @return steps from the start until the loss is back at or below its starting value
         *     after having risen; 0 without a dip, -1 if it never recovers
         */
        public long length() {
            double s = start();
            boolean rose = false;
            for (double[] p : points) {
                if (p[1] > s) {
                    rose = true;
                } else if (rose) {
                    return (long) (p[0] - points.get(0)[0]);
                }
            }
            return rose ? -1 : 0;
        }

        /** @return the last recorded loss */
        public double last() {
            return points.get(points.size() - 1)[1];
        }

        /** @return the recorded {@code {step, loss}} points */
        public List<double[]> points() {
            return List.copyOf(points);
        }
    }
}
