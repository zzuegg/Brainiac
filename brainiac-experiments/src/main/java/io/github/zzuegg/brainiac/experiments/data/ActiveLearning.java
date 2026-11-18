package io.github.zzuegg.brainiac.experiments.data;

import io.github.zzuegg.brainiac.experimental.data.ActiveLearner;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;

import java.util.SplittableRandom;

/**
 * Wave 8 item 8.6.13: pool-based active learning ({@link ActiveLearner}) on the toy digits of
 * {@link ToyTasks#digits}: a pool of 2000 unlabelled digits, 20 random labels to start, then
 * batches of 10 or 50 chosen by random sampling, predictive entropy, BALD, core-set, BADGE or
 * greedy leverage. The score is the area under the test-accuracy learning curve. A second pool
 * replaces 30% of its points with near-duplicates of a few pool points, the case where top-k
 * uncertainty sampling spends its batch on copies and the diversity-aware methods should not.
 */
@SuppressWarnings("exports")
public final class ActiveLearning {

    /**
     * One learning curve.
     *
     * @param curve test accuracy after each round
     * @param area the mean of the curve
     * @param millis wall time of the whole loop
     */
    public record Curve(double[] curve, double area, long millis) {

        /** @return the accuracy at the full budget */
        public double last() {
            return curve[curve.length - 1];
        }
    }

    private ActiveLearning() {
    }

    /**
     * Runs one acquisition.
     *
     * @param acquisition the acquisition function
     * @param duplicates replace 30% of the pool by near-duplicates
     * @param batch labels per round
     * @param budget the total number of labels
     * @param seed the seed
     * @return the learning curve
     */
    public static Curve run(ActiveLearner.Acquisition acquisition, boolean duplicates, int batch, int budget,
            long seed) {
        ToyTasks.Data pool = ToyTasks.digits(10, 200, 8, 0.08, seed, seed + 1);
        ToyTasks.Data test = ToyTasks.digits(10, 50, 8, 0.08, seed, seed + 2);
        float[][] x = pool.x().clone();
        int[] y = pool.y().clone();
        if (duplicates) {
            SplittableRandom r = new SplittableRandom(seed + 3);
            int sources = 10;
            int[] src = new int[sources];
            for (int s = 0; s < sources; s++) {
                src[s] = r.nextInt(x.length);
            }
            for (int i = 0; i < x.length; i++) {
                if (r.nextDouble() < 0.3) {
                    int s = src[r.nextInt(sources)];
                    float[] copy = x[s].clone();
                    for (int j = 0; j < copy.length; j++) {
                        copy[j] += (float) r.nextGaussian() * 0.02f;
                    }
                    x[i] = copy;
                    y[i] = pool.y()[s];
                }
            }
        }
        long start = System.nanoTime();
        ActiveLearner loop = ActiveLearner.of(() -> new ActiveLearner.MlpLearner(64, 32, 10, 60, seed + 4),
                acquisition, seed + 5);
        double[] curve = loop.run(x, y, test.x(), test.y(), 20, batch, budget);
        return new Curve(curve, ActiveLearner.area(curve), (System.nanoTime() - start) / 1_000_000);
    }
}
