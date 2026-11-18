package io.github.zzuegg.brainiac.experiments.privacy;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.PoisonedSets;
import io.github.zzuegg.brainiac.experimental.data.SyntheticRecsys;
import io.github.zzuegg.brainiac.experimental.data.TabularStressSuite;
import io.github.zzuegg.brainiac.experimental.dpmech.CorrelatedNoise;
import io.github.zzuegg.brainiac.experimental.dpmech.LocalDp;
import io.github.zzuegg.brainiac.experimental.dpmech.Mwem;
import io.github.zzuegg.brainiac.experimental.dpmech.PrivBayes;
import io.github.zzuegg.brainiac.experimental.dpmech.PrivateSelection;
import io.github.zzuegg.brainiac.experimental.dpmech.SparseVector;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.synth.MixedTable;
import io.github.zzuegg.brainiac.experimental.synth.SyntheticDataAudit;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 11 items 11.1.12 to 11.1.15: differential-privacy mechanisms beyond DP-SGD.
 *
 * <ul>
 *   <li>{@link #localDp}: frequency estimation from {@link PoisonedSets#zipf} or uniform value
 *       streams under local DP ({@link LocalDp} GRR, RAPPOR, OUE, OLH, raw and projected onto
 *       the simplex) against central Laplace histograms and the non-private histogram.</li>
 *   <li>{@link #selection} and {@link #hyperparameters}: {@link PrivateSelection} of a
 *       histogram mode and of the best of 16 {@link ToyTasks#digits} configurations by
 *       validation accuracy, against the non-private argmax and a uniform choice.</li>
 *   <li>{@link #sparseVector}: {@link SparseVector}'s measured privacy loss on a
 *       neighbouring pair of query vectors, correct against the known-broken variants.</li>
 *   <li>{@link #release}: {@link Mwem} and {@link PrivBayes} synthetic tables against
 *       independent Laplace marginals: 2-way workload error and train-on-synthetic,
 *       test-on-real accuracy ({@link SyntheticDataAudit}), with the real table as the upper
 *       bound. The table is {@link TabularStressSuite} (6 columns) cut into 4 bins, plus its
 *       binary label.</li>
 *   <li>{@link #prefixSums}: {@link CorrelatedNoise} prefix-sum error and noise state.</li>
 * </ul>
 */
public final class MechanismShootout {

    /**
     * A frequency-estimation row.
     *
     * @param mechanism the mechanism
     * @param mse the mean squared error of the frequency vector
     * @param projectedMse after projection onto the simplex
     * @param topRecall top-10 recall of the projected estimate
     * @param bits bits sent per user (0 for central mechanisms)
     */
    public record Frequency(String mechanism, double mse, double projectedMse, double topRecall, double bits) {
    }

    /**
     * A selection row.
     *
     * @param mechanism the mechanism
     * @param gap the expected utility gap to the best candidate
     */
    public record Selection(String mechanism, double gap) {
    }

    /**
     * A query-release row.
     *
     * @param mechanism the mechanism
     * @param workloadError the mean error over all 2-way marginals
     * @param tstr train-on-synthetic, test-on-real accuracy ({@code NaN} for marginals only)
     */
    public record Release(String mechanism, double workloadError, double tstr) {
    }

    /**
     * A prefix-sum row.
     *
     * @param mechanism the noise mechanism
     * @param rmse the root-mean-square prefix-sum error at unit noise multiplier
     * @param state noise values kept per model coordinate
     */
    public record Prefix(String mechanism, double rmse, int state) {
    }

    private MechanismShootout() {
    }

    /**
     * Local DP frequency estimation.
     *
     * @param users users
     * @param domain items
     * @param zipf Zipf ({@code s = 1.1}) rather than uniform frequencies
     * @param epsilon the budget
     * @param seed the seed
     * @return GRR, RAPPOR, OUE, OLH, CENTRAL_LAPLACE, NON_PRIVATE
     */
    public static List<Frequency> localDp(int users, int domain, boolean zipf, double epsilon, long seed) {
        PoisonedSets.ValueStream s = zipf ? PoisonedSets.zipf(users, domain, 1.1, seed)
                : PoisonedSets.uniform(users, domain, seed);
        double[] truth = histogram(s.values(), domain);
        List<Frequency> out = new ArrayList<>();
        for (LocalDp.Protocol p : LocalDp.Protocol.values()) {
            double[] est = LocalDp.estimate(p, s.values(), domain, epsilon, seed + 1 + p.ordinal());
            double[] proj = LocalDp.projectSimplex(est);
            out.add(new Frequency(p.name(), LocalDp.mse(est, truth), LocalDp.mse(proj, truth),
                    LocalDp.topRecall(proj, truth, 10), LocalDp.bits(p, domain, epsilon)));
        }
        double[] central = LocalDp.centralLaplace(s.values(), domain, epsilon, seed + 9);
        double[] cp = LocalDp.projectSimplex(central);
        out.add(new Frequency("CENTRAL_LAPLACE", LocalDp.mse(central, truth), LocalDp.mse(cp, truth),
                LocalDp.topRecall(cp, truth, 10), 0));
        out.add(new Frequency("NON_PRIVATE", 0, 0, 1, 0));
        return out;
    }

    static double[] histogram(int[] values, int k) {
        double[] h = new double[k];
        for (int v : values) {
            h[v] += 1.0 / values.length;
        }
        return h;
    }

    /**
     * Private choice of the mode of a Zipf histogram (counts, sensitivity 1).
     *
     * @param users users
     * @param candidates items
     * @param epsilon the budget
     * @param seed the seed
     * @return the four mechanisms, UNIFORM and NON_PRIVATE (gap 0)
     */
    public static List<Selection> selection(int users, int candidates, double epsilon, long seed) {
        PoisonedSets.ValueStream s = PoisonedSets.zipf(users, candidates, 1.1, seed);
        double[] counts = new double[candidates];
        for (int v : s.values()) {
            counts[v]++;
        }
        return gaps(counts, epsilon, 1, seed);
    }

    static List<Selection> gaps(double[] utility, double epsilon, double sensitivity, long seed) {
        List<Selection> out = new ArrayList<>();
        for (PrivateSelection.Mechanism m : PrivateSelection.Mechanism.values()) {
            out.add(new Selection(m.name(), PrivateSelection.expectedGap(m, utility, epsilon, sensitivity, 4000,
                    seed + m.ordinal())));
        }
        double max = Double.NEGATIVE_INFINITY;
        double mean = 0;
        for (double u : utility) {
            max = Math.max(max, u);
            mean += u / utility.length;
        }
        out.add(new Selection("UNIFORM", max - mean));
        out.add(new Selection("NON_PRIVATE", 0));
        return out;
    }

    /**
     * Private hyperparameter selection: 16 MLP configurations (hidden width × learning rate)
     * on digits scored by validation accuracy counts (sensitivity 1); the gap is in test
     * accuracy.
     *
     * @param epsilon the selection budget
     * @param seed the seed
     * @return the four mechanisms, UNIFORM and NON_PRIVATE (gaps in test accuracy to the
     *     configuration with the best test accuracy)
     */
    public static List<Selection> hyperparameters(double epsilon, long seed) {
        ToyTasks.Data train = ToyTasks.digits(10, 40, 8, 0.15, seed, seed + 1);
        ToyTasks.Data val = ToyTasks.digits(10, 30, 8, 0.15, seed, seed + 2);
        ToyTasks.Data test = ToyTasks.digits(10, 30, 8, 0.15, seed, seed + 3);
        int[] widths = {2, 4, 16, 64};
        float[] rates = {1e-4f, 1e-3f, 3e-3f, 3e-2f};
        double[] valCounts = new double[16];
        double[] testAcc = new double[16];
        int c = 0;
        for (int w : widths) {
            for (float lr : rates) {
                SimpleMlp m = SimpleMlp.relu(seed + c, 64, w, 10);
                Optimizer opt = Optimizer.adam(lr);
                Tensor x = train.inputs();
                Tensor ids = train.ids();
                for (int s = 0; s < 150; s++) {
                    Functional.crossEntropy(m.forward(Var.of(x)), ids).backward();
                    opt.step(m.parameters());
                }
                valCounts[c] = ToyTasks.accuracy(m.predict(val.inputs()), val.y()) * val.y().length;
                testAcc[c] = ToyTasks.accuracy(m.predict(test.inputs()), test.y());
                c++;
            }
        }
        double bestTest = 0;
        int argVal = 0;
        for (int i = 0; i < 16; i++) {
            bestTest = Math.max(bestTest, testAcc[i]);
            argVal = valCounts[i] > valCounts[argVal] ? i : argVal;
        }
        List<Selection> out = new ArrayList<>();
        for (PrivateSelection.Mechanism m : PrivateSelection.Mechanism.values()) {
            SplittableRandom r = new SplittableRandom(seed + 50 + m.ordinal());
            double acc = 0;
            int trials = 2000;
            for (int t = 0; t < trials; t++) {
                acc += testAcc[PrivateSelection.select(m, valCounts, epsilon, 1, r)] / trials;
            }
            out.add(new Selection(m.name(), bestTest - acc));
        }
        double mean = 0;
        for (double a : testAcc) {
            mean += a / 16;
        }
        out.add(new Selection("UNIFORM", bestTest - mean));
        out.add(new Selection("NON_PRIVATE", bestTest - testAcc[argVal]));
        return out;
    }

    /**
     * Measured privacy loss of AboveThreshold variants on the pair {@code q = (0^m, 1^m)},
     * {@code q' = (1^m, 0^m)} ({@code T = 0}, {@code Δ = 1}, {@code c = 1}).
     *
     * @param epsilon the claimed budget
     * @param m half the queries
     * @param trials runs per dataset
     * @param seed the seed
     * @return per variant, the largest measured {@code |log P(o|q)/P(o|q')|} over patterns
     *     seen at least 30 times
     */
    public static double[] sparseVector(double epsilon, int m, int trials, long seed) {
        double[] q = new double[2 * m];
        double[] qp = new double[2 * m];
        for (int i = 0; i < m; i++) {
            q[m + i] = 1;
            qp[i] = 1;
        }
        double[] out = new double[SparseVector.Variant.values().length];
        for (SparseVector.Variant v : SparseVector.Variant.values()) {
            out[v.ordinal()] = SparseVector.empiricalEpsilon(v, q, qp, 0, epsilon, 1, 1, trials, 30,
                    seed + v.ordinal());
        }
        return out;
    }

    /** The discretised table: 6 feature columns of 4 bins, then the binary label. */
    static int[][] table(int rows, long seed) {
        TabularStressSuite suite = new TabularStressSuite(4, 0.33, false, 7);
        SyntheticRecsys.Table t = suite.table(rows, seed);
        int cols = suite.columns();
        float[] x = t.features().toArray();
        float[] y = t.labels().toArray();
        int[][] out = new int[rows][cols + 1];
        for (int i = 0; i < rows; i++) {
            for (int c = 0; c < cols; c++) {
                out[i][c] = Math.clamp((int) Math.floor((x[i * cols + c] + 2) / 4 * 4), 0, 3);
            }
            out[i][cols] = (int) y[i];
        }
        return out;
    }

    static int[] cards(int columns) {
        int[] cards = new int[columns];
        java.util.Arrays.fill(cards, 4);
        cards[columns - 1] = 2;
        return cards;
    }

    static int[][] allPairs(int columns) {
        List<int[]> w = new ArrayList<>();
        for (int a = 0; a < columns; a++) {
            for (int b = a + 1; b < columns; b++) {
                w.add(new int[] {a, b});
            }
        }
        return w.toArray(new int[0][]);
    }

    /**
     * Private query release.
     *
     * @param epsilon the budget
     * @param rows real rows
     * @param seed the seed
     * @return INDEPENDENT_LAPLACE, MWEM, PRIVBAYES and REAL (non-private upper bound)
     */
    public static List<Release> release(double epsilon, int rows, long seed) {
        int[][] train = table(rows, seed);
        int[][] test = table(1000, seed + 1);
        int cols = train[0].length;
        int[] cards = cards(cols);
        int[][] workload = allPairs(cols);
        List<Release> out = new ArrayList<>();
        double[][] indep = Mwem.noisyMarginals(train, cards, workload, epsilon, seed + 2);
        out.add(new Release("INDEPENDENT_LAPLACE", Mwem.workloadError(train, cards, workload, indep), Double.NaN));
        double[] dist = Mwem.run(train, cards, workload, epsilon, 15, seed + 3);
        int[][] mw = Mwem.sample(dist, cards, rows, seed + 4);
        out.add(new Release("MWEM", Mwem.workloadError(train, cards, workload,
                Mwem.marginalsOf(mw, cards, workload, rows)), tstr(mw, test, cards)));
        int[][] pb = PrivBayes.fit(train, cards, 2, epsilon, 0.3, seed + 5).sample(rows, seed + 6);
        out.add(new Release("PRIVBAYES", Mwem.workloadError(train, cards, workload,
                Mwem.marginalsOf(pb, cards, workload, rows)), tstr(pb, test, cards)));
        out.add(new Release("REAL", 0, tstr(train, test, cards)));
        return out;
    }

    static double tstr(int[][] synthetic, int[][] test, int[] cards) {
        MixedTable tr = new MixedTable(new double[synthetic.length][0], synthetic, cards);
        MixedTable te = new MixedTable(new double[test.length][0], test, cards);
        return SyntheticDataAudit.utility(tr, te, cards.length - 1, tr.numericStats(), 40).accuracy();
    }

    /**
     * Prefix-sum error of correlated-noise mechanisms.
     *
     * @param steps {@code n}
     * @param participations {@code k}
     * @param separation steps between participations
     * @return INDEPENDENT, TREE, BANDED_MF and BLT
     */
    public static List<Prefix> prefixSums(int steps, int participations, int separation) {
        List<Prefix> out = new ArrayList<>();
        for (CorrelatedNoise m : mechanisms(steps, participations, separation)) {
            out.add(new Prefix(m.kind().name(), m.prefixRmse(participations, separation), m.state()));
        }
        return out;
    }

    /**
     * The four mechanisms, fitted for a run.
     *
     * @param steps {@code n}
     * @param participations {@code k}
     * @param separation steps between participations
     * @return independent, tree, banded (at most {@code separation} bands, up to 16) and BLT
     *     with three buffers
     */
    public static List<CorrelatedNoise> mechanisms(int steps, int participations, int separation) {
        return List.of(CorrelatedNoise.independent(steps), CorrelatedNoise.tree(steps),
                CorrelatedNoise.banded(steps, Math.max(2, Math.min(16, separation)), participations, separation, 300),
                CorrelatedNoise.blt(steps, 3, participations, separation, 400));
    }
}
