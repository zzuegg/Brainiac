package io.github.zzuegg.brainiac.experiments.evaluation;

import io.github.zzuegg.brainiac.experimental.eval.AdaptiveAnalyst;
import io.github.zzuegg.brainiac.experimental.eval.EvalStatistics;
import io.github.zzuegg.brainiac.experimental.eval.ReusableHoldout;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Error bars, paired tests and power (9.3.14) and adaptive overfitting of a reused holdout
 * (9.3.15), on simulated evaluations whose truth is known.
 * <ul>
 *   <li>{@link #errorBars}: a benchmark of question clusters (several questions share a
 *       context): a model answers question {@code q} of cluster {@code c} correctly with
 *       probability {@code σ(a + u_c + e_q)}, cluster effects {@code u_c ~ N(0, τ²)}, question
 *       effects {@code e_q ~ N(0, 0.5²)}; every replication draws a fresh question set. It
 *       measures the coverage of i.i.d. and clustered 95% intervals around the population
 *       accuracy, false "wins" between two identical models under each test, and the detection
 *       rate of a 2-point difference with paired and unpaired tests, plus the questions a
 *       power analysis asks for;</li>
 *   <li>{@link #boosting}: the boosting attack on random labels (true accuracy 1/2) against the
 *       plain holdout, the Ladder, the parameter-free Ladder and Thresholdout;</li>
 *   <li>{@link #honestSearch}: a greedy search over 200 nearest-centroid classifiers on random
 *       feature subsets of a weak-signal task, reported against fresh-data accuracy.</li>
 * </ul>
 * The evaluations are simulated rather than re-trained: a fresh question sample per
 * replication is exactly the variation the standard errors describe.
 */
public final class StatisticalReporting {

    private StatisticalReporting() {
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /** A simulated benchmark population. */
    private record Population(double tau, double questionSd) {
        double accuracy(double ability) {
            SplittableRandom r = new SplittableRandom(99);
            double s = 0;
            int n = 200_000;
            for (int i = 0; i < n; i++) {
                s += sigmoid(ability + tau * r.nextGaussian() + questionSd * r.nextGaussian());
            }
            return s / n;
        }
    }

    /**
     * Outcomes of {@link #errorBars}.
     *
     * @param iidCoverage coverage of the i.i.d. 95% interval of one model's accuracy
     * @param clusteredCoverage coverage of the clustered 95% interval
     * @param bootstrapCoverage coverage of the cluster bootstrap 95% interval
     * @param seRatio mean clustered SE over mean i.i.d. SE
     * @param falseWinsUnpaired false wins per 100 identical-model comparisons, unpaired i.i.d. z-test
     * @param falseWinsPaired same, paired i.i.d. z-test
     * @param falseWinsPairedClustered same, paired clustered z-test
     * @param falseWinsMcNemar same, McNemar
     * @param pointDifferences identical-model comparisons per 100 whose accuracies differ by at
     *     least one point (what a point-estimate table would read as a win)
     * @param powerPaired detection rate of the 2-point difference, paired clustered test
     * @param powerUnpaired detection rate, unpaired clustered test
     * @param questionsPaired questions a power analysis asks for (80% power), paired
     * @param questionsUnpaired same, unpaired
     */
    public record ErrorBars(double iidCoverage, double clusteredCoverage, double bootstrapCoverage, double seRatio,
            double falseWinsUnpaired, double falseWinsPaired, double falseWinsPairedClustered,
            double falseWinsMcNemar, double pointDifferences, double powerPaired, double powerUnpaired,
            long questionsPaired, long questionsUnpaired) {
    }

    /**
     * Coverage, false wins and power over replications.
     *
     * @param clusters clusters per question set
     * @param perCluster questions per cluster
     * @param tau the cluster effect's standard deviation
     * @param replications fresh question sets
     * @param seed the seed
     * @return the outcomes
     */
    public static ErrorBars errorBars(int clusters, int perCluster, double tau, int replications, long seed) {
        Population pop = new Population(tau, 0.5);
        double ability = 0.8;
        double truth = pop.accuracy(ability);
        double target = truth + 0.02;
        double lo = ability;
        double hi = ability + 1;
        for (int i = 0; i < 30; i++) {
            double mid = 0.5 * (lo + hi);
            if (pop.accuracy(mid) < target) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        double better = 0.5 * (lo + hi);
        SplittableRandom r = new SplittableRandom(seed);
        int n = clusters * perCluster;
        int[] cluster = new int[n];
        for (int i = 0; i < n; i++) {
            cluster[i] = i / perCluster;
        }
        int iid = 0;
        int cl = 0;
        int boot = 0;
        double seIid = 0;
        double seCl = 0;
        int[] falseWins = new int[4];
        int pointDiff = 0;
        int detPaired = 0;
        int detUnpaired = 0;
        double varPaired = 0;
        double varUnpaired = 0;
        for (int k = 0; k < replications; k++) {
            double[] logit = new double[n];
            for (int c = 0; c < clusters; c++) {
                double u = tau * r.nextGaussian();
                for (int q = 0; q < perCluster; q++) {
                    logit[c * perCluster + q] = u + 0.5 * r.nextGaussian();
                }
            }
            double[] a = answers(logit, ability, r);
            double[] b = answers(logit, ability, r);
            double[] c = answers(logit, better, r);
            EvalStatistics.Interval i1 = EvalStatistics.iid(a);
            EvalStatistics.Interval i2 = EvalStatistics.clustered(a, cluster);
            iid += i1.covers(truth, 1.96) ? 1 : 0;
            cl += i2.covers(truth, 1.96) ? 1 : 0;
            seIid += i1.se() / replications;
            seCl += i2.se() / replications;
            if (k < Math.min(replications, 100)) {
                double[] bi = EvalStatistics.clusterBootstrap(a, cluster, 300, 0.95, seed + k);
                boot += bi[0] <= truth && truth <= bi[1] ? 1 : 0;
            }
            falseWins[0] += EvalStatistics.unpaired(a, b, null).pValue() < 0.05 ? 1 : 0;
            falseWins[1] += EvalStatistics.paired(a, b, null).pValue() < 0.05 ? 1 : 0;
            falseWins[2] += EvalStatistics.paired(a, b, cluster).pValue() < 0.05 ? 1 : 0;
            falseWins[3] += EvalStatistics.mcNemar(a, b) < 0.05 ? 1 : 0;
            pointDiff += Math.abs(i1.mean() - EvalStatistics.iid(b).mean()) >= 0.01 ? 1 : 0;
            EvalStatistics.Interval p = EvalStatistics.paired(c, a, cluster);
            EvalStatistics.Interval up = EvalStatistics.unpaired(c, a, cluster);
            detPaired += p.mean() > 0 && p.pValue() < 0.05 ? 1 : 0;
            detUnpaired += up.mean() > 0 && up.pValue() < 0.05 ? 1 : 0;
            varPaired += p.se() * p.se() * n / replications;
            varUnpaired += up.se() * up.se() * n / replications;
        }
        double f = 100.0 / replications;
        return new ErrorBars(iid / (double) replications, cl / (double) replications,
                boot / (double) Math.min(replications, 100), seCl / seIid, falseWins[0] * f, falseWins[1] * f,
                falseWins[2] * f, falseWins[3] * f, pointDiff * f, detPaired / (double) replications,
                detUnpaired / (double) replications, EvalStatistics.questionsNeeded(0.02, varPaired, 0.05, 0.8),
                EvalStatistics.questionsNeeded(0.02, varUnpaired, 0.05, 0.8));
    }

    private static double[] answers(double[] logit, double ability, SplittableRandom r) {
        double[] x = new double[logit.length];
        for (int i = 0; i < x.length; i++) {
            x[i] = r.nextDouble() < sigmoid(ability + logit[i]) ? 1 : 0;
        }
        return x;
    }

    /**
     * Reported and true accuracy of the final submission of an adaptive analysis.
     *
     * @param mechanism the holdout mechanism
     * @param reported mean reported score
     * @param truth mean fresh-data score
     */
    public record Adaptive(String mechanism, double reported, double truth) {
        @Override
        public String toString() {
            return String.format("%s: reported %.3f, true %.3f", mechanism, reported, truth);
        }
    }

    private static List<Supplier<ReusableHoldout>> mechanisms(long seed) {
        return List.of(ReusableHoldout::plain, () -> ReusableHoldout.ladder(0.01),
                ReusableHoldout::parameterFreeLadder, () -> ReusableHoldout.thresholdout(0.04, 0.01, 1000, seed));
    }

    private static int hashBit(long a, long b, long c) {
        long h = a * 0x9E3779B97F4A7C15L ^ b * 0xC2B2AE3D27D4EB4FL ^ c * 0x165667B19E3779F9L;
        h ^= h >>> 33;
        h *= 0xFF51AFD7ED558CCDL;
        h ^= h >>> 33;
        return (int) (h & 1);
    }

    /**
     * The boosting attack on random labels.
     *
     * @param n examples in the training set and the holdout
     * @param queries random submissions
     * @param replications repeated attacks (fresh labels)
     * @param seed the seed
     * @return per mechanism, the mean reported and true accuracy of the final vote
     */
    public static List<Adaptive> boosting(int n, int queries, int replications, long seed) {
        List<Adaptive> out = new ArrayList<>();
        List<Supplier<ReusableHoldout>> mechs = mechanisms(seed);
        double[][] acc = new double[mechs.size()][2];
        SplittableRandom r = new SplittableRandom(seed);
        for (int k = 0; k < replications; k++) {
            int[] train = labels(n, r);
            int[] holdout = labels(n, r);
            int[] fresh = labels(4 * n, r);
            long salt = r.nextLong();
            for (int m = 0; m < mechs.size(); m++) {
                AdaptiveAnalyst.Outcome o = AdaptiveAnalyst.boostingAttack(mechs.get(m).get(), train, holdout, fresh,
                        queries, i -> (set, index) -> hashBit(salt + i, set, index));
                acc[m][0] += o.reported() / replications;
                acc[m][1] += o.truth() / replications;
            }
        }
        for (int m = 0; m < mechs.size(); m++) {
            out.add(new Adaptive(mechs.get(m).get().kind().name(), acc[m][0], acc[m][1]));
        }
        return out;
    }

    private static int[] labels(int n, SplittableRandom r) {
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            y[i] = r.nextInt(2);
        }
        return y;
    }

    /**
     * An honest greedy search over nearest-centroid classifiers on random feature subsets of
     * a weak-signal task (20 features, 3 informative), holdout of {@code n} points.
     *
     * @param n training and holdout points
     * @param candidates classifiers searched
     * @param replications repeated searches
     * @param seed the seed
     * @return per mechanism, the mean reported and true accuracy of the chosen classifier
     */
    public static List<Adaptive> honestSearch(int n, int candidates, int replications, long seed) {
        List<Supplier<ReusableHoldout>> mechs = mechanisms(seed);
        double[][] acc = new double[mechs.size()][2];
        SplittableRandom r = new SplittableRandom(seed);
        int d = 20;
        for (int k = 0; k < replications; k++) {
            double[][][] sets = {null, null, null};
            int[][] ys = new int[3][];
            int[] sizes = {n, n, 4 * n};
            for (int s = 0; s < 3; s++) {
                sets[s] = new double[sizes[s]][d];
                ys[s] = new int[sizes[s]];
                for (int i = 0; i < sizes[s]; i++) {
                    ys[s][i] = r.nextInt(2);
                    for (int j = 0; j < d; j++) {
                        sets[s][i][j] = r.nextGaussian() + (j < 3 ? 0.35 * (2 * ys[s][i] - 1) : 0);
                    }
                }
            }
            List<double[]> tr = new ArrayList<>();
            List<double[]> ho = new ArrayList<>();
            List<double[]> fr = new ArrayList<>();
            for (int c = 0; c < candidates; c++) {
                boolean[] use = new boolean[d];
                for (int j = 0; j < d; j++) {
                    use[j] = r.nextDouble() < 0.3;
                }
                double[][] centroid = new double[2][d];
                int[] count = new int[2];
                for (int i = 0; i < n; i++) {
                    count[ys[0][i]]++;
                    for (int j = 0; j < d; j++) {
                        centroid[ys[0][i]][j] += sets[0][i][j];
                    }
                }
                for (int y = 0; y < 2; y++) {
                    for (int j = 0; j < d; j++) {
                        centroid[y][j] /= Math.max(1, count[y]);
                    }
                }
                for (int s = 0; s < 3; s++) {
                    double[] score = new double[sizes[s]];
                    for (int i = 0; i < sizes[s]; i++) {
                        double d0 = 0;
                        double d1 = 0;
                        for (int j = 0; j < d; j++) {
                            if (use[j]) {
                                d0 += Math.pow(sets[s][i][j] - centroid[0][j], 2);
                                d1 += Math.pow(sets[s][i][j] - centroid[1][j], 2);
                            }
                        }
                        score[i] = (d1 < d0 ? 1 : 0) == ys[s][i] ? 1 : 0;
                    }
                    (s == 0 ? tr : s == 1 ? ho : fr).add(score);
                }
            }
            for (int m = 0; m < mechs.size(); m++) {
                AdaptiveAnalyst.Outcome o = AdaptiveAnalyst.greedySearch(mechs.get(m).get(), tr, ho, fr);
                acc[m][0] += o.reported() / replications;
                acc[m][1] += o.truth() / replications;
            }
        }
        List<Adaptive> out = new ArrayList<>();
        for (int m = 0; m < mechs.size(); m++) {
            out.add(new Adaptive(mechs.get(m).get().kind().name(), acc[m][0], acc[m][1]));
        }
        return out;
    }
}
