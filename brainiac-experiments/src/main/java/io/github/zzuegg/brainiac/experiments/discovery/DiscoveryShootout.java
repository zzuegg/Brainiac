package io.github.zzuegg.brainiac.experiments.discovery;

import io.github.zzuegg.brainiac.experimental.data.DiscoveryBenchmarks;
import io.github.zzuegg.brainiac.experimental.discovery.CausalGraphs;
import io.github.zzuegg.brainiac.experimental.discovery.CauseEffect;
import io.github.zzuegg.brainiac.experimental.discovery.CiTest;
import io.github.zzuegg.brainiac.experimental.discovery.Fci;
import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;
import io.github.zzuegg.brainiac.experimental.discovery.Lingam;
import io.github.zzuegg.brainiac.experimental.discovery.PcAlgorithm;
import io.github.zzuegg.brainiac.experimental.discovery.ScoreOrdering;
import io.github.zzuegg.brainiac.experimental.discovery.TriOpt;
import io.github.zzuegg.brainiac.experimental.ml.Dagma;
import io.github.zzuegg.brainiac.experimental.ml.GreedyEquivalenceSearch;
import io.github.zzuegg.brainiac.experimental.ml.Notears;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 11 items 11.3.1 to 11.3.4: causal discovery on {@link DiscoveryBenchmarks} SEMs, scored
 * against the true graph.
 * <ul>
 *   <li>{@link #constraintBased} (11.3.1): PC with Meek orientation (Fisher-z, or the kernel CI
 *       test on nonlinear data) against the PC skeleton alone, GES, NOTEARS, DAGMA and the empty
 *       graph on standardised data (removing the varsortability NOTEARS exploits), by SHD against
 *       the true CPDAG, orientation precision/recall and runtime. Continuous optimisers' DAGs are
 *       mapped to their CPDAGs before scoring.</li>
 *   <li>{@link #hidden} (11.3.1): FCI against PC with hidden confounders, by ancestral arrowhead
 *       and tail precision and the distance to the oracle PAG (FCI on a d-separation oracle).</li>
 *   <li>{@link #nonGaussian} (11.3.2, 11.3.3): DirectLiNGAM, ICA-LiNGAM and TriOpt against
 *       NOTEARS, DAGMA and PC on linear SEMs, by SHD against the true DAG (PC's undirected edges
 *       count as errors) and the share of correctly ordered ancestral pairs.</li>
 *   <li>{@link #nonlinearOrdering} (11.3.3): SCORE against NOTEARS, DAGMA and kernel PC on
 *       nonlinear additive-noise models, by order divergence and SHD.</li>
 *   <li>{@link #pairs} (11.3.4): ANM, IGCI (slope and entropy) and RECI against naive HSIC,
 *       pairwise LiNGAM and coin flips on bivariate pairs, by accuracy and the area under the
 *       accuracy-decision-rate curve.</li>
 * </ul>
 */
public final class DiscoveryShootout {

    private DiscoveryShootout() {
    }

    private static void add(Map<String, double[]> out, String key, double[] v, int runs) {
        double[] acc = out.computeIfAbsent(key, s -> new double[v.length]);
        for (int i = 0; i < v.length; i++) {
            acc[i] += v[i] / runs;
        }
    }

    private static double[][] binary(double[][] w) {
        int d = w.length;
        double[][] g = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                g[i][j] = w[i][j] != 0 ? 1 : 0;
            }
        }
        return g;
    }

    private static double[] cpdagScore(double[][] dag, double[][] cpdag, long ms) {
        double[] o = CausalGraphs.orientation(dag, cpdag);
        return new double[] {CausalGraphs.cpdagShd(dag, cpdag), o[0], o[1], ms};
    }

    private static double[] timed(double[][] dag, Function<Void, double[][]> run, boolean isDag) {
        long t0 = System.nanoTime();
        double[][] est = run.apply(null);
        long ms = (System.nanoTime() - t0) / 1_000_000;
        return cpdagScore(dag, isDag ? GreedyEquivalenceSearch.cpdag(binary(est)) : est, ms);
    }

    /**
     * PC against score-based and continuous learners.
     *
     * @param d variables
     * @param n samples
     * @param nonlinear nonlinear ANM data (kernel CI test) instead of linear-Gaussian (Fisher-z)
     * @param graphs random graphs averaged
     * @param seed the seed
     * @return method to {@code {SHD vs CPDAG, orientation precision, orientation recall, ms}}
     */
    public static Map<String, double[]> constraintBased(int d, int n, boolean nonlinear, int graphs, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int g = 0; g < graphs; g++) {
            double[][] w = DiscoveryBenchmarks.randomDag(d, 2, seed + g);
            double[][] dag = binary(w);
            double[][] raw = nonlinear ? DiscoveryBenchmarks.nonlinearAnm(dag, n, seed + 100 + g)
                    : DiscoveryBenchmarks.linearSem(w, n, DiscoveryBenchmarks.Noise.GAUSSIAN, false, seed + 100 + g);
            double[][] x = LeastSquares.standardise(raw);
            CiTest test = nonlinear ? CiTest.kernel(x, 200, 0.1) : CiTest.fisherZ(x);
            add(out, "empty graph", cpdagScore(dag, new double[d][d], 0), graphs);
            long t0 = System.nanoTime();
            PcAlgorithm.Result pc = new PcAlgorithm(0.01, nonlinear ? 2 : 3).fit(test, d);
            long ms = (System.nanoTime() - t0) / 1_000_000;
            add(out, "PC skeleton only", cpdagScore(dag, pc.skeleton().matrix(), ms), graphs);
            add(out, "PC + Meek", cpdagScore(dag, pc.cpdag(), ms), graphs);
            add(out, "GES", timed(dag, v -> new GreedyEquivalenceSearch(1, 4).fit(x), true), graphs);
            add(out, "NOTEARS", timed(dag, v -> new Notears(0.05, 0.3, 400, 12).fit(x), true), graphs);
            add(out, "DAGMA", timed(dag, v -> new Dagma(0.05, 0.3, 1500, 4).fit(x), true), graphs);
        }
        return out;
    }

    /**
     * FCI against PC with hidden confounders.
     *
     * @param d observed variables
     * @param hidden latent confounders
     * @param n samples
     * @param graphs graphs averaged
     * @param seed the seed
     * @return method to {@code {arrowhead precision, arrowhead recall, tail precision, distance to
     *     oracle PAG}}
     */
    public static Map<String, double[]> hidden(int d, int hidden, int n, int graphs, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int g = 0; g < graphs; g++) {
            DiscoveryBenchmarks.Hidden h = DiscoveryBenchmarks.confounded(d, 2, hidden, n, seed + g);
            double[][] x = LeastSquares.standardise(h.data());
            CiTest test = CiTest.fisherZ(x);
            int[][] oracle = new Fci(0.5, 3, 8).fit(CausalGraphs.oracle(h.dag(), h.observed()), d);
            int[][] fci = new Fci(0.01, 3, 8).fit(test, d);
            int[][] pc = CausalGraphs.marks(new PcAlgorithm(0.01, 3).fit(test, d).cpdag());
            for (Object[] row : new Object[][] {{"FCI", fci}, {"PC (CPDAG read as marks)", pc},
                {"FCI on the d-separation oracle", oracle}}) {
                int[][] m = (int[][]) row[1];
                double[] s = CausalGraphs.ancestralScores(m, h.dag(), h.observed());
                add(out, (String) row[0], new double[] {s[0], s[1], s[2], CausalGraphs.pagDistance(m, oracle)},
                        graphs);
            }
        }
        return out;
    }

    /**
     * LiNGAM and TriOpt on linear SEMs.
     *
     * @param d variables
     * @param n samples
     * @param noise the noise distribution
     * @param standardise whether the data are standardised before fitting
     * @param graphs graphs averaged
     * @param seed the seed
     * @return method to {@code {SHD vs DAG, ordered-pair share, ms}}
     */
    public static Map<String, double[]> nonGaussian(int d, int n, DiscoveryBenchmarks.Noise noise,
            boolean standardise, int graphs, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int g = 0; g < graphs; g++) {
            double[][] w = DiscoveryBenchmarks.randomDag(d, 2, seed + g);
            double[][] dag = binary(w);
            double[][] raw = DiscoveryBenchmarks.linearSem(w, n, noise, true, seed + 50 + g);
            double[][] x = standardise ? LeastSquares.standardise(raw) : raw;
            long t0 = System.nanoTime();
            Lingam.Result direct = Lingam.direct(x);
            add(out, "DirectLiNGAM", dagScore(dag, direct.dag(0.05), direct.order(), t0), graphs);
            t0 = System.nanoTime();
            Lingam.Result ica = Lingam.ica(x, 200, seed + g);
            add(out, "ICA-LiNGAM", dagScore(dag, ica.dag(0.05), ica.order(), t0), graphs);
            t0 = System.nanoTime();
            Lingam.Result tri = TriOpt.fit(x);
            add(out, "TriOpt", dagScore(dag, tri.dag(0.05), tri.order(), t0), graphs);
            t0 = System.nanoTime();
            double[][] nt = binary(new Notears(0.05, 0.3, 400, 12).fit(x));
            add(out, "NOTEARS", dagScore(dag, nt, DiscoveryBenchmarks.topologicalOrder(nt), t0), graphs);
            t0 = System.nanoTime();
            double[][] dg = binary(new Dagma(0.05, 0.3, 1500, 4).fit(x));
            add(out, "DAGMA", dagScore(dag, dg, DiscoveryBenchmarks.topologicalOrder(dg), t0), graphs);
            t0 = System.nanoTime();
            double[][] pc = new PcAlgorithm(0.01, 3).fit(CiTest.fisherZ(x), d).cpdag();
            long ms = (System.nanoTime() - t0) / 1_000_000;
            add(out, "PC (CPDAG)", new double[] {Notears.shd(dag, pc), Double.NaN, ms}, graphs);
        }
        return out;
    }

    private static double[] dagScore(double[][] dag, double[][] est, int[] order, long t0) {
        long ms = (System.nanoTime() - t0) / 1_000_000;
        return new double[] {Notears.shd(dag, est), CausalGraphs.orderedPairs(order, dag), ms};
    }

    /**
     * SCORE on nonlinear additive-noise models.
     *
     * @param d variables
     * @param n samples
     * @param graphs graphs averaged
     * @param seed the seed
     * @return method to {@code {order divergence, SHD vs DAG, ms}}
     */
    public static Map<String, double[]> nonlinearOrdering(int d, int n, int graphs, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int g = 0; g < graphs; g++) {
            double[][] dag = binary(DiscoveryBenchmarks.randomDag(d, 2, seed + g));
            double[][] x = DiscoveryBenchmarks.nonlinearAnm(dag, n, seed + 70 + g);
            double[][] xs = LeastSquares.standardise(x);
            long t0 = System.nanoTime();
            ScoreOrdering.Result score = new ScoreOrdering(1e-3, 1e-3, 1e-3).fit(x);
            add(out, "SCORE", orderScore(dag, score.dag(), score.order(), t0), graphs);
            t0 = System.nanoTime();
            double[][] nt = binary(new Notears(0.05, 0.3, 400, 12).fit(xs));
            add(out, "NOTEARS", orderScore(dag, nt, DiscoveryBenchmarks.topologicalOrder(nt), t0), graphs);
            t0 = System.nanoTime();
            double[][] dg = binary(new Dagma(0.05, 0.3, 1500, 4).fit(xs));
            add(out, "DAGMA", orderScore(dag, dg, DiscoveryBenchmarks.topologicalOrder(dg), t0), graphs);
            int[] random = new int[d];
            java.util.Arrays.setAll(random, i -> i);
            SplittableRandom r = new SplittableRandom(seed + 999 + g);
            for (int i = d - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int tmp = random[i];
                random[i] = random[j];
                random[j] = tmp;
            }
            add(out, "random order", new double[] {CausalGraphs.orderDivergence(random, dag), Double.NaN, 0}, graphs);
        }
        return out;
    }

    private static double[] orderScore(double[][] dag, double[][] est, int[] order, long t0) {
        long ms = (System.nanoTime() - t0) / 1_000_000;
        return new double[] {CausalGraphs.orderDivergence(order, dag), Notears.shd(dag, est), ms};
    }

    /**
     * Bivariate cause-effect pairs.
     *
     * @param kind the mechanism class
     * @param count pairs
     * @param n samples per pair
     * @param seed the seed
     * @return method to {@code {accuracy, AUC of the accuracy-decision-rate curve}}
     */
    public static Map<String, double[]> pairs(DiscoveryBenchmarks.PairKind kind, int count, int n, long seed) {
        Map<String, Function<DiscoveryBenchmarks.Pair, Double>> methods = new LinkedHashMap<>();
        methods.put("ANM (kernel ridge + HSIC)", p -> CauseEffect.anm(p.x(), p.y()));
        methods.put("IGCI slope", p -> CauseEffect.igciSlope(p.x(), p.y()));
        methods.put("IGCI entropy", p -> CauseEffect.igciEntropy(p.x(), p.y()));
        methods.put("RECI", p -> CauseEffect.reci(p.x(), p.y()));
        methods.put("naive HSIC (linear residuals)", p -> CauseEffect.naiveHsic(p.x(), p.y()));
        methods.put("LiNGAM pairwise", p -> CauseEffect.lingam(p.x(), p.y()));
        SplittableRandom coin = new SplittableRandom(seed);
        methods.put("random guess", p -> coin.nextBoolean() ? 1.0 : -1.0);
        double[][] scores = new double[methods.size()][count];
        boolean[] truth = new boolean[count];
        for (int k = 0; k < count; k++) {
            DiscoveryBenchmarks.Pair p = DiscoveryBenchmarks.pair(kind, n, seed + 1000 + k);
            truth[k] = p.xCausesY();
            int m = 0;
            for (Function<DiscoveryBenchmarks.Pair, Double> f : methods.values()) {
                scores[m++][k] = f.apply(p);
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        int m = 0;
        for (String name : methods.keySet()) {
            out.put(name, CauseEffect.accuracyCurve(scores[m++], truth));
        }
        return out;
    }
}
