package io.github.zzuegg.brainiac.experimental.discovery;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * PCMCI (Runge et al., Science Advances 2019,
 * <a href="https://arxiv.org/abs/1702.07007">arXiv:1702.07007</a>) for lagged causal discovery
 * in multivariate time series.
 *
 * <ol>
 *   <li>PC1 condition selection: for every target {@code X^j_t}, start from all lagged candidates
 *       {@code X^i_{t-tau}}, {@code 1 <= tau <= tauMax}, and in rounds {@code p = 0, 1, ...}
 *       remove a candidate that tests independent of the target given the {@code p} strongest
 *       other candidates (level {@code alphaPc}); the survivors are a superset of the parents.</li>
 *   <li>MCI: test {@code X^i_{t-tau} indep X^j_t | P(X^j_t) \ {X^i_{t-tau}}, P(X^i_{t-tau})},
 *       conditioning on the parents of both ends (the source's parents shifted by {@code tau},
 *       at most {@code maxSourceParents} of them). Conditioning on the source's own past is what
 *       keeps false positives near the nominal level under strong autocorrelation.</li>
 * </ol>
 *
 * The CI test is either the partial correlation (residuals of OLS on the conditioning set,
 * t-test) or the residual-HSIC {@link CiTest#kernel kernel test}.
 */
public final class Pcmci {

    /** The CI test used. */
    public enum Test {
        /** Partial correlation with a t-test. */
        PARTIAL_CORRELATION,
        /** Kernel ridge residuals tested by HSIC. */
        KERNEL
    }

    private final int tauMax;
    private final double alphaPc;
    private final int maxSourceParents;
    private final Test test;

    /**
     * Creates the learner.
     *
     * @param tauMax the largest lag
     * @param alphaPc the level of the condition-selection stage, for example {@code 0.2}
     * @param maxSourceParents the largest number of source parents in the MCI step
     * @param test the CI test
     */
    public Pcmci(int tauMax, double alphaPc, int maxSourceParents, Test test) {
        this.tauMax = tauMax;
        this.alphaPc = alphaPc;
        this.maxSourceParents = maxSourceParents;
        this.test = test;
    }

    private record Node(int series, int lag) {
    }

    private record Outcome(double pValue, double statistic) {
    }

    private double[] column(double[][] x, Node v, int offset) {
        double[] out = new double[x.length - offset];
        for (int t = 0; t < out.length; t++) {
            out[t] = x[t + offset - v.lag()][v.series()];
        }
        return out;
    }

    private Outcome ci(double[][] x, Node a, Node b, List<Node> z, int offset) {
        double[] va = column(x, a, offset);
        double[] vb = column(x, b, offset);
        int n = va.length;
        double[][] zm = new double[n][z.size()];
        for (int k = 0; k < z.size(); k++) {
            double[] c = column(x, z.get(k), offset);
            for (int t = 0; t < n; t++) {
                zm[t][k] = c[t];
            }
        }
        if (test == Test.KERNEL) {
            double[][] data = new double[n][2 + z.size()];
            int[] s = new int[z.size()];
            for (int t = 0; t < n; t++) {
                data[t][0] = va[t];
                data[t][1] = vb[t];
                System.arraycopy(zm[t], 0, data[t], 2, z.size());
            }
            for (int k = 0; k < s.length; k++) {
                s[k] = 2 + k;
            }
            double p = CiTest.kernel(data, 250, 0.1).pValue(0, 1, s);
            return new Outcome(p, 1 - p);
        }
        double[] ra = z.isEmpty() ? va : LeastSquares.fit(zm, va, true, 1e-8).residuals();
        double[] rb = z.isEmpty() ? vb : LeastSquares.fit(zm, vb, true, 1e-8).residuals();
        double r = LeastSquares.correlation(ra, rb);
        double df = n - 2 - z.size();
        double t = r * Math.sqrt(df / Math.max(1e-12, 1 - r * r));
        return new Outcome(Distributions.studentTwoSided(t, df), Math.abs(r));
    }

    private List<Node> parents(double[][] x, int j, int offset) {
        int d = x[0].length;
        List<Node> cand = new ArrayList<>();
        Map<Node, Double> strength = new HashMap<>();
        for (int i = 0; i < d; i++) {
            for (int tau = 1; tau <= tauMax; tau++) {
                Node v = new Node(i, tau);
                cand.add(v);
                strength.put(v, Double.POSITIVE_INFINITY);
            }
        }
        Node target = new Node(j, 0);
        for (int p = 0; p < cand.size(); p++) {
            List<Node> keep = new ArrayList<>();
            List<Node> snapshot = new ArrayList<>(cand);
            for (Node v : snapshot) {
                List<Node> z = new ArrayList<>();
                for (Node u : cand) {
                    if (!u.equals(v) && z.size() < p) {
                        z.add(u);
                    }
                }
                if (z.size() < p) {
                    keep.add(v);
                    continue;
                }
                Outcome o = ci(x, v, target, z, offset);
                if (o.pValue() > alphaPc) {
                    cand.remove(v);
                } else {
                    strength.put(v, Math.min(strength.get(v), o.statistic()));
                    keep.add(v);
                }
            }
            cand = keep;
            cand.sort(Comparator.comparingDouble((Node v) -> strength.get(v)).reversed());
            if (p >= cand.size() - 1) {
                break;
            }
        }
        return cand;
    }

    /**
     * Runs PCMCI.
     *
     * @param series {@code [T][d]}
     * @return {@code p[i][j][tau - 1]}: the MCI p-value of {@code X^i_{t-tau} -> X^j_t}
     */
    public double[][][] fit(double[][] series) {
        int d = series[0].length;
        int offset = 2 * tauMax;
        List<List<Node>> parents = new ArrayList<>();
        for (int j = 0; j < d; j++) {
            parents.add(parents(series, j, offset));
        }
        double[][][] p = new double[d][d][tauMax];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                for (int tau = 1; tau <= tauMax; tau++) {
                    Node src = new Node(i, tau);
                    List<Node> z = new ArrayList<>();
                    for (Node u : parents.get(j)) {
                        if (!u.equals(src)) {
                            z.add(u);
                        }
                    }
                    int added = 0;
                    for (Node u : parents.get(i)) {
                        if (added >= maxSourceParents) {
                            break;
                        }
                        Node shifted = new Node(u.series(), u.lag() + tau);
                        if (!z.contains(shifted) && !shifted.equals(src)) {
                            z.add(shifted);
                            added++;
                        }
                    }
                    p[i][j][tau - 1] = ci(series, src, new Node(j, 0), z, offset).pValue();
                }
            }
        }
        return p;
    }

    /**
     * The smallest p-value over lags.
     *
     * @param p {@code [d][d][tauMax]}
     * @return {@code [d][d]}
     */
    public static double[][] minOverLags(double[][][] p) {
        int d = p.length;
        double[][] out = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                double m = 1;
                for (double v : p[i][j]) {
                    m = Math.min(m, v);
                }
                out[i][j] = m;
            }
        }
        return out;
    }
}
