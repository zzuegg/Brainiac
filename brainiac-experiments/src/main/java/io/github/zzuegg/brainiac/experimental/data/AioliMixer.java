package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

/**
 * Aioli (Chen et al., "Aioli: A Unified Optimization Framework for Language Model Data Mixing",
 * ICLR 2025, <a href="https://arxiv.org/abs/2411.05735">arXiv:2411.05735</a>). Every mixing
 * method minimises the average loss under a "mixing law" linking the per-group loss drop to the
 * proportions; Aioli fits the law's parameters online, during the one training run, and moves the
 * proportions by exponentiated gradient descent.
 * <ul>
 *   <li>Training runs in rounds of {@code K + 1} chunks. Chunk {@code k < K} of a round trains on
 *       a probe mixture {@code (1 - ε) p + ε e_k}; the last chunk trains on {@code p}. After every
 *       chunk the caller reports each group's validation loss.</li>
 *   <li>The linear dynamic mixing law {@code L_i(t+1) = L_i(t) - Σ_j A_ij m_j} is fitted by ridge
 *       regression of the observed loss drops on the mixtures used, over the recent
 *       history.</li>
 *   <li>The proportions then follow exponentiated gradient descent on the average loss:
 *       {@code p_j ← p_j exp(η g_j / max|g|)}, {@code g_j = Σ_i A_ij}, renormalised (the
 *       gradient is scaled by its largest entry so η is unitless).</li>
 * </ul>
 */
public final class AioliMixer {

    private final int groups;
    private final double eta;
    private final double ridge;
    private final double probe;
    private final int history;
    private final Deque<double[][]> observations = new ArrayDeque<>();
    private double[] p;
    private double[] current;
    private double[] previousLoss;
    private int chunk;
    private double[][] lastA;

    /**
     * Creates the mixer, starting uniform.
     *
     * @param groups the number of groups
     * @param eta the exponentiated-gradient step, for example 0.3
     * @param ridge the ridge penalty of the law fit, for example 1e-4
     * @param probe the probe share ε, for example 0.5
     * @param history the observations kept for the fit, for example 4 rounds' worth
     */
    public AioliMixer(int groups, double eta, double ridge, double probe, int history) {
        this.groups = groups;
        this.eta = eta;
        this.ridge = ridge;
        this.probe = probe;
        this.history = history;
        this.p = new double[groups];
        Arrays.fill(p, 1.0 / groups);
        this.current = mixtureFor(0);
    }

    private double[] mixtureFor(int k) {
        if (k >= groups) {
            return p.clone();
        }
        double[] m = new double[groups];
        for (int j = 0; j < groups; j++) {
            m[j] = (1 - probe) * p[j] + (j == k ? probe : 0);
        }
        return m;
    }

    /** @return the mixture to train the next chunk on */
    public double[] proportions() {
        return current.clone();
    }

    /** @return the current (non-probe) proportions */
    public double[] weights() {
        return p.clone();
    }

    /** @return the last fitted mixing-law matrix, or null before the first fit */
    public double[][] law() {
        return lastA == null ? null : lastA.clone();
    }

    /**
     * Reports the per-group validation losses after training a chunk on {@link #proportions()}.
     *
     * @param losses each group's loss
     * @return the mixture for the next chunk
     */
    public double[] observe(double[] losses) {
        if (previousLoss != null) {
            double[] drop = new double[groups];
            for (int i = 0; i < groups; i++) {
                drop[i] = previousLoss[i] - losses[i];
            }
            observations.addLast(new double[][] {current.clone(), drop});
            while (observations.size() > history) {
                observations.removeFirst();
            }
        }
        previousLoss = losses.clone();
        chunk++;
        if (chunk % (groups + 1) == 0 && observations.size() >= groups) {
            updateProportions();
        }
        current = mixtureFor(chunk % (groups + 1));
        return current.clone();
    }

    private void updateProportions() {
        int n = observations.size();
        double[][] m = new double[n][];
        double[][] d = new double[n][];
        int r = 0;
        for (double[][] o : observations) {
            m[r] = o[0];
            d[r++] = o[1];
        }
        double[][] gram = new double[groups][groups];
        for (int a = 0; a < groups; a++) {
            for (int b = 0; b < groups; b++) {
                for (int k = 0; k < n; k++) {
                    gram[a][b] += m[k][a] * m[k][b];
                }
            }
            gram[a][a] += ridge;
        }
        double[][] a = new double[groups][];
        for (int i = 0; i < groups; i++) {
            double[] rhs = new double[groups];
            for (int j = 0; j < groups; j++) {
                for (int k = 0; k < n; k++) {
                    rhs[j] += m[k][j] * d[k][i];
                }
            }
            a[i] = CurvatureProducts.solveSpd(gram, rhs);
        }
        lastA = a;
        double[] g = new double[groups];
        double scale = 0;
        for (int j = 0; j < groups; j++) {
            for (int i = 0; i < groups; i++) {
                g[j] += a[i][j];
            }
            scale = Math.max(scale, Math.abs(g[j]));
        }
        if (scale == 0) {
            return;
        }
        double sum = 0;
        for (int j = 0; j < groups; j++) {
            p[j] *= Math.exp(eta * g[j] / scale);
            sum += p[j];
        }
        for (int j = 0; j < groups; j++) {
            p[j] = Math.max(1e-4, p[j] / sum);
        }
        double s2 = Arrays.stream(p).sum();
        for (int j = 0; j < groups; j++) {
            p[j] /= s2;
        }
    }

    @Override
    public String toString() {
        return "AioliMixer" + Arrays.toString(p);
    }
}
