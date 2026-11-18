package io.github.zzuegg.brainiac.experimental.tabular;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.estimators.MutualInformation;
import io.github.zzuegg.brainiac.experimental.ml.AgglomerativeClustering;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Mutual-information-guided structured corruption for tabular self-supervised learning, after
 * Lawson et al. (ICML 2026, <a href="https://icml.cc/virtual/2026/71753">poster 71753</a>). Random
 * per-feature corruption is easy to undo when features are redundant: a corrupted column can be
 * read back from its correlated neighbours, so the pretext task teaches little. This corruptor
 * <ol>
 *   <li>estimates the pairwise mutual information of the features with the KSG estimator
 *       ({@link MutualInformation#ksg}) on a subsample,</li>
 *   <li>groups the features by agglomerative clustering ({@link AgglomerativeClustering}, average
 *       linkage) of their mutual-information profiles, so mutually dependent features fall into
 *       one group, and</li>
 *   <li>corrupts whole groups at once: {@link Replacement#GROUP_MARGINAL} copies the group's
 *       values jointly from one random other row (a draw from the group's joint marginal);
 *       {@link Replacement#GROUP_CVAE} samples them from a small conditional VAE given the
 *       remaining features, so the replacement is plausible in context.</li>
 * </ol>
 * It plugs into {@link TabularPretraining} as a {@link TabularPretraining.Corruptor}.
 */
public final class StructuredCorruption implements TabularPretraining.Corruptor {

    /** How a corrupted group is filled. */
    public enum Replacement {
        /** The group's values from one random other row. */
        GROUP_MARGINAL,
        /** A sample of a conditional VAE given the uncorrupted features. */
        GROUP_CVAE
    }

    private final float[][] pool;
    private final int[] group;
    private final int groups;
    private final double rate;
    private final Replacement replacement;
    private final double[][] mi;
    private Model cvaeEncoder;
    private Model cvaeDecoder;
    private int latent;

    private StructuredCorruption(float[][] pool, int[] group, int groups, double rate, Replacement replacement,
            double[][] mi) {
        this.pool = pool;
        this.group = group;
        this.groups = groups;
        this.rate = rate;
        this.replacement = replacement;
        this.mi = mi;
    }

    /**
     * Pairwise KSG mutual information between columns.
     *
     * @param rows the rows {@code [n][d]}
     * @param k the KSG neighbour count, such as 3
     * @return the symmetric matrix {@code [d][d]} (diagonal zero)
     */
    public static double[][] mutualInformation(float[][] rows, int k) {
        int d = rows[0].length;
        double[][][] cols = new double[d][rows.length][1];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < d; j++) {
                cols[j][i][0] = rows[i][j] + 1e-6 * Math.sin(1000.0 * i + j);
            }
        }
        double[][] out = new double[d][d];
        for (int a = 0; a < d; a++) {
            for (int b = a + 1; b < d; b++) {
                out[a][b] = Math.max(0, MutualInformation.ksg(cols[a], cols[b], k));
                out[b][a] = out[a][b];
            }
        }
        return out;
    }

    /**
     * Fits the grouping (and, for {@link Replacement#GROUP_CVAE}, the conditional VAE).
     *
     * @param rows the unlabelled training rows (standardised)
     * @param groups the number of feature groups
     * @param rate the target share of features corrupted per row
     * @param replacement how corrupted groups are filled
     * @param sample rows used for the mutual information, such as 300
     * @param seed the seed
     * @return the corruptor
     */
    public static StructuredCorruption fit(float[][] rows, int groups, double rate, Replacement replacement,
            int sample, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] sub = new float[Math.min(sample, rows.length)][];
        for (int i = 0; i < sub.length; i++) {
            sub[i] = rows[r.nextInt(rows.length)];
        }
        double[][] mi = mutualInformation(sub, 3);
        int d = rows[0].length;
        double[][] profile = new double[d][d];
        for (int a = 0; a < d; a++) {
            for (int b = 0; b < d; b++) {
                profile[a][b] = a == b ? maxRow(mi[a]) : mi[a][b];
            }
        }
        int[] g = AgglomerativeClustering.cluster(profile, Math.min(groups, d), AgglomerativeClustering.Linkage.AVERAGE);
        StructuredCorruption c = new StructuredCorruption(rows, g, Math.min(groups, d), rate, replacement, mi);
        if (replacement == Replacement.GROUP_CVAE) {
            c.trainCvae(rows, 200, seed + 1);
        }
        return c;
    }

    private static double maxRow(double[] row) {
        double m = 0;
        for (double v : row) {
            m = Math.max(m, v);
        }
        return m;
    }

    /** @return the group of every feature */
    public int[] groups() {
        return group.clone();
    }

    /** @return the estimated pairwise mutual information */
    public double[][] mutualInformation() {
        return mi;
    }

    /** Draws a group mask: whole groups at random until about {@code rate * d} features are covered. */
    private boolean[] drawMask(SplittableRandom r) {
        int d = group.length;
        boolean[] m = new boolean[d];
        int target = Math.max(1, (int) Math.round(rate * d));
        int covered = 0;
        int[] order = new int[groups];
        for (int i = 0; i < groups; i++) {
            order[i] = i;
        }
        for (int i = groups - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = order[i];
            order[i] = order[j];
            order[j] = t;
        }
        for (int gi = 0; gi < groups && covered < target; gi++) {
            int size = 0;
            for (int j = 0; j < d; j++) {
                size += group[j] == order[gi] ? 1 : 0;
            }
            if (covered > 0 && covered + size > target * 1.5 && r.nextBoolean()) {
                continue;
            }
            for (int j = 0; j < d; j++) {
                if (group[j] == order[gi]) {
                    m[j] = true;
                    covered++;
                }
            }
        }
        return m;
    }

    @Override
    public TabularPretraining.Corrupted corrupt(float[][] rows, SplittableRandom random) {
        int d = rows[0].length;
        float[][] v = new float[rows.length][];
        float[][] mask = new float[rows.length][d];
        for (int i = 0; i < rows.length; i++) {
            boolean[] m = drawMask(random);
            v[i] = rows[i].clone();
            for (int j = 0; j < d; j++) {
                mask[i][j] = m[j] ? 1 : 0;
            }
            if (replacement == Replacement.GROUP_MARGINAL) {
                // one donor row per group keeps the group's joint distribution
                int[] donor = new int[groups];
                for (int g = 0; g < groups; g++) {
                    donor[g] = random.nextInt(pool.length);
                }
                for (int j = 0; j < d; j++) {
                    if (m[j]) {
                        v[i][j] = pool[donor[group[j]]][j];
                    }
                }
            }
        }
        if (replacement == Replacement.GROUP_CVAE) {
            float[] gen = cvaeSample(v, mask, random);
            for (int i = 0; i < rows.length; i++) {
                for (int j = 0; j < d; j++) {
                    if (mask[i][j] > 0) {
                        v[i][j] = gen[i * d + j];
                    }
                }
            }
        }
        return new TabularPretraining.Corrupted(v, mask);
    }

    /** Input of the CVAE's networks: the visible features (masked ones zeroed) and the mask. */
    private static Tensor conditioning(float[][] rows, float[][] mask) {
        int n = rows.length;
        int d = rows[0].length;
        float[] c = new float[n * 2 * d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                c[i * 2 * d + j] = mask[i][j] > 0 ? 0 : rows[i][j];
                c[i * 2 * d + d + j] = mask[i][j];
            }
        }
        return Tensor.of(Shape.of(n, 2 * d), c);
    }

    /**
     * Trains the conditional VAE: encoder {@code q(z | x, visible, mask)}, decoder
     * {@code p(x_masked | z, visible, mask)}, squared reconstruction error on the masked entries
     * plus the KL divergence to the standard normal prior.
     */
    private void trainCvae(float[][] rows, int steps, long seed) {
        int d = rows[0].length;
        latent = Math.max(2, d / 4);
        cvaeEncoder = Model.sequential().input(3 * d).dense(32, Activation.RELU).dense(2 * latent).seed(seed).build();
        cvaeDecoder = Model.sequential().input(latent + 2 * d).dense(32, Activation.RELU).dense(d).seed(seed + 1)
                .build();
        List<Var> params = new ArrayList<>(cvaeEncoder.parameters());
        params.addAll(cvaeDecoder.parameters());
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int b = Math.min(64, rows.length);
        for (int s = 0; s < steps; s++) {
            float[][] xb = new float[b][];
            float[][] mb = new float[b][d];
            for (int i = 0; i < b; i++) {
                xb[i] = rows[r.nextInt(rows.length)];
                boolean[] m = drawMask(r);
                for (int j = 0; j < d; j++) {
                    mb[i][j] = m[j] ? 1 : 0;
                }
            }
            Tensor cond = conditioning(xb, mb);
            Tensor x = TabularPretraining.tensor(xb);
            Tensor mask = TabularPretraining.tensor(mb);
            Tensor eps = Tensor.normal(Shape.of(b, latent), 0, 1, r);
            opt.minimize(params, () -> {
                Var stats = cvaeEncoder.forward(Var.concat(1, Var.of(x), Var.of(cond)), true);
                Var mu = stats.slice(1, 0, latent);
                Var logVar = stats.slice(1, latent, 2 * latent).clamp(-6, 4);
                Var z = mu.add(logVar.mul(0.5f).exp().mul(eps));
                Var rec = cvaeDecoder.forward(Var.concat(1, z, Var.of(cond)), true);
                Var nll = rec.sub(x).square().mul(mask).sum().mul(1f / b);
                Var kl = mu.square().add(logVar.exp()).sub(logVar).sub(1f).sum().mul(0.5f / b);
                return nll.add(kl);
            });
        }
    }

    private float[] cvaeSample(float[][] rows, float[][] mask, SplittableRandom r) {
        Tensor cond = conditioning(rows, mask);
        Tensor z = Tensor.normal(Shape.of(rows.length, latent), 0, 1, r);
        return Var.noGrad(() -> cvaeDecoder.forward(Var.concat(1, Var.of(z), Var.of(cond)), false)).value()
                .toArray();
    }
}
