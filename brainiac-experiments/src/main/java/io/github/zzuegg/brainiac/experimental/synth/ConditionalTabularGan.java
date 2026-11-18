package io.github.zzuegg.brainiac.experimental.synth;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.GaussianMixture;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * CTGAN (Xu et al., "Modeling Tabular Data using Conditional GAN", NeurIPS 2019,
 * <a href="https://arxiv.org/abs/1907.00503">arXiv:1907.00503</a>).
 * <ul>
 *   <li>mode-specific normalisation: each numeric column gets a 1-D {@link GaussianMixture}
 *       ({@code modes} components); a value is written as the one-hot of a mode drawn from its
 *       posterior plus {@code α = (x - μ_k) / (4 σ_k)};</li>
 *   <li>a conditional generator with training-by-sampling: each step picks a categorical column
 *       uniformly and a category with probability proportional to the log of its frequency, takes
 *       real rows with that category, and asks the generator for rows with it (a condition vector
 *       fed to both networks, plus a cross-entropy penalty that the generated column matches it).
 *       Rare categories are seen far more often than their frequency, which is what helps the
 *       minority class;</li>
 *   <li>the generator emits {@code tanh} for each α and softmax blocks for the modes and
 *       categories (CTGAN uses Gumbel-softmax); the discriminator is a PacGAN critic that sees
 *       {@code pac} rows at once, which punishes a generator that collapses onto few rows, and is
 *       trained with the non-saturating GAN loss (CTGAN uses WGAN-GP); both networks use Adam with
 *       β = (0.5, 0.9).</li>
 * </ul>
 */
public final class ConditionalTabularGan {

    private final int numeric;
    private final int[] categories;
    private final int modes;
    private final int noise;
    private final int hidden;
    private final int pac;
    private double[][] mu;
    private double[][] sigma;
    private SimpleMlp generator;
    private SimpleMlp discriminator;
    private int repr;
    private int cond;
    private double[][] frequency;

    /**
     * Creates an untrained model.
     *
     * @param numeric numeric columns
     * @param categories categories per categorical column
     * @param modes mixture components per numeric column
     * @param noise generator noise dimensions
     * @param hidden network width
     * @param pac rows the discriminator sees together (PacGAN; CTGAN uses 10)
     */
    public ConditionalTabularGan(int numeric, int[] categories, int modes, int noise, int hidden, int pac) {
        this.numeric = numeric;
        this.categories = categories.clone();
        this.modes = modes;
        this.noise = noise;
        this.hidden = hidden;
        this.pac = Math.max(1, pac);
    }

    private Var critic(Var rows, Var c) {
        Var x = Var.concat(1, rows, c);
        int n = x.shape().dim(0);
        return discriminator.forward(x.reshape(n / pac, pac * x.shape().dim(1)));
    }

    private double[] encodeRow(MixedTable t, int i, SplittableRandom r) {
        double[] v = new double[repr];
        int k = 0;
        for (int j = 0; j < numeric; j++) {
            double x = t.numeric()[i][j];
            double[] post = new double[modes];
            double s = 0;
            for (int m = 0; m < modes; m++) {
                double z = (x - mu[j][m]) / sigma[j][m];
                post[m] = Math.exp(-0.5 * z * z) / sigma[j][m] + 1e-300;
                s += post[m];
            }
            double u = r.nextDouble() * s;
            int mode = 0;
            while (mode < modes - 1 && (u -= post[mode]) > 0) {
                mode++;
            }
            v[k] = Math.max(-0.99, Math.min(0.99, (x - mu[j][mode]) / (4 * sigma[j][mode])));
            v[k + 1 + mode] = 1;
            k += 1 + modes;
        }
        for (int c = 0; c < categories.length; c++) {
            v[k + t.categorical()[i][c]] = 1;
            k += categories[c];
        }
        return v;
    }

    private Var generate(Var z, Var c) {
        Var out = generator.forward(Var.concat(1, z, c));
        List<Var> parts = new ArrayList<>();
        int k = 0;
        for (int j = 0; j < numeric; j++) {
            parts.add(out.slice(1, k, k + 1).tanh());
            parts.add(softmax(out.slice(1, k + 1, k + 1 + modes)));
            k += 1 + modes;
        }
        for (int cat : categories) {
            parts.add(softmax(out.slice(1, k, k + cat)));
            k += cat;
        }
        return Var.concat(1, parts.toArray(new Var[0]));
    }

    private static Var softmax(Var logits) {
        return logits.sub(logits.logsumexp(1, true)).exp();
    }

    private int catOffset(int column) {
        int k = numeric * (1 + modes);
        for (int c = 0; c < column; c++) {
            k += categories[c];
        }
        return k;
    }

    private int condOffset(int column) {
        int k = 0;
        for (int c = 0; c < column; c++) {
            k += categories[c];
        }
        return k;
    }

    /**
     * Trains on a table.
     *
     * @param table the real table
     * @param iterations GAN steps
     * @param batchSize rows per step (rounded down to a multiple of {@code pac})
     * @param lr the Adam rate of both networks
     * @param seed the seed
     */
    public void fit(MixedTable table, int iterations, int batchSize, float lr, long seed) {
        int batch = Math.max(pac, batchSize / pac * pac);
        SplittableRandom r = new SplittableRandom(seed);
        mu = new double[numeric][modes];
        sigma = new double[numeric][modes];
        for (int j = 0; j < numeric; j++) {
            double[][] col = new double[table.rows()][1];
            for (int i = 0; i < col.length; i++) {
                col[i][0] = table.numeric()[i][j];
            }
            GaussianMixture g = GaussianMixture.fit(col, modes, 60, 1e-3, seed + j);
            for (int m = 0; m < modes; m++) {
                mu[j][m] = g.mean(m)[0];
                sigma[j][m] = Math.sqrt(Math.max(1e-6, g.covariance(m)[0][0]));
            }
        }
        repr = numeric * (1 + modes);
        cond = 0;
        for (int c : categories) {
            repr += c;
            cond += c;
        }
        frequency = new double[categories.length][];
        List<List<List<Integer>>> rowsWith = new ArrayList<>();
        for (int c = 0; c < categories.length; c++) {
            frequency[c] = new double[categories[c]];
            List<List<Integer>> lists = new ArrayList<>();
            for (int k = 0; k < categories[c]; k++) {
                lists.add(new ArrayList<>());
            }
            for (int i = 0; i < table.rows(); i++) {
                frequency[c][table.categorical()[i][c]]++;
                lists.get(table.categorical()[i][c]).add(i);
            }
            rowsWith.add(lists);
        }
        generator = SimpleMlp.relu(seed + 100, noise + cond, hidden, hidden, repr);
        discriminator = SimpleMlp.relu(seed + 200, pac * (repr + cond), hidden, hidden, 1);
        Optimizer gOpt = Optimizer.adam(lr, 0.5f, 0.9f, 1e-8f);
        Optimizer dOpt = Optimizer.adam(lr, 0.5f, 0.9f, 1e-8f);
        for (int it = 0; it < iterations; it++) {
            float[] cv = new float[batch * cond];
            float[] real = new float[batch * repr];
            int[] col = new int[batch];
            int[] cat = new int[batch];
            for (int b = 0; b < batch; b++) {
                int c = r.nextInt(categories.length);
                double[] w = new double[categories[c]];
                double s = 0;
                for (int k = 0; k < w.length; k++) {
                    w[k] = frequency[c][k] > 0 ? Math.log(1 + frequency[c][k]) : 0;
                    s += w[k];
                }
                double u = r.nextDouble() * s;
                int k = 0;
                while (k < w.length - 1 && (u -= w[k]) > 0) {
                    k++;
                }
                List<Integer> pool = rowsWith.get(c).get(k);
                int row = pool.get(r.nextInt(pool.size()));
                col[b] = c;
                cat[b] = k;
                cv[b * cond + condOffset(c) + k] = 1;
                double[] enc = encodeRow(table, row, r);
                for (int q = 0; q < repr; q++) {
                    real[b * repr + q] = (float) enc[q];
                }
            }
            Var c = Var.of(Tensor.of(Shape.of(batch, cond), cv));
            Var fake = generate(Var.of(Tensor.normal(Shape.of(batch, noise), 0, 1, r)), c);
            Var dReal = critic(Var.of(Tensor.of(Shape.of(batch, repr), real)), c);
            Var dFake = critic(Var.of(fake.value()), c);
            dReal.neg().softplus().mean().add(dFake.softplus().mean()).backward();
            dOpt.step(discriminator.parameters());
            for (Var p : generator.parameters()) {
                p.zeroGrad();
            }
            Var fake2 = generate(Var.of(Tensor.normal(Shape.of(batch, noise), 0, 1, r)), c);
            Var gLoss = critic(fake2, c).neg().softplus().mean();
            float[] pick = new float[batch * repr];
            for (int b = 0; b < batch; b++) {
                pick[b * repr + catOffset(col[b]) + cat[b]] = 1;
            }
            Var condCe = fake2.add(1e-6f).log().mul(Var.of(Tensor.of(Shape.of(batch, repr), pick))).sum(1, false)
                    .mean().neg();
            gLoss.add(condCe).backward();
            gOpt.step(generator.parameters());
            for (Var p : discriminator.parameters()) {
                p.zeroGrad();
            }
        }
    }

    /**
     * Samples rows, with condition vectors drawn by the real category frequencies.
     *
     * @param rows rows to generate
     * @param seed the seed
     * @return the synthetic table
     */
    public MixedTable sample(int rows, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] cv = new float[rows * cond];
        for (int i = 0; i < rows; i++) {
            int c = r.nextInt(categories.length);
            double s = 0;
            for (double f : frequency[c]) {
                s += f;
            }
            double u = r.nextDouble() * s;
            int k = 0;
            while (k < categories[c] - 1 && (u -= frequency[c][k]) > 0) {
                k++;
            }
            cv[i * cond + condOffset(c) + k] = 1;
        }
        float[] g = Var.noGrad(() -> generate(Var.of(Tensor.normal(Shape.of(rows, noise), 0, 1, r)),
                Var.of(Tensor.of(Shape.of(rows, cond), cv)))).value().toArray();
        double[][] x = new double[rows][numeric];
        int[][] cat = new int[rows][categories.length];
        for (int i = 0; i < rows; i++) {
            int k = 0;
            for (int j = 0; j < numeric; j++) {
                double a = g[i * repr + k];
                int mode = 0;
                for (int m = 1; m < modes; m++) {
                    mode = g[i * repr + k + 1 + m] > g[i * repr + k + 1 + mode] ? m : mode;
                }
                x[i][j] = a * 4 * sigma[j][mode] + mu[j][mode];
                k += 1 + modes;
            }
            for (int c = 0; c < categories.length; c++) {
                int best = 0;
                for (int q = 1; q < categories[c]; q++) {
                    best = g[i * repr + k + q] > g[i * repr + k + best] ? q : best;
                }
                cat[i][c] = best;
                k += categories[c];
            }
        }
        return new MixedTable(x, cat, categories.clone());
    }

    @Override
    public String toString() {
        return "ConditionalTabularGan(" + numeric + " numeric, " + categories.length + " categorical)";
    }
}
