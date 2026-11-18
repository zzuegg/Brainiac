package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Deep embedded clustering: DEC (Xie, Girshick and Farhadi, ICML 2016) and IDEC (Guo et al.,
 * IJCAI 2017).
 *
 * <p>An autoencoder is pretrained on the data; k-means in its latent space gives the initial
 * cluster centres {@code mu_j}. Then encoder and centres are refined together: each latent
 * {@code z_i} gets a soft assignment from a Student-t kernel,
 * {@code q_ij ∝ (1 + |z_i - mu_j|^2 / alpha)^-((alpha + 1) / 2)} ({@link #softAssignment}),
 * and the loss is {@code KL(P || Q)} to a sharpened target
 * {@code p_ij ∝ q_ij^2 / f_j} with cluster frequencies {@code f_j = sum_i q_ij}
 * ({@link #targetDistribution}), recomputed every few steps on the whole data set. The target
 * emphasises confident assignments and normalises cluster sizes, a self-training loop. DEC
 * drops the decoder for refinement; IDEC keeps the reconstruction loss,
 * {@code L = L_rec + gamma KL}, so the latent space keeps the data's local structure and is
 * not distorted by early wrong assignments.
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1511.06335">arXiv:1511.06335</a>;
 * <a href="https://doi.org/10.24963/ijcai.2017/243">doi:10.24963/ijcai.2017/243</a>.
 */
public final class DeepEmbeddedClustering {

    /** Which refinement. */
    public enum Mode {
        /** KL only (the decoder is dropped). */
        DEC,
        /** KL plus reconstruction. */
        IDEC
    }

    /**
     * Training settings.
     *
     * @param hidden the autoencoder's hidden width
     * @param latent the latent width
     * @param pretrainSteps autoencoder steps
     * @param refineSteps clustering steps
     * @param batch rows per step
     * @param gamma IDEC's weight of the clustering loss (0.1 in the paper)
     * @param learningRate Adam's learning rate
     * @param targetInterval steps between target-distribution updates
     */
    public record Settings(int hidden, int latent, int pretrainSteps, int refineSteps, int batch, float gamma,
            float learningRate, int targetInterval) {

        /**
         * Small defaults for 8x8 images.
         *
         * @return 64 hidden, 10 latent, 400 pretraining and 300 refinement steps of 64 rows
         */
        public static Settings small() {
            return new Settings(64, 10, 400, 300, 64, 0.1f, 3e-3f, 20);
        }
    }

    /**
     * A fitted clustering.
     *
     * @param labels the final hard assignments
     * @param initialLabels k-means on the pretrained latents (DEC's starting point, the baseline)
     * @param latents the final latents {@code [n][latent]}
     */
    public record Result(int[] labels, int[] initialLabels, double[][] latents) {
    }

    private DeepEmbeddedClustering() {
    }

    /**
     * The Student-t soft assignment of latents to centres.
     *
     * @param z latents {@code [n, d]}
     * @param centres {@code [k, d]}
     * @param alpha the degrees of freedom, 1 in the paper
     * @return {@code q [n, k]}, rows summing to one
     */
    public static Var softAssignment(Var z, Var centres, float alpha) {
        Var zz = z.square().sum(1, true);
        Var cc = centres.square().sum(1, true).transpose();
        Var d2 = zz.add(cc).sub(z.matmul(centres.transpose()).mul(2f)).maximum(0f);
        Var kernel = d2.div(alpha).add(1f).log().mul(-(alpha + 1) / 2).exp();
        return kernel.div(kernel.sum(1, true));
    }

    /**
     * DEC's sharpened target {@code p_ij = (q_ij^2 / f_j) / sum_j' (q_ij'^2 / f_j')}.
     *
     * @param q soft assignments {@code [n, k]}
     * @return the target, same shape
     */
    public static Tensor targetDistribution(Tensor q) {
        Tensor w = q.square().div(q.sum(0, true));
        return w.div(w.sum(1, true));
    }

    /**
     * Pretrains the autoencoder, initialises centres by k-means and refines.
     *
     * @param x the data {@code [n][d]}
     * @param k the number of clusters
     * @param mode DEC or IDEC
     * @param s the settings
     * @param seed the seed
     * @return the clustering
     */
    public static Result fit(float[][] x, int k, Mode mode, Settings s, long seed) {
        int n = x.length;
        int d = x[0].length;
        float[] flat = new float[n * d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(x[i], 0, flat, i * d, d);
        }
        Tensor all = Tensor.of(Shape.of(n, d), flat);
        Model encoder = Model.sequential().input(d).dense(s.hidden(), Activation.RELU)
                .dense(s.latent(), Activation.IDENTITY).seed(seed).build();
        Model decoder = Model.sequential().input(s.latent()).dense(s.hidden(), Activation.RELU)
                .dense(d, Activation.IDENTITY).seed(seed + 1).build();
        List<Var> ae = new ArrayList<>(encoder.parameters());
        ae.addAll(decoder.parameters());
        Optimizer opt = Optimizer.adam(s.learningRate());
        SplittableRandom r = new SplittableRandom(seed + 2);
        for (int step = 0; step < s.pretrainSteps(); step++) {
            Tensor b = batch(flat, n, d, s.batch(), r, null);
            Var loss = decoder.forward(encoder.forward(Var.of(b))).sub(b).square().mean();
            loss.backward();
            opt.step(ae);
        }
        double[][] latents = latents(encoder, all);
        KMeans init = KMeans.fit(latents, k, KMeans.Init.PLUS_PLUS, 10, 100, seed + 3);
        float[] c = new float[k * s.latent()];
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < s.latent(); l++) {
                c[j * s.latent() + l] = (float) init.centres()[j][l];
            }
        }
        Var centres = Var.param(Tensor.of(Shape.of(k, s.latent()), c));
        List<Var> trained = new ArrayList<>(encoder.parameters());
        trained.add(centres);
        if (mode == Mode.IDEC) {
            trained.addAll(decoder.parameters());
        }
        Optimizer refine = Optimizer.adam(s.learningRate() / 3);
        Tensor target = null;
        for (int step = 0; step < s.refineSteps(); step++) {
            if (step % s.targetInterval() == 0) {
                target = targetDistribution(
                        Var.noGrad(() -> softAssignment(encoder.forward(Var.of(all)), centres, 1f)).value());
            }
            int[] rows = new int[s.batch()];
            Tensor b = batch(flat, n, d, s.batch(), r, rows);
            Var z = encoder.forward(Var.of(b));
            Var q = softAssignment(z, centres, 1f);
            Tensor p = rowsOf(target, rows);
            Var kl = Var.of(p.mul(p.add(1e-10f).log())).sub(q.add(1e-10f).log().mul(p)).sum(1, false).mean();
            Var loss = mode == Mode.IDEC ? decoder.forward(z).sub(b).square().mean().add(kl.mul(s.gamma())) : kl;
            loss.backward();
            refine.step(trained);
        }
        Tensor q = Var.noGrad(() -> softAssignment(encoder.forward(Var.of(all)), centres, 1f)).value();
        return new Result(q.argmax(1).toIntArray(), init.labels(), latents(encoder, all));
    }

    private static double[][] latents(Model encoder, Tensor all) {
        Tensor z = Var.noGrad(() -> encoder.forward(Var.of(all))).value();
        return MlMatrices.matrix(z);
    }

    private static Tensor batch(float[] flat, int n, int d, int size, SplittableRandom r, int[] rows) {
        float[] b = new float[size * d];
        for (int i = 0; i < size; i++) {
            int row = r.nextInt(n);
            if (rows != null) {
                rows[i] = row;
            }
            System.arraycopy(flat, row * d, b, i * d, d);
        }
        return Tensor.of(Shape.of(size, d), b);
    }

    private static Tensor rowsOf(Tensor t, int[] rows) {
        int k = t.shape().dim(1);
        float[] v = t.toArray();
        float[] out = new float[rows.length * k];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(v, rows[i] * k, out, i * k, k);
        }
        return Tensor.of(Shape.of(rows.length, k), out);
    }
}
