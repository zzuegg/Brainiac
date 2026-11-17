package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A variational autoencoder over binary vectors (Bernoulli decoder) with the three classic
 * refinements of the plain VAE:
 * <ul>
 *   <li><b>beta-VAE</b> (Higgins et al., 2017): the KL term weighted by {@code beta > 1}, trading
 *       reconstruction for a more factorised, disentangled code;</li>
 *   <li><b>IWAE</b> (Burda, Grosse and Salakhutdinov,
 *       <a href="https://arxiv.org/abs/1509.00519">arXiv:1509.00519</a>): the importance-weighted
 *       bound {@code log mean_k p(x, z_k) / q(z_k | x)} with {@code k} samples, tighter than the
 *       ELBO ({@code k = 1}) and, evaluated with large {@code k}, the standard test
 *       log-likelihood estimate ({@link #logLikelihood});</li>
 *   <li><b>ladder VAE</b> (Sonderby et al., <a href="https://arxiv.org/abs/1602.02282">arXiv:1602.02282</a>):
 *       two stochastic layers; the inference network's bottom-up Gaussian for the lower
 *       latent is merged, precision-weighted, with the generative top-down prior
 *       {@code p(z1 | z2)}, so both directions share the work and the top layer stays active.</li>
 * </ul>
 */
public final class HierarchicalVae {
    private static final float LOG_2PI = (float) Math.log(2 * Math.PI);
    private final int input;
    private final int top;
    private final int bottom;
    private final boolean ladder;
    private final GenerativeOps.Mlp up1;
    private final GenerativeOps.Mlp up2;
    private final GenerativeOps.Mlp prior1;
    private final GenerativeOps.Mlp decoder;

    private HierarchicalVae(int input, int hidden, int bottom, int top, boolean ladder, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.input = input;
        this.top = top;
        this.bottom = bottom;
        this.ladder = ladder;
        // bottom-up: x -> (mu, log var) of the lower latent and a hidden summary for the top one
        up1 = new GenerativeOps.Mlp(r, false, input, hidden, 2 * bottom + hidden);
        up2 = ladder ? new GenerativeOps.Mlp(r, false, hidden, hidden, 2 * top) : null;
        prior1 = ladder ? new GenerativeOps.Mlp(r, false, top, hidden, 2 * bottom) : null;
        decoder = new GenerativeOps.Mlp(r, false, bottom, hidden, hidden, input);
    }

    /**
     * A plain one-layer VAE (use it with {@code beta} for beta-VAE, or {@code k > 1} for IWAE).
     *
     * @param input the data dimension
     * @param hidden the hidden width
     * @param latent the latent dimension
     * @param seed the initialization seed
     * @return the model
     */
    public static HierarchicalVae plain(int input, int hidden, int latent, long seed) {
        return new HierarchicalVae(input, hidden, latent, 0, false, seed);
    }

    /**
     * A two-layer ladder VAE.
     *
     * @param input the data dimension
     * @param hidden the hidden width
     * @param bottom the lower latent dimension
     * @param top the upper latent dimension
     * @param seed the initialization seed
     * @return the model
     */
    public static HierarchicalVae ladder(int input, int hidden, int bottom, int top, long seed) {
        return new HierarchicalVae(input, hidden, bottom, top, true, seed);
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(up1.parameters());
        if (ladder) {
            p.addAll(up2.parameters());
            p.addAll(prior1.parameters());
        }
        p.addAll(decoder.parameters());
        return p;
    }

    /** log N(z; mu, exp(logVar)) summed over the last axis: {@code [n, 1]}. */
    private static Var logNormal(Var z, Var mu, Var logVar) {
        return z.sub(mu).square().div(logVar.exp()).add(logVar).add(LOG_2PI).sum(1, true).mul(-0.5f);
    }

    private static Var standardLogNormal(Var z) {
        return z.square().add(LOG_2PI).sum(1, true).mul(-0.5f);
    }

    private static Var sample(Var mu, Var logVar, RandomGenerator random) {
        return mu.add(logVar.mul(0.5f).exp().mul(Tensor.normal(mu.value().shape(), 0, 1, random)));
    }

    /**
     * One pass of encoder, sampling and decoder.
     *
     * @return {@code {log p(x | z), log p(z), log q(z | x), latent mean for probing}}, each {@code [n, 1]}
     *     but the last
     */
    private Var[] pass(Tensor x, RandomGenerator random) {
        Var xv = Var.of(x);
        Var h = up1.apply(xv);
        Var mu1 = h.slice(1, 0, bottom);
        Var lv1 = h.slice(1, bottom, 2 * bottom).clamp(-8f, 8f);
        Var logPrior;
        Var logPost;
        Var z1;
        Var probe;
        if (ladder) {
            Var d = GenerativeOps.silu(h.slice(1, 2 * bottom, h.shape().dim(1)));
            Var t = up2.apply(d);
            Var mu2 = t.slice(1, 0, top);
            Var lv2 = t.slice(1, top, 2 * top).clamp(-8f, 8f);
            Var z2 = sample(mu2, lv2, random);
            Var p = prior1.apply(z2);
            Var pmu = p.slice(1, 0, bottom);
            Var plv = p.slice(1, bottom, 2 * bottom).clamp(-8f, 8f);
            // precision-weighted merge of the bottom-up estimate and the top-down prior
            Var precQ = lv1.neg().exp();
            Var precP = plv.neg().exp();
            Var prec = precQ.add(precP);
            Var qmu = mu1.mul(precQ).add(pmu.mul(precP)).div(prec);
            Var qlv = prec.log().neg();
            z1 = sample(qmu, qlv, random);
            logPrior = standardLogNormal(z2).add(logNormal(z1, pmu, plv));
            logPost = logNormal(z2, mu2, lv2).add(logNormal(z1, qmu, qlv));
            probe = mu2;
        } else {
            z1 = sample(mu1, lv1, random);
            logPrior = standardLogNormal(z1);
            logPost = logNormal(z1, mu1, lv1);
            probe = mu1;
        }
        Var logits = decoder.apply(z1);
        Var logLik = xv.mul(logits).sub(logits.softplus()).sum(1, true);
        return new Var[] {logLik, logPrior, logPost, probe};
    }

    /**
     * The negative beta-weighted ELBO, averaged over the batch.
     *
     * @param x binary data {@code [n, input]} with entries 0 or 1
     * @param beta the KL weight; 1 for the plain VAE
     * @param random the sampling noise
     * @return the scalar loss
     */
    public Var elboLoss(Tensor x, float beta, RandomGenerator random) {
        Var[] p = pass(x, random);
        return p[0].neg().add(p[2].sub(p[1]).mul(beta)).mean();
    }

    /**
     * The negative importance-weighted bound with {@code k} samples per example, averaged
     * over the batch.
     *
     * @param x binary data {@code [n, input]}
     * @param k the number of importance samples
     * @param random the sampling noise
     * @return the scalar loss
     */
    public Var iwaeLoss(Tensor x, int k, RandomGenerator random) {
        int n = x.shape().dim(0);
        Var[] p = pass(repeat(x, k), random);
        Var logW = p[0].add(p[1]).sub(p[2]).reshape(k, n);
        Var m = Var.of(logW.value().max(0, true));
        Var lse = logW.sub(m).exp().sum(0, true).log().add(m).sub((float) Math.log(k));
        return lse.mean().neg();
    }

    /** Stacks {@code k} copies of the batch: row {@code j * n + i} is example {@code i}. */
    private static Tensor repeat(Tensor x, int k) {
        Tensor[] parts = new Tensor[k];
        java.util.Arrays.fill(parts, x);
        return Tensor.concat(0, parts);
    }

    /**
     * The IWAE estimate of {@code log p(x)} per example, the standard test log-likelihood.
     *
     * @param x binary data {@code [n, input]}
     * @param k importance samples, such as 500 or 5000
     * @param random the sampling noise
     * @return nats per example
     */
    public double[] logLikelihood(Tensor x, int k, RandomGenerator random) {
        int n = x.shape().dim(0);
        double[] max = new double[n];
        java.util.Arrays.fill(max, Double.NEGATIVE_INFINITY);
        List<float[]> chunks = new ArrayList<>();
        int chunk = Math.max(1, 4096 / n);
        for (int done = 0; done < k; done += chunk) {
            int c = Math.min(chunk, k - done);
            Var[] p = pass(repeat(x, c), random);
            float[] lw = p[0].add(p[1]).sub(p[2]).value().toArray();
            chunks.add(lw);
            for (int i = 0; i < lw.length; i++) {
                max[i % n] = Math.max(max[i % n], lw[i]);
            }
        }
        double[] sum = new double[n];
        for (float[] lw : chunks) {
            for (int i = 0; i < lw.length; i++) {
                sum[i % n] += Math.exp(lw[i] - max[i % n]);
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = max[i] + Math.log(sum[i] / k);
        }
        return out;
    }

    /**
     * The posterior means of the top latent layer (the only one for a plain VAE).
     *
     * @param x binary data {@code [n, input]}
     * @return {@code [n, latent]}
     */
    public Tensor encode(Tensor x) {
        return pass(x, new SplittableRandom(0))[3].value();
    }

    /**
     * Counts active units of the top layer: latents whose posterior mean varies across the
     * data with variance above 0.01 (Burda et al.'s criterion).
     *
     * @param x binary data
     * @return the number of active units
     */
    public int activeUnits(Tensor x) {
        float[] m = encode(x).toArray();
        int n = x.shape().dim(0);
        int d = m.length / n;
        int active = 0;
        for (int j = 0; j < d; j++) {
            double mean = 0;
            for (int i = 0; i < n; i++) {
                mean += m[i * d + j];
            }
            mean /= n;
            double var = 0;
            for (int i = 0; i < n; i++) {
                var += (m[i * d + j] - mean) * (m[i * d + j] - mean);
            }
            active += var / n > 0.01 ? 1 : 0;
        }
        return active;
    }

    /**
     * Decodes samples of the prior to Bernoulli means.
     *
     * @param n how many
     * @param random the noise source
     * @return {@code [n, input]} probabilities
     */
    public Tensor sample(int n, RandomGenerator random) {
        Var z;
        if (ladder) {
            Var z2 = Var.of(Tensor.normal(Shape.of(n, top), 0, 1, random));
            Var p = prior1.apply(z2);
            z = sample(p.slice(1, 0, bottom), p.slice(1, bottom, 2 * bottom).clamp(-8f, 8f), random);
        } else {
            z = Var.of(Tensor.normal(Shape.of(n, bottom), 0, 1, random));
        }
        return decoder.apply(z).value().sigmoid();
    }

    /** @return the data dimension */
    public int inputSize() {
        return input;
    }
}
