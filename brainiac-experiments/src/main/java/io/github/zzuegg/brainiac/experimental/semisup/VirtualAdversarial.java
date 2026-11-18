package io.github.zzuegg.brainiac.experimental.semisup;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Virtual adversarial training (Miyato, Maeda, Koyama, Ishii, TPAMI 2018,
 * <a href="https://arxiv.org/abs/1704.03976">arXiv:1704.03976</a>): the worst-case
 * augmentation instead of a random one. For each input it estimates, by power iteration on a
 * finite-difference Hessian-vector product, the perturbation {@code r} of norm {@code epsilon}
 * that changes the prediction most in KL divergence, and penalises {@code KL(p(x) || p(x + r))}.
 * It needs no labels and no hand-designed augmentations, so it also applies to tabular data; an
 * entropy-minimisation term ({@link SoftTargets#entropy}) is usually added (VAT+EntMin).
 */
public final class VirtualAdversarial {

    private final float xi;
    private final float epsilon;
    private final int powerIterations;
    private final float entropyWeight;

    /**
     * Creates the regulariser.
     *
     * @param xi the finite-difference step of the power iteration, such as 1e-6 to 10 (scaled by
     *     the input scale)
     * @param epsilon the norm of the adversarial perturbation per row
     * @param powerIterations power-iteration steps, 1 in the paper
     * @param entropyWeight the weight of the entropy-minimisation term, 0 for plain VAT
     */
    public VirtualAdversarial(float xi, float epsilon, int powerIterations, float entropyWeight) {
        this.xi = xi;
        this.epsilon = epsilon;
        this.powerIterations = powerIterations;
        this.entropyWeight = entropyWeight;
    }

    /**
     * Finds the virtual adversarial perturbation of a batch.
     *
     * @param model maps {@code [batch, features]} inputs to logits
     * @param x the inputs
     * @param cleanLogits the model's logits on {@code x}, the target distribution
     * @param random the source of the random start direction
     * @return {@code r_adv}, one row of norm {@code epsilon} per input
     */
    public Tensor perturbation(Function<Var, Var> model, Tensor x, Tensor cleanLogits, SplittableRandom random) {
        Tensor p = SoftTargets.tensor(SoftTargets.probabilities(cleanLogits, 1f));
        int n = x.shape().dim(0);
        int d = Math.toIntExact(x.size() / n);
        float[] dir = new float[n * d];
        for (int i = 0; i < dir.length; i++) {
            dir[i] = (float) random.nextGaussian();
        }
        normalise(dir, n, d, 1f);
        for (int it = 0; it < powerIterations; it++) {
            float[] scaled = dir.clone();
            for (int i = 0; i < scaled.length; i++) {
                scaled[i] *= xi;
            }
            Var r = Var.param(Tensor.of(x.shape(), scaled));
            Var kl = SoftTargets.kl(p, model.apply(Var.of(x).add(r)));
            dir = Var.gradients(kl, List.of(r), false).get(0).value().toArray();
            normalise(dir, n, d, 1f);
        }
        normalise(dir, n, d, epsilon);
        return Tensor.of(x.shape(), dir);
    }

    /**
     * The VAT loss (plus the entropy term if configured).
     *
     * @param model maps inputs to logits
     * @param x unlabelled (or all) inputs
     * @param random the random source
     * @return {@code KL(p(x) || p(x + r_adv)) + entropyWeight * H(p(x))}, differentiable in the model
     */
    public Var loss(Function<Var, Var> model, Tensor x, SplittableRandom random) {
        Var clean = model.apply(Var.of(x));
        Tensor rAdv = perturbation(model, x, clean.value(), random);
        Tensor p = SoftTargets.tensor(SoftTargets.probabilities(clean.value(), 1f));
        Var out = SoftTargets.kl(p, model.apply(Var.of(x.add(rAdv))));
        if (entropyWeight > 0) {
            out = out.add(SoftTargets.entropy(clean).mul(entropyWeight));
        }
        return out;
    }

    private static void normalise(float[] v, int n, int d, float norm) {
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < d; j++) {
                s += v[i * d + j] * v[i * d + j];
            }
            float f = (float) (norm / Math.sqrt(s + 1e-12));
            for (int j = 0; j < d; j++) {
                v[i * d + j] *= f;
            }
        }
    }
}
