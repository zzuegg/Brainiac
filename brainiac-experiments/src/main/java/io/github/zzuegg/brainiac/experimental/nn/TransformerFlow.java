package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * TarFlow (Zhai et al. 2024, "Normalizing Flows are Capable Generative Models",
 * <a href="https://arxiv.org/abs/2412.06329">arXiv:2412.06329</a>): a normalizing flow made of
 * autoregressive blocks, each a causal transformer over the data's patches. Block {@code b}
 * maps patch {@code x_i} to {@code z_i = (x_i - mu_i) exp(-alpha_i)}, where
 * {@code mu_i, alpha_i} come from the transformer reading patches {@code < i} only, so the
 * Jacobian is triangular and {@code log det = -sum alpha}. Successive blocks reverse the
 * patch order. Sampling inverts each block patch by patch.
 *
 * <p>The paper's two tricks are included: train on data with Gaussian noise of scale
 * {@code sigma} added (the model is a density of noisy data), and after sampling remove the
 * noise with one score step, {@code x + sigma^2 grad log p(x)}, the gradient taken through
 * the flow by autograd.
 *
 * <p>Input {@code [batch, patches * patchSize]}, patches in order.
 */
public final class TransformerFlow {

    private final int patches;
    private final int patchSize;
    private final List<Model> blocks = new ArrayList<>();

    /**
     * @param patches the number of patches (sequence length)
     * @param patchSize values per patch
     * @param blocks flow blocks, each reversing the order of the previous
     * @param width the transformer width
     * @param layers transformer layers per block
     * @param heads attention heads
     * @param seed the seed
     */
    public TransformerFlow(int patches, int patchSize, int blocks, int width, int layers, int heads, long seed) {
        this.patches = patches;
        this.patchSize = patchSize;
        for (int b = 0; b < blocks; b++) {
            Model.Builder m = Model.sequential().input(patches, patchSize).dense(width).positionalEncoding();
            for (int l = 0; l < layers; l++) {
                m.add(TransformerBlock.of(heads).causal(true));
            }
            Model model = m.layerNorm().dense(2 * patchSize).seed(seed + b).build();
            List<Var> p = model.parameters();
            for (int i = p.size() - 2; i < p.size(); i++) {   // start as the identity map
                p.get(i).assign(Tensor.zeros(p.get(i).value().shape()));
            }
            this.blocks.add(model);
        }
    }

    /** @return every trainable parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Model m : blocks) {
            p.addAll(m.parameters());
        }
        return p;
    }

    /** @return the data dimension */
    public int dimension() {
        return patches * patchSize;
    }

    /** Shifts patches right by one so that position i sees patches before i. */
    private Var shifted(Var x, int batch) {
        Var zero = Var.of(Tensor.zeros(Shape.of(batch, 1, patchSize)));
        return Var.concat(1, zero, x.slice(1, 0, patches - 1));
    }

    private Var reversed(Var x) {
        Var[] parts = new Var[patches];
        for (int i = 0; i < patches; i++) {
            parts[i] = x.slice(1, patches - 1 - i, patches - i);
        }
        return Var.concat(1, parts);
    }

    /**
     * The exact log-density of each sample.
     *
     * @param x data {@code [batch, dimension]}
     * @return {@code [batch]} log-likelihoods
     */
    public Var logProb(Var x) {
        int batch = x.shape().dim(0);
        Var h = x.reshape(batch, patches, patchSize);
        Var logDet = null;
        for (int b = 0; b < blocks.size(); b++) {
            Var out = blocks.get(b).forward(shifted(h, batch), true);
            Var mu = out.slice(2, 0, patchSize);
            Var alpha = out.slice(2, patchSize, 2 * patchSize);
            h = h.sub(mu).mul(alpha.neg().exp());
            Var ld = alpha.reshape(batch, patches * patchSize).sum(-1, false).neg();
            logDet = logDet == null ? ld : logDet.add(ld);
            h = reversed(h);
        }
        int dim = dimension();
        Var z = h.reshape(batch, dim);
        Var logNormal = z.square().sum(-1, false).mul(-0.5f).sub((float) (0.5 * dim * Math.log(2 * Math.PI)));
        return logNormal.add(logDet);
    }

    /**
     * Draws samples by inverting every block patch by patch.
     *
     * @param n the number of samples
     * @param random the noise source
     * @return {@code [n, dimension]}
     */
    public Tensor sample(int n, RandomGenerator random) {
        float[] h = Tensor.normal(Shape.of(n, dimension()), 0, 1, random).toArray();
        for (int b = blocks.size() - 1; b >= 0; b--) {
            h = reverse(h, n);   // undo the reversal that followed block b
            float[] x = new float[h.length];
            for (int i = 0; i < patches; i++) {
                Var in = shifted(Var.of(Tensor.of(Shape.of(n, patches, patchSize), x)), n);
                float[] out = blocks.get(b).forward(in, false).value().toArray();
                for (int s = 0; s < n; s++) {
                    for (int k = 0; k < patchSize; k++) {
                        int o = (s * patches + i) * 2 * patchSize;
                        int at = (s * patches + i) * patchSize + k;
                        x[at] = (float) (h[at] * Math.exp(out[o + patchSize + k]) + out[o + k]);
                    }
                }
            }
            h = x;
        }
        return Tensor.of(Shape.of(n, dimension()), h);
    }

    private float[] reverse(float[] h, int n) {
        float[] out = new float[h.length];
        for (int s = 0; s < n; s++) {
            for (int i = 0; i < patches; i++) {
                System.arraycopy(h, (s * patches + i) * patchSize, out, (s * patches + patches - 1 - i) * patchSize,
                        patchSize);
            }
        }
        return out;
    }

    /**
     * One score-based denoising step, {@code x + sigma^2 grad log p(x)}: for a flow trained on
     * data with noise of scale {@code sigma}, this is Tweedie's estimate of the clean data.
     *
     * @param x noisy samples {@code [batch, dimension]}
     * @param sigma the training noise scale
     * @return the denoised samples
     */
    public Tensor denoise(Tensor x, float sigma) {
        Var v = Var.param(x);
        Var logp = logProb(v).sum();
        Tensor grad = Var.gradients(logp, List.of(v), false).get(0).value();
        for (Var p : parameters()) {
            p.zeroGrad();
        }
        return x.add(grad.mul(sigma * sigma));
    }
}
