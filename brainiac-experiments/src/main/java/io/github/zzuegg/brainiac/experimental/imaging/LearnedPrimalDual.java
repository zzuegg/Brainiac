package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.UNet2d;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Learned tomographic reconstruction from sinograms.
 *
 * <ul>
 *   <li>{@link Variant#LEARNED_PRIMAL_DUAL} (Adler &amp; Öktem, IEEE TMI 2018,
 *       <a href="https://arxiv.org/abs/1707.06474">arXiv:1707.06474</a>): a primal-dual hybrid
 *       gradient scheme unrolled for a fixed number of iterations in which both proximal steps
 *       are small CNNs ({@link ImagingCnn}) with their own weights per iteration; the dual
 *       update sees the dual memory, the forward projection of the primal and the data, the
 *       primal update sees the primal memory and the back-projection of the dual, both through
 *       the exact {@link RadonTransform} and its adjoint as differentiable operations. The primal
 *       memory starts from the filtered back-projection (the paper starts from zero), which
 *       shortens training a lot at the small budgets used here.</li>
 *   <li>{@link Variant#FBP_CONVNET} (Jin et al., IEEE TIP 2017,
 *       <a href="https://arxiv.org/abs/1611.03679">arXiv:1611.03679</a>): filtered
 *       back-projection followed by a residual {@link UNet2d} post-processor.</li>
 * </ul>
 * The operator is scaled to unit norm inside the network so the CNNs see values of order one.
 */
public final class LearnedPrimalDual {

    /** The architectures. */
    public enum Variant {
        /** Unrolled primal-dual with CNN updates in both domains. */
        LEARNED_PRIMAL_DUAL,
        /** FBP followed by a residual U-Net. */
        FBP_CONVNET
    }

    private final Variant variant;
    private final RadonTransform radon;
    private final double scale;
    private final int primal;
    private final int dual;
    private final List<ImagingCnn> primalNets = new ArrayList<>();
    private final List<ImagingCnn> dualNets = new ArrayList<>();
    private final UNet2d unet;

    /**
     * Creates an untrained network.
     *
     * @param variant the architecture
     * @param radon the forward operator
     * @param iterations unrolled iterations (learned primal-dual)
     * @param memory primal and dual memory channels (learned primal-dual)
     * @param hidden hidden channels of each CNN (U-Net base channels for FBPConvNet)
     * @param seed the seed
     */
    public LearnedPrimalDual(Variant variant, RadonTransform radon, int iterations, int memory, int hidden,
            long seed) {
        this.variant = variant;
        this.radon = radon;
        this.scale = 1 / Math.sqrt(radon.normSquared(15));
        this.primal = memory;
        this.dual = memory;
        SplittableRandom r = new SplittableRandom(seed);
        if (variant == Variant.LEARNED_PRIMAL_DUAL) {
            for (int i = 0; i < iterations; i++) {
                dualNets.add(new ImagingCnn(dual + 2, hidden, dual, 3, false, r));
                primalNets.add(new ImagingCnn(primal + 1, hidden, primal, 3, false, r));
            }
            unet = null;
        } else {
            unet = new UNet2d(1, 1, hidden, 2, true, 0, r);
        }
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        if (unet != null) {
            p.addAll(unet.parameters());
        }
        for (int i = 0; i < primalNets.size(); i++) {
            p.addAll(dualNets.get(i).parameters());
            p.addAll(primalNets.get(i).parameters());
        }
        return p;
    }

    /** @return the number of trainable values */
    public long parameterCount() {
        long n = 0;
        for (Var v : parameters()) {
            n += v.value().size();
        }
        return n;
    }

    /**
     * Reconstructs a batch.
     *
     * @param sinograms {@code [b, 1, angles, side]}
     * @return images {@code [b, 1, side, side]}
     */
    public Var forward(Tensor sinograms) {
        int b = sinograms.shape().dim(0);
        int side = radon.side();
        if (variant == Variant.FBP_CONVNET) {
            Var fbp = Var.of(radon.fbp(sinograms, RadonTransform.Filter.RAM_LAK));
            return fbp.add(unet.forward(fbp));
        }
        float s = (float) scale;
        Var g = Var.of(sinograms.mul(s));
        Tensor fbp = radon.fbp(sinograms, RadonTransform.Filter.SHEPP_LOGAN);
        Tensor[] copies = new Tensor[primal];
        java.util.Arrays.fill(copies, fbp);
        Var h = Var.of(Tensor.concat(1, copies));
        Var f = Var.of(Tensor.zeros(Shape.of(b, dual, radon.angles(), side)));
        for (int i = 0; i < primalNets.size(); i++) {
            Var proj = radon.apply(h.slice(1, primal > 1 ? 1 : 0, primal > 1 ? 2 : 1)).mul(s);
            f = f.add(dualNets.get(i).forward(Var.concat(1, f, proj, g)));
            Var back = radon.adjoint(f.slice(1, 0, 1)).mul(s);
            h = h.add(primalNets.get(i).forward(Var.concat(1, h, back)));
        }
        return h.slice(1, 0, 1);
    }

    /**
     * Trains on image-sinogram pairs with the mean squared error.
     *
     * @param images clean images {@code [n, 1, side, side]}
     * @param sinograms their measurements {@code [n, 1, angles, side]}
     * @param steps gradient steps
     * @param batch examples per step
     * @param learningRate Adam's rate
     * @param seed the seed of the batches
     * @return the last loss
     */
    public double fit(Tensor images, Tensor sinograms, int steps, int batch, float learningRate, long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        int n = images.shape().dim(0);
        double last = 0;
        List<Var> params = parameters();
        for (int s = 0; s < steps; s++) {
            int start = r.nextInt(Math.max(1, n - batch + 1));
            Tensor x = images.slice(0, start, start + batch);
            Tensor y = sinograms.slice(0, start, start + batch);
            Var loss = forward(y).sub(Var.of(x)).square().mean();
            last = loss.value().item();
            loss.backward();
            opt.step(params);
        }
        return last;
    }

    /**
     * Reconstructs without gradients.
     *
     * @param sinograms {@code [b, 1, angles, side]}
     * @return images
     */
    public Tensor reconstruct(Tensor sinograms) {
        return Var.noGrad(() -> forward(sinograms)).value();
    }
}
