package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Unrolled model-based networks for undersampled MRI, trained end to end through the
 * {@link MriOperator}.
 *
 * <ul>
 *   <li>{@link Variant#MODL} (Aggarwal, Mani &amp; Jacob, IEEE TMI 2019,
 *       <a href="https://arxiv.org/abs/1712.02862">arXiv:1712.02862</a>): alternate one CNN
 *       denoiser {@code z = D(x)}, shared across iterations, with the data-consistency step
 *       {@code x = argmin |A x - y|^2 + mu |x - z|^2}, solved by a few conjugate-gradient
 *       iterations written in autograd operations (so the learned {@code mu} and the denoiser
 *       get gradients through the solver).</li>
 *   <li>{@link Variant#VARNET} (Hammernik et al., MRM 2018; Sriram et al., MICCAI 2020,
 *       <a href="https://arxiv.org/abs/2004.06688">arXiv:2004.06688</a>): unrolled gradient
 *       steps {@code x - eta_t A^T (A x - y) - R_t(x)} with a learned soft data-consistency
 *       weight {@code eta_t} and an untied CNN regulariser per step; because the transform is
 *       unitary, the image-domain step equals the end-to-end variational network's k-space soft
 *       data consistency {@code k - eta_t M (k - y)}.</li>
 * </ul>
 * Both take the operator at call time, so they can be tested on a different sampling mask than
 * they were trained with.
 */
public final class UnrolledMri {

    /** The architectures. */
    public enum Variant {
        /** Shared denoiser with conjugate-gradient data consistency. */
        MODL,
        /** Unrolled gradient steps with learned regularisers and soft data consistency. */
        VARNET
    }

    private final Variant variant;
    private final int iterations;
    private final int cgIterations;
    private final List<ImagingCnn> nets = new ArrayList<>();
    private final List<Var> weights = new ArrayList<>();

    /**
     * Creates an untrained network.
     *
     * @param variant the architecture
     * @param iterations unrolled iterations
     * @param cgIterations conjugate-gradient iterations per data step (MoDL)
     * @param hidden hidden channels of the CNNs
     * @param seed the seed
     */
    public UnrolledMri(Variant variant, int iterations, int cgIterations, int hidden, long seed) {
        this.variant = variant;
        this.iterations = iterations;
        this.cgIterations = cgIterations;
        SplittableRandom r = new SplittableRandom(seed);
        int nNets = variant == Variant.MODL ? 1 : iterations;
        for (int i = 0; i < nNets; i++) {
            nets.add(new ImagingCnn(1, hidden, 1, 3, variant == Variant.MODL, r));
        }
        int nWeights = variant == Variant.MODL ? 1 : iterations;
        for (int i = 0; i < nWeights; i++) {
            weights.add(Var.param(Tensor.scalar(0.5f)));
        }
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (ImagingCnn n : nets) {
            p.addAll(n.parameters());
        }
        p.addAll(weights);
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

    private static Var dot(Var a, Var b) {
        return a.mul(b).sum(3, true).sum(2, true).sum(1, true);
    }

    /**
     * Conjugate gradients for {@code (A^T A + mu I) x = b} in autograd operations.
     *
     * @param op the operator
     * @param b the right-hand side {@code [batch, 1, h, w]}
     * @param mu the shift (a scalar)
     * @param x0 the start
     * @param steps the iterations
     * @return the approximate solution
     */
    public static Var conjugateGradient(MriOperator op, Var b, Var mu, Var x0, int steps) {
        Var x = x0;
        Var r = b.sub(op.normal(x).add(x.mul(mu)));
        Var p = r;
        Var rr = dot(r, r);
        for (int k = 0; k < steps; k++) {
            Var ap = op.normal(p).add(p.mul(mu));
            Var alpha = rr.div(dot(p, ap).add(1e-12f));
            x = x.add(p.mul(alpha));
            r = r.sub(ap.mul(alpha));
            Var next = dot(r, r);
            p = r.add(p.mul(next.div(rr.add(1e-12f))));
            rr = next;
        }
        return x;
    }

    /**
     * Reconstructs a batch.
     *
     * @param op the sampling operator
     * @param k masked k-space {@code [b, side, side, 2]}
     * @return images {@code [b, 1, side, side]}
     */
    public Var forward(MriOperator op, Tensor k) {
        Var y = Var.of(k);
        Var atb = op.adjoint(y);
        Var x = atb;
        if (variant == Variant.MODL) {
            Var mu = weights.get(0).softplus();
            for (int i = 0; i < iterations; i++) {
                Var z = nets.get(0).forward(x);
                x = conjugateGradient(op, atb.add(z.mul(mu)), mu, x, cgIterations);
            }
            return x;
        }
        for (int i = 0; i < iterations; i++) {
            Var eta = weights.get(i).softplus();
            Var grad = op.normal(x).sub(atb);
            Var reg = nets.get(i).forward(x);
            x = x.sub(grad.mul(eta)).sub(reg);
        }
        return x;
    }

    /**
     * Trains on image and k-space pairs with the mean squared error.
     *
     * @param op the operator
     * @param images {@code [n, 1, side, side]}
     * @param kspace {@code [n, side, side, 2]}
     * @param steps gradient steps
     * @param batch examples per step
     * @param learningRate Adam's rate
     * @param seed the seed of the batches
     * @return the last loss
     */
    public double fit(MriOperator op, Tensor images, Tensor kspace, int steps, int batch, float learningRate,
            long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        int n = images.shape().dim(0);
        List<Var> params = parameters();
        double last = 0;
        for (int s = 0; s < steps; s++) {
            int start = r.nextInt(Math.max(1, n - batch + 1));
            Var loss = forward(op, kspace.slice(0, start, start + batch)).sub(Var.of(images.slice(0, start,
                    start + batch))).square().mean();
            last = loss.value().item();
            loss.backward();
            opt.step(params);
        }
        return last;
    }

    /**
     * Reconstructs without gradients.
     *
     * @param op the operator
     * @param k k-space {@code [b, side, side, 2]}
     * @return images
     */
    public Tensor reconstruct(MriOperator op, Tensor k) {
        return Var.noGrad(() -> forward(op, k)).value();
    }

    /**
     * Stacks flat measurements of one operator into a batch.
     *
     * @param op the operator
     * @param measurements flat {@code [re, im]} measurements
     * @return {@code [n, side, side, 2]}
     */
    public static Tensor stack(MriOperator op, List<double[]> measurements) {
        Tensor[] parts = new Tensor[measurements.size()];
        for (int i = 0; i < parts.length; i++) {
            parts[i] = op.pack(measurements.get(i));
        }
        return parts.length == 1 ? parts[0] : Tensor.concat(0, parts);
    }
}
