package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A RealNVP normalizing flow over vectors {@code [batch, dim]}: an invertible stack of affine
 * coupling layers mapping data {@code x} to a standard normal {@code z}. Each coupling keeps
 * half the features ({@code mask = 1}) and transforms the other half as
 * {@code y = x * exp(s(x_kept)) + t(x_kept)}, with {@code s, t} from a small MLP; the
 * Jacobian is triangular, so the exact log-likelihood is
 * {@code log N(z) + sum(s)} and sampling is the inverse pass from Gaussian noise. Masks
 * alternate so every feature gets transformed. Trained by maximizing {@link #logProb}.
 *
 * <p>Paper: Dinh, Sohl-Dickstein, Bengio, "Density estimation using Real NVP" (2016),
 * <a href="https://arxiv.org/abs/1605.08803">arXiv:1605.08803</a>.
 */
public final class RealNvpFlow {

    /** One affine coupling layer. */
    private static final class Coupling {
        final Tensor mask;
        final Tensor inverseMask;
        final Var w1;
        final Var b1;
        final Var w2;
        final Var b2;
        final Var ws;
        final Var bs;
        final Var wt;
        final Var bt;
        final Var scale;

        Coupling(int dim, int hidden, float[] mask, RandomGenerator random) {
            this.mask = Tensor.of(Shape.of(dim), mask);
            this.inverseMask = this.mask.neg().add(1f);
            w1 = TechniqueOps.weight(random, dim, hidden);
            b1 = TechniqueOps.filled(0f, hidden);
            w2 = TechniqueOps.weight(random, hidden, hidden);
            b2 = TechniqueOps.filled(0f, hidden);
            ws = TechniqueOps.filled(0f, hidden, dim);
            bs = TechniqueOps.filled(0f, dim);
            wt = TechniqueOps.filled(0f, hidden, dim);
            bt = TechniqueOps.filled(0f, dim);
            scale = TechniqueOps.filled(1f, dim);
        }

        /** {@code s} (bounded with tanh and a learned scale) and {@code t}, zero on kept features. */
        Var[] st(Var kept) {
            Var h = kept.matmul(w1).add(b1).relu();
            h = h.matmul(w2).add(b2).relu();
            Var s = h.matmul(ws).add(bs).tanh().mul(scale).mul(inverseMask);
            Var t = h.matmul(wt).add(bt).mul(inverseMask);
            return new Var[] {s, t};
        }

        List<Var> parameters() {
            return List.of(w1, b1, w2, b2, ws, bs, wt, bt, scale);
        }
    }

    private final int dim;
    private final List<Coupling> couplings = new ArrayList<>();

    /**
     * Creates the flow; it starts as the identity.
     *
     * @param dim the data dimension, at least 2
     * @param layers the number of coupling layers (alternating masks), 4 to 8 typically
     * @param hidden the hidden width of each coupling's MLP
     * @param random the random source
     */
    public RealNvpFlow(int dim, int layers, int hidden, RandomGenerator random) {
        this.dim = dim;
        for (int l = 0; l < layers; l++) {
            float[] mask = new float[dim];
            for (int i = 0; i < dim; i++) {
                mask[i] = (i + l) % 2 == 0 ? 1f : 0f;
            }
            couplings.add(new Coupling(dim, hidden, mask, random));
        }
    }

    /**
     * Maps data to the latent space.
     *
     * @param x data {@code [batch, dim]}
     * @return {@code {z, logDet}}: the latents {@code [batch, dim]} and the log-determinant of
     *     the Jacobian per sample {@code [batch]}
     */
    public Var[] forward(Var x) {
        Var logDet = null;
        for (Coupling c : couplings) {
            Var kept = x.mul(c.mask);
            Var[] st = c.st(kept);
            x = kept.add(x.mul(c.inverseMask).mul(st[0].exp()).add(st[1]));
            Var ld = st[0].sum(-1, false);
            logDet = logDet == null ? ld : logDet.add(ld);
        }
        return new Var[] {x, logDet};
    }

    /**
     * The exact log-density of each sample.
     *
     * @param x data {@code [batch, dim]}
     * @return {@code [batch]} log-likelihoods
     */
    public Var logProb(Var x) {
        Var[] zl = forward(x);
        Var logNormal = zl[0].square().sum(-1, false).mul(-0.5f).sub((float) (0.5 * dim * Math.log(2 * Math.PI)));
        return logNormal.add(zl[1]);
    }

    /**
     * Maps latents back to data: the exact inverse of {@link #forward}.
     *
     * @param z latents {@code [batch, dim]}
     * @return data {@code [batch, dim]}
     */
    public Tensor inverse(Tensor z) {
        Var y = Var.of(z);
        for (int i = couplings.size() - 1; i >= 0; i--) {
            Coupling c = couplings.get(i);
            Var kept = y.mul(c.mask);
            Var[] st = c.st(kept);
            y = kept.add(y.sub(st[1]).mul(st[0].neg().exp()).mul(c.inverseMask));
        }
        return y.value();
    }

    /**
     * Draws samples.
     *
     * @param n the number of samples
     * @param random the noise source
     * @return {@code [n, dim]}
     */
    public Tensor sample(int n, RandomGenerator random) {
        return inverse(Tensor.normal(Shape.of(n, dim), 0, 1, random));
    }

    /** @return every trainable parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Coupling c : couplings) {
            p.addAll(c.parameters());
        }
        return p;
    }
}
