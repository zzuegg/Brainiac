package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Flows;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A masked autoregressive flow (MAF, Papamakarios et al.,
 * <a href="https://arxiv.org/abs/1705.07057">arXiv:1705.07057</a>): a stack of affine
 * autoregressive transforms {@code u_i = (x_i - mu_i(x_<i)) exp(-alpha_i(x_<i))}, each
 * computed in one pass by a MADE network (Germain et al., 2015) whose weight masks only let
 * output {@code i} see inputs before {@code i} in the layer's order. Density evaluation is
 * one parallel pass; sampling needs one pass per dimension. Every dimension is conditioned
 * on all earlier ones, a richer dependency than a coupling layer's split halves. The order is
 * reversed between layers.
 */
public final class MaskedAutoregressiveFlow implements DensityFlow {
    private final int dim;
    private final List<Made> layers = new ArrayList<>();

    private MaskedAutoregressiveFlow(int dim, int count, int hidden, long seed) {
        if (dim < 2 || count < 1 || hidden < dim - 1) {
            throw new IllegalArgumentException("MAF needs dim >= 2, at least one layer and hidden >= dim - 1");
        }
        this.dim = dim;
        SplittableRandom random = new SplittableRandom(seed);
        for (int l = 0; l < count; l++) {
            int[] order = new int[dim];
            for (int i = 0; i < dim; i++) {
                order[i] = l % 2 == 0 ? i : dim - 1 - i;
            }
            layers.add(new Made(dim, hidden, order, random));
        }
    }

    /**
     * Creates the flow; zero-initialised output layers make it start as the identity.
     *
     * @param dim the data dimension, at least 2
     * @param layers the number of autoregressive layers
     * @param hidden the MADE hidden width, at least {@code dim - 1}
     * @param seed the initialization seed
     * @return the flow
     */
    public static MaskedAutoregressiveFlow of(int dim, int layers, int hidden, long seed) {
        return new MaskedAutoregressiveFlow(dim, layers, hidden, seed);
    }

    @Override
    public Flows.Result forward(Var x) {
        Var h = x;
        Var logDet = null;
        for (Made m : layers) {
            Var[] ma = m.apply(h);
            h = h.sub(ma[0]).mul(ma[1].neg().exp());
            Var ld = ma[1].sum(1, false).neg();
            logDet = logDet == null ? ld : logDet.add(ld);
        }
        return new Flows.Result(h, logDet);
    }

    @Override
    public Tensor inverse(Tensor z) {
        int b = z.shape().dim(0);
        Tensor u = z;
        for (int l = layers.size() - 1; l >= 0; l--) {
            Made m = layers.get(l);
            float[] uv = u.toArray();
            float[] x = new float[b * dim];
            for (int step = 0; step < dim; step++) {
                int i = m.order[step];
                Var[] ma = m.apply(Var.of(Tensor.of(Shape.of(b, dim), x.clone())));
                float[] mu = ma[0].value().toArray();
                float[] alpha = ma[1].value().toArray();
                for (int r = 0; r < b; r++) {
                    int k = r * dim + i;
                    x[k] = (float) (uv[k] * Math.exp(alpha[k]) + mu[k]);
                }
            }
            u = Tensor.of(Shape.of(b, dim), x);
        }
        return u;
    }

    @Override
    public int dimension() {
        return dim;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Made m : layers) {
            p.addAll(m.parameters());
        }
        return p;
    }

    /** A two-hidden-layer MADE emitting a shift and a log-scale per dimension. */
    private static final class Made {
        final int[] order;
        final Tensor m1;
        final Tensor m2;
        final Tensor m3;
        final Var w1;
        final Var b1;
        final Var w2;
        final Var b2;
        final Var wMu;
        final Var bMu;
        final Var wAlpha;
        final Var bAlpha;

        Made(int dim, int hidden, int[] order, SplittableRandom random) {
            this.order = order;
            int[] inDegree = new int[dim];
            for (int step = 0; step < dim; step++) {
                inDegree[order[step]] = step + 1;
            }
            int[] h1 = new int[hidden];
            int[] h2 = new int[hidden];
            for (int j = 0; j < hidden; j++) {
                h1[j] = 1 + j % Math.max(1, dim - 1);
                h2[j] = 1 + (j + 1) % Math.max(1, dim - 1);
            }
            m1 = mask(inDegree, h1, false);
            m2 = mask(h1, h2, false);
            m3 = mask(h2, inDegree, true);
            w1 = TechniqueOps.weight(random, dim, hidden);
            b1 = TechniqueOps.filled(0f, hidden);
            w2 = TechniqueOps.weight(random, hidden, hidden);
            b2 = TechniqueOps.filled(0f, hidden);
            wMu = TechniqueOps.filled(0f, hidden, dim);
            bMu = TechniqueOps.filled(0f, dim);
            wAlpha = TechniqueOps.filled(0f, hidden, dim);
            bAlpha = TechniqueOps.filled(0f, dim);
        }

        /** Connection masks: {@code out >= in} for hidden layers, {@code out > in} for the output. */
        private static Tensor mask(int[] in, int[] out, boolean strict) {
            float[] m = new float[in.length * out.length];
            for (int i = 0; i < in.length; i++) {
                for (int j = 0; j < out.length; j++) {
                    m[i * out.length + j] = (strict ? out[j] > in[i] : out[j] >= in[i]) ? 1 : 0;
                }
            }
            return Tensor.of(Shape.of(in.length, out.length), m);
        }

        Var[] apply(Var x) {
            Var h = GenerativeOps.silu(x.matmul(w1.mul(m1)).add(b1));
            h = GenerativeOps.silu(h.matmul(w2.mul(m2)).add(b2));
            Var mu = h.matmul(wMu.mul(m3)).add(bMu);
            Var alpha = h.matmul(wAlpha.mul(m3)).add(bAlpha).tanh().mul(3f);
            return new Var[] {mu, alpha};
        }

        List<Var> parameters() {
            return List.of(w1, b1, w2, b2, wMu, bMu, wAlpha, bAlpha);
        }
    }
}
