package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * VeRA, vector-based random matrix adaptation (Kopiczko et al. 2023,
 * <a href="https://arxiv.org/abs/2310.11454">arXiv:2310.11454</a>), around a frozen core
 * {@link Dense}: {@code y = act(x W0 + b0 + ((x A) * d) B * b)}. The low-rank matrices
 * {@code A [in, r]} and {@code B [r, out]} are <em>frozen random</em> and shared by every
 * adapted layer (sliced from one pair, regenerated from a seed, so they cost no storage);
 * only the scaling vectors {@code d [r]} (start 0.1) and {@code b [out]} (start 0) train.
 * That is about {@code r + out} values per layer instead of LoRA's {@code r (in + out)},
 * so a high rank is cheap.
 */
public final class VeraDense implements Layer {

    /** The frozen random pair shared by every adapter made from it. */
    public static final class Shared {
        private final Tensor a;
        private final Tensor b;

        /**
         * @param maxIn the widest input of any adapted layer
         * @param maxOut the widest output
         * @param rank the shared rank
         * @param seed regenerates the matrices
         */
        public Shared(int maxIn, int maxOut, int rank, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            float ba = (float) Math.sqrt(6.0 / maxIn);
            float bb = (float) Math.sqrt(6.0 / rank);
            this.a = Tensor.uniform(Shape.of(maxIn, rank), -ba, ba, r);
            this.b = Tensor.uniform(Shape.of(rank, maxOut), -bb, bb, r);
        }

        /**
         * @param base the layer to adapt
         * @return a VeRA adapter around it
         */
        public VeraDense adapt(Dense base) {
            return new VeraDense(base, this);
        }
    }

    private final Dense base;
    private final Tensor a;
    private final Tensor b;
    private final Var d;
    private final Var scale;

    private VeraDense(Dense base, Shared shared) {
        this.base = base;
        Tensor w = base.weight().value();
        int in = w.shape().dim(0);
        int out = w.shape().dim(1);
        int rank = shared.a.shape().dim(1);
        this.a = shared.a.slice(0, 0, in);
        this.b = shared.b.slice(1, 0, out);
        this.d = Var.param(Tensor.full(Shape.of(rank), 0.1f));
        this.scale = Var.param(Tensor.zeros(Shape.of(out)));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var w0 = Var.of(base.weight().value());
        Var y = input.matmul(w0).add(base.bias().value());
        Var delta = input.matmul(Var.of(a)).mul(d).matmul(Var.of(b)).mul(scale);
        return base.activation().forward(y.add(delta), training);
    }

    @Override
    public List<Var> parameters() {
        return List.of(d, scale);
    }

    @Override
    public int outputSize(int inputSize) {
        return base.outputSize(inputSize);
    }

    @Override
    public String toString() {
        return "VeraDense(" + base + ", rank " + d.shape().dim(0) + ")";
    }
}
