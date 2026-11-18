package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * PACT, parameterised clipping activation (Choi et al. 2018, <a href="https://arxiv.org/abs/1805.06085">arXiv:1805.06085</a>):
 * a ReLU clipped at a <em>learned</em> level {@code alpha} and quantised to {@code k} bits,
 *
 * <pre>
 * y = clip(x, 0, alpha),   y_q = round(y (2^k - 1) / alpha) alpha / (2^k - 1)
 * </pre>
 *
 * The rounding is passed straight through, and {@code dy/d alpha = 1[x >= alpha]}, so the clipping
 * level adapts to the bit width: a low level resolves small activations finely, a high one keeps
 * the outliers. It was the standard 4-bit activation recipe before LSQ. A {@link Layer} with a
 * {@link #spec} so it replaces a ReLU in a sequential model.
 */
public final class PactActivation implements Layer {

    private final int bits;
    private final Var alpha;

    /**
     * @param bits the activation bits, at least 1
     * @param initialAlpha the starting clipping level, for example the activations' upper range
     */
    public PactActivation(int bits, float initialAlpha) {
        if (bits < 1) {
            throw new IllegalArgumentException("bits must be positive");
        }
        this.bits = bits;
        this.alpha = Var.param(Tensor.full(Shape.of(1, 1), initialAlpha));
    }

    /**
     * @param bits the activation bits
     * @param initialAlpha the starting clipping level
     * @return a spec keeping the input shape
     */
    public static LayerSpec spec(int bits, float initialAlpha) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new PactActivation(bits, initialAlpha);
            }
        };
    }

    /** @return the learned clipping level {@code [1, 1]} */
    public Var alpha() {
        return alpha;
    }

    @Override
    public Var forward(Var x) {
        Var a = alpha.reshape(onesShape(x.shape().rank()));
        Var clipped = x.relu().sub(x.sub(a).relu());   // min(relu(x), alpha) for alpha > 0
        float levels = (1 << bits) - 1;
        float al = Math.max(1e-6f, alpha.value().item());
        Tensor q = clipped.value().mul(levels / al).round().mul(al / levels);
        return Functional.straightThrough(Var.of(q), clipped);
    }

    @Override
    public List<Var> parameters() {
        return List.of(alpha);
    }

    private static int[] onesShape(int rank) {
        int[] d = new int[rank];
        java.util.Arrays.fill(d, 1);
        return d;
    }

    @Override
    public String toString() {
        return "PactActivation(" + bits + " bits)";
    }
}
