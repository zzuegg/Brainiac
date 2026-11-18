package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * DoReFa-Net quantisers (Zhou et al. 2016, <a href="https://arxiv.org/abs/1606.06160">arXiv:1606.06160</a>)
 * for quantisation-aware training with straight-through rounding:
 * <ul>
 * <li>{@link #weight}: {@code w' = tanh(w) / (2 max|tanh(w)|) + 1/2} maps the weights into
 *     {@code [0, 1]}, which is quantised uniformly to {@code k} bits and mapped back by
 *     {@code 2 q - 1}; the tanh limits the influence of large weights on the range;</li>
 * <li>{@link #activation}: {@code clip(x, 0, 1)} quantised to {@code k} bits.</li>
 * </ul>
 * DoReFa also quantises gradients; that part is left out (training uses full-precision
 * gradients, as most later QAT work does).
 */
public final class DoReFaQuantizer {

    private DoReFaQuantizer() {
    }

    /**
     * @param x values in {@code [0, 1]}
     * @param bits the bits
     * @return {@code round(x (2^k - 1)) / (2^k - 1)} with a straight-through gradient
     */
    public static Var quantizeUnit(Var x, int bits) {
        float levels = (1 << bits) - 1;
        Tensor q = x.value().mul(levels).round().div(levels);
        return Functional.straightThrough(Var.of(q), x);
    }

    /**
     * The DoReFa weight quantiser.
     *
     * @param w the weight
     * @param bits the bits, at least 1
     * @return the quantised weight in {@code [-1, 1]}
     */
    public static Var weight(Var w, int bits) {
        Var t = w.tanh();
        float max = Math.max(1e-8f, t.value().abs().max().item());
        Var unit = t.div(2 * max).add(0.5f);
        return quantizeUnit(unit, bits).mul(2f).sub(1f);
    }

    /**
     * The DoReFa activation quantiser.
     *
     * @param x the activations
     * @param bits the bits
     * @return {@code clip(x, 0, 1)} quantised
     */
    public static Var activation(Var x, int bits) {
        return quantizeUnit(x.clamp(0f, 1f), bits);
    }
}
