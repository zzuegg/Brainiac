package io.github.zzuegg.brainiac.experimental.ood;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Post-hoc OOD scores from a trained classifier's logits, higher meaning "more
 * in-distribution":
 *
 * <ul>
 *   <li>{@link #msp}: the maximum softmax probability, the baseline of Hendrycks and Gimpel
 *       (ICLR 2017, <a href="https://arxiv.org/abs/1610.02136">arXiv:1610.02136</a>).</li>
 *   <li>{@link #maxLogit}: the largest logit, which avoids softmax saturation (Hendrycks et
 *       al., ICML 2022, <a href="https://arxiv.org/abs/1911.11132">arXiv:1911.11132</a>).</li>
 *   <li>{@link #energy}: the negative free energy {@code T log sum exp(f / T)}, aligned with
 *       the input density (Liu et al., NeurIPS 2020,
 *       <a href="https://arxiv.org/abs/2010.03759">arXiv:2010.03759</a>).</li>
 *   <li>{@link #odin}: temperature scaling with a large {@code T} plus a small input step that
 *       raises the top softmax score, which helps in-distribution inputs more than OOD ones
 *       (Liang et al., ICLR 2018, <a href="https://arxiv.org/abs/1706.02690">arXiv:1706.02690</a>);
 *       it costs one backward and a second forward pass.</li>
 * </ul>
 */
public final class LogitScores {

    private LogitScores() {
    }

    /**
     * @param logits {@code [n, classes]}
     * @return the maximum softmax probability per row
     */
    public static double[] msp(Tensor logits) {
        return mspAt(logits, 1);
    }

    /**
     * @param logits {@code [n, classes]}
     * @return the largest logit per row
     */
    public static double[] maxLogit(Tensor logits) {
        return logits.max(1, false).toDoubleArray();
    }

    /**
     * @param logits {@code [n, classes]}
     * @param temperature the energy temperature, usually 1
     * @return {@code T logsumexp(logits / T)} per row (the negative energy)
     */
    public static double[] energy(Tensor logits, float temperature) {
        return logits.div(temperature).logsumexp(1, false).mul(temperature).toDoubleArray();
    }

    /**
     * The ODIN score.
     *
     * @param logits maps inputs to logits (with a graph to the inputs)
     * @param x the inputs
     * @param temperature the softmax temperature, about 1000 in the paper
     * @param epsilon the input step size
     * @return the maximum temperature-scaled softmax probability at the perturbed inputs
     */
    public static double[] odin(UnaryOperator<Var> logits, Tensor x, float temperature, float epsilon) {
        Var in = Var.param(x);
        Var z = logits.apply(in).div(temperature);
        Var logP = z.sub(z.logsumexp(1, true));
        Var top = logP.max(1, false).sum();   // log S_yhat(x; T), summed over the batch
        Tensor g = Var.gradients(top, List.of(in), false).getFirst().value();
        Tensor perturbed = x.add(g.sign().mul(epsilon));
        Tensor zz = Var.noGrad(() -> logits.apply(Var.of(perturbed)).value());
        return mspAt(zz, temperature);
    }

    /**
     * The maximum softmax probability at a temperature.
     *
     * @param logits {@code [n, classes]}
     * @param temperature the temperature
     * @return per row
     */
    public static double[] mspAt(Tensor logits, float temperature) {
        Tensor z = logits.div(temperature);
        return z.max(1, false).sub(z.logsumexp(1, false)).exp().toDoubleArray();
    }
}
