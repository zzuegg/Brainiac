package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayDeque;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Grokfast (Lee et al. 2024, <a href="https://arxiv.org/abs/2405.20233">arXiv:2405.20233</a>):
 * treats each parameter's gradient over training steps as a signal, splits it into a fast
 * component (which drives memorisation) and a slow one (which drives generalisation), and
 * amplifies the slow one: {@code g' = g + lambda * lowpass(g)}. On algorithmic tasks such as
 * modular addition the delayed generalisation of grokking then arrives many times sooner.
 *
 * <p>Two low-pass filters, as in the paper:
 * <ul>
 *   <li>{@link #ema}: {@code e = alpha * e + (1 - alpha) * g} (Grokfast-EMA, one extra tensor
 *       per parameter, the default choice);</li>
 *   <li>{@link #movingAverage}: the mean of the last {@code window} gradients (Grokfast-MA,
 *       {@code window} tensors per parameter); gradients pass unchanged until the window
 *       is full.</li>
 * </ul>
 * It is a {@link GradientTransform}, so it goes in front of any optimizer with
 * {@link #wrap(Optimizer)} or {@link GradientTransform#wrap}.
 */
public final class Grokfast implements GradientTransform {

    private final float alpha;
    private final int window;
    private final float lambda;
    private final Map<Var, Tensor> ema = new IdentityHashMap<>();
    private final Map<Var, ArrayDeque<Tensor>> history = new IdentityHashMap<>();

    private Grokfast(float alpha, int window, float lambda) {
        if (lambda < 0 || alpha < 0 || alpha >= 1) {
            throw new IllegalArgumentException("need 0 <= alpha < 1 and lambda >= 0");
        }
        this.alpha = alpha;
        this.window = window;
        this.lambda = lambda;
    }

    /**
     * Grokfast-EMA.
     *
     * @param alpha the filter's momentum, 0.98 in the paper
     * @param lambda the amplification of the slow component, 2 in the paper
     * @return the transform
     */
    public static Grokfast ema(float alpha, float lambda) {
        return new Grokfast(alpha, 0, lambda);
    }

    /**
     * Grokfast-MA.
     *
     * @param window gradients averaged, 100 in the paper
     * @param lambda the amplification of the slow component, 5 in the paper
     * @return the transform
     */
    public static Grokfast movingAverage(int window, float lambda) {
        if (window < 1) {
            throw new IllegalArgumentException("window must be positive");
        }
        return new Grokfast(0, window, lambda);
    }

    @Override
    public Tensor apply(Var parameter, Tensor grad) {
        if (window == 0) {
            Tensor old = ema.get(parameter);
            Tensor e = old == null ? grad.add(0f) : old.mul(alpha).add(grad.mul(1 - alpha));
            // a copy, computed now: the optimizer may release the gradient after its step,
            // and the running filter must not grow a lazy chain
            Tensor.evaluate(List.of(e));
            ema.put(parameter, e);
            return grad.add(e.mul(lambda));
        }
        ArrayDeque<Tensor> q = history.computeIfAbsent(parameter, k -> new ArrayDeque<>());
        Tensor copy = grad.add(0f);
        Tensor.evaluate(List.of(copy));    // the optimizer may release the gradient after its step
        q.addLast(copy);
        if (q.size() > window) {
            q.removeFirst();
        }
        if (q.size() < window) {
            return grad;
        }
        Tensor sum = null;
        for (Tensor t : q) {
            sum = sum == null ? t : sum.add(t);
        }
        return grad.add(sum.mul(lambda / window));
    }

    /**
     * Returns the slow component kept for a parameter.
     *
     * @param parameter the parameter
     * @return its filtered gradient, or {@code null} before its first gradient (EMA only)
     */
    public Tensor slow(Var parameter) {
        return ema.get(parameter);
    }

    /**
     * Drops the filter state of a parameter that is no longer trained.
     *
     * @param parameter the parameter
     */
    public void forget(Var parameter) {
        ema.remove(parameter);
        history.remove(parameter);
    }

    /**
     * @param inner the optimizer that receives the filtered gradients
     * @return {@code inner} with this filter in front
     */
    public Optimizer wrap(Optimizer inner) {
        return GradientTransform.wrap(inner, this);
    }

    @Override
    public String toString() {
        return window == 0 ? "Grokfast-EMA(alpha=" + alpha + ", lambda=" + lambda + ")"
                : "Grokfast-MA(window=" + window + ", lambda=" + lambda + ")";
    }
}
