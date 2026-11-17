package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The neural long-term memory of Titans (Behrouz, Zhong and Mirrokni 2025,
 * <a href="https://arxiv.org/abs/2501.00663">arXiv:2501.00663</a>): a small MLP {@code M}
 * that keeps learning at inference time. Writing a key/value pair takes one gradient step on
 * the associative loss {@code ||M(k) - v||^2}, with momentum ("surprise") and weight decay
 * ("forgetting"):
 * <pre>
 * S_t = eta S_(t-1) - theta grad loss(M_(t-1); k_t, v_t)     past surprise + momentary surprise
 * M_t = (1 - alpha) M_(t-1) + S_t                            forget, then memorize
 * </pre>
 * A pair the memory already predicts has a small gradient and barely changes it; a
 * surprising one changes it a lot. Reading is a plain forward pass {@code M(q)}. The rates
 * can depend on the data through a {@link Gates} function, as in the paper. The inner
 * gradient is computed with autograd, so any depth works. For an end-to-end trainable
 * layer use {@link TitansLayer}.
 */
public final class TitansMemory {

    /** Data-dependent rates for one write. */
    @FunctionalInterface
    public interface Gates {
        /**
         * @param keys the keys being written {@code [n, keySize]}
         * @return {@code {theta, eta, alpha}}: learning rate, momentum and forgetting
         */
        float[] of(Tensor keys);
    }

    private final List<Tensor> weights = new ArrayList<>();
    private final List<Tensor> momentum = new ArrayList<>();
    private final int keySize;
    private final int valueSize;
    private Gates gates;
    private float lastSurprise;

    /**
     * Creates a memory: a linear map when {@code hidden == 0}, else a two-layer MLP with a tanh
     * hidden layer.
     *
     * @param keySize the key size
     * @param hidden the hidden units, or 0 for a linear memory
     * @param valueSize the value size
     * @param random the random source for the initial weights
     */
    public TitansMemory(int keySize, int hidden, int valueSize, SplittableRandom random) {
        this.keySize = keySize;
        this.valueSize = valueSize;
        if (hidden == 0) {
            weights.add(Tensor.zeros(Shape.of(keySize, valueSize)));
        } else {
            weights.add(Init.XAVIER.create(Shape.of(keySize, hidden), keySize, hidden, random));
            weights.add(Tensor.zeros(Shape.of(hidden, valueSize)));
        }
        for (Tensor w : weights) {
            momentum.add(Tensor.zeros(w.shape()));
        }
        gates(0.5f, 0.6f, 0.0f);
    }

    /**
     * Fixed rates.
     *
     * @param theta the learning rate of a write
     * @param eta the momentum (how long surprise carries on)
     * @param alpha the forgetting rate per write, in [0, 1]
     * @return this memory
     */
    public TitansMemory gates(float theta, float eta, float alpha) {
        float[] g = {theta, eta, alpha};
        this.gates = k -> g;
        return this;
    }

    /**
     * Data-dependent rates.
     *
     * @param gates computes the rates from the keys of each write
     * @return this memory
     */
    public TitansMemory gates(Gates gates) {
        this.gates = java.util.Objects.requireNonNull(gates, "gates");
        return this;
    }

    private Var apply(List<Var> w, Var k) {
        Var h = k.matmul(w.get(0));
        if (w.size() == 1) {
            return h;
        }
        return h.tanh().matmul(w.get(1));
    }

    /**
     * Reads the memory.
     *
     * @param queries {@code [n, keySize]}
     * @return {@code [n, valueSize]}
     */
    public Tensor read(Tensor queries) {
        List<Var> w = new ArrayList<>();
        for (Tensor t : weights) {
            w.add(Var.of(t));
        }
        return apply(w, Var.of(queries)).value();
    }

    /**
     * Memorizes key/value pairs with one surprise-driven step (all rows form one chunk).
     *
     * @param keys {@code [n, keySize]}
     * @param values {@code [n, valueSize]}
     * @return the loss before the step, how surprising the pairs were
     */
    public float write(Tensor keys, Tensor values) {
        if (keys.shape().dim(1) != keySize || values.shape().dim(1) != valueSize) {
            throw new IllegalArgumentException("expected keys of " + keySize + " and values of " + valueSize);
        }
        List<Var> w = new ArrayList<>();
        for (Tensor t : weights) {
            w.add(Var.param(t));
        }
        Var loss = apply(w, Var.of(keys)).sub(values).square().sum(1, false).mean();
        float before = loss.value().item();
        loss.backward();
        float[] g = gates.of(keys);
        float theta = g[0];
        float eta = g[1];
        float alpha = g[2];
        double norm = 0;
        for (int i = 0; i < weights.size(); i++) {
            Tensor grad = w.get(i).grad();
            norm += grad.square().sum().item();
            Tensor s = momentum.get(i).mul(eta).sub(grad.mul(theta));
            momentum.set(i, s);
            weights.set(i, weights.get(i).mul(1 - alpha).add(s));
        }
        lastSurprise = (float) Math.sqrt(norm);
        return before;
    }

    /** @return the norm of the momentary-surprise gradient of the last write */
    public float lastSurprise() {
        return lastSurprise;
    }

    /** Forgets everything: zeroes the output layer and the momentum. */
    public void reset() {
        int last = weights.size() - 1;
        weights.set(last, Tensor.zeros(weights.get(last).shape()));
        for (int i = 0; i < momentum.size(); i++) {
            momentum.set(i, Tensor.zeros(momentum.get(i).shape()));
        }
    }

    @Override
    public String toString() {
        return "TitansMemory(" + weights.size() + " layers, " + keySize + " -> " + valueSize + ")";
    }
}
