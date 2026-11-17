package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.SplittableRandom;

/**
 * A discrete message channel made differentiable with the Gumbel-softmax (Jang, Gu and Poole,
 * <a href="https://arxiv.org/abs/1611.01144">1611.01144</a>; Maddison et al.,
 * <a href="https://arxiv.org/abs/1611.00712">1611.00712</a>), as used for emergent
 * communication by Havrylov and Titov (<a href="https://arxiv.org/abs/1705.11192">1705.11192</a>).
 * A message is {@code length} symbols from a vocabulary of {@code vocab}; the sender emits
 * {@code [batch, length * vocab]} logits. During training each symbol is sampled as
 * {@code softmax((logits + g) / tau)} with Gumbel noise {@code g}; with {@code hard} the
 * forward pass sends the exact one-hot while the gradient flows through the soft sample
 * (straight-through). At test time the channel sends the argmax symbols.
 */
public final class GumbelChannel {

    private final int length;
    private final int vocab;
    private final SplittableRandom random;
    private float temperature;

    private GumbelChannel(int length, int vocab, float temperature, long seed) {
        this.length = length;
        this.vocab = vocab;
        this.temperature = temperature;
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param length symbols per message
     * @param vocab the vocabulary size
     * @param temperature the relaxation temperature
     * @param seed the noise seed
     * @return the channel
     */
    public static GumbelChannel of(int length, int vocab, float temperature, long seed) {
        return new GumbelChannel(length, vocab, temperature, seed);
    }

    /**
     * @param tau the new temperature (annealing)
     * @return this channel
     */
    public GumbelChannel temperature(float tau) {
        this.temperature = tau;
        return this;
    }

    /** @return the width of an encoded message, {@code length * vocab} */
    public int width() {
        return length * vocab;
    }

    /**
     * A sampled, differentiable message.
     *
     * @param logits {@code [batch, length * vocab]}
     * @param hard send exact one-hots forward (straight-through gradients)
     * @return the message, {@code [batch, length * vocab]}
     */
    public Var sample(Var logits, boolean hard) {
        int batch = logits.shape().dim(0);
        float[] g = new float[batch * length * vocab];
        for (int i = 0; i < g.length; i++) {
            g[i] = (float) -Math.log(-Math.log(random.nextDouble(1e-10, 1)));
        }
        Var soft = Activation.SOFTMAX.forward(logits.add(Tensor.of(Shape.of(batch, length * vocab), g))
                .div(temperature).reshape(batch, length, vocab)).reshape(batch, length * vocab);
        if (!hard) {
            return soft;
        }
        Tensor oneHot = Tensor.of(Shape.of(batch, length * vocab), argmaxOneHot(soft.value().toArray()));
        return soft.sub(soft.detach()).add(oneHot);
    }

    /**
     * The deterministic message: the argmax symbol at every position, one-hot encoded.
     *
     * @param logits {@code [batch, length * vocab]}
     * @return the message
     */
    public Tensor greedy(Tensor logits) {
        return Tensor.of(logits.shape(), argmaxOneHot(logits.toArray()));
    }

    /**
     * @param message a one-hot message row, {@code length * vocab} long
     * @return its symbols
     */
    public int[] symbols(float[] message) {
        int[] s = new int[length];
        for (int p = 0; p < length; p++) {
            int best = 0;
            for (int v = 1; v < vocab; v++) {
                if (message[p * vocab + v] > message[p * vocab + best]) {
                    best = v;
                }
            }
            s[p] = best;
        }
        return s;
    }

    private float[] argmaxOneHot(float[] x) {
        float[] out = new float[x.length];
        for (int base = 0; base < x.length; base += vocab) {
            int best = base;
            for (int k = base + 1; k < base + vocab; k++) {
                if (x[k] > x[best]) {
                    best = k;
                }
            }
            out[best] = 1;
        }
        return out;
    }
}
