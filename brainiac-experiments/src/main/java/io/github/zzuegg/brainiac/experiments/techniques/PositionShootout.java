package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GluFeedForward;
import io.github.zzuegg.brainiac.experimental.nn.PositionalAttention;
import io.github.zzuegg.brainiac.experimental.nn.PositionalAttention.Position;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Compares attention position encodings on a task that needs relative positions: a causal
 * one-layer transformer must output, at every step, the token seen {@code lag} steps earlier.
 * Identical tokens appear at many positions, so content alone cannot solve it; NoPE has to
 * infer distances from the causal mask, while RoPE, xPos, ALiBi, T5 and CoPE encode them.
 * Reports the accuracy over the steps that have a token {@code lag} back.
 */
public final class PositionShootout {

    private PositionShootout() {
    }

    /**
     * Trains one model and returns its accuracy on fresh sequences.
     *
     * @param position the position encoding
     * @param lag how far back to copy from
     * @param steps training steps
     * @param seed the seed
     * @return the test accuracy
     */
    public static float run(Position position, int lag, int steps, long seed) {
        int vocab = 6;
        int length = 12;
        int d = 32;
        SplittableRandom rnd = new SplittableRandom(seed);
        Var embed = Var.param(Tensor.normal(Shape.of(vocab, d), 0, 1, rnd));
        PositionalAttention attention = new PositionalAttention(
                PositionalAttention.Settings.of(d, 4, true, position, true), rnd);
        LayerNorm n1 = LayerNorm.of(d);
        LayerNorm n2 = LayerNorm.of(d);
        GluFeedForward ffn = new GluFeedForward(GluFeedForward.Kind.SWIGLU, d, 2 * d, rnd);
        Var out = Var.param(Tensor.normal(Shape.of(d, vocab), 0, 0.1f, rnd));
        List<Var> params = new ArrayList<>(List.of(embed, out));
        params.addAll(attention.parameters());
        params.addAll(n1.parameters());
        params.addAll(n2.parameters());
        params.addAll(ffn.parameters());
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < steps; s++) {
            Tensor[] batch = batch(32, length, vocab, lag, rnd);
            loss(batch, embed, attention, n1, n2, ffn, out, length, vocab).backward();
            opt.step(params);
        }
        Tensor[] test = batch(256, length, vocab, lag, new SplittableRandom(seed + 1));
        Var logits = logits(test[0], embed, attention, n1, n2, ffn, out, length);
        Tensor pred = logits.value().argmax(2).slice(1, lag, length);
        return pred.equalTo(test[1].slice(1, lag, length)).mean().item();
    }

    private static Tensor[] batch(int n, int length, int vocab, int lag, SplittableRandom rnd) {
        float[] x = new float[n * length];
        float[] y = new float[n * length];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < length; t++) {
                x[i * length + t] = rnd.nextInt(vocab);
            }
            for (int t = 0; t < length; t++) {
                y[i * length + t] = t >= lag ? x[i * length + t - lag] : -1;
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, length), x), Tensor.of(Shape.of(n, length), y)};
    }

    private static Var logits(Tensor ids, Var embed, PositionalAttention attention, LayerNorm n1, LayerNorm n2,
            GluFeedForward ffn, Var out, int length) {
        Var h = embed.gather(0, ids);                         // [n, length, d]
        h = h.add(attention.forward(n1.forward(h)));
        h = h.add(ffn.forward(n2.forward(h)));
        return h.matmul(out);
    }

    private static Var loss(Tensor[] batch, Var embed, PositionalAttention attention, LayerNorm n1, LayerNorm n2,
            GluFeedForward ffn, Var out, int length, int vocab) {
        Var l = logits(batch[0], embed, attention, n1, n2, ffn, out, length);
        int n = batch[0].shape().dim(0);
        return Functional.crossEntropy(l.reshape(n * length, vocab), batch[1].reshape(n * length));
    }
}
