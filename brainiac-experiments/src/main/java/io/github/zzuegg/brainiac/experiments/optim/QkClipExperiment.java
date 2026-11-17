package io.github.zzuegg.brainiac.experiments.optim;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.MuonClip;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.MultiHeadAttention;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * MuonClip against Muon on a task that drives attention logits up: a one-layer causal
 * transformer (token and position embeddings, one core {@link MultiHeadAttention}, a
 * linear read-out) must output, at every position, the <em>first</em> token of the
 * sequence. The solution is a hard attention onto position 0, so with a high Muon
 * learning rate the query-key logits grow steadily; {@link MuonClip} caps them at
 * {@code tau} by rescaling {@code W_q} and {@code W_k} after each step. Reported: the
 * largest attention logit over training and the final held-out loss and accuracy.
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class QkClipExperiment {

    private static final int VOCAB = 16;
    private static final int STEPS_PER_SEQUENCE = 8;
    private static final int FEATURES = 32;
    private static final int HEADS = 2;

    private QkClipExperiment() {
    }

    /**
     * The outcome of one run.
     *
     * @param maxLogit the largest attention logit seen after any step
     * @param finalLoss the held-out cross-entropy at the end
     * @param accuracy the held-out accuracy at the end
     * @param clips the steps at which QK-clip rescaled a head (0 without it)
     */
    public record Result(float maxLogit, float finalLoss, float accuracy, int clips) {
    }

    /**
     * Trains the model.
     *
     * @param threshold QK-clip's {@code tau}, or 0 for plain Muon
     * @param muonRate Muon's learning rate (Adam gets a tenth of it for the embeddings)
     * @param steps training steps
     * @param seed the seed
     * @return the result
     */
    public static Result run(float threshold, float muonRate, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Var tokens = Var.param(Tensor.normal(Shape.of(VOCAB, FEATURES), 0, 1, r));
        Var positions = Var.param(Tensor.normal(Shape.of(STEPS_PER_SEQUENCE, FEATURES), 0, 1, r));
        float headSize = FEATURES / HEADS;
        MultiHeadAttention attention = new MultiHeadAttention(new MultiHeadAttention.Settings(FEATURES, FEATURES, HEADS,
                HEADS, (int) headSize, (int) headSize, FEATURES, false, true, 0, false, 0, 0, false), Init.XAVIER, r);
        Var readout = Var.param(Init.XAVIER.create(Shape.of(FEATURES, VOCAB), FEATURES, VOCAB, r));
        List<Var> params = new ArrayList<>(List.of(tokens, positions, readout));
        params.addAll(attention.parameters());
        Optimizer muonAndAdam = Optimizer.groups(Optimizer.adam(muonRate / 10),
                Optimizer.Group.of(List.of(readout), Optimizer.muon(muonRate, 0, 0.95f, false)),
                Optimizer.Group.of(attention.parameters(), Optimizer.muon(muonRate, 0, 0.95f, false)));
        MuonClip clip = threshold > 0 ? new MuonClip(muonAndAdam, threshold).watch(attention) : null;
        Optimizer opt = clip != null ? clip : muonAndAdam;
        float maxLogit = 0;
        for (int s = 0; s < steps; s++) {
            Tensor[] b = batch(32, r);
            Var x = embed(tokens, positions, b[0]);
            if (clip != null) {
                clip.record(attention, x.value());
            }
            Var logits = attention.forward(List.of(x), true).matmul(readout);
            Functional.crossEntropy(logits.reshape(32 * STEPS_PER_SEQUENCE, VOCAB), b[1]).backward();
            opt.step(params);
            float[] max = MuonClip.maxLogits(attention.parameters().get(0).value(),
                    attention.parameters().get(1).value(), x.value(), HEADS, HEADS, (int) headSize,
                    1 / (float) Math.sqrt(headSize), true);
            for (float m : max) {
                maxLogit = Math.max(maxLogit, m);
            }
        }
        Tensor[] test = batch(512, new SplittableRandom(seed + 1));
        Var logits = attention.forward(List.of(embed(tokens, positions, test[0])), false).matmul(readout)
                .reshape(512 * STEPS_PER_SEQUENCE, VOCAB);
        float loss = Functional.crossEntropy(logits, test[1]).value().item();
        float acc = logits.value().argmax(1).equalTo(test[1]).mean().item();
        return new Result(maxLogit, loss, acc, clip == null ? 0 : clip.clips());
    }

    private static Var embed(Var tokens, Var positions, Tensor oneHot) {
        return Var.of(oneHot).matmul(tokens).add(positions);
    }

    /** One-hot sequences {@code [n, steps, vocab]} and targets {@code [n * steps]}: the first token. */
    private static Tensor[] batch(int n, SplittableRandom r) {
        float[] x = new float[n * STEPS_PER_SEQUENCE * VOCAB];
        float[] y = new float[n * STEPS_PER_SEQUENCE];
        for (int i = 0; i < n; i++) {
            int first = -1;
            for (int t = 0; t < STEPS_PER_SEQUENCE; t++) {
                int tok = r.nextInt(VOCAB);
                first = t == 0 ? tok : first;
                x[(i * STEPS_PER_SEQUENCE + t) * VOCAB + tok] = 1;
                y[i * STEPS_PER_SEQUENCE + t] = first;
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, STEPS_PER_SEQUENCE, VOCAB), x),
            Tensor.of(Shape.of(n * STEPS_PER_SEQUENCE), y)};
    }
}
