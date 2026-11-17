package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Medusa (Cai et al. 2024, <a href="https://arxiv.org/abs/2401.10774">arXiv:2401.10774</a>):
 * extra decoding heads on a frozen model's last hidden state, head {@code k} predicting the
 * token {@code k + 1} places ahead. Each step the model's own next token and the heads'
 * guesses for the ones after are checked in one forward pass, and the longest run the model
 * agrees with is kept, so greedy output is unchanged but takes fewer passes, without a
 * separate draft model.
 *
 * <p>Each head is one residual block {@code h + SiLU(h W_k + b_k)} followed by the model's
 * own output head (final norm and projection), as in Medusa-1; {@code W_k} starts at zero,
 * so every head starts as a copy of the next-token head. Only the heads train
 * ({@link #loss}). Candidates are a single chain (the top-1 of each head) rather than
 * Medusa's tree.
 */
public final class MedusaHeads {

    private final HookedDecoder model;
    private final List<Var> weights = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();

    /**
     * @param model the frozen decoder
     * @param heads the number of extra heads
     * @param seed the seed of the small random start
     */
    public MedusaHeads(HookedDecoder model, int heads, long seed) {
        this.model = model;
        int d = model.settings().dim();
        SplittableRandom r = new SplittableRandom(seed);
        for (int k = 0; k < heads; k++) {
            weights.add(Var.param(Tensor.normal(Shape.of(d, d), 0, 1e-3f, r)));
            biases.add(Var.param(Tensor.zeros(Shape.of(d))));
        }
    }

    /** @return the heads' parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(weights);
        p.addAll(biases);
        return p;
    }

    /** @return the number of extra heads */
    public int heads() {
        return weights.size();
    }

    private Var head(int k, Var hidden) {
        Var h = hidden.add(hidden.matmul(weights.get(k)).add(biases.get(k)).activation(
                io.github.zzuegg.brainiac.tensor.spi.ActivationOp.SILU, 0));
        return model.project(h);
    }

    /**
     * The heads' training loss: head {@code k} at position {@code t} predicts token
     * {@code t + k + 2}, from the frozen model's hidden states (detached).
     *
     * @param tokens {@code [batch][steps]}
     * @return the mean cross-entropy over heads
     */
    public Var loss(int[][] tokens) {
        int batch = tokens.length;
        int steps = tokens[0].length;
        Var hidden = model.forward(tokens).hidden().get(model.layers()).detach();
        Var total = null;
        for (int k = 0; k < heads(); k++) {
            int ahead = k + 2;
            int[][] target = new int[batch][steps];
            for (int b = 0; b < batch; b++) {
                for (int t = 0; t < steps; t++) {
                    target[b][t] = t + ahead < steps ? tokens[b][t + ahead] : -1;
                }
            }
            Var l = HookedDecoder.loss(head(k, hidden), target);
            total = total == null ? l : total.add(l);
        }
        return total.div(heads());
    }

    /**
     * @param tokens the generated tokens
     * @param forwardPasses the model passes used
     */
    public record Result(int[] tokens, int forwardPasses) {
    }

    /**
     * Greedy decoding accelerated by the heads; the output equals plain greedy decoding.
     *
     * @param prompt the prompt
     * @param maxNew the tokens to generate
     * @return the tokens and passes
     */
    public Result generate(int[] prompt, int maxNew) {
        int[] context = prompt.clone();
        int passes = 0;
        int[] guesses = new int[0];
        List<Integer> out = new ArrayList<>();
        int vocab = model.vocabSize();
        while (out.size() < maxNew) {
            int[] input = Arrays.copyOf(context, context.length + guesses.length);
            System.arraycopy(guesses, 0, input, context.length, guesses.length);
            HookedDecoder.Run run = model.forward(new int[][] {input});
            passes++;
            float[] logits = run.logits().value().toArray();
            // the model's prediction after the context, then after each guess
            int accepted = 0;
            int base = context.length - 1;
            int next = LogitMath.argmax(Arrays.copyOfRange(logits, base * vocab, (base + 1) * vocab));
            while (accepted < guesses.length && guesses[accepted] == next) {
                accepted++;
                int pos = base + accepted;
                next = LogitMath.argmax(Arrays.copyOfRange(logits, pos * vocab, (pos + 1) * vocab));
            }
            for (int i = 0; i < accepted && out.size() < maxNew; i++) {
                out.add(guesses[i]);
            }
            if (out.size() < maxNew) {
                out.add(next);
            }
            int keep = context.length + accepted;
            context = Arrays.copyOf(input, keep + 1);
            context[keep] = next;
            // the heads guess from the hidden state where 'next' was predicted
            Var hidden = run.hidden().get(model.layers()).slice(1, keep - 1, keep);
            guesses = new int[heads()];
            for (int k = 0; k < heads(); k++) {
                guesses[k] = LogitMath.argmax(head(k, hidden).value().toArray());
            }
        }
        return new Result(out.stream().mapToInt(Integer::intValue).toArray(), passes);
    }
}
