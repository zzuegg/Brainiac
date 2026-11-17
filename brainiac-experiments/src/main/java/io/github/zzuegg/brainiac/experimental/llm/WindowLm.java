package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A tiny autoregressive language model for reasoning experiments: the last {@code window}
 * tokens (one-hot, right-aligned) plus a one-hot of the prefix length go through an MLP that
 * predicts the next token. It is a real causal LM (it factorizes {@code p(x) = prod p(x_t |
 * x_<t)}), just small enough that RL-for-reasoning recipes such as GRPO, DPO or SPIN train in
 * a second on a CPU.
 *
 * <p>Every training method in this package works through {@link #logProbs}, which returns
 * the per-token log-probabilities of completions as an autograd {@link Var}, so losses are
 * written once and differentiated automatically.
 *
 * <pre>{@code
 * WindowLm lm = WindowLm.of(tokenizer.size(), 8, 64, 1);
 * int[][] out = lm.sample(new int[][] {prompt}, 3, endToken, 1.0, random);
 * }</pre>
 *
 * <p>It implements {@link LanguageModel}, so the core {@code Generator} (sampling options,
 * stop tokens, beam search) drives it like any other model.
 */
public final class WindowLm implements LanguageModel {

    private final int vocab;
    private final int window;
    private final int hidden;
    private final Model model;

    private WindowLm(int vocab, int window, int hidden, Model model) {
        this.vocab = vocab;
        this.window = window;
        this.hidden = hidden;
        this.model = model;
    }

    /**
     * @param vocab the vocabulary size
     * @param window how many previous tokens the model sees
     * @param hidden the width of the two hidden layers
     * @param seed the initialization seed
     * @return a freshly initialized model
     */
    public static WindowLm of(int vocab, int window, int hidden, long seed) {
        Model m = Model.sequential()
                .input(window * vocab + window + 1)
                .dense(hidden, Activation.TANH)
                .dense(hidden, Activation.TANH)
                .dense(vocab)
                .seed(seed)
                .build();
        return new WindowLm(vocab, window, hidden, m);
    }

    @Override
    public int vocabSize() {
        return vocab;
    }

    @Override
    public State newState(int batch) {
        int[][] rows = new int[batch][];
        Arrays.fill(rows, new int[0]);
        return new History(rows, 0);
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        History h = (History) state;
        List<int[]> prefixes = new ArrayList<>();
        for (int b = 0; b < h.rows.length; b++) {
            int[] row = h.rows[b];
            for (int t = 0; t < tokens[b].length; t++) {
                if (padding == null || !padding[b][t]) {
                    row = Arrays.copyOf(row, row.length + 1);
                    row[row.length - 1] = tokens[b][t];
                }
            }
            h.rows[b] = row;
            prefixes.add(row);
        }
        h.length += tokens.length == 0 ? 0 : tokens[0].length;
        return logits(prefixes).value();
    }

    /** The tokens each sequence has read; the model only looks at the last window of them. */
    private static final class History implements State {
        private final int[][] rows;
        private int length;

        History(int[][] rows, int length) {
            this.rows = rows;
            this.length = length;
        }

        @Override
        public int length() {
            return length;
        }

        @Override
        public State select(int[] keep) {
            int[][] picked = new int[keep.length][];
            for (int i = 0; i < keep.length; i++) {
                picked[i] = rows[keep[i]];
            }
            return new History(picked, length);
        }
    }

    /** @return the context window */
    public int window() {
        return window;
    }

    /** @return the underlying network */
    public Model model() {
        return model;
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return model.parameters();
    }

    /** @return an independent copy with the same weights, for example a frozen reference policy */
    public WindowLm copy() {
        WindowLm c = of(vocab, window, hidden, 0);
        LlmTensors.copy(model.parameters(), c.model.parameters());
        return c;
    }

    /**
     * Overwrites this model's weights with another's.
     *
     * @param other a model of the same size
     */
    public void load(WindowLm other) {
        LlmTensors.copy(other.model.parameters(), model.parameters());
    }

    /**
     * Encodes a prefix into the network input.
     *
     * @param prefix the tokens so far
     * @return the input features
     */
    float[] encode(int[] prefix) {
        float[] x = new float[window * vocab + window + 1];
        int n = Math.min(window, prefix.length);
        for (int i = 0; i < n; i++) {
            int token = prefix[prefix.length - n + i];
            int slot = window - n + i;
            x[slot * vocab + token] = 1;
        }
        x[window * vocab + Math.min(window, prefix.length)] = 1;
        return x;
    }

    /**
     * @param prefixes token prefixes
     * @return the next-token logits {@code [n, vocab]}
     */
    public Var logits(List<int[]> prefixes) {
        float[][] rows = new float[prefixes.size()][];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = encode(prefixes.get(i));
        }
        return model.forward(Var.of(LlmTensors.rows(rows)), true);
    }

    /**
     * @param prefix the tokens so far
     * @return the next-token logits
     */
    public float[] logits(int[] prefix) {
        return LlmTensors.predict(model, encode(prefix));
    }

    /**
     * The per-token log-probabilities of completions, flattened in order: completion 0's
     * tokens first, then completion 1's, and so on.
     *
     * @param prompts the prompts
     * @param completions what followed each prompt
     * @return {@code [totalTokens, 1]}
     */
    public Var logProbs(List<int[]> prompts, List<int[]> completions) {
        List<int[]> prefixes = new ArrayList<>();
        int[] targets = new int[totalTokens(completions)];
        int k = 0;
        for (int s = 0; s < prompts.size(); s++) {
            int[] p = prompts.get(s);
            int[] c = completions.get(s);
            for (int t = 0; t < c.length; t++) {
                int[] prefix = Arrays.copyOf(p, p.length + t);
                System.arraycopy(c, 0, prefix, p.length, t);
                prefixes.add(prefix);
                targets[k++] = c[t];
            }
        }
        return LlmTensors.pick(LlmTensors.logSoftmax(logits(prefixes)), targets);
    }

    /**
     * Sums per-token values of each completion: a matrix {@code [sequences, tokens]} that
     * maps the output of {@link #logProbs} to sequence totals by multiplication.
     *
     * @param completions the completions
     * @param normalize divide each row by the completion length (a mean instead of a sum)
     * @return the summing matrix
     */
    public static Tensor sequenceSum(List<int[]> completions, boolean normalize) {
        int total = totalTokens(completions);
        float[] m = new float[completions.size() * total];
        int k = 0;
        for (int s = 0; s < completions.size(); s++) {
            int len = completions.get(s).length;
            for (int t = 0; t < len; t++) {
                m[s * total + k++] = normalize ? 1f / len : 1f;
            }
        }
        return Tensor.of(Shape.of(completions.size(), total), m);
    }

    /**
     * @param prompts the prompts
     * @param completions what followed each prompt
     * @return {@code log p(completion | prompt)} for each pair, {@code [n, 1]}
     */
    public Var sequenceLogProbs(List<int[]> prompts, List<int[]> completions) {
        return Var.of(sequenceSum(completions, false)).matmul(logProbs(prompts, completions));
    }

    /**
     * @param prompt a prompt
     * @param completion what followed
     * @return {@code log p(completion | prompt)}
     */
    public double sequenceLogProb(int[] prompt, int[] completion) {
        if (completion.length == 0) {
            return 0;
        }
        return LlmTensors.values(sequenceLogProbs(List.of(prompt), List.of(completion)))[0];
    }

    static int totalTokens(List<int[]> completions) {
        int n = 0;
        for (int[] c : completions) {
            n += c.length;
        }
        return n;
    }

    /**
     * Samples completions for a batch of prompts, one network call per position for the
     * whole batch. A completion stops after the end token or at {@code maxNew} tokens.
     *
     * @param prompts the prompts
     * @param maxNew the most tokens to add
     * @param eos the end token
     * @param temperature softmax temperature; 0 picks the most likely token
     * @param random the random source
     * @return the completions, each including its end token if one was produced
     */
    public int[][] sample(int[][] prompts, int maxNew, int eos, double temperature, SplittableRandom random) {
        int n = prompts.length;
        int[][] seqs = new int[n][];
        int[] lengths = new int[n];
        boolean[] done = new boolean[n];
        for (int i = 0; i < n; i++) {
            seqs[i] = Arrays.copyOf(prompts[i], prompts[i].length + maxNew);
            lengths[i] = prompts[i].length;
        }
        for (int t = 0; t < maxNew; t++) {
            List<int[]> live = new ArrayList<>();
            List<Integer> who = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (!done[i]) {
                    live.add(Arrays.copyOf(seqs[i], lengths[i]));
                    who.add(i);
                }
            }
            if (live.isEmpty()) {
                break;
            }
            float[] logits = LlmTensors.values(logits(live));
            for (int j = 0; j < who.size(); j++) {
                int i = who.get(j);
                int tok = LlmTensors.sample(Arrays.copyOfRange(logits, j * vocab, (j + 1) * vocab), temperature, random);
                seqs[i][lengths[i]++] = tok;
                if (tok == eos) {
                    done[i] = true;
                }
            }
        }
        int[][] out = new int[n][];
        for (int i = 0; i < n; i++) {
            out[i] = Arrays.copyOfRange(seqs[i], prompts[i].length, lengths[i]);
        }
        return out;
    }

    /**
     * @param prompt a prompt
     * @param maxNew the most tokens to add
     * @param eos the end token
     * @return the most likely completion, token by token
     */
    public int[] greedy(int[] prompt, int maxNew, int eos) {
        return sample(new int[][] {prompt}, maxNew, eos, 0, new SplittableRandom(0))[0];
    }

    /**
     * One supervised step: next-token cross-entropy on the completions.
     *
     * @param prompts the prompts
     * @param completions the target completions
     * @param optimizer the optimizer
     * @return the mean loss per token
     */
    public float supervisedStep(List<int[]> prompts, List<int[]> completions, Optimizer optimizer) {
        Var loss = logProbs(prompts, completions).mean().neg();
        float value = loss.value().item();
        loss.backward();
        optimizer.step(parameters());
        return value;
    }
}
