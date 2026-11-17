package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.List;
import java.util.Objects;

/**
 * A small verifier (value head) that reads a token sequence and outputs the probability that
 * it is good: an outcome reward model when it reads whole solutions, a value function for
 * partial ones. Verifiers are what turn extra test-time compute into accuracy: best-of-N,
 * self-critique loops and tree search all rank candidates with one
 * (Cobbe et al., <a href="https://arxiv.org/abs/2110.14168">2110.14168</a>).
 *
 * <p>It sees the last {@code window} tokens one-hot plus the sequence length, through an MLP,
 * and trains with binary cross-entropy.
 */
public final class SequenceScorer {

    private final int vocab;
    private final int window;
    private final Model model;
    private Optimizer optimizer = Optimizer.adam(3e-3f);

    private SequenceScorer(int vocab, int window, Model model) {
        this.vocab = vocab;
        this.window = window;
        this.model = model;
    }

    /**
     * @param vocab the vocabulary size
     * @param window how many trailing tokens it reads
     * @param hidden the hidden width
     * @param seed the initialization seed
     * @return an untrained scorer
     */
    public static SequenceScorer of(int vocab, int window, int hidden, long seed) {
        Model m = Model.sequential()
                .input(window * vocab + window + 1)
                .dense(hidden, Activation.TANH)
                .dense(hidden, Activation.TANH)
                .dense(1)
                .seed(seed)
                .build();
        return new SequenceScorer(vocab, window, m);
    }

    /**
     * @param o the optimizer used by {@link #fit}
     * @return this scorer
     */
    public SequenceScorer optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /** @return the network */
    public Model model() {
        return model;
    }

    float[] encode(int[] tokens) {
        float[] x = new float[window * vocab + window + 1];
        int n = Math.min(window, tokens.length);
        for (int i = 0; i < n; i++) {
            x[(window - n + i) * vocab + tokens[tokens.length - n + i]] = 1;
        }
        x[window * vocab + n] = 1;
        return x;
    }

    /**
     * @param tokens a sequence
     * @return the probability that it is good
     */
    public double score(int[] tokens) {
        double z = LlmTensors.predict(model, encode(tokens))[0];
        return 1 / (1 + Math.exp(-z));
    }

    /**
     * One step of binary cross-entropy.
     *
     * @param sequences the sequences
     * @param labels 1 for good, 0 for bad (soft labels are fine)
     * @return the mean loss before the step
     */
    public float step(List<int[]> sequences, float[] labels) {
        float[][] rows = new float[sequences.size()][];
        for (int i = 0; i < rows.length; i++) {
            rows[i] = encode(sequences.get(i));
        }
        Var z = model.forward(Var.of(LlmTensors.rows(rows)), true);
        // BCE from logits: softplus(z) - y z
        Var loss = z.softplus().sub(z.mul(LlmTensors.column(labels))).mean();
        float v = loss.value().item();
        loss.backward();
        optimizer.step(model.parameters());
        return v;
    }

    /**
     * Trains for a number of full-batch steps.
     *
     * @param sequences the sequences
     * @param labels their labels
     * @param steps the number of steps
     * @return the final loss
     */
    public float fit(List<int[]> sequences, float[] labels, int steps) {
        float v = Float.NaN;
        for (int i = 0; i < steps; i++) {
            v = step(sequences, labels);
        }
        return v;
    }

    /** @return the vocabulary size */
    public int vocabSize() {
        return vocab;
    }
}
