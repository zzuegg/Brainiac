package io.github.zzuegg.brainiac.experimental.code;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A small conditional "code model": an autoregressive policy over a DSL's primitives,
 * conditioned on a fixed-size feature vector (the input/output examples, optionally followed by
 * feedback such as a failed program and what it did). It samples programs with their
 * log-likelihoods, which is what resampling, self-repair ({@link RepairLoop}) and reranking
 * ({@link ExecutionReranker}) need from a code language model; {@code ProgramGenerator}
 * (RobustFill) offers only beam search with execution filtering.
 *
 * <p>The network is an MLP over the condition, a one-hot of the program so far (one slot per
 * position) and the position, predicting the next primitive or the end of the program.
 * {@link #encodeExamples} turns input/output lists into a condition.
 */
public final class ProgramSampler {

    /**
     * Sampled programs.
     *
     * @param programs the programs
     * @param logProbs their log-likelihoods under the sampler
     */
    public record Samples(List<List<String>> programs, double[] logProbs) {
    }

    private final List<String> primitives;
    private final int maxLength;
    private final int conditionSize;
    private final Model net;

    private ProgramSampler(List<String> primitives, int maxLength, int conditionSize, int hidden, long seed) {
        this.primitives = List.copyOf(primitives);
        this.maxLength = maxLength;
        this.conditionSize = conditionSize;
        int in = conditionSize + maxLength * (primitives.size() + 1) + maxLength + 1;
        this.net = Model.sequential().input(in).dense(hidden, Activation.TANH).dense(hidden, Activation.TANH)
                .dense(primitives.size() + 1).seed(seed).build();
    }

    /**
     * @param primitives the DSL's primitives
     * @param maxLength the longest program
     * @param conditionSize the length of the condition vector
     * @param hidden the hidden width
     * @param seed the seed
     * @return an untrained sampler
     */
    public static ProgramSampler of(List<String> primitives, int maxLength, int conditionSize, int hidden, long seed) {
        return new ProgramSampler(primitives, maxLength, conditionSize, hidden, seed);
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return net.parameters();
    }

    /** @return the primitives */
    public List<String> primitives() {
        return primitives;
    }

    /** @return the longest program */
    public int maxLength() {
        return maxLength;
    }

    /**
     * Length of {@link #encodeExamples}' output.
     *
     * @param examples examples encoded
     * @param maxList list positions encoded
     * @param values distinct values (larger ones share a slot)
     * @return the feature count
     */
    public static int exampleFeatures(int examples, int maxList, int values) {
        return examples * 2 * (maxList * (values + 1) + maxList + 1);
    }

    /**
     * Encodes lists one-hot per position, plus their length, input then output per example.
     *
     * @param inputs the example inputs (the first {@code examples} are used)
     * @param outputs the example outputs ({@code null} entries encode as empty)
     * @param examples examples encoded
     * @param maxList list positions encoded
     * @param values distinct values
     * @return the features
     */
    public static float[] encodeExamples(int[][] inputs, int[][] outputs, int examples, int maxList, int values) {
        int per = maxList * (values + 1) + maxList + 1;
        float[] f = new float[examples * 2 * per];
        for (int e = 0; e < Math.min(examples, inputs.length); e++) {
            encodeList(f, (2 * e) * per, inputs[e], maxList, values);
            encodeList(f, (2 * e + 1) * per, outputs[e], maxList, values);
        }
        return f;
    }

    /**
     * Writes one list one-hot at an offset.
     *
     * @param f the features
     * @param offset where the list starts
     * @param xs the list ({@code null} writes nothing)
     * @param maxList positions
     * @param values distinct values
     */
    public static void encodeList(float[] f, int offset, int[] xs, int maxList, int values) {
        if (xs == null) {
            return;
        }
        int n = Math.min(maxList, xs.length);
        for (int i = 0; i < n; i++) {
            int v = xs[i] >= 0 && xs[i] < values ? xs[i] : values;
            f[offset + i * (values + 1) + v] = 1;
        }
        f[offset + maxList * (values + 1) + n] = 1;
    }

    private float[] input(float[] condition, int[] prefix) {
        int p = primitives.size() + 1;
        float[] x = new float[conditionSize + maxLength * p + maxLength + 1];
        System.arraycopy(condition, 0, x, 0, Math.min(condition.length, conditionSize));
        for (int i = 0; i < Math.min(prefix.length, maxLength); i++) {
            x[conditionSize + i * p + prefix[i]] = 1;
        }
        x[conditionSize + maxLength * p + Math.min(prefix.length, maxLength)] = 1;
        return x;
    }

    private Var logSoftmax(List<float[]> rows) {
        int d = rows.get(0).length;
        float[] v = new float[rows.size() * d];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, v, i * d, d);
        }
        Var logits = net.forward(Var.of(Tensor.of(Shape.of(rows.size(), d), v)), true);
        return logits.sub(logits.logsumexp(1, true));
    }

    private int[] tokens(List<String> program) {
        int[] t = new int[program.size()];
        for (int i = 0; i < t.length; i++) {
            t[i] = primitives.indexOf(program.get(i));
        }
        return t;
    }

    /**
     * One supervised step: next-primitive cross-entropy, the end of the program included.
     *
     * @param conditions one condition per program
     * @param programs the target programs
     * @param optimizer the optimizer
     * @return the mean token loss
     */
    public float trainStep(List<float[]> conditions, List<List<String>> programs, Optimizer optimizer) {
        List<float[]> rows = new ArrayList<>();
        List<Integer> targets = new ArrayList<>();
        int end = primitives.size();
        for (int i = 0; i < programs.size(); i++) {
            int[] t = tokens(programs.get(i));
            for (int k = 0; k <= Math.min(t.length, maxLength - 1); k++) {
                rows.add(input(conditions.get(i), java.util.Arrays.copyOf(t, k)));
                targets.add(k < t.length ? t[k] : end);
            }
        }
        Var ls = logSoftmax(rows);
        float[] mask = new float[rows.size() * (end + 1)];
        for (int i = 0; i < targets.size(); i++) {
            mask[i * (end + 1) + targets.get(i)] = 1;
        }
        Var loss = ls.mul(Tensor.of(Shape.of(rows.size(), end + 1), mask)).sum().neg().mul(1f / rows.size());
        float v = loss.value().item();
        loss.backward();
        optimizer.step(parameters());
        return v;
    }

    /**
     * Samples programs.
     *
     * @param condition the condition
     * @param n how many
     * @param temperature the softmax temperature
     * @param random the random source
     * @return the programs and their log-likelihoods (at temperature 1)
     */
    public Samples sample(float[] condition, int n, double temperature, SplittableRandom random) {
        int end = primitives.size();
        int[][] progs = new int[n][maxLength];
        int[] len = new int[n];
        boolean[] done = new boolean[n];
        double[] logp = new double[n];
        for (int t = 0; t < maxLength; t++) {
            List<float[]> rows = new ArrayList<>();
            List<Integer> who = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (!done[i]) {
                    rows.add(input(condition, java.util.Arrays.copyOf(progs[i], len[i])));
                    who.add(i);
                }
            }
            if (rows.isEmpty()) {
                break;
            }
            float[] ls = Var.noGrad(() -> logSoftmax(rows)).value().toArray();
            for (int j = 0; j < who.size(); j++) {
                int i = who.get(j);
                double[] p = new double[end + 1];
                double sum = 0;
                for (int v = 0; v <= end; v++) {
                    p[v] = Math.exp(ls[j * (end + 1) + v] / temperature);
                    if (v == end && len[i] == 0) {
                        p[v] = 0;   // at least one primitive
                    }
                    sum += p[v];
                }
                double u = random.nextDouble() * sum;
                int tok = end;
                for (int v = 0; v <= end; v++) {
                    u -= p[v];
                    if (u <= 0) {
                        tok = v;
                        break;
                    }
                }
                logp[i] += ls[j * (end + 1) + tok];
                if (tok == end) {
                    done[i] = true;
                } else {
                    progs[i][len[i]++] = tok;
                    done[i] = len[i] >= maxLength;
                }
            }
        }
        List<List<String>> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<String> p = new ArrayList<>();
            for (int k = 0; k < len[i]; k++) {
                p.add(primitives.get(progs[i][k]));
            }
            out.add(p);
        }
        return new Samples(out, logp);
    }

    /**
     * @param condition the condition
     * @param program a program
     * @return its log-likelihood, the end included
     */
    public double logProb(float[] condition, List<String> program) {
        int[] t = tokens(program);
        int end = primitives.size();
        List<float[]> rows = new ArrayList<>();
        for (int k = 0; k <= Math.min(t.length, maxLength - 1); k++) {
            rows.add(input(condition, java.util.Arrays.copyOf(t, k)));
        }
        float[] ls = Var.noGrad(() -> logSoftmax(rows)).value().toArray();
        double s = 0;
        for (int k = 0; k < rows.size(); k++) {
            s += ls[k * (end + 1) + (k < t.length ? t[k] : end)];
        }
        return s;
    }

    @Override
    public String toString() {
        return "ProgramSampler(" + primitives.size() + " primitives, max " + maxLength + ")";
    }

    /**
     * @param o another sampler's parameters are copied in
     */
    public void load(ProgramSampler o) {
        Objects.requireNonNull(o);
        for (int i = 0; i < parameters().size(); i++) {
            parameters().get(i).assign(o.parameters().get(i).value());
        }
    }
}
