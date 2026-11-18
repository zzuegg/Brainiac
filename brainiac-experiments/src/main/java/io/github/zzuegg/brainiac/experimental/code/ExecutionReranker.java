package io.github.zzuegg.brainiac.experimental.code;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Execution-based reranking: which of many sampled programs to submit.
 * <ul>
 *   <li>{@link #filter}: keep the programs that reproduce the given examples.</li>
 *   <li>{@link #cluster}: AlphaCode's clustering (Li et al., Science 2022,
 *       <a href="https://arxiv.org/abs/2203.07814">arXiv:2203.07814</a>): run the survivors on
 *       extra inputs, group them by identical outputs, and submit from the largest cluster
 *       (its most likely member).</li>
 *   <li>{@link #dualAgreement}: CodeT (Chen et al., ICLR 2023,
 *       <a href="https://arxiv.org/abs/2207.10397">arXiv:2207.10397</a>): given candidate test
 *       cases, programs that pass the same tests form a consensus set scored by
 *       {@code |programs in it| * |tests they pass|}.</li>
 *   <li>{@link Verifier}: LEVER (Ni et al., ICML 2023,
 *       <a href="https://arxiv.org/abs/2302.08468">arXiv:2302.08468</a>): a small learned
 *       verifier reads the program together with its execution results and predicts
 *       correctness; programs with equal execution results pool their generation probability,
 *       and the score is {@code P(correct) * P(generated)}.</li>
 *   <li>{@link #likelihood}: the generation-likelihood baseline.</li>
 * </ul>
 */
public final class ExecutionReranker {

    /**
     * A sampled program.
     *
     * @param program the primitives
     * @param logProb its generation log-likelihood
     */
    public record Candidate(List<String> program, double logProb) {
    }

    /**
     * A test case.
     *
     * @param input the input
     * @param output the expected output
     */
    public record TestCase(int[] input, int[] output) {
    }

    private final ExecutionTracer tracer;

    private ExecutionReranker(ExecutionTracer tracer) {
        this.tracer = Objects.requireNonNull(tracer);
    }

    /**
     * @param tracer runs programs
     * @return the reranker
     */
    public static ExecutionReranker of(ExecutionTracer tracer) {
        return new ExecutionReranker(tracer);
    }

    private String signature(List<String> program, int[][] inputs) {
        StringBuilder sb = new StringBuilder();
        for (int[] in : inputs) {
            int[] o = tracer.run(program, in);
            sb.append(o == null ? "!" : Arrays.toString(o)).append('|');
        }
        return sb.toString();
    }

    /**
     * @param candidates the samples
     * @return the most likely one, or {@code null} if there are none
     */
    public Candidate likelihood(List<Candidate> candidates) {
        Candidate best = null;
        for (Candidate c : candidates) {
            best = best == null || c.logProb() > best.logProb() ? c : best;
        }
        return best;
    }

    /**
     * @param candidates the samples
     * @param inputs the given example inputs
     * @param outputs the expected outputs
     * @return those that reproduce every example
     */
    public List<Candidate> filter(List<Candidate> candidates, int[][] inputs, int[][] outputs) {
        List<Candidate> out = new ArrayList<>();
        for (Candidate c : candidates) {
            boolean ok = true;
            for (int i = 0; i < inputs.length && ok; i++) {
                int[] o = tracer.run(c.program(), inputs[i]);
                ok = o != null && Arrays.equals(o, outputs[i]);
            }
            if (ok) {
                out.add(c);
            }
        }
        return out;
    }

    /**
     * Groups programs by their outputs on extra inputs.
     *
     * @param candidates the samples (usually filtered first)
     * @param extraInputs inputs to run them on
     * @return the clusters, largest first
     */
    public List<List<Candidate>> clusters(List<Candidate> candidates, int[][] extraInputs) {
        Map<String, List<Candidate>> groups = new LinkedHashMap<>();
        for (Candidate c : candidates) {
            groups.computeIfAbsent(signature(c.program(), extraInputs), k -> new ArrayList<>()).add(c);
        }
        List<List<Candidate>> out = new ArrayList<>(groups.values());
        out.sort((a, b) -> Integer.compare(b.size(), a.size()));
        return out;
    }

    /**
     * AlphaCode: the most likely member of the largest behaviour cluster.
     *
     * @param candidates the samples (usually filtered first)
     * @param extraInputs inputs to run them on
     * @return the submission, or {@code null} if there are none
     */
    public Candidate cluster(List<Candidate> candidates, int[][] extraInputs) {
        List<List<Candidate>> cs = clusters(candidates, extraInputs);
        return cs.isEmpty() ? null : likelihood(cs.get(0));
    }

    /**
     * CodeT's dual agreement.
     *
     * @param candidates the samples
     * @param tests candidate test cases (from any source)
     * @return the submission, or {@code null} if there are none
     */
    public Candidate dualAgreement(List<Candidate> candidates, List<TestCase> tests) {
        Map<String, List<Candidate>> sets = new HashMap<>();
        Map<String, Integer> passed = new HashMap<>();
        for (Candidate c : candidates) {
            StringBuilder key = new StringBuilder();
            int n = 0;
            for (int t = 0; t < tests.size(); t++) {
                int[] o = tracer.run(c.program(), tests.get(t).input());
                boolean ok = o != null && Arrays.equals(o, tests.get(t).output());
                key.append(ok ? '1' : '0');
                n += ok ? 1 : 0;
            }
            sets.computeIfAbsent(key.toString(), k -> new ArrayList<>()).add(c);
            passed.put(key.toString(), n);
        }
        Candidate best = null;
        double bestScore = -1;
        for (Map.Entry<String, List<Candidate>> e : sets.entrySet()) {
            double score = (double) e.getValue().size() * passed.get(e.getKey());
            Candidate top = likelihood(e.getValue());
            if (score > bestScore || score == bestScore && best != null && top.logProb() > best.logProb()) {
                bestScore = score;
                best = top;
            }
        }
        return best;
    }

    // ------------------------------------------------------------------ LEVER

    /**
     * LEVER's learned verifier over a program and its execution results.
     */
    public static final class Verifier {
        private final ExecutionTracer tracer;
        private final int examples;
        private final int maxList;
        private final int values;
        private final int maxLength;
        private final Model net;
        private final Optimizer optimizer = Optimizer.adam(3e-3f);
        private final boolean execution;

        /**
         * @param tracer runs programs
         * @param execution whether the verifier sees execution results (LEVER) or only the
         *     program and the examples (a verifier without execution)
         * @param examples given examples read
         * @param maxList list positions encoded
         * @param values distinct values
         * @param maxLength the longest program
         * @param hidden the hidden width
         * @param seed the seed
         */
        public Verifier(ExecutionTracer tracer, boolean execution, int examples, int maxList, int values,
                int maxLength, int hidden, long seed) {
            this.tracer = Objects.requireNonNull(tracer);
            this.execution = execution;
            this.examples = examples;
            this.maxList = maxList;
            this.values = values;
            this.maxLength = maxLength;
            this.net = Model.sequential().input(features()).dense(hidden, Activation.TANH).dense(1).seed(seed)
                    .build();
        }

        private int features() {
            int per = maxList * (values + 1) + maxList + 1;
            return maxLength * tracer.primitives().size() + examples * (3 * per + 2);
        }

        /**
         * The verifier's input: the program one-hot by position, then per example the input,
         * the expected output, the program's actual output, whether it ran and whether it matched.
         *
         * @param program the program
         * @param inputs the given inputs
         * @param outputs the expected outputs
         * @return the features
         */
        public float[] encode(List<String> program, int[][] inputs, int[][] outputs) {
            int p = tracer.primitives().size();
            int per = maxList * (values + 1) + maxList + 1;
            float[] f = new float[features()];
            for (int i = 0; i < Math.min(program.size(), maxLength); i++) {
                f[i * p + tracer.primitives().indexOf(program.get(i))] = 1;
            }
            int off = maxLength * p;
            for (int e = 0; e < Math.min(examples, inputs.length); e++) {
                int[] actual = execution ? tracer.run(program, inputs[e]) : null;
                int base = off + e * (3 * per + 2);
                ProgramSampler.encodeList(f, base, inputs[e], maxList, values);
                ProgramSampler.encodeList(f, base + per, outputs[e], maxList, values);
                ProgramSampler.encodeList(f, base + 2 * per, actual, maxList, values);
                if (!execution) {
                    continue;
                }
                f[base + 3 * per] = actual == null ? 0 : 1;
                f[base + 3 * per + 1] = actual != null && Arrays.equals(actual, outputs[e]) ? 1 : 0;
            }
            return f;
        }

        private Var logits(List<float[]> xs) {
            int d = features();
            float[] v = new float[xs.size() * d];
            for (int i = 0; i < xs.size(); i++) {
                System.arraycopy(xs.get(i), 0, v, i * d, d);
            }
            return net.forward(Var.of(Tensor.of(Shape.of(xs.size(), d), v)), true);
        }

        /**
         * One binary cross-entropy step.
         *
         * @param xs encoded programs
         * @param correct whether each is right on hidden tests
         * @return the loss before the step
         */
        public float step(List<float[]> xs, boolean[] correct) {
            float[] s = new float[correct.length];
            for (int i = 0; i < s.length; i++) {
                s[i] = correct[i] ? 1 : -1;
            }
            Var z = logits(xs).mul(Tensor.of(Shape.of(s.length, 1), s));
            Var loss = z.neg().softplus().mean();
            float v = loss.value().item();
            loss.backward();
            optimizer.step(net.parameters());
            return v;
        }

        /**
         * @param x an encoded program
         * @return {@code P(correct)}
         */
        public double probability(float[] x) {
            float z = Var.noGrad(() -> logits(List.of(x))).value().toArray()[0];
            return 1 / (1 + Math.exp(-z));
        }
    }

    /**
     * LEVER's reranking: programs with the same outputs on the given inputs pool their
     * generation probability; each is scored by {@code P(correct) * P(generated)} and the best
     * group's most likely member is submitted.
     *
     * @param candidates the samples
     * @param verifier the trained verifier
     * @param inputs the given inputs
     * @param outputs the expected outputs
     * @return the submission, or {@code null} if there are none
     */
    public Candidate lever(List<Candidate> candidates, Verifier verifier, int[][] inputs, int[][] outputs) {
        Map<String, List<Candidate>> groups = new LinkedHashMap<>();
        for (Candidate c : candidates) {
            groups.computeIfAbsent(signature(c.program(), inputs), k -> new ArrayList<>()).add(c);
        }
        Candidate best = null;
        double bestScore = -1;
        for (List<Candidate> g : groups.values()) {
            double pGen = 0;
            for (Candidate c : g) {
                pGen += Math.exp(c.logProb());
            }
            Candidate top = likelihood(g);
            double score = verifier.probability(verifier.encode(top.program(), inputs, outputs)) * pGen;
            if (score > bestScore) {
                bestScore = score;
                best = top;
            }
        }
        return best;
    }
}
