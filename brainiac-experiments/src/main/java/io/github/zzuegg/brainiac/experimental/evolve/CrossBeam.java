package io.github.zzuegg.brainiac.experimental.evolve;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.SplittableRandom;

/**
 * CrossBeam: bottom-up synthesis in which a learned policy chooses which pool values each
 * primitive is applied to next. Every pool value is embedded by an MLP over its property
 * signature against the target ({@link Bustle#signature}); a primitive's score for a value
 * is the dot product of the value embedding with a learned primitive embedding, a pointer
 * over the pool. Each round applies every primitive to its {@code beam} best-scoring values
 * it has not been applied to yet, adds the new values (observational equivalence drops
 * duplicates) and repeats, so the search order is the policy's, not a fixed weight order.
 *
 * <p>Training is on-policy with teacher forcing, as in the paper: the search runs on a
 * random program's examples with the current policy, the ground-truth intermediate value is
 * added to the pool after every round, and the loss is the cross-entropy of the pointer
 * toward the ground-truth value for the ground-truth next primitive.
 *
 * <p>Paper: Shi, Dai, Ellis and Sutton, "CrossBeam: Learning to Search in Bottom-Up Program
 * Synthesis" (ICLR 2022), <a href="https://arxiv.org/abs/2203.10452">arXiv:2203.10452</a>.
 */
public final class CrossBeam {

    /**
     * A search outcome.
     *
     * @param program the program found, or {@code null}
     * @param evaluated primitive applications spent
     * @param pool the final pool size
     */
    public record Result(List<String> program, long evaluated, int pool) {
        /** @return whether a program was found */
        public boolean solved() {
            return program != null;
        }
    }

    private final BottomUpSynthesis search;
    private final Model encoder;
    private final Var ops;
    private final int hidden;

    /**
     * Creates an untrained policy.
     *
     * @param search supplies the primitives and their semantics
     * @param hidden the embedding width
     * @param seed the seed
     */
    public CrossBeam(BottomUpSynthesis search, int hidden, long seed) {
        this.search = search;
        this.hidden = hidden;
        this.encoder = Model.sequential().input(Bustle.SIGNATURE).dense(hidden, Activation.RELU).dense(hidden)
                .seed(seed).build();
        this.ops = Var.param(Tensor.normal(Shape.of(hidden, search.primitives().size()), 0,
                (float) (1 / Math.sqrt(hidden)), new SplittableRandom(seed + 1)));
    }

    /** The growing pool of one search. */
    private final class Pool {
        final int[][] inputs;
        final int[][] outputs;
        final List<int[][]> values = new ArrayList<>();
        final List<List<String>> programs = new ArrayList<>();
        final List<float[]> signatures = new ArrayList<>();
        final Map<String, Integer> index = new HashMap<>();

        Pool(int[][] inputs, int[][] outputs) {
            this.inputs = inputs;
            this.outputs = outputs;
            add(inputs, List.of());
        }

        int add(int[][] value, List<String> program) {
            String key = Arrays.deepToString(value);
            Integer old = index.get(key);
            if (old != null) {
                return -1 - old;
            }
            values.add(value);
            programs.add(program);
            signatures.add(Bustle.signature(value, inputs, outputs));
            index.put(key, values.size() - 1);
            return values.size() - 1;
        }

        Var embeddings(int from, int to) {
            float[] x = new float[(to - from) * Bustle.SIGNATURE];
            for (int i = from; i < to; i++) {
                System.arraycopy(signatures.get(i), 0, x, (i - from) * Bustle.SIGNATURE, Bustle.SIGNATURE);
            }
            return encoder.forward(Var.of(Tensor.of(Shape.of(to - from, Bustle.SIGNATURE), x)), true);
        }
    }

    /**
     * Policy-guided search.
     *
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @param maxLength the longest program
     * @param beam values per primitive per round
     * @param budget the most primitive applications
     * @return the outcome
     */
    public Result search(int[][] inputs, int[][] outputs, int maxLength, int beam, long budget) {
        List<String> prims = search.primitives();
        int k = prims.size();
        Pool pool = new Pool(inputs, outputs);
        List<float[]> scores = new ArrayList<>();
        List<PriorityQueue<Integer>> queues = new ArrayList<>();
        for (int op = 0; op < k; op++) {
            int o = op;
            queues.add(new PriorityQueue<>((a, b) -> Float.compare(scores.get(b)[o], scores.get(a)[o])));
        }
        long cost = 0;
        int scored = 0;
        while (true) {
            if (scored < pool.values.size()) {
                int from = scored;
                int to = pool.values.size();
                float[] s = Var.noGrad(() -> pool.embeddings(from, to).matmul(ops)).value().toArray();
                for (int i = from; i < to; i++) {
                    scores.add(Arrays.copyOfRange(s, (i - from) * k, (i - from + 1) * k));
                    if (pool.programs.get(i).size() < maxLength) {
                        for (PriorityQueue<Integer> q : queues) {
                            q.add(i);
                        }
                    }
                }
                scored = to;
            }
            boolean any = false;
            for (int op = 0; op < k; op++) {
                PriorityQueue<Integer> q = queues.get(op);
                for (int b = 0; b < beam && !q.isEmpty(); b++) {
                    int v = q.poll();
                    any = true;
                    if (++cost > budget) {
                        return new Result(null, budget, pool.values.size());
                    }
                    int[][] out = search.apply(prims.get(op), pool.values.get(v));
                    if (out == null) {
                        continue;
                    }
                    List<String> program = new ArrayList<>(pool.programs.get(v));
                    program.add(prims.get(op));
                    if (Arrays.deepEquals(out, outputs)) {
                        return new Result(List.copyOf(program), cost, pool.values.size());
                    }
                    pool.add(out, program);
                }
            }
            if (!any) {
                return new Result(null, cost, pool.values.size());
            }
        }
    }

    /**
     * On-policy training with teacher forcing on random programs.
     *
     * @param source generates the random programs and examples
     * @param steps training programs
     * @param maxLength the longest training program
     * @param examples examples per program
     * @param beam values per primitive per round during training
     * @param learningRate the Adam rate
     * @param random the source
     * @return the share of the last 100 steps' ground-truth values the pointer ranked first
     */
    public double train(GuidedEnumerator source, int steps, int maxLength, int examples, int beam, float learningRate,
            SplittableRandom random) {
        Optimizer opt = Optimizer.adam(learningRate);
        List<String> prims = search.primitives();
        int k = prims.size();
        int right = 0;
        int total = 0;
        List<Var> params = new ArrayList<>(encoder.parameters());
        params.add(ops);
        for (int s = 0; s < steps; s++) {
            Object[] t = source.randomTask(2 + random.nextInt(Math.max(1, maxLength - 1)), examples, random);
            if (t == null) {
                continue;
            }
            int[] idx = (int[]) t[0];
            int[][] in = (int[][]) t[1];
            int[][] out = (int[][]) t[2];
            Pool pool = new Pool(in, out);
            int truth = 0;
            Var loss = null;
            List<String> program = new ArrayList<>();
            for (int r = 0; r < idx.length; r++) {
                int op = idx[r];
                int n = pool.values.size();
                Var logits = pool.embeddings(0, n).matmul(ops.slice(1, op, op + 1)).reshape(1, n);
                float[] lv = logits.value().toArray();
                int arg = 0;
                for (int i = 1; i < n; i++) {
                    arg = lv[i] > lv[arg] ? i : arg;
                }
                if (s >= steps - 100) {
                    right += arg == truth ? 1 : 0;
                    total++;
                }
                Var term = logits.logSoftmax(1).slice(1, truth, truth + 1).neg();
                loss = loss == null ? term : loss.add(term);
                float[][] all = new float[n][];
                Var emb = Var.noGrad(() -> pool.embeddings(0, n).matmul(ops));
                float[] ev = emb.value().toArray();
                for (int i = 0; i < n; i++) {
                    all[i] = Arrays.copyOfRange(ev, i * k, (i + 1) * k);
                }
                for (int o = 0; o < k; o++) {
                    Integer[] order = new Integer[n];
                    for (int i = 0; i < n; i++) {
                        order[i] = i;
                    }
                    int oo = o;
                    Arrays.sort(order, (a, b) -> Float.compare(all[b][oo], all[a][oo]));
                    for (int b = 0; b < Math.min(beam, n); b++) {
                        int[][] v = search.apply(prims.get(o), pool.values.get(order[b]));
                        if (v != null) {
                            List<String> p = new ArrayList<>(pool.programs.get(order[b]));
                            p.add(prims.get(o));
                            pool.add(v, p);
                        }
                    }
                }
                program.add(prims.get(op));
                int[][] next = search.apply(prims.get(op), pool.values.get(truth));
                int at = pool.add(next, List.copyOf(program));
                truth = at >= 0 ? at : -1 - at;
            }
            if (loss != null) {
                loss.mul(1f / idx.length).backward();
                opt.step(params);
            }
        }
        return right / (double) Math.max(1, total);
    }
}
