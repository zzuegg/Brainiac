package io.github.zzuegg.brainiac.experimental.evolve;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.BiFunction;

/**
 * Neural-guided enumerative synthesis of straight-line list programs, as in DeepCoder:
 * an {@link AttributePredictor} reads a task's input/output examples and predicts which
 * primitives the solution uses, and that prediction orders an exact enumerative search.
 * The search stays complete (the network never decides what is correct, only what to try
 * first), so guidance can only change how many programs are enumerated, not whether a
 * found program is right.
 *
 * <p>The enumerator ({@link #enumerate}) is breadth-first over program length with
 * observational-equivalence pruning: a partial program whose outputs on the examples equal
 * those of one already seen is dropped, because any extension of it behaves like an
 * extension of the earlier one. Cost is counted in primitive applications to the whole
 * example set, the unit {@code SelfImprovingCoders} charges. {@link #sortAndAdd} is
 * DeepCoder's "sort and add" schedule: enumerate with only the {@code k} most probable
 * primitives, and if that fails add the next most probable one and enumerate again.
 *
 * <p>Paper: Balog, Gaunt, Brockschmidt, Nowozin, Tarlow, "DeepCoder: Learning to Write
 * Programs" (ICLR 2017), <a href="https://arxiv.org/abs/1611.01989">arXiv:1611.01989</a>.
 */
public final class GuidedEnumerator {

    /**
     * A search outcome.
     *
     * @param program the program found, or {@code null}
     * @param evaluated primitive applications spent
     */
    public record Result(List<String> program, long evaluated) {
        /** @return whether a program was found */
        public boolean solved() {
            return program != null;
        }
    }

    private final List<String> primitives;
    private final BiFunction<String, int[], int[]> semantics;

    /**
     * Creates the enumerator for a DSL.
     *
     * @param primitives the primitive names
     * @param semantics applies a primitive to a list; {@code null} means the program fails
     */
    public GuidedEnumerator(List<String> primitives, BiFunction<String, int[], int[]> semantics) {
        this.primitives = List.copyOf(primitives);
        this.semantics = semantics;
    }

    /** @return the primitive names */
    public List<String> primitives() {
        return primitives;
    }

    /**
     * Breadth-first enumeration with observational-equivalence pruning.
     *
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @param maxLength the longest program
     * @param allowed the primitives to use, in the order to try them
     * @param budget the most primitive applications to spend
     * @return the first (shortest) program that reproduces every example
     */
    public Result enumerate(int[][] inputs, int[][] outputs, int maxLength, List<String> allowed, long budget) {
        return enumerate(inputs, outputs, maxLength, allowed, null, budget);
    }

    /**
     * Breadth-first enumeration in which each level's programs are tried best-first by the
     * summed log-probability of their primitives: the search is as complete as the plain
     * one, but within the length that holds the solution it reaches it sooner.
     *
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @param maxLength the longest program
     * @param allowed the primitives to use
     * @param probabilities per primitive of {@link #primitives()}, the chance the solution
     *     uses it, or {@code null} to keep the given order
     * @param budget the most primitive applications to spend
     * @return the first program that reproduces every example
     */
    public Result enumerate(int[][] inputs, int[][] outputs, int maxLength, List<String> allowed,
            double[] probabilities, long budget) {
        if (probabilities != null) {
            List<String> sorted = new ArrayList<>(allowed);
            sorted.sort((a, b) -> Double.compare(probabilities[primitives.indexOf(b)],
                    probabilities[primitives.indexOf(a)]));
            allowed = sorted;
        }
        long cost = 0;
        List<int[][]> frontier = new ArrayList<>();
        List<List<String>> programs = new ArrayList<>();
        frontier.add(inputs);
        programs.add(List.of());
        Set<String> seen = new HashSet<>();
        seen.add(Arrays.deepToString(inputs));
        for (int len = 1; len <= maxLength; len++) {
            List<int[][]> next = new ArrayList<>();
            List<List<String>> nextPrograms = new ArrayList<>();
            for (int f = 0; f < frontier.size(); f++) {
                int[][] in = frontier.get(f);
                for (String op : allowed) {
                    if (++cost > budget) {
                        return new Result(null, budget);
                    }
                    int[][] out = new int[in.length][];
                    boolean right = true;
                    boolean failed = false;
                    for (int e = 0; e < in.length && !failed; e++) {
                        out[e] = semantics.apply(op, in[e]);
                        failed = out[e] == null;
                        right &= !failed && Arrays.equals(out[e], outputs[e]);
                    }
                    if (failed) {
                        continue;
                    }
                    List<String> program = new ArrayList<>(programs.get(f));
                    program.add(op);
                    if (right) {
                        return new Result(List.copyOf(program), cost);
                    }
                    if (len < maxLength && seen.add(Arrays.deepToString(out))) {
                        next.add(out);
                        nextPrograms.add(program);
                    }
                }
            }
            if (probabilities != null) {
                Integer[] order = new Integer[next.size()];
                double[] score = new double[next.size()];
                for (int i = 0; i < order.length; i++) {
                    order[i] = i;
                    for (String op : nextPrograms.get(i)) {
                        score[i] += Math.log(probabilities[primitives.indexOf(op)] + 1e-6);
                    }
                }
                Arrays.sort(order, (a, b) -> Double.compare(score[b], score[a]));
                List<int[][]> f2 = new ArrayList<>();
                List<List<String>> p2 = new ArrayList<>();
                for (int i : order) {
                    f2.add(next.get(i));
                    p2.add(nextPrograms.get(i));
                }
                next = f2;
                nextPrograms = p2;
            }
            frontier = next;
            programs = nextPrograms;
        }
        return new Result(null, cost);
    }

    /**
     * Unguided enumeration over every primitive.
     *
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @param maxLength the longest program
     * @param budget the most primitive applications to spend
     * @return the outcome
     */
    public Result enumerate(int[][] inputs, int[][] outputs, int maxLength, long budget) {
        return enumerate(inputs, outputs, maxLength, primitives, budget);
    }

    /**
     * DeepCoder's sort-and-add search: enumerate over the {@code startK} most probable
     * primitives, then over one more each time the search fails, until the budget runs out.
     *
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @param maxLength the longest program
     * @param scores a probability per primitive (higher is tried first, and orders each
     *     level), as from {@link AttributePredictor#predict}
     * @param startK how many primitives the first round uses
     * @param budget the most primitive applications to spend over all rounds
     * @return the outcome, with the cost of every round counted
     */
    public Result sortAndAdd(int[][] inputs, int[][] outputs, int maxLength, double[] scores, int startK,
            long budget) {
        Integer[] order = new Integer[primitives.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[b], scores[a]));
        long spent = 0;
        for (int k = Math.min(startK, order.length); k <= order.length; k++) {
            List<String> active = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                active.add(primitives.get(order[i]));
            }
            Result r = enumerate(inputs, outputs, maxLength, active, scores, budget - spent);
            spent += r.evaluated();
            if (r.solved()) {
                return new Result(r.program(), spent);
            }
            if (spent >= budget) {
                break;
            }
        }
        return new Result(null, spent);
    }

    /**
     * DeepCoder's attribute predictor: an MLP encodes each input/output example, the
     * encodings are averaged over the examples, and a sigmoid layer predicts, per primitive,
     * whether the program uses it. Lists are encoded as one-hot values (0..{@code VALUES-1}
     * plus one overflow bucket) at the first {@code POSITIONS} positions, plus their lengths.
     */
    public static final class AttributePredictor {
        /** Positions of a list that are encoded. */
        public static final int POSITIONS = 8;
        /** Distinct values encoded; larger ones share an overflow bucket. */
        public static final int VALUES = 20;
        private static final int PER_LIST = POSITIONS * (VALUES + 1) + 1;

        private final int classes;
        private final Model encoder;
        private final Model head;
        private final Optimizer optimizer = Optimizer.adam(2e-3f);

        /**
         * Creates an untrained predictor.
         *
         * @param primitives how many primitives the DSL has
         * @param hidden the hidden width
         * @param seed the seed
         */
        public AttributePredictor(int primitives, int hidden, long seed) {
            this.classes = primitives;
            encoder = Model.sequential().input(2 * PER_LIST).dense(hidden, Activation.RELU)
                    .dense(hidden, Activation.RELU).seed(seed).build();
            head = Model.sequential().input(hidden).dense(primitives, Activation.IDENTITY).seed(seed + 1).build();
        }

        /**
         * The fixed-size encoding of one list.
         *
         * @param xs the list
         * @param into the destination
         * @param offset where to write {@code PER_LIST} values
         */
        static void encodeList(int[] xs, float[] into, int offset) {
            for (int p = 0; p < Math.min(POSITIONS, xs.length); p++) {
                into[offset + p * (VALUES + 1) + Math.min(VALUES, Math.max(0, xs[p]))] = 1;
            }
            into[offset + POSITIONS * (VALUES + 1)] = xs.length / 8f;
        }

        private Var logits(List<int[][]> inputs, List<int[][]> outputs) {
            int b = inputs.size();
            int e = inputs.get(0).length;
            float[] x = new float[b * e * 2 * PER_LIST];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < e; j++) {
                    int off = (i * e + j) * 2 * PER_LIST;
                    encodeList(inputs.get(i)[j], x, off);
                    encodeList(outputs.get(i)[j], x, off + PER_LIST);
                }
            }
            Var h = encoder.forward(Var.of(Tensor.of(Shape.of(b * e, 2 * PER_LIST), x)), true);
            int hidden = h.shape().dim(1);
            return head.forward(h.reshape(b, e, hidden).mean(1, false), true);
        }

        /**
         * One training step on programs with known primitives (binary cross-entropy).
         *
         * @param inputs per task, its example inputs (same count for every task)
         * @param outputs per task, its outputs
         * @param used per task, a 0/1 flag per primitive
         * @return the loss
         */
        public float trainStep(List<int[][]> inputs, List<int[][]> outputs, List<float[]> used) {
            Var z = logits(inputs, outputs);
            float[] y = new float[used.size() * classes];
            for (int i = 0; i < used.size(); i++) {
                System.arraycopy(used.get(i), 0, y, i * classes, classes);
            }
            Var loss = z.softplus().sub(z.mul(Tensor.of(Shape.of(used.size(), classes), y))).mean();
            float v = loss.value().item();
            loss.backward();
            List<Var> params = new ArrayList<>(encoder.parameters());
            params.addAll(head.parameters());
            optimizer.step(params);
            return v;
        }

        /**
         * The probability that the solution uses each primitive.
         *
         * @param inputs the task's example inputs
         * @param outputs its outputs
         * @return one probability per primitive
         */
        public double[] predict(int[][] inputs, int[][] outputs) {
            float[] z = Var.noGrad(() -> logits(java.util.Collections.singletonList(inputs), java.util.Collections.singletonList(outputs)).sigmoid().value().toArray());
            double[] out = new double[z.length];
            for (int i = 0; i < z.length; i++) {
                out[i] = z[i];
            }
            return out;
        }
    }

    /**
     * Draws a random program and examples for training the predictor.
     *
     * @param length the program length
     * @param examples how many examples
     * @param random the random source
     * @return {@code {program indices, inputs, outputs}} as an {@code Object[]}, or
     *     {@code null} if the program fails or leaves every input unchanged
     */
    public Object[] randomTask(int length, int examples, SplittableRandom random) {
        List<String> program = new ArrayList<>();
        int[] idx = new int[length];
        for (int i = 0; i < length; i++) {
            idx[i] = random.nextInt(primitives.size());
            program.add(primitives.get(idx[i]));
        }
        int[][] in = new int[examples][];
        int[][] out = new int[examples][];
        boolean changes = false;
        for (int e = 0; e < examples; e++) {
            in[e] = new int[5 + random.nextInt(3)];
            for (int i = 0; i < in[e].length; i++) {
                in[e][i] = random.nextInt(10);
            }
            int[] xs = in[e];
            for (String op : program) {
                xs = xs == null ? null : semantics.apply(op, xs);
            }
            if (xs == null) {
                return null;
            }
            out[e] = xs;
            changes |= !Arrays.equals(xs, in[e]);
        }
        return changes ? new Object[] {idx, in, out} : null;
    }
}
