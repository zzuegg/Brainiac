package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.GraphOfThoughts;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Graph of Thoughts against Tree of Thoughts on the paper's sorting task (item 5.3): sort 32
 * digits with a noisy model. The simulated model sorts a list of {@code n} elements with a
 * per-element error rate that grows with {@code n} (a dropped, duplicated or miscopied
 * element), as an LLM's error rate grows with the length it must attend over; merging two
 * sorted lists is a local operation and errs four times less. A thought is scored, as in the
 * paper, by its errors against its own input: descents in the output plus the per-digit
 * count mismatch.
 * <ul>
 *   <li><b>IO</b>: one call sorts the whole list.</li>
 *   <li><b>ToT</b>: levels of {@code k} attempts, keeping the best and refining it, until
 *       the call budget is spent or the score is perfect.</li>
 *   <li><b>GoT</b>: split into four chunks of eight, sort each three times and keep the
 *       best, merge pairs (best of two), and merge the halves with the rest of the budget
 *       (best of the attempts).</li>
 * </ul>
 */
public final class GraphOfThoughtsSorting {

    /**
     * Results over the problems.
     *
     * @param ioErrors mean errors of one call
     * @param totErrors mean errors of Tree of Thoughts
     * @param gotErrors mean errors of Graph of Thoughts
     * @param ioSolved share sorted exactly by one call
     * @param totSolved share sorted exactly by ToT
     * @param gotSolved share sorted exactly by GoT
     * @param totCalls mean model calls of ToT
     * @param gotCalls mean model calls of GoT
     */
    public record Result(double ioErrors, double totErrors, double gotErrors, double ioSolved, double totSolved,
            double gotSolved, double totCalls, double gotCalls) {
    }

    /**
     * A thought: a list and the input it should be a sorting of.
     *
     * @param input the input
     * @param output the current answer
     */
    public record Part(int[] input, int[] output) {
    }

    private GraphOfThoughtsSorting() {
    }

    /**
     * @param p a thought
     * @return descents plus per-digit count mismatches; 0 for a correct sort
     */
    public static int errors(Part p) {
        int e = 0;
        for (int i = 1; i < p.output().length; i++) {
            e += p.output()[i] < p.output()[i - 1] ? 1 : 0;
        }
        int[] counts = new int[10];
        for (int x : p.input()) {
            counts[x]++;
        }
        for (int x : p.output()) {
            counts[x]--;
        }
        for (int c : counts) {
            e += Math.abs(c);
        }
        return e;
    }

    /** The simulated model: a correct answer, corrupted elementwise at the given rate. */
    static int[] noisy(int[] correct, double rate, SplittableRandom r) {
        List<Integer> out = new ArrayList<>();
        for (int x : correct) {
            if (r.nextDouble() >= rate) {
                out.add(x);
                continue;
            }
            switch (r.nextInt(3)) {
                case 0 -> out.add(r.nextInt(10));   // miscopied
                case 1 -> { }                         // dropped
                default -> {
                    out.add(x);                       // duplicated
                    out.add(x);
                }
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    static int[] sorted(int[] xs) {
        int[] s = xs.clone();
        Arrays.sort(s);
        return s;
    }

    /**
     * @param problems how many lists
     * @param length their length (a multiple of 4)
     * @param rate the sorting error rate per element per element of length
     * @param budget the model calls ToT may use
     * @param seed the seed
     * @return the comparison
     */
    public static Result run(int problems, int length, double rate, int budget, long seed) {
        SplittableRandom data = new SplittableRandom(seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        double[] err = new double[3];
        double[] solved = new double[3];
        double totCalls = 0;
        double gotCalls = 0;
        for (int pi = 0; pi < problems; pi++) {
            int[] input = new int[length];
            for (int i = 0; i < length; i++) {
                input[i] = data.nextInt(10);
            }
            // the model's calls
            java.util.function.UnaryOperator<Part> sort = p -> new Part(p.input(),
                    noisy(sorted(p.input()), Math.min(0.5, rate * p.input().length), r));
            java.util.function.UnaryOperator<Part> refine = sort;   // re-sorting a whole list is as hard
            java.util.function.Function<List<Part>, Part> merge = parts -> {
                int[] in = parts.stream().flatMapToInt(q -> Arrays.stream(q.input())).toArray();
                int[] joined = sorted(parts.stream().flatMapToInt(q -> Arrays.stream(q.output())).toArray());
                return new Part(in, noisy(joined, Math.min(0.5, rate * in.length / 4), r));
            };
            Part problem = new Part(input, input);

            // IO
            GraphOfThoughts<Part> io = GraphOfThoughts.of(p -> -errors(p));
            Part ioOut = io.generate(io.root(problem), 1, sort).getFirst().content();
            record(err, solved, 0, ioOut);

            // ToT: k attempts per level, keep the best, refine it in the next level
            GraphOfThoughts<Part> tot = GraphOfThoughts.of(p -> -errors(p));
            int k = 4;
            GraphOfThoughts.Thought<Part> best = tot.keepBest(tot.generate(tot.root(problem), k, sort), 1).getFirst();
            while (tot.calls() + k <= budget && best.score() < 0) {
                GraphOfThoughts.Thought<Part> next = tot.keepBest(tot.generate(best, k, refine), 1).getFirst();
                if (next.score() > best.score()) {
                    best = next;
                }
            }
            record(err, solved, 1, best.content());
            totCalls += tot.calls();

            // GoT: split, sort the chunks, merge
            GraphOfThoughts<Part> got = GraphOfThoughts.of(p -> -errors(p));
            List<GraphOfThoughts.Thought<Part>> chunks = got.split(got.root(problem), p -> {
                List<Part> out = new ArrayList<>();
                int c = p.input().length / 4;
                for (int i = 0; i < 4; i++) {
                    int[] in = Arrays.copyOfRange(p.input(), i * c, (i + 1) * c);
                    out.add(new Part(in, in));
                }
                return out;
            });
            List<GraphOfThoughts.Thought<Part>> sortedChunks = new ArrayList<>();
            for (GraphOfThoughts.Thought<Part> c : chunks) {
                sortedChunks.add(got.keepBest(got.generate(c, 3, sort), 1).getFirst());
            }
            GraphOfThoughts.Thought<Part> left = got.aggregate(sortedChunks.subList(0, 2), 2, merge);
            GraphOfThoughts.Thought<Part> right = got.aggregate(sortedChunks.subList(2, 4), 2, merge);
            // the rest of the budget goes to attempts at the final merge
            GraphOfThoughts.Thought<Part> whole = got.aggregate(List.of(left, right), Math.max(1, budget - got.calls()),
                    merge);
            record(err, solved, 2, whole.content());
            gotCalls += got.calls();
        }
        double n = problems;
        return new Result(err[0] / n, err[1] / n, err[2] / n, solved[0] / n, solved[1] / n, solved[2] / n,
                totCalls / n, gotCalls / n);
    }

    private static void record(double[] err, double[] solved, int i, Part p) {
        err[i] += errors(p);
        solved[i] += Arrays.equals(p.output(), sorted(p.input())) ? 1 : 0;
    }
}
