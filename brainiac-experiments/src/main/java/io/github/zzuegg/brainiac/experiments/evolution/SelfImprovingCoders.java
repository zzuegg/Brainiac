package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experimental.evolve.SelfModifyingArchive;
import io.github.zzuegg.brainiac.experiments.reasoning.ListPrograms;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.OptionalDouble;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * A toy Darwin Gödel Machine (item 5.6): coding agents improve their own scaffold. An agent
 * is a {@link Scaffold} for an enumerative program synthesizer over {@link ListPrograms}:
 * how long the programs it writes may be, whether it prunes programs that behave like one it
 * has already written (observational equivalence, which costs a hash per program), and
 * whether it skips pointless repeats such as {@code sort sort}. The benchmark is a set of
 * synthesis tasks with a fixed compute budget per task, scored as the share solved minus a
 * charge for the compute spent, as a real coding benchmark charges for tokens.
 *
 * <p>The agent also carries its own edit policy (weights over which part of itself to change)
 * and modifies itself with it, editing those weights too. The landscape is deceptive: longer
 * programs without pruning burn the budget, and pruning alone only adds cost, so each single
 * edit from the starting agent scores lower; only the two together pay off. The open-ended
 * {@link SelfModifyingArchive} keeps the weaker intermediate agents as stepping stones;
 * greedy hill-climbing never accepts them.
 */
public final class SelfImprovingCoders {

    /**
     * A coding agent's scaffold.
     *
     * @param maxLength the longest program it writes
     * @param prune whether it prunes observationally equivalent programs
     * @param skipRepeats whether it skips idempotent or self-cancelling repeats
     * @param editWeights its own policy over edits: length, prune, skip, policy
     */
    public record Scaffold(int maxLength, boolean prune, boolean skipRepeats, double[] editWeights) {
        /**
         * The agent edits itself, picking what to change with its own policy.
         *
         * @param random the random source
         * @return the edited agent
         */
        public Scaffold modifySelf(SplittableRandom random) {
            double total = Arrays.stream(editWeights).sum();
            double u = random.nextDouble() * total;
            int edit = 0;
            while (edit < editWeights.length - 1 && (u -= editWeights[edit]) > 0) {
                edit++;
            }
            return switch (edit) {
                case 0 -> new Scaffold(Math.clamp(maxLength + (random.nextBoolean() ? 1 : -1), 1, 6), prune,
                        skipRepeats, editWeights);
                case 1 -> new Scaffold(maxLength, !prune, skipRepeats, editWeights);
                case 2 -> new Scaffold(maxLength, prune, !skipRepeats, editWeights);
                default -> {
                    double[] w = editWeights.clone();
                    int i = random.nextInt(w.length);
                    w[i] = Math.clamp(w[i] * (random.nextBoolean() ? 2 : 0.5), 0.05, 20);
                    yield new Scaffold(maxLength, prune, skipRepeats, w);
                }
            };
        }

        String key() {
            return maxLength + "/" + prune + "/" + skipRepeats;
        }

        @Override
        public String toString() {
            return "Scaffold(len " + maxLength + ", prune " + prune + ", skip " + skipRepeats + ")";
        }
    }

    /**
     * Scores over generations.
     *
     * @param archive best score after each generation, open-ended archive
     * @param greedy best score after each generation, hill-climbing
     * @param archiveBest the archive's best agent
     * @param greedyBest where hill-climbing ended
     * @param archiveSize agents in the archive at the end
     * @param initial the starting agent's score
     */
    public record Result(double[] archive, double[] greedy, String archiveBest, String greedyBest, int archiveSize,
            double initial) {
    }

    private static final Set<String> IDEMPOTENT = Set.of("sort", "uniq", "even", "odd", "head2");
    private static final Set<String> INVOLUTION = Set.of("rev", "swap");

    private final List<ListPrograms.Task> benchmark;
    private final int budget;
    private final double costWeight;
    private final Map<String, Double> cache = new HashMap<>();

    /**
     * @param benchmark the synthesis tasks
     * @param budget primitive applications allowed per task
     * @param costWeight the charge for spending the whole budget on every task
     */
    public SelfImprovingCoders(List<ListPrograms.Task> benchmark, int budget, double costWeight) {
        this.benchmark = List.copyOf(benchmark);
        this.budget = budget;
        this.costWeight = costWeight;
    }

    /**
     * @param seed the seed
     * @return ten tasks each of one, two and three concepts (two, four and six primitives)
     */
    public static List<ListPrograms.Task> defaultBenchmark(long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<ListPrograms.Task> out = new ArrayList<>();
        for (ListPrograms.Concept c : ListPrograms.CONCEPTS) {
            out.add(ListPrograms.task(List.of(c), 6, random));
            out.add(ListPrograms.task(2, 6, random));
            out.add(ListPrograms.task(3, 6, random));
        }
        return out;
    }

    /**
     * Runs the agent's synthesizer on one task.
     *
     * @param s the scaffold
     * @param task the task
     * @return {solved ? 1 : 0, primitive applications spent}
     */
    long[] solve(Scaffold s, ListPrograms.Task task) {
        List<String> prims = ListPrograms.primitives();
        long cost = 0;
        int perProgram = s.prune() ? 2 : 1;   // pruning hashes every program it writes
        List<int[][]> frontier = new ArrayList<>();
        List<String> lastOp = new ArrayList<>();
        frontier.add(task.inputs());
        lastOp.add("");
        Set<String> seen = new HashSet<>();
        seen.add(Arrays.deepToString(task.inputs()));
        for (int len = 1; len <= s.maxLength(); len++) {
            List<int[][]> next = new ArrayList<>();
            List<String> nextOp = new ArrayList<>();
            for (int f = 0; f < frontier.size(); f++) {
                String last = lastOp.get(f);
                for (String op : prims) {
                    if (s.skipRepeats() && op.equals(last) && (IDEMPOTENT.contains(op) || INVOLUTION.contains(op))) {
                        continue;
                    }
                    cost += perProgram;
                    if (cost > budget) {
                        return new long[] {0, budget};
                    }
                    int[][] in = frontier.get(f);
                    int[][] out = new int[in.length][];
                    boolean right = true;
                    for (int e = 0; e < in.length; e++) {
                        out[e] = ListPrograms.apply(op, in[e]);
                        right &= Arrays.equals(out[e], task.outputs()[e]);
                    }
                    if (right) {
                        return new long[] {1, cost};
                    }
                    if (len < s.maxLength() && (!s.prune() || seen.add(Arrays.deepToString(out)))) {
                        next.add(out);
                        nextOp.add(op);
                    }
                }
            }
            frontier = next;
            lastOp = nextOp;
        }
        return new long[] {0, cost};
    }

    /**
     * The benchmark score: share solved minus the compute charge.
     *
     * @param s the agent
     * @return its score (empty never: every scaffold here runs)
     */
    public OptionalDouble evaluate(Scaffold s) {
        return OptionalDouble.of(cache.computeIfAbsent(s.key(), k -> {
            double solved = 0;
            double cost = 0;
            for (ListPrograms.Task t : benchmark) {
                long[] r = solve(s, t);
                solved += r[0];
                cost += r[1];
            }
            return solved / benchmark.size() - costWeight * cost / ((double) budget * benchmark.size());
        }));
    }

    /**
     * @return the score of every scaffold (length 1 to 6, prune, skip), for inspection
     */
    public Map<String, Double> landscape() {
        Map<String, Double> out = new java.util.TreeMap<>();
        for (int len = 1; len <= 6; len++) {
            for (boolean prune : new boolean[] {false, true}) {
                for (boolean skip : new boolean[] {false, true}) {
                    Scaffold s = new Scaffold(len, prune, skip, new double[] {1, 1, 1, 1});
                    out.put(s.key(), evaluate(s).orElseThrow());
                }
            }
        }
        return out;
    }

    /**
     * @param generations generations of self-modification
     * @param children children per generation
     * @param budget primitive applications per task
     * @param costWeight the compute charge
     * @param seed the seed
     * @return scores over generations for the archive and hill-climbing
     */
    public static Result run(int generations, int children, int budget, double costWeight, long seed) {
        SelfImprovingCoders bench = new SelfImprovingCoders(defaultBenchmark(seed), budget, costWeight);
        Scaffold start = new Scaffold(2, false, false, new double[] {1, 1, 1, 1});
        SelfModifyingArchive<Scaffold> dgm = SelfModifyingArchive.of(start, bench::evaluate,
                Scaffold::modifySelf);
        double[] archive = dgm.run(generations, children, new SplittableRandom(seed));
        SelfModifyingArchive.Climb<Scaffold> greedy = SelfModifyingArchive.hillClimb(start, bench::evaluate,
                Scaffold::modifySelf, generations, children, new SplittableRandom(seed));
        return new Result(archive, greedy.best(), dgm.best().agent().toString(), greedy.agent().toString(),
                dgm.archive().size(), bench.evaluate(start).orElseThrow());
    }
}
