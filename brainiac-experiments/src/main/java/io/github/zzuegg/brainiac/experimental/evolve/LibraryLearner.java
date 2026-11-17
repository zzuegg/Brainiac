package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.function.Predicate;

/**
 * DreamCoder-style library learning (Ellis et al.,
 * <a href="https://arxiv.org/abs/2006.08381">2006.08381</a>) for straight-line programs:
 * sequences of primitive names, such as list or string transformations. It alternates two
 * phases:
 * <ul>
 *   <li><b>wake</b>: for each unsolved task, enumerate programs over the current library in
 *       order of decreasing prior probability, up to a budget of evaluations, and keep the
 *       first one the task accepts;</li>
 *   <li><b>sleep (abstraction)</b>: look for program fragments, contiguous runs of tokens,
 *       that recur across the solutions of different tasks; add those that compress the
 *       solutions most as new primitives, and rewrite the solutions with them;</li>
 *   <li><b>sleep (dreaming)</b>: refit the prior, a smoothed unigram over library tokens,
 *       on the rewritten solutions, a minimal recognition model that makes the enumerator
 *       try the tokens that solved tasks before.</li>
 * </ul>
 * A program that needs six primitives is out of reach of the enumerator, but becomes a
 * two-token program once the right abstractions exist; solved tasks feed new abstractions,
 * which solve harder tasks, which feed deeper abstractions.
 */
public final class LibraryLearner {

    /**
     * A library entry.
     *
     * @param name its token
     * @param expansion the base primitives it stands for
     */
    public record Entry(String name, List<String> expansion) {
    }

    /**
     * A solved task.
     *
     * @param program the program over the library at the time it was found
     * @param expanded the same program in base primitives
     */
    public record Solution(List<String> program, List<String> expanded) {
    }

    /**
     * The wake phase's outcome.
     *
     * @param solved tasks solved so far (this round and earlier)
     * @param newlySolved tasks solved in this round
     * @param evaluations programs evaluated in this round
     * @param evaluationsPerSolve mean evaluations of the tasks solved this round
     */
    public record Wake(int solved, int newlySolved, long evaluations, double evaluationsPerSolve) {
    }

    private final List<Entry> library = new ArrayList<>();
    private final Map<String, Double> logPrior = new HashMap<>();
    private final Map<Integer, Solution> solutions = new LinkedHashMap<>();
    private int maxLength = 4;
    private int budget = 2000;
    private int maxFragment = 4;
    private int minUses = 2;
    private int maxNewPerSleep = 8;
    private int abstractions;

    private LibraryLearner(List<String> primitives) {
        for (String p : primitives) {
            library.add(new Entry(p, List.of(p)));
        }
        refitPrior();
    }

    /**
     * @param primitives the base primitives
     * @return a learner whose library is just the primitives, with a uniform prior
     */
    public static LibraryLearner of(List<String> primitives) {
        return new LibraryLearner(primitives);
    }

    /**
     * @param tokens the longest program the enumerator writes, in library tokens (4 by default)
     * @return this learner
     */
    public LibraryLearner maxLength(int tokens) {
        this.maxLength = tokens;
        return this;
    }

    /**
     * @param evaluations programs evaluated per task per wake (2000 by default)
     * @return this learner
     */
    public LibraryLearner budget(int evaluations) {
        this.budget = evaluations;
        return this;
    }

    /**
     * @param longest the longest fragment considered, in tokens (4 by default)
     * @param uses the fewest distinct tasks a fragment must occur in (2 by default)
     * @param perSleep the most abstractions added per sleep (8 by default)
     * @return this learner
     */
    public LibraryLearner abstraction(int longest, int uses, int perSleep) {
        this.maxFragment = longest;
        this.minUses = uses;
        this.maxNewPerSleep = perSleep;
        return this;
    }

    /** @return the library, primitives first */
    public List<Entry> library() {
        return List.copyOf(library);
    }

    /** @return the solutions found, by task index */
    public Map<Integer, Solution> solutions() {
        return Map.copyOf(solutions);
    }

    private Entry entry(String name) {
        for (Entry e : library) {
            if (e.name().equals(name)) {
                return e;
            }
        }
        throw new IllegalArgumentException("unknown token " + name);
    }

    /**
     * @param program library tokens
     * @return the program in base primitives
     */
    public List<String> expand(List<String> program) {
        List<String> out = new ArrayList<>();
        for (String t : program) {
            out.addAll(entry(t).expansion());
        }
        return out;
    }

    private record Partial(List<String> tokens, double logP) {
    }

    /**
     * Enumerates programs best-first by prior until one solves the task or the budget runs out.
     *
     * @param task accepts a program in base primitives
     * @param evaluations receives the number of programs evaluated
     * @return the solution, or {@code null}
     */
    public Solution search(Predicate<List<String>> task, long[] evaluations) {
        PriorityQueue<Partial> queue = new PriorityQueue<>((a, b) -> Double.compare(b.logP(), a.logP()));
        queue.add(new Partial(List.of(), 0));
        int used = 0;
        while (!queue.isEmpty() && used < budget) {
            Partial p = queue.poll();
            if (!p.tokens().isEmpty()) {
                used++;
                List<String> expanded = expand(p.tokens());
                if (task.test(expanded)) {
                    evaluations[0] += used;
                    return new Solution(p.tokens(), expanded);
                }
            }
            if (p.tokens().size() < maxLength) {
                for (Entry e : library) {
                    List<String> next = new ArrayList<>(p.tokens());
                    next.add(e.name());
                    queue.add(new Partial(next, p.logP() + logPrior.get(e.name())));
                }
            }
        }
        evaluations[0] += used;
        return null;
    }

    /**
     * The wake phase: searches every task not solved yet.
     *
     * @param tasks the tasks, each accepting a program in base primitives
     * @return what this round did
     */
    public Wake wake(List<? extends Predicate<List<String>>> tasks) {
        long[] evaluations = {0};
        int fresh = 0;
        long freshEvals = 0;
        for (int i = 0; i < tasks.size(); i++) {
            if (solutions.containsKey(i)) {
                continue;
            }
            long before = evaluations[0];
            Solution s = search(tasks.get(i), evaluations);
            if (s != null) {
                solutions.put(i, s);
                fresh++;
                freshEvals += evaluations[0] - before;
            }
        }
        return new Wake(solutions.size(), fresh, evaluations[0], fresh == 0 ? 0 : freshEvals / (double) fresh);
    }

    /**
     * The sleep phase: adds the fragments that best compress the solutions as new library
     * entries, rewrites the solutions with them and refits the prior.
     *
     * @return the entries added
     */
    public List<Entry> sleep() {
        List<Entry> added = new ArrayList<>();
        for (int round = 0; round < maxNewPerSleep; round++) {
            Map<List<String>, Set<Integer>> occurrences = new HashMap<>();
            for (Map.Entry<Integer, Solution> s : solutions.entrySet()) {
                List<String> base = s.getValue().expanded();
                for (int len = 2; len <= maxFragment; len++) {
                    for (int at = 0; at + len <= base.size(); at++) {
                        occurrences.computeIfAbsent(List.copyOf(base.subList(at, at + len)), k -> new HashSet<>())
                                .add(s.getKey());
                    }
                }
            }
            List<String> best = null;
            int bestGain = 0;
            for (Map.Entry<List<String>, Set<Integer>> e : occurrences.entrySet()) {
                int uses = e.getValue().size();
                // tokens saved over all solutions, minus one for the new library entry
                int gain = uses * (e.getKey().size() - 1) - 1;
                if (uses < minUses || gain <= 0 || inLibrary(e.getKey())) {
                    continue;
                }
                if (gain > bestGain || gain == bestGain && e.getKey().toString().compareTo(best.toString()) < 0) {
                    best = e.getKey();
                    bestGain = gain;
                }
            }
            if (best == null) {
                break;
            }
            Entry entry = new Entry("f" + abstractions++ + best, List.copyOf(best));
            library.add(entry);
            added.add(entry);
        }
        rewriteSolutions();
        refitPrior();
        return added;
    }

    private boolean inLibrary(List<String> expansion) {
        for (Entry e : library) {
            if (e.expansion().equals(expansion)) {
                return true;
            }
        }
        return false;
    }

    /** Rewrites each solution greedily with the longest matching entries. */
    private void rewriteSolutions() {
        for (Map.Entry<Integer, Solution> s : solutions.entrySet()) {
            List<String> base = s.getValue().expanded();
            List<String> out = new ArrayList<>();
            int at = 0;
            while (at < base.size()) {
                Entry best = null;
                for (Entry e : library) {
                    int len = e.expansion().size();
                    if (at + len <= base.size() && base.subList(at, at + len).equals(e.expansion())
                            && (best == null || len > best.expansion().size())) {
                        best = e;
                    }
                }
                Objects.requireNonNull(best, "a primitive is missing from the library");
                out.add(best.name());
                at += best.expansion().size();
            }
            s.setValue(new Solution(out, base));
        }
    }

    /** Refits the unigram prior to the token counts of the solutions, add-one smoothed. */
    private void refitPrior() {
        Map<String, Integer> counts = new HashMap<>();
        int total = 0;
        for (Solution s : solutions.values()) {
            for (String t : s.program()) {
                counts.merge(t, 1, Integer::sum);
                total++;
            }
        }
        logPrior.clear();
        for (Entry e : library) {
            logPrior.put(e.name(), Math.log((counts.getOrDefault(e.name(), 0) + 1.0) / (total + library.size())));
        }
    }
}
