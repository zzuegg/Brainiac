package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Ways to spend more inference compute for better answers from the same model, the "test-time
 * scaling" knobs (Snell et al., <a href="https://arxiv.org/abs/2408.03314">2408.03314</a>):
 * <ul>
 *   <li>{@link #bestOfN}: sample N answers, keep the one a verifier likes best.</li>
 *   <li>{@link #selfConsistency}: sample N reasoning chains and majority-vote the final
 *       answer (Wang et al., <a href="https://arxiv.org/abs/2203.11171">2203.11171</a>);
 *       {@link #weightedVote} weights the votes by verifier scores.</li>
 *   <li>{@link #critiqueLoop}: a self-critique loop that resamples while the verifier
 *       scores the answer low, so easy questions cost one sample and hard ones more.</li>
 *   <li>{@link #stepBeamSearch} and {@link #stepDepthFirst}: Tree of Thoughts (Yao et al.,
 *       <a href="https://arxiv.org/abs/2305.10601">2305.10601</a>): search over partial
 *       solutions one reasoning step at a time, guided by a value function such as a
 *       {@link ProcessRewardModel}; breadth-first with a beam, or depth-first with pruning
 *       and backtracking.</li>
 * </ul>
 */
public final class TestTimeCompute {

    private TestTimeCompute() {
    }

    /** Draws complete answers for a prompt. */
    @FunctionalInterface
    public interface Sampler {
        /**
         * @param prompt the prompt
         * @param n how many answers
         * @param random the random source
         * @return the answers
         */
        int[][] sample(int[] prompt, int n, SplittableRandom random);

        /**
         * @param lm a language model
         * @param maxNew the most tokens per answer
         * @param end the end token
         * @param temperature the sampling temperature
         * @return a sampler that draws from the model
         */
        static Sampler of(WindowLm lm, int maxNew, int end, double temperature) {
            return (prompt, n, random) -> {
                int[][] ps = new int[n][];
                Arrays.fill(ps, prompt);
                return lm.sample(ps, maxNew, end, temperature, random);
            };
        }
    }

    /** Scores a (possibly partial) answer: a verifier or value function. */
    @FunctionalInterface
    public interface Scorer {
        /**
         * @param prompt the prompt
         * @param completion the answer so far
         * @return higher is better
         */
        double score(int[] prompt, int[] completion);
    }

    /**
     * @param prompt the prompt
     * @param sampler draws answers
     * @param n how many to draw
     * @param verifier ranks them
     * @param random the random source
     * @return the best-scored answer
     */
    public static int[] bestOfN(int[] prompt, Sampler sampler, int n, Scorer verifier, SplittableRandom random) {
        int[][] answers = sampler.sample(prompt, n, random);
        int best = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < answers.length; i++) {
            double s = verifier.score(prompt, answers[i]);
            if (s > bestScore) {
                bestScore = s;
                best = i;
            }
        }
        return answers[best];
    }

    /**
     * @param answers the answers; {@code null}s (unparsable) do not vote
     * @param <T> the answer type
     * @return the most common answer (ties go to the one seen first), or {@code null}
     */
    public static <T> T majorityVote(List<T> answers) {
        Map<T, Integer> counts = new LinkedHashMap<>();
        for (T a : answers) {
            if (a != null) {
                counts.merge(a, 1, Integer::sum);
            }
        }
        return counts.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(null);
    }

    /**
     * @param prompt the prompt
     * @param sampler draws reasoning chains
     * @param n how many
     * @param extract the final answer of a chain, or {@code null}
     * @param random the random source
     * @param <T> the answer type
     * @return the majority answer
     */
    public static <T> T selfConsistency(int[] prompt, Sampler sampler, int n, Function<int[], T> extract,
            SplittableRandom random) {
        List<T> votes = new ArrayList<>();
        for (int[] a : sampler.sample(prompt, n, random)) {
            votes.add(extract.apply(a));
        }
        return majorityVote(votes);
    }

    /**
     * Self-consistency where each vote counts its verifier score.
     *
     * @param prompt the prompt
     * @param sampler draws reasoning chains
     * @param n how many
     * @param extract the final answer of a chain, or {@code null}
     * @param verifier weights each chain
     * @param random the random source
     * @param <T> the answer type
     * @return the answer with the highest total score
     */
    public static <T> T weightedVote(int[] prompt, Sampler sampler, int n, Function<int[], T> extract,
            Scorer verifier, SplittableRandom random) {
        Map<T, Double> total = new HashMap<>();
        for (int[] a : sampler.sample(prompt, n, random)) {
            T v = extract.apply(a);
            if (v != null) {
                total.merge(v, verifier.score(prompt, a), Double::sum);
            }
        }
        return total.entrySet().stream().max(Map.Entry.comparingByValue()).map(Map.Entry::getKey).orElse(null);
    }

    /**
     * The result of a critique loop.
     *
     * @param completion the accepted (or best) answer
     * @param samples how many answers were drawn
     * @param score its verifier score
     */
    public record Critique(int[] completion, int samples, double score) {
    }

    /**
     * Samples one answer at a time and keeps resampling while the verifier's score stays
     * below a threshold, up to a budget; returns the first accepted answer or else the best.
     *
     * @param prompt the prompt
     * @param sampler draws answers
     * @param verifier the critic
     * @param threshold the score that accepts an answer
     * @param maxSamples the budget
     * @param random the random source
     * @return what was accepted and what it cost
     */
    public static Critique critiqueLoop(int[] prompt, Sampler sampler, Scorer verifier, double threshold,
            int maxSamples, SplittableRandom random) {
        int[] best = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 1; i <= maxSamples; i++) {
            int[] a = sampler.sample(prompt, 1, random)[0];
            double s = verifier.score(prompt, a);
            if (s > bestScore) {
                bestScore = s;
                best = a;
            }
            if (s >= threshold) {
                return new Critique(a, i, s);
            }
        }
        return new Critique(best, maxSamples, bestScore);
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static boolean ends(int[] c, int end) {
        return c.length > 0 && c[c.length - 1] == end;
    }

    private record Node(int[] completion, double value) {
    }

    private static List<int[]> proposeSteps(WindowLm lm, int[] prompt, int[] partial, int k, int maxStepTokens,
            int[] stops, double temperature, SplittableRandom random) {
        int[][] ps = new int[k][];
        Arrays.fill(ps, concat(prompt, partial));
        List<int[]> unique = new ArrayList<>();
        for (int[] step : lm.sampleUntil(ps, maxStepTokens, stops, temperature, random)) {
            if (step.length > 0 && unique.stream().noneMatch(u -> Arrays.equals(u, step))) {
                unique.add(step);
            }
        }
        return unique;
    }

    /**
     * Breadth-first Tree-of-Thoughts search with a beam: every level proposes
     * {@code expansions} next steps for each partial solution in the beam, scores the
     * extended solutions with the value function and keeps the best {@code beamWidth}.
     *
     * @param lm proposes steps
     * @param prompt the prompt
     * @param value scores partial and complete solutions
     * @param beamWidth partial solutions kept per level
     * @param expansions steps proposed per partial solution
     * @param maxSteps the deepest level
     * @param maxStepTokens the most tokens per step
     * @param stepEnd the token that ends a step
     * @param end the token that ends a solution
     * @param random the random source
     * @return the best complete solution found, or the best partial one
     */
    public static int[] stepBeamSearch(WindowLm lm, int[] prompt, Scorer value, int beamWidth, int expansions,
            int maxSteps, int maxStepTokens, int stepEnd, int end, SplittableRandom random) {
        Objects.requireNonNull(value);
        List<Node> beam = List.of(new Node(new int[0], 0));
        int[] stops = {stepEnd, end};
        for (int depth = 0; depth < maxSteps; depth++) {
            List<Node> next = new ArrayList<>();
            boolean open = false;
            for (Node n : beam) {
                if (ends(n.completion, end)) {
                    next.add(n);
                    continue;
                }
                open = true;
                for (int[] step : proposeSteps(lm, prompt, n.completion, expansions, maxStepTokens, stops, 1.0,
                        random)) {
                    int[] c = concat(n.completion, step);
                    next.add(new Node(c, value.score(prompt, c)));
                }
            }
            if (!open) {
                break;
            }
            next.sort(Comparator.comparingDouble(Node::value).reversed());
            beam = next.subList(0, Math.min(beamWidth, next.size()));
        }
        return beam.stream().filter(n -> ends(n.completion, end)).findFirst().orElse(beam.get(0)).completion;
    }

    /**
     * Depth-first Tree-of-Thoughts search: extend the most promising step first, prune steps
     * whose value falls below a threshold, and backtrack when a branch dies, until a complete
     * solution is found or the budget of value calls runs out.
     *
     * @param lm proposes steps
     * @param prompt the prompt
     * @param value scores partial and complete solutions
     * @param threshold the value below which a branch is pruned
     * @param expansions steps proposed at each node
     * @param maxSteps the deepest level
     * @param maxStepTokens the most tokens per step
     * @param stepEnd the token that ends a step
     * @param end the token that ends a solution
     * @param budget the most value calls
     * @param random the random source
     * @return the first complete solution above the threshold, or the best seen
     */
    public static int[] stepDepthFirst(WindowLm lm, int[] prompt, Scorer value, double threshold, int expansions,
            int maxSteps, int maxStepTokens, int stepEnd, int end, int budget, SplittableRandom random) {
        Search s = new Search(lm, prompt, value, threshold, expansions, maxSteps, maxStepTokens,
                new int[] {stepEnd, end}, end, budget, random);
        int[] found = s.dfs(new int[0], 0);
        return found != null ? found : s.best;
    }

    private static final class Search {
        private final WindowLm lm;
        private final int[] prompt;
        private final Scorer value;
        private final double threshold;
        private final int expansions;
        private final int maxSteps;
        private final int maxStepTokens;
        private final int[] stops;
        private final int end;
        private final SplittableRandom random;
        private int budget;
        private int[] best = new int[0];
        private double bestValue = Double.NEGATIVE_INFINITY;

        Search(WindowLm lm, int[] prompt, Scorer value, double threshold, int expansions, int maxSteps,
                int maxStepTokens, int[] stops, int end, int budget, SplittableRandom random) {
            this.lm = lm;
            this.prompt = prompt;
            this.value = value;
            this.threshold = threshold;
            this.expansions = expansions;
            this.maxSteps = maxSteps;
            this.maxStepTokens = maxStepTokens;
            this.stops = stops;
            this.end = end;
            this.budget = budget;
            this.random = random;
        }

        int[] dfs(int[] partial, int depth) {
            if (depth >= maxSteps || budget <= 0) {
                return null;
            }
            List<Node> children = new ArrayList<>();
            for (int[] step : proposeSteps(lm, prompt, partial, expansions, maxStepTokens, stops, 1.0, random)) {
                if (budget-- <= 0) {
                    break;
                }
                int[] c = concat(partial, step);
                double v = value.score(prompt, c);
                if (ends(c, end) && v > bestValue) {
                    bestValue = v;
                    best = c;
                }
                children.add(new Node(c, v));
            }
            children.sort(Comparator.comparingDouble(Node::value).reversed());
            for (Node child : children) {
                if (child.value < threshold) {
                    break;   // prune: the rest are worse
                }
                if (ends(child.completion, end)) {
                    return child.completion;
                }
                int[] found = dfs(child.completion, depth + 1);
                if (found != null) {
                    return found;
                }
            }
            return null;
        }
    }
}
