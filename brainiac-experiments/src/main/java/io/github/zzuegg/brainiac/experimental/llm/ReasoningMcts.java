package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.experimental.rl.Mcts;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Monte Carlo tree search over reasoning steps, as in rStar (Qi et al.,
 * <a href="https://arxiv.org/abs/2408.06195">2408.06195</a>), rStar-Math (Guan et al.,
 * <a href="https://arxiv.org/abs/2501.04519">2501.04519</a>) and LLM-MCTS: a state is a
 * partial solution, an action is one of {@code k} next steps the language model proposes
 * for it, and the search is the PUCT tree search of {@link Mcts}, with a process reward model
 * (or any {@link TestTimeCompute.Scorer}) as the value of partial solutions and the reward of
 * finished ones. Unlike Tree-of-Thoughts beam or depth-first search, visits flow to the
 * branches whose <em>backed-up</em> values are high, so a step that looked good but leads
 * only to bad continuations loses its visits.
 *
 * <p>{@link #solve} searches from the empty solution, commits the most visited step, and
 * searches again from there, until the solution ends; it returns the best-valued finished
 * solution the search met on the way (or the committed path, if asked). Proposals and values are cached per
 * partial solution, so {@link #valueCalls} counts distinct value-function calls, the cost
 * that compares fairly with best-of-N and tree search.
 */
public final class ReasoningMcts {

    private final WindowLm lm;
    private final TestTimeCompute.Scorer value;
    private int expansions = 4;
    private int simulations = 16;
    private int maxSteps = 3;
    private int maxStepTokens = 4;
    private final int stepEnd;
    private final int end;
    private float cPuct = 1.25f;
    private boolean returnBest = true;
    private int valueCalls;
    private int stepSamples;

    private ReasoningMcts(WindowLm lm, TestTimeCompute.Scorer value, int stepEnd, int end) {
        this.lm = Objects.requireNonNull(lm);
        this.value = Objects.requireNonNull(value);
        this.stepEnd = stepEnd;
        this.end = end;
    }

    /**
     * @param lm proposes steps
     * @param value scores partial and complete solutions, higher is better (a PRM, typically in 0..1)
     * @param stepEnd the token that ends a step
     * @param end the token that ends a solution
     * @return a search with 4 proposals per node, 16 simulations per committed step, 3 steps
     */
    public static ReasoningMcts of(WindowLm lm, TestTimeCompute.Scorer value, int stepEnd, int end) {
        return new ReasoningMcts(lm, value, stepEnd, end);
    }

    /**
     * @param k steps proposed per node
     * @return this search
     */
    public ReasoningMcts expansions(int k) {
        this.expansions = k;
        return this;
    }

    /**
     * @param n simulations per committed step
     * @return this search
     */
    public ReasoningMcts simulations(int n) {
        this.simulations = n;
        return this;
    }

    /**
     * @param steps the deepest solution, in steps
     * @param tokens the most tokens per step
     * @return this search
     */
    public ReasoningMcts limits(int steps, int tokens) {
        this.maxSteps = steps;
        this.maxStepTokens = tokens;
        return this;
    }

    /**
     * @param c the PUCT exploration constant
     * @return this search
     */
    public ReasoningMcts cPuct(float c) {
        this.cPuct = c;
        return this;
    }

    /**
     * @param best whether {@link #solve} returns the best-valued finished solution seen
     *     anywhere in the search (the default, as rStar picks the best trajectory) rather
     *     than the path of most-visited steps
     * @return this search
     */
    public ReasoningMcts returnBestFinished(boolean best) {
        this.returnBest = best;
        return this;
    }

    /** @return distinct value-function calls so far */
    public int valueCalls() {
        return valueCalls;
    }

    /** @return steps sampled from the model so far */
    public int stepSamples() {
        return stepSamples;
    }

    private record State(int[] tokens, int steps) {
    }

    private final class Problem implements Mcts.Model<State> {
        private final int[] prompt;
        private final SplittableRandom random;
        private final Map<String, List<int[]>> proposals = new HashMap<>();
        private final Map<String, Double> values = new HashMap<>();
        private int[] bestFinished;
        private double bestFinishedValue = Double.NEGATIVE_INFINITY;

        Problem(int[] prompt, SplittableRandom random) {
            this.prompt = prompt;
            this.random = random;
        }

        @Override
        public int actions() {
            return expansions;
        }

        boolean finished(State s) {
            return s.tokens().length > 0 && s.tokens()[s.tokens().length - 1] == end || s.steps() >= maxSteps;
        }

        double valueOf(State s) {
            return values.computeIfAbsent(Arrays.toString(s.tokens()), k -> {
                valueCalls++;
                return value.score(prompt, s.tokens());
            });
        }

        List<int[]> steps(State s) {
            return proposals.computeIfAbsent(Arrays.toString(s.tokens()), k -> {
                int[][] ps = new int[expansions][];
                int[] context = Arrays.copyOf(prompt, prompt.length + s.tokens().length);
                System.arraycopy(s.tokens(), 0, context, prompt.length, s.tokens().length);
                Arrays.fill(ps, context);
                stepSamples += expansions;
                List<int[]> unique = new ArrayList<>();
                for (int[] step : lm.sampleUntil(ps, maxStepTokens, new int[] {stepEnd, end}, 1.0, random)) {
                    if (step.length > 0 && unique.stream().noneMatch(u -> Arrays.equals(u, step))) {
                        unique.add(step);
                    }
                }
                return unique;
            });
        }

        @Override
        public Mcts.Evaluation evaluate(State s) {
            if (finished(s)) {
                return new Mcts.Evaluation(new float[expansions], 0, true);
            }
            List<int[]> steps = steps(s);
            float[] priors = new float[expansions];
            for (int a = 0; a < steps.size(); a++) {
                priors[a] = 1;
            }
            return new Mcts.Evaluation(priors, (float) (s.tokens().length == 0 ? 0.5 : valueOf(s)), false);
        }

        @Override
        public Mcts.Step<State> step(State s, int action) {
            int[] step = steps(s).get(action);
            int[] next = Arrays.copyOf(s.tokens(), s.tokens().length + step.length);
            System.arraycopy(step, 0, next, s.tokens().length, step.length);
            State child = new State(next, s.steps() + 1);
            if (!finished(child)) {
                return new Mcts.Step<>(child, 0);   // open solutions are valued when expanded
            }
            double v = valueOf(child);   // a finished solution is rewarded once with its score
            if (v > bestFinishedValue) {
                bestFinishedValue = v;
                bestFinished = next;
            }
            return new Mcts.Step<>(child, (float) v);
        }
    }

    /**
     * Searches step by step.
     *
     * @param prompt the problem
     * @param random the random source
     * @return the solution
     */
    public int[] solve(int[] prompt, SplittableRandom random) {
        Problem problem = new Problem(prompt, random);
        Mcts<State> search = Mcts.of(problem).simulations(simulations).cPuct(cPuct);
        State s = new State(new int[0], 0);
        while (!problem.finished(s)) {
            if (problem.steps(s).isEmpty()) {
                break;
            }
            Mcts.Result r = search.search(s, random);
            s = problem.step(s, r.best()).state();
        }
        return returnBest && problem.bestFinished != null ? problem.bestFinished : s.tokens();
    }
}
