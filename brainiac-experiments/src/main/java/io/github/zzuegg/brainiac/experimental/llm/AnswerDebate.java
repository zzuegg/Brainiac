package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Multi-agent debate and ensemble voting over a shared pool of candidate answers (debate as in
 * Du et al., <a href="https://arxiv.org/abs/2305.14325">2305.14325</a>; voting as in
 * self-consistency, <a href="https://arxiv.org/abs/2203.11171">2203.11171</a>). Each agent can
 * score how likely it finds an answer. Three ways to combine them:
 * <ul>
 *   <li>{@link #vote}: every agent names its favourite, the most named wins;</li>
 *   <li>{@link #productOfExperts}: the answer with the highest summed log-likelihood wins;</li>
 *   <li>{@link #debate}: rounds in which every agent sees the others' current answers and
 *       re-picks by its own log-likelihood plus {@code agreement} per agent that agrees, so
 *       confident agents keep their answers and unsure ones are persuaded; then a vote.</li>
 * </ul>
 * Tiny language models cannot read each other's arguments, so the debate happens in answer
 * space: what is exchanged is the answers, not the reasons.
 */
public final class AnswerDebate {

    /** An agent that scores answers. */
    @FunctionalInterface
    public interface Debater {
        /**
         * @param prompt the question
         * @param answer a candidate answer
         * @return the agent's log-likelihood of the answer
         */
        double logLikelihood(int[] prompt, int[] answer);
    }

    /**
     * @param answer the chosen answer
     * @param rounds the per-round answers of every agent (indices into the candidates)
     */
    public record Verdict(int[] answer, List<int[]> rounds) {
    }

    private final List<Debater> agents;
    private double agreement = 2;

    private AnswerDebate(List<Debater> agents) {
        this.agents = List.copyOf(agents);
    }

    /**
     * @param agents the agents
     * @return the panel
     */
    public static AnswerDebate of(List<Debater> agents) {
        return new AnswerDebate(Objects.requireNonNull(agents));
    }

    /**
     * @param bonus the log-likelihood bonus per agreeing peer
     * @return this panel
     */
    public AnswerDebate agreement(double bonus) {
        this.agreement = bonus;
        return this;
    }

    private static List<int[]> distinct(List<int[]> candidates) {
        Map<String, int[]> unique = new LinkedHashMap<>();
        for (int[] c : candidates) {
            unique.putIfAbsent(Arrays.toString(c), c);
        }
        return new ArrayList<>(unique.values());
    }

    private double[][] scores(int[] prompt, List<int[]> pool) {
        double[][] s = new double[agents.size()][pool.size()];
        for (int i = 0; i < agents.size(); i++) {
            for (int c = 0; c < pool.size(); c++) {
                s[i][c] = agents.get(i).logLikelihood(prompt, pool.get(c));
            }
        }
        return s;
    }

    private static int argmax(double[] v) {
        int b = 0;
        for (int i = 1; i < v.length; i++) {
            if (v[i] > v[b]) {
                b = i;
            }
        }
        return b;
    }

    private static int majority(int[] picks, double[][] s) {
        Map<Integer, Integer> counts = new HashMap<>();
        for (int p : picks) {
            counts.merge(p, 1, Integer::sum);
        }
        int best = picks[0];
        double bestKey = Double.NEGATIVE_INFINITY;
        for (Map.Entry<Integer, Integer> e : counts.entrySet()) {
            double total = 0;
            for (double[] row : s) {
                total += row[e.getKey()];
            }
            double key = e.getValue() * 1e6 + total;
            if (key > bestKey) {
                bestKey = key;
                best = e.getKey();
            }
        }
        return best;
    }

    /**
     * @param prompt the question
     * @param candidates candidate answers (duplicates are merged)
     * @return the answer most agents like best (ties broken by summed log-likelihood)
     */
    public int[] vote(int[] prompt, List<int[]> candidates) {
        List<int[]> pool = distinct(candidates);
        double[][] s = scores(prompt, pool);
        int[] picks = new int[agents.size()];
        for (int i = 0; i < picks.length; i++) {
            picks[i] = argmax(s[i]);
        }
        return pool.get(majority(picks, s));
    }

    /**
     * @param prompt the question
     * @param candidates candidate answers
     * @return the answer with the highest summed log-likelihood
     */
    public int[] productOfExperts(int[] prompt, List<int[]> candidates) {
        List<int[]> pool = distinct(candidates);
        double[][] s = scores(prompt, pool);
        double[] sum = new double[pool.size()];
        for (double[] row : s) {
            for (int c = 0; c < sum.length; c++) {
                sum[c] += row[c];
            }
        }
        return pool.get(argmax(sum));
    }

    /**
     * @param prompt the question
     * @param candidates candidate answers
     * @param rounds debate rounds after the opening answers
     * @return the verdict
     */
    public Verdict debate(int[] prompt, List<int[]> candidates, int rounds) {
        List<int[]> pool = distinct(candidates);
        double[][] s = scores(prompt, pool);
        int[] picks = new int[agents.size()];
        for (int i = 0; i < picks.length; i++) {
            picks[i] = argmax(s[i]);
        }
        List<int[]> history = new ArrayList<>();
        history.add(picks.clone());
        for (int r = 0; r < rounds; r++) {
            int[] next = new int[picks.length];
            for (int i = 0; i < picks.length; i++) {
                double[] v = s[i].clone();
                for (int j = 0; j < picks.length; j++) {
                    if (j != i) {
                        v[picks[j]] += agreement;
                    }
                }
                next[i] = argmax(v);
            }
            picks = next;
            history.add(picks.clone());
        }
        return new Verdict(pool.get(majority(picks, s)), history);
    }
}
