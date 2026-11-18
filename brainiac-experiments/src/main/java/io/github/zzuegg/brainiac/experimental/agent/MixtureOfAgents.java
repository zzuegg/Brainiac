package io.github.zzuegg.brainiac.experimental.agent;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Mixture-of-Agents (Wang et al. 2024, <a href="https://arxiv.org/abs/2406.04692">arXiv:2406.04692</a>).
 * Several proposer agents answer a question; in every following layer each agent sees all
 * answers of the previous layer and writes an improved one; a final aggregator synthesises the
 * last layer's answers. Unlike a vote it can combine partial solutions (one agent's right half
 * with another's), and unlike debate there is no adversarial framing.
 *
 * <p>Agents are functions of the question and the previous layer's answers (empty for the
 * first layer). Token cost is counted as every call's input (question plus the answers it
 * reads) plus its output.
 */
public final class MixtureOfAgents {

    /** One agent. */
    @FunctionalInterface
    public interface Agent {
        /**
         * @param question the question
         * @param previous the previous layer's answers (empty in the first layer)
         * @return this agent's answer
         */
        int[] respond(int[] question, List<int[]> previous);
    }

    /**
     * A finished run.
     *
     * @param answer the aggregator's answer
     * @param layers every layer's answers
     * @param tokens tokens read and written by all calls
     * @param calls agent calls
     */
    public record Answer(int[] answer, List<List<int[]>> layers, long tokens, int calls) {
    }

    private final List<List<Agent>> layers;
    private final Agent aggregator;

    private MixtureOfAgents(List<List<Agent>> layers, Agent aggregator) {
        if (layers.isEmpty()) {
            throw new IllegalArgumentException("at least one proposer layer");
        }
        List<List<Agent>> copy = new ArrayList<>();
        for (List<Agent> l : layers) {
            copy.add(List.copyOf(l));
        }
        this.layers = List.copyOf(copy);
        this.aggregator = Objects.requireNonNull(aggregator);
    }

    /**
     * @param layers the proposer layers, first to last
     * @param aggregator the final aggregator
     * @return the mixture
     */
    public static MixtureOfAgents of(List<List<Agent>> layers, Agent aggregator) {
        return new MixtureOfAgents(layers, aggregator);
    }

    /**
     * @param question the question
     * @return the aggregated answer and its cost
     */
    public Answer answer(int[] question) {
        List<int[]> previous = List.of();
        List<List<int[]>> history = new ArrayList<>();
        long tokens = 0;
        int calls = 0;
        for (List<Agent> layer : layers) {
            List<int[]> next = new ArrayList<>();
            for (Agent a : layer) {
                int[] out = a.respond(question, previous);
                tokens += cost(question, previous, out);
                calls++;
                next.add(out);
            }
            previous = List.copyOf(next);
            history.add(previous);
        }
        int[] out = aggregator.respond(question, previous);
        tokens += cost(question, previous, out);
        return new Answer(out, history, tokens, calls + 1);
    }

    private static long cost(int[] question, List<int[]> read, int[] written) {
        long n = question.length + written.length;
        for (int[] r : read) {
            n += r.length;
        }
        return n;
    }

    /**
     * The usual aggregation prompt: the question, then each answer after a separator.
     *
     * @param question the question
     * @param answers the answers
     * @param separator the separator token
     * @return the prompt
     */
    public static int[] prompt(int[] question, List<int[]> answers, int separator) {
        List<Integer> out = new ArrayList<>();
        for (int t : question) {
            out.add(t);
        }
        for (int[] a : answers) {
            out.add(separator);
            for (int t : a) {
                out.add(t);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    @Override
    public String toString() {
        return "MixtureOfAgents(" + layers.size() + " layers)";
    }
}
