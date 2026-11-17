package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.function.UnaryOperator;

/**
 * Graph of Thoughts (Besta et al., <a href="https://arxiv.org/abs/2308.09687">2308.09687</a>):
 * reasoning as a dataflow graph of thoughts instead of a chain or a tree. Besides
 * <em>generating</em> several thoughts from one and <em>keeping the best</em> (all a tree of
 * thoughts can do), a thought can <em>aggregate</em> several others (merge sorted halves,
 * combine partial answers) and be <em>refined</em> in a loop. Problems that decompose, such as
 * sorting a long list, are solved as many small, reliable model calls whose results are
 * merged, instead of one long, error-prone one.
 *
 * <p>This class is the graph-of-operations executor: each operation takes and returns
 * thoughts, records them (with their parents) in the graph, scores them, and counts the
 * model calls, so strategies can be compared at equal cost. The model calls are supplied as
 * functions (prompts to a language model, or anything else).
 *
 * @param <T> the type of a thought's content
 */
public final class GraphOfThoughts<T> {

    /**
     * A vertex of the graph.
     *
     * @param id its index
     * @param content the thought
     * @param score its score, higher is better
     * @param parents the thoughts it was made from
     * @param <T> the content type
     */
    public record Thought<T>(int id, T content, double score, List<Integer> parents) {
    }

    private final ToDoubleFunction<T> scorer;
    private final List<Thought<T>> graph = new ArrayList<>();
    private int calls;

    private GraphOfThoughts(ToDoubleFunction<T> scorer) {
        this.scorer = Objects.requireNonNull(scorer);
    }

    /**
     * @param scorer scores a thought (higher is better); not counted as a model call
     * @param <T> the content type
     * @return an empty graph
     */
    public static <T> GraphOfThoughts<T> of(ToDoubleFunction<T> scorer) {
        return new GraphOfThoughts<>(scorer);
    }

    private Thought<T> add(T content, List<Integer> parents) {
        Thought<T> t = new Thought<>(graph.size(), content, scorer.applyAsDouble(content), List.copyOf(parents));
        graph.add(t);
        return t;
    }

    /**
     * Adds a root thought, such as the problem itself.
     *
     * @param content the thought
     * @return it, scored
     */
    public Thought<T> root(T content) {
        return add(content, List.of());
    }

    /**
     * Generate: one model call per parent that returns several thoughts (such as splitting
     * a problem into parts).
     *
     * @param parent the thought
     * @param call the model call
     * @return the new thoughts
     */
    public List<Thought<T>> split(Thought<T> parent, Function<T, List<T>> call) {
        calls++;
        List<Thought<T>> out = new ArrayList<>();
        for (T c : call.apply(parent.content())) {
            out.add(add(c, List.of(parent.id())));
        }
        return out;
    }

    /**
     * Generate: {@code k} independent model calls from the same parent.
     *
     * @param parent the thought
     * @param k how many
     * @param call the model call
     * @return the new thoughts
     */
    public List<Thought<T>> generate(Thought<T> parent, int k, UnaryOperator<T> call) {
        List<Thought<T>> out = new ArrayList<>();
        for (int i = 0; i < k; i++) {
            calls++;
            out.add(add(call.apply(parent.content()), List.of(parent.id())));
        }
        return out;
    }

    /**
     * KeepBest: the {@code n} highest-scored thoughts.
     *
     * @param thoughts the candidates
     * @param n how many
     * @return the best, best first
     */
    public List<Thought<T>> keepBest(List<Thought<T>> thoughts, int n) {
        List<Thought<T>> s = new ArrayList<>(thoughts);
        s.sort(Comparator.comparingDouble(Thought<T>::score).reversed());
        return List.copyOf(s.subList(0, Math.min(n, s.size())));
    }

    /**
     * Aggregate: {@code tries} model calls that each merge the same thoughts; the best
     * merge is kept.
     *
     * @param parts the thoughts to merge
     * @param tries how many attempts
     * @param call the model call
     * @return the best merge
     */
    public Thought<T> aggregate(List<Thought<T>> parts, int tries, Function<List<T>, T> call) {
        List<T> contents = parts.stream().map(Thought::content).toList();
        List<Integer> ids = parts.stream().map(Thought::id).toList();
        Thought<T> best = null;
        for (int i = 0; i < tries; i++) {
            calls++;
            Thought<T> t = add(call.apply(contents), ids);
            if (best == null || t.score() > best.score()) {
                best = t;
            }
        }
        return best;
    }

    /**
     * Refine: a loop of model calls that improve a thought, keeping an improvement only if
     * it scores higher, stopping early at {@code target}.
     *
     * @param thought the thought
     * @param rounds the most calls
     * @param target a score at which to stop
     * @param call the model call
     * @return the best version
     */
    public Thought<T> refine(Thought<T> thought, int rounds, double target, UnaryOperator<T> call) {
        Thought<T> best = thought;
        for (int i = 0; i < rounds && best.score() < target; i++) {
            calls++;
            Thought<T> t = add(call.apply(best.content()), List.of(best.id()));
            if (t.score() > best.score()) {
                best = t;
            }
        }
        return best;
    }

    /** @return the model calls so far */
    public int calls() {
        return calls;
    }

    /** @return every thought, in the order made */
    public List<Thought<T>> graph() {
        return List.copyOf(graph);
    }
}
