package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.data.VectorIndex;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;

/**
 * Voyager's skill library and automatic curriculum (Wang et al.,
 * <a href="https://arxiv.org/abs/2305.16291">2305.16291</a>), for any program type. An agent
 * that solves a task stores the verified program as a <em>skill</em>, keyed by an embedding
 * of the task's description; for the next task it retrieves the most similar skills (with a
 * core {@link VectorIndex}) and hands them to its program writer, which can call or compose
 * them. Skills therefore compound: a task that needs three earlier solutions glued together
 * becomes a short program over skills instead of a long one over primitives.
 *
 * <p>{@link #solve} is Voyager's iterative prompting: up to a budget of proposals, each one
 * seeing the retrieved skills and the critiques of the earlier failures, until the verifier
 * (Voyager's self-verification) accepts one. {@link Curriculum} is the automatic curriculum:
 * it proposes the easiest task not yet solved, skips tasks that keep failing and revisits
 * them once the library has grown.
 *
 * @param <P> the program type
 */
public final class SkillLibrary<P> {

    /**
     * A verified skill.
     *
     * @param id its index
     * @param description what it does (the task it solved)
     * @param program the program
     * @param <P> the program type
     */
    public record Skill<P>(int id, String description, P program) {
    }

    /**
     * Writes a program, as the code-writing model does.
     *
     * @param <P> the program type
     */
    @FunctionalInterface
    public interface Writer<P> {
        /**
         * @param task the task's description
         * @param skills the retrieved skills, most similar first
         * @param critiques what went wrong with the earlier proposals for this task
         * @param random the random source
         * @return a proposal
         */
        P write(String task, List<Skill<P>> skills, List<String> critiques, SplittableRandom random);
    }

    /**
     * Checks a proposal, as the environment and the self-verification critic do.
     *
     * @param <P> the program type
     */
    @FunctionalInterface
    public interface Verifier<P> {
        /**
         * @param program the proposal
         * @return {@code null} if it solves the task, otherwise a critique
         */
        String critique(P program);
    }

    /**
     * The result of one task.
     *
     * @param success whether a proposal was accepted
     * @param program the accepted proposal, or the last one
     * @param proposals how many were written
     * @param <P> the program type
     */
    public record Attempt<P>(boolean success, P program, int proposals) {
    }

    private final Function<String, float[]> embedder;
    private final int dim;
    private final VectorIndex index;
    private final List<Skill<P>> skills = new ArrayList<>();

    private SkillLibrary(Function<String, float[]> embedder, int dim) {
        this.embedder = Objects.requireNonNull(embedder);
        this.dim = dim;
        this.index = VectorIndex.flat(dim, VectorIndex.Metric.COSINE, Device.cpu());
    }

    /**
     * @param embedder embeds descriptions
     * @param dim the embedding width
     * @param <P> the program type
     * @return an empty library
     */
    public static <P> SkillLibrary<P> of(Function<String, float[]> embedder, int dim) {
        return new SkillLibrary<>(embedder, dim);
    }

    /**
     * Stores a verified skill.
     *
     * @param description what it does
     * @param program the program
     * @return the skill
     */
    public Skill<P> add(String description, P program) {
        Skill<P> s = new Skill<>(skills.size(), description, program);
        skills.add(s);
        index.add(Tensor.of(Shape.of(1, dim), embedder.apply(description), Device.cpu()));
        return s;
    }

    /**
     * @param query a task description
     * @param k how many
     * @return the most similar skills, most similar first
     */
    public List<Skill<P>> retrieve(String query, int k) {
        if (skills.isEmpty() || k <= 0) {
            return List.of();
        }
        List<Skill<P>> out = new ArrayList<>();
        for (VectorIndex.Hit h : index.search(Tensor.of(embedder.apply(query)), Math.min(k, skills.size()))) {
            out.add(skills.get(h.id()));
        }
        return out;
    }

    /** @return every skill, in the order learned */
    public List<Skill<P>> skills() {
        return List.copyOf(skills);
    }

    /** @return the number of skills */
    public int size() {
        return skills.size();
    }

    /**
     * Iterative prompting on one task: retrieve, propose, verify, refine; a verified
     * program is added to the library under the task's description.
     *
     * @param task the task's description
     * @param writer writes proposals
     * @param verifier checks them
     * @param retrieve how many skills to retrieve
     * @param budget the most proposals
     * @param random the random source
     * @return the attempt
     */
    public Attempt<P> solve(String task, Writer<P> writer, Verifier<P> verifier, int retrieve, int budget,
            SplittableRandom random) {
        List<Skill<P>> context = retrieve(task, retrieve);
        List<String> critiques = new ArrayList<>();
        P last = null;
        for (int i = 1; i <= budget; i++) {
            last = writer.write(task, context, List.copyOf(critiques), random);
            String c = verifier.critique(last);
            if (c == null) {
                add(task, last);
                return new Attempt<>(true, last, i);
            }
            critiques.add(c);
        }
        return new Attempt<>(false, last, budget);
    }

    /**
     * Voyager's automatic curriculum over a pool of candidate tasks: always propose the
     * unsolved task with the fewest failures so far, easiest first among equals, so the
     * agent works at the frontier of what it can do and returns to hard tasks as its skills
     * grow.
     *
     * @param <T> the task type
     */
    public static final class Curriculum<T> {
        private final List<T> pool;
        private final ToDoubleFunction<T> difficulty;
        private final Map<T, Integer> failures = new HashMap<>();
        private final List<T> solved = new ArrayList<>();

        private Curriculum(List<T> pool, ToDoubleFunction<T> difficulty) {
            this.pool = new ArrayList<>(pool);
            this.difficulty = Objects.requireNonNull(difficulty);
        }

        /**
         * @param pool the candidate tasks
         * @param difficulty an estimate of each task's difficulty (lower is easier)
         * @param <T> the task type
         * @return the curriculum
         */
        public static <T> Curriculum<T> of(List<T> pool, ToDoubleFunction<T> difficulty) {
            return new Curriculum<>(pool, difficulty);
        }

        /** @return the next task, or {@code null} when every task is solved */
        public T next() {
            T best = null;
            double bestKey = Double.POSITIVE_INFINITY;
            for (T t : pool) {
                double key = failures.getOrDefault(t, 0) * 1000.0 + difficulty.applyAsDouble(t);
                if (key < bestKey) {
                    bestKey = key;
                    best = t;
                }
            }
            return best;
        }

        /**
         * @param task the task tried
         * @param success whether it was solved
         */
        public void report(T task, boolean success) {
            if (success) {
                pool.remove(task);
                solved.add(task);
            } else {
                failures.merge(task, 1, Integer::sum);
            }
        }

        /** @return the tasks solved so far, in order */
        public List<T> solved() {
            return List.copyOf(solved);
        }
    }
}
