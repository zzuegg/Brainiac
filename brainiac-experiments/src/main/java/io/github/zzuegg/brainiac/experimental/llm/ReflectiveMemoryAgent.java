package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * One agent loop that fuses Reflexion with MemGPT-style paged memory (item 5.10): the
 * verbal lessons of {@link ReflexionLoop} (Shinn et al.,
 * <a href="https://arxiv.org/abs/2303.11366">2303.11366</a>) no longer die with the task.
 * Every reflection, and a note of every success, is written into a long-term
 * {@link MemoryStream} (Park et al., <a href="https://arxiv.org/abs/2304.03442">2304.03442</a>);
 * when a new task arrives, the agent searches that store by relevance (with recency and
 * importance, as the stream ranks), pages the hits into its working context, a
 * {@link VirtualContextPager} (Packer et al.,
 * <a href="https://arxiv.org/abs/2310.08560">2310.08560</a>), and hands them to the actor
 * together with the current task's own reflections. Over a long run of tasks that recur or
 * resemble each other, lessons learned once are reused instead of relearned.
 *
 * @param <A> the type of an attempt
 */
public final class ReflectiveMemoryAgent<A> {

    /** Separates the task from the lesson in a stored memory. */
    public static final String SEPARATOR = " | ";

    /**
     * What one task cost.
     *
     * @param outcome the Reflexion outcome
     * @param retrieved the long-term memories paged in before the first trial
     * @param <A> the attempt type
     */
    public record Episode<A>(ReflexionLoop.Outcome<A> outcome, List<String> retrieved) {
    }

    private final ReflexionLoop.Actor<A> actor;
    private final MemoryStream longTerm;
    private final VirtualContextPager context;
    private Function<A, String> describe = String::valueOf;
    private int retrieve = 4;
    private int episodicLessons = 3;
    private double time;

    private ReflectiveMemoryAgent(ReflexionLoop.Actor<A> actor, MemoryStream longTerm, VirtualContextPager context) {
        this.actor = Objects.requireNonNull(actor);
        this.longTerm = Objects.requireNonNull(longTerm);
        this.context = Objects.requireNonNull(context);
    }

    /**
     * @param actor makes attempts from the lessons it is given (retrieved ones first, each
     *     written {@code task | lesson}, then the current task's own)
     * @param longTerm the store reflections are written to
     * @param context the working context retrieved memories and outcomes are paged into
     * @param <A> the attempt type
     * @return the agent
     */
    public static <A> ReflectiveMemoryAgent<A> of(ReflexionLoop.Actor<A> actor, MemoryStream longTerm,
            VirtualContextPager context) {
        return new ReflectiveMemoryAgent<>(actor, longTerm, context);
    }

    /**
     * @param k long-term memories retrieved per task (4 by default)
     * @return this agent
     */
    public ReflectiveMemoryAgent<A> retrieve(int k) {
        this.retrieve = k;
        return this;
    }

    /**
     * @param lessons the current task's reflections kept between trials (3 by default)
     * @return this agent
     */
    public ReflectiveMemoryAgent<A> episodicLessons(int lessons) {
        this.episodicLessons = lessons;
        return this;
    }

    /**
     * @param fn writes an accepted attempt into the success note
     * @return this agent
     */
    public ReflectiveMemoryAgent<A> describe(Function<A, String> fn) {
        this.describe = Objects.requireNonNull(fn);
        return this;
    }

    /** @return the long-term store */
    public MemoryStream longTerm() {
        return longTerm;
    }

    /** @return the working context */
    public VirtualContextPager context() {
        return context;
    }

    /**
     * Solves one task: retrieve, run Reflexion with the retrieved lessons, write back.
     *
     * @param task the task's description, also the retrieval query
     * @param evaluator judges attempts
     * @param reflector turns failures into lessons
     * @param maxTrials the trial budget
     * @param random the random source
     * @return the episode
     */
    public Episode<A> solve(String task, ReflexionLoop.Evaluator<A> evaluator, ReflexionLoop.Reflector<A> reflector,
            int maxTrials, SplittableRandom random) {
        time++;
        List<String> retrieved = new ArrayList<>();
        for (MemoryStream.Retrieved r : longTerm.retrieve(task, retrieve, time)) {
            retrieved.add(r.memory().text());
        }
        context.add("user", task);
        if (!retrieved.isEmpty()) {
            context.add("function", "memory_search -> " + String.join("; ", retrieved));
        }
        ReflexionLoop.Actor<A> withMemory = (episodic, r) -> {
            List<String> lessons = new ArrayList<>(retrieved);
            for (String e : episodic) {
                lessons.add(task + SEPARATOR + e);
            }
            return actor.attempt(lessons, r);
        };
        ReflexionLoop.Outcome<A> o = ReflexionLoop.of(withMemory, evaluator, reflector).memorySize(episodicLessons)
                .run(maxTrials, random);
        for (String lesson : o.memory()) {
            longTerm.add(task + SEPARATOR + lesson, MemoryStream.Kind.REFLECTION, time, 5, List.of());
        }
        if (o.success()) {
            longTerm.add(task + SEPARATOR + "solved by " + describe.apply(o.last()), MemoryStream.Kind.REFLECTION,
                    time, 8, List.of());
        }
        context.add("assistant", (o.success() ? "solved " : "failed ") + task + " in " + o.trials() + " trials");
        return new Episode<>(o, List.copyOf(retrieved));
    }
}
