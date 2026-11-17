package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Reflexion (Shinn et al., <a href="https://arxiv.org/abs/2303.11366">2303.11366</a>): an
 * agent improves across trials of one task not by changing weights but by writing lessons.
 * After every failed attempt, the evaluator's feedback (a failing test, a "too high") is
 * turned by a reflector into a short verbal reflection, kept in an episodic memory of the
 * last few lessons, and handed to the actor on the next trial.
 *
 * @param <A> the type of an attempt
 */
public final class ReflexionLoop<A> {

    /**
     * Makes an attempt, given the lessons so far.
     *
     * @param <A> the attempt type
     */
    @FunctionalInterface
    public interface Actor<A> {
        /**
         * @param memory the reflections so far, oldest first
         * @param random the random source
         * @return an attempt
         */
        A attempt(List<String> memory, SplittableRandom random);
    }

    /**
     * Judges an attempt, as the environment or a unit test would.
     *
     * @param <A> the attempt type
     */
    @FunctionalInterface
    public interface Evaluator<A> {
        /**
         * @param attempt the attempt
         * @return the feedback
         */
        Feedback evaluate(A attempt);
    }

    /**
     * Turns a failure into a lesson.
     *
     * @param <A> the attempt type
     */
    @FunctionalInterface
    public interface Reflector<A> {
        /**
         * @param attempt the failed attempt
         * @param feedback what the evaluator said
         * @return the reflection to remember
         */
        String reflect(A attempt, Feedback feedback);
    }

    /**
     * @param success whether the attempt solved the task
     * @param signal the evaluator's message
     */
    public record Feedback(boolean success, String signal) {
    }

    /**
     * @param success whether some trial succeeded
     * @param trials the trials used
     * @param last the last attempt
     * @param memory the reflections at the end
     * @param <A> the attempt type
     */
    public record Outcome<A>(boolean success, int trials, A last, List<String> memory) {
    }

    private final Actor<A> actor;
    private final Evaluator<A> evaluator;
    private final Reflector<A> reflector;
    private int memorySize = 3;

    private ReflexionLoop(Actor<A> actor, Evaluator<A> evaluator, Reflector<A> reflector) {
        this.actor = Objects.requireNonNull(actor);
        this.evaluator = Objects.requireNonNull(evaluator);
        this.reflector = Objects.requireNonNull(reflector);
    }

    /**
     * @param actor makes attempts
     * @param evaluator judges them
     * @param reflector writes lessons
     * @param <A> the attempt type
     * @return the loop
     */
    public static <A> ReflexionLoop<A> of(Actor<A> actor, Evaluator<A> evaluator, Reflector<A> reflector) {
        return new ReflexionLoop<>(actor, evaluator, reflector);
    }

    /**
     * @param lessons how many reflections to keep (0: plain retrying)
     * @return this loop
     */
    public ReflexionLoop<A> memorySize(int lessons) {
        this.memorySize = lessons;
        return this;
    }

    /**
     * @param maxTrials the most attempts
     * @param random the random source
     * @return the outcome
     */
    public Outcome<A> run(int maxTrials, SplittableRandom random) {
        List<String> memory = new ArrayList<>();
        A last = null;
        for (int t = 1; t <= maxTrials; t++) {
            last = actor.attempt(List.copyOf(memory), random);
            Feedback f = evaluator.evaluate(last);
            if (f.success()) {
                return new Outcome<>(true, t, last, List.copyOf(memory));
            }
            if (memorySize > 0) {
                memory.add(reflector.reflect(last, f));
                if (memory.size() > memorySize) {
                    memory.removeFirst();
                }
            }
        }
        return new Outcome<>(false, maxTrials, last, List.copyOf(memory));
    }
}
