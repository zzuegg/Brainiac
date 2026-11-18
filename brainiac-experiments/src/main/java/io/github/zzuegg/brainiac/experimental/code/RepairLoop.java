package io.github.zzuegg.brainiac.experimental.code;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Self-repair with execution feedback, with honest budget accounting (Chen et al., "Teaching
 * large language models to self-debug", ICLR 2024,
 * <a href="https://arxiv.org/abs/2304.05128">arXiv:2304.05128</a>; Olausson et al., "Is
 * self-repair a silver bullet for code generation?", ICLR 2024,
 * <a href="https://arxiv.org/abs/2306.09896">arXiv:2306.09896</a>). A program is generated and
 * run on the given examples; if it fails, the failure is fed back (the wrong outputs, or the
 * whole execution trace) and the model writes a repaired program, up to {@code repairs} times
 * per initial program, then a fresh program is sampled. Every model call, generation or repair,
 * costs one sample, so the result is a solve rate at a given total budget, comparable with
 * plain resampling ({@link Feedback#NONE}) at the same budget: Olausson et al.'s point is that
 * repair helps only when the feedback is good enough to beat simply drawing more samples.
 */
public final class RepairLoop {

    /** What the repairer sees about a failure. */
    public enum Feedback {
        /** No repair: resample (pass@k). */
        NONE,
        /** The failed program and its actual outputs on the examples (the diff). */
        OUTPUT_DIFF,
        /** The failed program, its outputs and its line-by-line states on the examples. */
        TRACE
    }

    /** Writes a program for a task. */
    @FunctionalInterface
    public interface Generator {
        /**
         * @param inputs the example inputs
         * @param outputs the expected outputs
         * @param random the random source
         * @return a program
         */
        List<String> generate(int[][] inputs, int[][] outputs, SplittableRandom random);
    }

    /** Repairs a failed program from feedback. */
    @FunctionalInterface
    public interface Repairer {
        /**
         * @param inputs the example inputs
         * @param outputs the expected outputs
         * @param failed the failed program
         * @param traces its execution on each example
         * @param feedback how much of the traces the repairer may use
         * @param random the random source
         * @return a repaired program
         */
        List<String> repair(int[][] inputs, int[][] outputs, List<String> failed, List<ExecutionTracer.Trace> traces,
                Feedback feedback, SplittableRandom random);
    }

    /**
     * The outcome of one task.
     *
     * @param solvedAt the sample (1-based) at which a program passed every example, or 0
     * @param program the passing program, or the last attempt
     * @param repairsTried repair calls made
     */
    public record Result(int solvedAt, List<String> program, int repairsTried) {

        /**
         * @param budget a sample budget
         * @return whether the task was solved within it
         */
        public boolean solvedWithin(int budget) {
            return solvedAt > 0 && solvedAt <= budget;
        }
    }

    private final ExecutionTracer tracer;
    private final Generator generator;
    private final Repairer repairer;
    private int repairs = 1;

    private RepairLoop(ExecutionTracer tracer, Generator generator, Repairer repairer) {
        this.tracer = Objects.requireNonNull(tracer);
        this.generator = Objects.requireNonNull(generator);
        this.repairer = repairer;
    }

    /**
     * @param tracer runs and traces programs
     * @param generator writes initial programs
     * @param repairer repairs failed ones ({@code null} for resampling only)
     * @return the loop
     */
    public static RepairLoop of(ExecutionTracer tracer, Generator generator, Repairer repairer) {
        return new RepairLoop(tracer, generator, repairer);
    }

    /**
     * @param n repair attempts per initial program, each on the previous attempt
     * @return this loop
     */
    public RepairLoop repairs(int n) {
        this.repairs = n;
        return this;
    }

    /**
     * @param program a program
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @return whether it reproduces every example
     */
    public boolean passes(List<String> program, int[][] inputs, int[][] outputs) {
        for (int i = 0; i < inputs.length; i++) {
            int[] o = tracer.run(program, inputs[i]);
            if (o == null || !Arrays.equals(o, outputs[i])) {
                return false;
            }
        }
        return true;
    }

    /**
     * Runs the loop on one task until it is solved or the budget is spent.
     *
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @param budget the most model calls
     * @param feedback the feedback ({@link Feedback#NONE} resamples)
     * @param random the random source
     * @return the outcome
     */
    public Result run(int[][] inputs, int[][] outputs, int budget, Feedback feedback, SplittableRandom random) {
        int used = 0;
        int repairCalls = 0;
        List<String> last = List.of();
        while (used < budget) {
            List<String> p = generator.generate(inputs, outputs, random);
            used++;
            last = p;
            if (passes(p, inputs, outputs)) {
                return new Result(used, p, repairCalls);
            }
            if (feedback == Feedback.NONE || repairer == null) {
                continue;
            }
            for (int k = 0; k < repairs && used < budget; k++) {
                List<ExecutionTracer.Trace> traces = new ArrayList<>();
                for (int[] in : inputs) {
                    traces.add(tracer.trace(p, in));
                }
                p = repairer.repair(inputs, outputs, p, traces, feedback, random);
                used++;
                repairCalls++;
                last = p;
                if (passes(p, inputs, outputs)) {
                    return new Result(used, p, repairCalls);
                }
            }
        }
        return new Result(0, last, repairCalls);
    }
}
