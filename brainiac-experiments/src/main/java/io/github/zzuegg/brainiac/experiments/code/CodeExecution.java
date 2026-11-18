package io.github.zzuegg.brainiac.experiments.code;

import io.github.zzuegg.brainiac.experimental.code.ExecutionReranker;
import io.github.zzuegg.brainiac.experimental.code.ExecutionTracer;
import io.github.zzuegg.brainiac.experimental.code.ProgramSampler;
import io.github.zzuegg.brainiac.experimental.code.RepairLoop;
import io.github.zzuegg.brainiac.experimental.code.TraceLm;
import io.github.zzuegg.brainiac.experimental.evolve.GuidedEnumerator;
import io.github.zzuegg.brainiac.experimental.llm.ReflexionLoop;
import io.github.zzuegg.brainiac.experiments.reasoning.ListPrograms;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Wave 9, 9.2.14 to 9.2.16: code models on the list DSL of {@link ListPrograms}, whose programs
 * can be run and traced ({@link ExecutionTracer}).
 * <ul>
 *   <li>{@link #executionPrediction} (9.2.14): predict a program's output directly, through a
 *       scratchpad of states, or through CWM-style traces ({@link TraceLm}), trained on programs
 *       of 1 to 3 primitives and tested on up to 6.</li>
 *   <li>{@link #repair} (9.2.15): a {@link ProgramSampler} writes programs (random programs of 1
 *       or 2 primitives, its training distribution) from three examples; at
 *       equal total samples, plain resampling is compared with a {@link RepairLoop} whose
 *       repairer sees the output diff or the full trace, with {@link ReflexionLoop}'s text-only
 *       reflection (a memory of failed attempts) and with {@link GuidedEnumerator}.</li>
 *   <li>{@link #rerank} (9.2.16): with two (under-specified) given examples and five hidden
 *       ones, which of n sampled programs to submit: by likelihood, filtered then random,
 *       AlphaCode clustering, CodeT dual agreement, a LEVER verifier with execution results and
 *       the same verifier without them ({@link ExecutionReranker}).</li>
 * </ul>
 */
public final class CodeExecution {

    /** The primitives of 9.2.14: every one keeps the list length and values below 10. */
    static final List<String> TRACE_PRIMITIVES = List.of("rev", "sort", "swap", "+1", "+2", "*3");
    private static final int VALUES = 10;
    /** Input values of 9.2.14 are below this. */
    private static final int TRACE_INPUT_VALUES = 5;
    /** The longest program tested in 9.2.14. */
    private static final int TRACE_MAX_LENGTH = 6;

    private CodeExecution() {
    }

    private static int[] smallList(SplittableRandom r) {
        int[] xs = new int[2 + r.nextInt(2)];
        for (int i = 0; i < xs.length; i++) {
            xs[i] = r.nextInt(TRACE_INPUT_VALUES);
        }
        return xs;
    }

    private static int[] randomList(SplittableRandom r, int min, int max) {
        int[] xs = new int[min + r.nextInt(max - min + 1)];
        for (int i = 0; i < xs.length; i++) {
            xs[i] = r.nextInt(VALUES);
        }
        return xs;
    }

    private static List<String> randomProgram(List<String> prims, int length, SplittableRandom r) {
        List<String> p = new ArrayList<>();
        for (int i = 0; i < length; i++) {
            p.add(prims.get(r.nextInt(prims.size())));
        }
        return p;
    }

    // ------------------------------------------------------------------------- 9.2.14

    /**
     * One 9.2.14 model.
     *
     * @param accuracyByLength exact-output accuracy for programs of 1, 2, ..., 6 primitives
     * @param tokensPerProgram mean target tokens per test program
     */
    public record PredictionRow(double[] accuracyByLength, double tokensPerProgram) {

        /** @return mean accuracy on the training lengths (1 to 3) */
        public double inDistribution() {
            return (accuracyByLength[0] + accuracyByLength[1] + accuracyByLength[2]) / 3;
        }

        /** @return mean accuracy on the longer programs (4 to 6) */
        public double longer() {
            double s = 0;
            for (int i = 3; i < accuracyByLength.length; i++) {
                s += accuracyByLength[i];
            }
            return s / (accuracyByLength.length - 3);
        }
    }

    /**
     * Runs 9.2.14: lists of 2 or 3 values below 5, programs of 1 to 3 primitives in training
     * and 1 to 6 in testing.
     *
     * @param steps training steps per model (32 programs each)
     * @param testPerLength test programs per length
     * @param seed the seed
     * @return model name to its row
     */
    public static Map<String, PredictionRow> executionPrediction(int steps, int testPerLength, long seed) {
        ExecutionTracer tracer = ExecutionTracer.of(TRACE_PRIMITIVES, ListPrograms::apply, VALUES);
        Map<String, PredictionRow> out = new LinkedHashMap<>();
        for (ExecutionTracer.Format f : ExecutionTracer.Format.values()) {
            TraceLm lm = TraceLm.of(tracer, f, 16, 64, seed);
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom r = new SplittableRandom(seed + 1);
            for (int s = 0; s < steps; s++) {
                List<List<String>> ps = new ArrayList<>();
                List<int[]> in = new ArrayList<>();
                for (int i = 0; i < 32; i++) {
                    ps.add(randomProgram(TRACE_PRIMITIVES, 1 + r.nextInt(3), r));
                    in.add(smallList(r));
                }
                lm.trainStep(ps, in, opt);
            }
            double[] acc = new double[TRACE_MAX_LENGTH];
            double tokens = 0;
            SplittableRandom tr = new SplittableRandom(seed + 2);
            for (int len = 1; len <= TRACE_MAX_LENGTH; len++) {
                for (int k = 0; k < testPerLength; k++) {
                    List<String> p = randomProgram(TRACE_PRIMITIVES, len, tr);
                    int[] x = smallList(tr);
                    ExecutionTracer.Trace t = tracer.trace(p, x);
                    tokens += tracer.target(f, t).length / ((double) TRACE_MAX_LENGTH * testPerLength);
                    int[] pred = lm.predict(p, x, tracer.target(ExecutionTracer.Format.TRACE, t).length + 8);
                    acc[len - 1] += pred != null && Arrays.equals(pred, t.output()) ? 1.0 / testPerLength : 0;
                }
            }
            out.put(switch (f) {
                case DIRECT -> "direct";
                case STATES -> "states scratchpad";
                case TRACE -> "CWM-style trace";
            }, new PredictionRow(acc, tokens));
        }
        return out;
    }

    // ------------------------------------------------------------------------- 9.2.15

    private static final int EXAMPLES = 3;
    private static final int MAX_PROGRAM = 5;
    private static final int LIST_SLOTS = 10;

    /** Example inputs and outputs of a random program, or {@code null} if it misbehaves. */
    private static int[][][] examples(List<String> program, int count, SplittableRandom r) {
        int[][] in = new int[count][];
        int[][] out = new int[count][];
        for (int e = 0; e < count; e++) {
            in[e] = randomList(r, 5, 7);
            out[e] = ListPrograms.execute(program, in[e]);
            if (out[e] == null) {
                return null;
            }
        }
        return new int[][][] {in, out};
    }

    /**
     * A synthesis task: a random program of 1 or 2 primitives (the generator's training
     * distribution) and its examples.
     *
     * @param program the hidden program
     * @param inputs the example inputs
     * @param outputs the example outputs
     */
    private record CodeTask(List<String> program, int[][] inputs, int[][] outputs) {

        boolean solvedBy(List<String> p) {
            for (int e = 0; e < inputs.length; e++) {
                int[] o = ListPrograms.execute(p, inputs[e]);
                if (o == null || !Arrays.equals(o, outputs[e])) {
                    return false;
                }
            }
            return true;
        }
    }

    private static CodeTask randomTask(int examples, SplittableRandom r) {
        int[][][] ex = null;
        List<String> p = null;
        while (ex == null) {
            p = randomProgram(ListPrograms.primitives(), 1 + r.nextInt(2), r);
            ex = examples(p, examples, r);
        }
        return new CodeTask(p, ex[0], ex[1]);
    }

    private static float[] condition(int[][] in, int[][] out, int examples) {
        return ProgramSampler.encodeExamples(in, out, examples, LIST_SLOTS, VALUES);
    }

    private static int conditionSize(int examples) {
        return ProgramSampler.exampleFeatures(examples, LIST_SLOTS, VALUES);
    }

    /** Trains a generator on random programs of 1 or 2 primitives (the test tasks' lengths). */
    static ProgramSampler generator(int examples, int steps, long seed) {
        List<String> prims = ListPrograms.primitives();
        ProgramSampler g = ProgramSampler.of(prims, MAX_PROGRAM, conditionSize(examples), 128, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            List<float[]> cs = new ArrayList<>();
            List<List<String>> ps = new ArrayList<>();
            while (ps.size() < 32) {
                List<String> p = randomProgram(prims, 1 + r.nextInt(2), r);
                int[][][] ex = examples(p, examples, r);
                if (ex != null) {
                    cs.add(condition(ex[0], ex[1], examples));
                    ps.add(p);
                }
            }
            g.trainStep(cs, ps, opt);
        }
        return g;
    }

    private static List<String> mutate(List<String> p, SplittableRandom r) {
        List<String> prims = ListPrograms.primitives();
        List<String> c = new ArrayList<>(p);
        int kind = r.nextInt(3);
        if (kind == 0 && c.size() < MAX_PROGRAM) {
            c.add(r.nextInt(c.size() + 1), prims.get(r.nextInt(prims.size())));
        } else if (kind == 1 && c.size() > 1) {
            c.remove(r.nextInt(c.size()));
        } else {
            c.set(r.nextInt(c.size()), prims.get(r.nextInt(prims.size())));
        }
        return c;
    }

    private static int repairConditionSize(RepairLoop.Feedback mode) {
        int per = LIST_SLOTS * (VALUES + 1) + LIST_SLOTS + 1;
        int size = conditionSize(EXAMPLES) + MAX_PROGRAM * ListPrograms.primitives().size() + EXAMPLES * per;
        return mode == RepairLoop.Feedback.TRACE ? size + MAX_PROGRAM * per : size;
    }

    /** The repairer's condition: examples, the failed program, its outputs and (TRACE) its states. */
    private static float[] repairCondition(int[][] in, int[][] out, List<String> failed,
            List<ExecutionTracer.Trace> traces, RepairLoop.Feedback mode) {
        int per = LIST_SLOTS * (VALUES + 1) + LIST_SLOTS + 1;
        float[] f = new float[repairConditionSize(mode)];
        float[] ex = condition(in, out, EXAMPLES);
        System.arraycopy(ex, 0, f, 0, ex.length);
        int off = ex.length;
        int p = ListPrograms.primitives().size();
        for (int i = 0; i < Math.min(MAX_PROGRAM, failed.size()); i++) {
            f[off + i * p + ListPrograms.primitives().indexOf(failed.get(i))] = 1;
        }
        off += MAX_PROGRAM * p;
        for (int e = 0; e < EXAMPLES; e++) {
            ProgramSampler.encodeList(f, off + e * per, traces.get(e).output(), LIST_SLOTS, VALUES);
        }
        off += EXAMPLES * per;
        if (mode == RepairLoop.Feedback.TRACE) {
            List<ExecutionTracer.Frame> frames = traces.get(0).frames();
            for (int k = 0; k < Math.min(MAX_PROGRAM, frames.size()); k++) {
                ProgramSampler.encodeList(f, off + k * per, frames.get(k).after(), LIST_SLOTS, VALUES);
            }
        }
        return f;
    }

    /** Trains a repairer on single-edit bugs of random programs. */
    static ProgramSampler repairer(ExecutionTracer tracer, RepairLoop.Feedback mode, int steps, long seed) {
        ProgramSampler g = ProgramSampler.of(ListPrograms.primitives(), MAX_PROGRAM, repairConditionSize(mode), 128,
                seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            List<float[]> cs = new ArrayList<>();
            List<List<String>> ps = new ArrayList<>();
            while (ps.size() < 32) {
                List<String> target = randomProgram(ListPrograms.primitives(), 1 + r.nextInt(4), r);
                int[][][] ex = examples(target, EXAMPLES, r);
                if (ex == null) {
                    continue;
                }
                List<String> bug = mutate(target, r);
                List<ExecutionTracer.Trace> traces = new ArrayList<>();
                for (int[] x : ex[0]) {
                    traces.add(tracer.trace(bug, x));
                }
                cs.add(repairCondition(ex[0], ex[1], bug, traces, mode));
                ps.add(target);
            }
            g.trainStep(cs, ps, opt);
        }
        return g;
    }

    /**
     * 9.2.15's result.
     *
     * @param firstSample the generator's single-sample solve rate
     * @param passAt method name to its solve rate at 1, 2, 4, 8 and 16 total samples
     */
    public record RepairResult(double firstSample, Map<String, double[]> passAt) {
    }

    private static final int[] BUDGETS = {1, 2, 4, 8, 16};

    /**
     * Runs 9.2.15.
     *
     * @param generatorSteps training steps of the generator (sets its first-sample solve rate)
     * @param repairSteps training steps of each repairer
     * @param tasks test tasks
     * @param seed the seed
     * @return the pass@t curves
     */
    public static RepairResult repair(int generatorSteps, int repairSteps, int tasks, long seed) {
        ExecutionTracer tracer = ExecutionTracer.of(ListPrograms.primitives(), ListPrograms::apply, VALUES);
        ProgramSampler gen = generator(EXAMPLES, generatorSteps, seed);
        Map<RepairLoop.Feedback, ProgramSampler> repairers = new LinkedHashMap<>();
        repairers.put(RepairLoop.Feedback.OUTPUT_DIFF, repairer(tracer, RepairLoop.Feedback.OUTPUT_DIFF, repairSteps,
                seed + 1));
        repairers.put(RepairLoop.Feedback.TRACE, repairer(tracer, RepairLoop.Feedback.TRACE, repairSteps, seed + 1));
        RepairLoop.Generator g = (in, out, r) -> gen.sample(condition(in, out, EXAMPLES), 1, 1.0, r).programs().get(0);
        RepairLoop.Repairer rep = (in, out, failed, traces, mode, r) -> repairers.get(mode)
                .sample(repairCondition(in, out, failed, traces, mode), 1, 1.0, r).programs().get(0);
        RepairLoop loop = RepairLoop.of(tracer, g, rep).repairs(1);
        SplittableRandom tr = new SplittableRandom(seed + 2);
        List<CodeTask> test = new ArrayList<>();
        while (test.size() < tasks) {
            test.add(randomTask(EXAMPLES, tr));
        }
        int max = BUDGETS[BUDGETS.length - 1];
        Map<String, double[]> curves = new LinkedHashMap<>();
        double first = 0;
        for (RepairLoop.Feedback mode : RepairLoop.Feedback.values()) {
            double[] c = new double[BUDGETS.length];
            SplittableRandom r = new SplittableRandom(seed + 3);
            for (CodeTask t : test) {
                RepairLoop.Result res = loop.run(t.inputs(), t.outputs(), max, mode, r);
                for (int b = 0; b < BUDGETS.length; b++) {
                    c[b] += res.solvedWithin(BUDGETS[b]) ? 1.0 / test.size() : 0;
                }
            }
            if (mode == RepairLoop.Feedback.NONE) {
                first = c[0];
            }
            curves.put(switch (mode) {
                case NONE -> "resampling";
                case OUTPUT_DIFF -> "repair, output diff";
                case TRACE -> "repair, trace";
            }, c);
        }
        // Reflexion: a text memory of failed attempts, which the actor avoids repeating
        double[] refl = new double[BUDGETS.length];
        SplittableRandom rr = new SplittableRandom(seed + 3);
        for (CodeTask t : test) {
            ReflexionLoop<List<String>> reflexion = ReflexionLoop.of((List<String> memory, SplittableRandom r) -> {
                Set<String> avoid = new HashSet<>(memory);
                List<String> p = g.generate(t.inputs(), t.outputs(), r);
                for (int k = 0; k < 4 && avoid.contains("failed: " + p); k++) {
                    p = g.generate(t.inputs(), t.outputs(), r);
                }
                return p;
            }, p -> new ReflexionLoop.Feedback(loop.passes(p, t.inputs(), t.outputs()), "wrong output"),
                    (p, fb) -> "failed: " + p).memorySize(max);
            ReflexionLoop.Outcome<List<String>> o = reflexion.run(max, rr);
            for (int b = 0; b < BUDGETS.length; b++) {
                refl[b] += o.success() && o.trials() <= BUDGETS[b] ? 1.0 / test.size() : 0;
            }
        }
        curves.put("Reflexion, text memory", refl);
        GuidedEnumerator enumerator = new GuidedEnumerator(ListPrograms.primitives(), ListPrograms::apply);
        double[] en = new double[BUDGETS.length];
        for (CodeTask t : test) {
            for (int b = 0; b < BUDGETS.length; b++) {
                GuidedEnumerator.Result res = enumerator.enumerate(t.inputs(), t.outputs(), MAX_PROGRAM, BUDGETS[b]);
                en[b] += res.solved() ? 1.0 / test.size() : 0;
            }
        }
        curves.put("GuidedEnumerator", en);
        return new RepairResult(first, curves);
    }

    // ------------------------------------------------------------------------- 9.2.16

    /**
     * Runs 9.2.16.
     *
     * @param given given examples per task (2: under-specified; 5: fully specified)
     * @param samples programs sampled per task
     * @param generatorSteps training steps of the generator
     * @param tasks test tasks
     * @param seed the seed
     * @return method name to its hidden-test top-1 accuracy
     */
    public static Map<String, Double> rerank(int given, int[] samples, int generatorSteps, int tasks, long seed) {
        ExecutionTracer tracer = ExecutionTracer.of(ListPrograms.primitives(), ListPrograms::apply, VALUES);
        ExecutionReranker rr = ExecutionReranker.of(tracer);
        ProgramSampler gen = generator(given, generatorSteps, seed);
        // LEVER verifiers, trained on the generator's samples for training tasks
        ExecutionReranker.Verifier lever = new ExecutionReranker.Verifier(tracer, true, given, LIST_SLOTS, VALUES,
                MAX_PROGRAM, 64, seed + 4);
        ExecutionReranker.Verifier blind = new ExecutionReranker.Verifier(tracer, false, given, LIST_SLOTS, VALUES,
                MAX_PROGRAM, 64, seed + 4);
        SplittableRandom r = new SplittableRandom(seed + 5);
        List<float[]> lx = new ArrayList<>();
        List<float[]> bx = new ArrayList<>();
        List<Boolean> ly = new ArrayList<>();
        for (int k = 0; k < 30; k++) {
            CodeTask t = randomTask(given + 5, r);
            int[][] in = Arrays.copyOf(t.inputs(), given);
            int[][] out = Arrays.copyOf(t.outputs(), given);
            for (List<String> p : gen.sample(condition(in, out, given), 8, 1.0, r).programs()) {
                lx.add(lever.encode(p, in, out));
                bx.add(blind.encode(p, in, out));
                ly.add(t.solvedBy(p));
            }
        }
        boolean[] y = new boolean[ly.size()];
        for (int i = 0; i < y.length; i++) {
            y[i] = ly.get(i);
        }
        for (int s = 0; s < 100; s++) {
            lever.step(lx, y);
            blind.step(bx, y);
        }
        int maxN = Arrays.stream(samples).max().orElse(16);
        SplittableRandom tr = new SplittableRandom(seed + 6);
        Map<String, Double> out = new LinkedHashMap<>();
        for (int n : samples) {
            for (String m : new String[] {"likelihood", "filter then random", "AlphaCode clustering",
                "CodeT dual agreement", "LEVER verifier", "verifier without execution"}) {
                out.put(m + " @" + n, 0.0);
            }
        }
        for (int k = 0; k < tasks; k++) {
            CodeTask t = randomTask(given + 5, tr);
            int[][] in = Arrays.copyOf(t.inputs(), given);
            int[][] outs = Arrays.copyOf(t.outputs(), given);
            int[][] hiddenIn = Arrays.copyOfRange(t.inputs(), given, given + 5);
            int[][] hiddenOut = Arrays.copyOfRange(t.outputs(), given, given + 5);
            ProgramSampler.Samples all = gen.sample(condition(in, outs, given), maxN, 1.0, tr);
            int[][] extra = new int[4][];
            for (int e = 0; e < extra.length; e++) {
                extra[e] = randomList(tr, 5, 7);
            }
            for (int n : samples) {
                List<ExecutionReranker.Candidate> cands = new ArrayList<>();
                for (int i = 0; i < n; i++) {
                    cands.add(new ExecutionReranker.Candidate(all.programs().get(i), all.logProbs()[i]));
                }
                List<ExecutionReranker.Candidate> kept = rr.filter(cands, in, outs);
                List<ExecutionReranker.TestCase> tests = new ArrayList<>();
                Set<String> seen = new HashSet<>();
                for (ExecutionReranker.Candidate c : kept) {
                    for (int[] x : extra) {
                        int[] o = tracer.run(c.program(), x);
                        if (o != null && seen.add(Arrays.toString(x) + Arrays.toString(o))) {
                            tests.add(new ExecutionReranker.TestCase(x, o));
                        }
                    }
                }
                Map<String, ExecutionReranker.Candidate> picks = new LinkedHashMap<>();
                picks.put("likelihood", rr.likelihood(cands));
                picks.put("filter then random", kept.isEmpty() ? null : kept.get(tr.nextInt(kept.size())));
                picks.put("AlphaCode clustering", rr.cluster(kept, extra));
                picks.put("CodeT dual agreement", rr.dualAgreement(kept, tests));
                picks.put("LEVER verifier", rr.lever(cands, lever, in, outs));
                ExecutionReranker.Candidate bestBlind = null;
                double bestScore = -1;
                for (ExecutionReranker.Candidate c : cands) {
                    double sc = blind.probability(blind.encode(c.program(), in, outs)) * Math.exp(c.logProb());
                    if (sc > bestScore) {
                        bestScore = sc;
                        bestBlind = c;
                    }
                }
                picks.put("verifier without execution", bestBlind);
                for (Map.Entry<String, ExecutionReranker.Candidate> e : picks.entrySet()) {
                    ExecutionReranker.Candidate c = e.getValue();
                    boolean ok = c != null;
                    for (int i = 0; ok && i < hiddenIn.length; i++) {
                        int[] o = tracer.run(c.program(), hiddenIn[i]);
                        ok = o != null && Arrays.equals(o, hiddenOut[i]);
                    }
                    out.merge(e.getKey() + " @" + n, ok ? 1.0 / tasks : 0, Double::sum);
                }
            }
        }
        return out;
    }
}
