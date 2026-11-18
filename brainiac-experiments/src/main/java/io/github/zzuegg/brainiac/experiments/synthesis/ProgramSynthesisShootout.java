package io.github.zzuegg.brainiac.experiments.synthesis;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.evolve.BottomUpSynthesis;
import io.github.zzuegg.brainiac.experimental.evolve.Bustle;
import io.github.zzuegg.brainiac.experimental.evolve.CrossBeam;
import io.github.zzuegg.brainiac.experimental.evolve.GuidedEnumerator;
import io.github.zzuegg.brainiac.experimental.evolve.LibraryLearner;
import io.github.zzuegg.brainiac.experimental.evolve.Probe;
import io.github.zzuegg.brainiac.experimental.evolve.ProgramEvolution;
import io.github.zzuegg.brainiac.experimental.llm.ProgramGenerator;
import io.github.zzuegg.brainiac.experimental.nn.TransformerSeq2Seq;
import io.github.zzuegg.brainiac.experiments.reasoning.ListPrograms;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Predicate;

/**
 * Program synthesis from input/output examples on the {@link ListPrograms} DSL: which
 * search finds a program that reproduces five examples and generalises to three held-out
 * inputs, and how many candidate programs it tries first.
 *
 * <ul>
 *   <li>{@link Method#ENUMERATE}: breadth-first enumeration with observational-equivalence
 *       pruning ({@link GuidedEnumerator#enumerate}), the factored-out enumerator of
 *       {@code SelfImprovingCoders};</li>
 *   <li>{@link Method#DEEPCODER}: the same enumerator under DeepCoder's sort-and-add
 *       schedule, guided by an attribute predictor trained on random programs, and
 *       {@link Method#DEEPCODER_ORDERED}, which keeps every primitive but tries each
 *       level's programs best-first by the predicted probabilities;</li>
 *   <li>{@link Method#LIBRARY_LEARNER}: DreamCoder-style best-first search
 *       ({@link LibraryLearner}) after two wake/sleep rounds on the training programs;</li>
 *   <li>{@link Method#EVOLUTION}: AutoML-Zero's regularized evolution
 *       ({@link ProgramEvolution#regularizedEvolution}) on the share of examples right;</li>
 *   <li>{@link Method#ROBUSTFILL}: neural program generation ({@link ProgramGenerator})
 *       with execution-filtered beam search, at beam widths 1 and 16;</li>
 *   <li>{@link Method#INDUCTION}: neural program induction, a transformer that maps two
 *       examples and a new input straight to the output, with no program in between;</li>
 *   <li>{@link #bottomUp} (items 11.2.12-11.2.13): weighted bottom-up enumeration
 *       ({@link Method#BOTTOM_UP}), BUSTLE's learned value weights ({@link Method#BUSTLE}),
 *       CrossBeam's learned search policy ({@link Method#CROSSBEAM}) and Probe's
 *       just-in-time grammar ({@link Method#PROBE}), against plain enumeration and
 *       DeepCoder's ordered search at several budgets.</li>
 * </ul>
 *
 * <p>Training programs have 1-4 primitives; test programs 2-5, so length 5 is longer than
 * anything the networks saw. The cost unit is programs tried: the enumerator extends a
 * prefix by one primitive per program, the library learner and evolution run whole
 * programs, and the beam search counts the candidates it executes.
 */
public final class ProgramSynthesisShootout {

    /** The synthesisers. */
    public enum Method {
        /** Unguided enumeration with observational-equivalence pruning. */
        ENUMERATE,
        /** Neural-guided sort-and-add enumeration. */
        DEEPCODER,
        /** Every primitive, each level ordered by the predicted probabilities (no restarts). */
        DEEPCODER_ORDERED,
        /** DreamCoder-style library learning. */
        LIBRARY_LEARNER,
        /** Regularized evolution. */
        EVOLUTION,
        /** Seq2seq program generation with beam search and execution filtering. */
        ROBUSTFILL,
        /** Direct output prediction (program induction). */
        INDUCTION,
        /** Unguided weighted bottom-up enumeration ({@link BottomUpSynthesis}). */
        BOTTOM_UP,
        /** Bottom-up with BUSTLE's learned value reweighting. */
        BUSTLE,
        /** Bottom-up with CrossBeam's learned argument-selection policy. */
        CROSSBEAM,
        /** Bottom-up with Probe's just-in-time grammar learning (no pretraining). */
        PROBE
    }

    /**
     * One method's scores.
     *
     * @param method the method
     * @param label a variant label (such as the beam width)
     * @param solved the share of test tasks whose program also fits the held-out inputs
     * @param consistent the share for which a program fitting the shown examples was found
     * @param meanTried mean programs tried per task (budget when unsolved)
     * @param millis wall time of the search (training excluded)
     */
    public record Row(Method method, String label, double solved, double consistent, double meanTried, long millis) {
        @Override
        public String toString() {
            return String.format("%s%s solved %.2f (consistent %.2f) tried %.0f in %d ms", method,
                    label.isEmpty() ? "" : "[" + label + "]", solved, consistent, meanTried, millis);
        }
    }

    /**
     * A test task.
     *
     * @param inputs the shown example inputs
     * @param outputs their outputs
     * @param heldIn held-out inputs
     * @param heldOut their outputs
     * @param hidden the program that made them
     */
    public record Task(int[][] inputs, int[][] outputs, int[][] heldIn, int[][] heldOut, List<String> hidden) {
        boolean generalises(List<String> program) {
            for (int i = 0; i < heldIn.length; i++) {
                int[] o = ListPrograms.execute(program, heldIn[i]);
                if (o == null || !Arrays.equals(o, heldOut[i])) {
                    return false;
                }
            }
            return true;
        }

        boolean fits(List<String> program) {
            for (int i = 0; i < inputs.length; i++) {
                int[] o = ListPrograms.execute(program, inputs[i]);
                if (o == null || !Arrays.equals(o, outputs[i])) {
                    return false;
                }
            }
            return true;
        }
    }

    private static final int SHOWN = 5;
    private static final int HELD = 3;

    private ProgramSynthesisShootout() {
    }

    /** @return the enumerator over the {@link ListPrograms} primitives */
    public static GuidedEnumerator enumerator() {
        return new GuidedEnumerator(ListPrograms.primitives(), ListPrograms::apply);
    }

    /**
     * Random test tasks.
     *
     * @param n how many
     * @param minLength the shortest hidden program
     * @param maxLength the longest
     * @param seed the seed
     * @return the tasks
     */
    public static List<Task> tasks(int n, int minLength, int maxLength, long seed) {
        GuidedEnumerator g = enumerator();
        SplittableRandom r = new SplittableRandom(seed);
        List<Task> out = new ArrayList<>();
        while (out.size() < n) {
            int len = minLength + r.nextInt(maxLength - minLength + 1);
            Object[] t = g.randomTask(len, SHOWN + HELD, r);
            if (t == null) {
                continue;
            }
            int[][] in = (int[][]) t[1];
            int[][] o = (int[][]) t[2];
            List<String> hidden = new ArrayList<>();
            for (int i : (int[]) t[0]) {
                hidden.add(ListPrograms.primitives().get(i));
            }
            out.add(new Task(Arrays.copyOf(in, SHOWN), Arrays.copyOf(o, SHOWN), Arrays.copyOfRange(in, SHOWN, in.length),
                    Arrays.copyOfRange(o, SHOWN, o.length), List.copyOf(hidden)));
        }
        return out;
    }

    /**
     * Trains DeepCoder's attribute predictor on random programs of 1-4 primitives.
     *
     * @param steps training steps of 32 programs
     * @param seed the seed
     * @return the predictor
     */
    public static GuidedEnumerator.AttributePredictor trainPredictor(int steps, long seed) {
        GuidedEnumerator g = enumerator();
        int p = g.primitives().size();
        GuidedEnumerator.AttributePredictor predictor = new GuidedEnumerator.AttributePredictor(p, 96, seed);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            List<int[][]> ins = new ArrayList<>();
            List<int[][]> outs = new ArrayList<>();
            List<float[]> used = new ArrayList<>();
            while (used.size() < 32) {
                Object[] t = g.randomTask(1 + r.nextInt(4), SHOWN, r);
                if (t == null) {
                    continue;
                }
                float[] u = new float[p];
                for (int i : (int[]) t[0]) {
                    u[i] = 1;
                }
                ins.add((int[][]) t[1]);
                outs.add((int[][]) t[2]);
                used.add(u);
            }
            predictor.trainStep(ins, outs, used);
        }
        return predictor;
    }

    /**
     * Runs the search-based methods.
     *
     * @param tasks the test tasks
     * @param budget the most programs each method may try per task
     * @param predictorSteps training steps of DeepCoder's predictor
     * @param seed the seed
     * @return one row per method: enumerate, DeepCoder, DeepCoder ordered, library learner, evolution
     */
    public static List<Row> searchers(List<Task> tasks, long budget, int predictorSteps, long seed) {
        GuidedEnumerator g = enumerator();
        GuidedEnumerator.AttributePredictor predictor = trainPredictor(predictorSteps, seed);
        LibraryLearner library = trainLibrary(seed, (int) Math.min(budget, 3000));
        List<Row> rows = new ArrayList<>();
        for (Method m : new Method[] {Method.ENUMERATE, Method.DEEPCODER, Method.DEEPCODER_ORDERED,
                Method.LIBRARY_LEARNER, Method.EVOLUTION}) {
            long t0 = System.nanoTime();
            double solved = 0;
            double consistent = 0;
            double tried = 0;
            SplittableRandom r = new SplittableRandom(seed + 5);
            for (Task t : tasks) {
                List<String> found;
                long cost;
                switch (m) {
                    case ENUMERATE -> {
                        GuidedEnumerator.Result res = g.enumerate(t.inputs(), t.outputs(), 5, budget);
                        found = res.program();
                        cost = res.evaluated();
                    }
                    case DEEPCODER -> {
                        GuidedEnumerator.Result res = g.sortAndAdd(t.inputs(), t.outputs(), 5,
                                predictor.predict(t.inputs(), t.outputs()), 8, budget);
                        found = res.program();
                        cost = res.evaluated();
                    }
                    case DEEPCODER_ORDERED -> {
                        GuidedEnumerator.Result res = g.enumerate(t.inputs(), t.outputs(), 5, g.primitives(),
                                predictor.predict(t.inputs(), t.outputs()), budget);
                        found = res.program();
                        cost = res.evaluated();
                    }
                    case LIBRARY_LEARNER -> {
                        long[] evals = {0};
                        LibraryLearner.Solution s = library.search(t::fits, evals);
                        found = s == null ? null : s.expanded();
                        cost = evals[0];
                    }
                    default -> {
                        long[] at = {budget};
                        List<List<String>> best = new ArrayList<>();
                        best.add(null);
                        int[] count = {0};
                        ProgramEvolution.regularizedEvolution(List.of(ListPrograms.primitives().get(0)),
                                ProgramSynthesisShootout::mutate, p -> {
                                    count[0]++;
                                    double f = fitness(t, p);
                                    if (f >= 1 && best.get(0) == null) {
                                        best.set(0, p);
                                        at[0] = count[0];
                                    }
                                    return f;
                                }, 50, 5, (int) budget - 1, r);
                        found = best.get(0);
                        cost = at[0];
                    }
                }
                tried += found == null ? budget : cost;
                consistent += found != null ? 1 : 0;
                solved += found != null && t.generalises(found) ? 1 : 0;
            }
            rows.add(new Row(m, "", solved / tasks.size(), consistent / tasks.size(), tried / tasks.size(),
                    (System.nanoTime() - t0) / 1_000_000));
        }
        return rows;
    }

    /** One search's result on one task: programs tried and the program, if any. */
    private record Attempt(long cost, List<String> program) {
    }

    /**
     * The bottom-up family (items 11.2.12-11.2.13) against plain enumeration and DeepCoder's
     * ordered search. Every search is deterministic and only truncated by the budget, so each
     * runs once at the largest budget and the smaller budgets are read off the cost at which
     * it succeeded.
     *
     * @param tasks the test tasks
     * @param budgets the budgets to report, ascending
     * @param predictorSteps training steps of DeepCoder's predictor
     * @param bustleSteps training steps of BUSTLE's usefulness model (16 programs each)
     * @param crossBeamSteps CrossBeam training programs
     * @param seed the seed
     * @return per method one row per budget (label {@code "budget B"}) and one for the tasks
     *     of length 4-5 at the largest budget (label {@code "len 4-5"})
     */
    public static List<Row> bottomUp(List<Task> tasks, long[] budgets, int predictorSteps, int bustleSteps,
            int crossBeamSteps, long seed) {
        GuidedEnumerator g = enumerator();
        BottomUpSynthesis bu = new BottomUpSynthesis(ListPrograms.primitives(), ListPrograms::apply);
        long max = budgets[budgets.length - 1];
        GuidedEnumerator.AttributePredictor predictor = trainPredictor(predictorSteps, seed);
        Bustle bustle = new Bustle(bu, 32, 3.0, seed);
        bustle.train(g, bustleSteps, 16, 4, SHOWN, 3e-3f, new SplittableRandom(seed + 11));
        CrossBeam crossBeam = new CrossBeam(bu, 32, seed);
        crossBeam.train(g, crossBeamSteps, 4, SHOWN, 2, 3e-3f, new SplittableRandom(seed + 12));
        Probe probe = new Probe(bu, 8);
        List<Row> rows = new ArrayList<>();
        for (Method m : new Method[] {Method.ENUMERATE, Method.DEEPCODER_ORDERED, Method.BOTTOM_UP, Method.BUSTLE,
                Method.CROSSBEAM, Method.PROBE}) {
            long t0 = System.nanoTime();
            List<Attempt> attempts = new ArrayList<>();
            for (Task t : tasks) {
                attempts.add(switch (m) {
                    case ENUMERATE -> {
                        GuidedEnumerator.Result r = g.enumerate(t.inputs(), t.outputs(), 5, max);
                        yield new Attempt(r.evaluated(), r.program());
                    }
                    case DEEPCODER_ORDERED -> {
                        GuidedEnumerator.Result r = g.enumerate(t.inputs(), t.outputs(), 5, g.primitives(),
                                predictor.predict(t.inputs(), t.outputs()), max);
                        yield new Attempt(r.evaluated(), r.program());
                    }
                    case BOTTOM_UP -> {
                        BottomUpSynthesis.Result r = bu.search(t.inputs(), t.outputs(), 5, max);
                        yield new Attempt(r.evaluated(), r.program());
                    }
                    case BUSTLE -> {
                        BottomUpSynthesis.Result r = bustle.search(t.inputs(), t.outputs(), 5, max);
                        yield new Attempt(r.evaluated(), r.program());
                    }
                    case CROSSBEAM -> {
                        CrossBeam.Result r = crossBeam.search(t.inputs(), t.outputs(), 5, 8, max);
                        yield new Attempt(r.evaluated(), r.program());
                    }
                    default -> {
                        Probe.Result r = probe.search(t.inputs(), t.outputs(), 5, max);
                        yield new Attempt(r.evaluated(), r.program());
                    }
                });
            }
            long millis = (System.nanoTime() - t0) / 1_000_000;
            for (long b : budgets) {
                rows.add(score(m, "budget " + b, tasks, attempts, b, millis, 0));
            }
            rows.add(score(m, "len 4-5", tasks, attempts, max, millis, 4));
        }
        return rows;
    }

    private static Row score(Method m, String label, List<Task> tasks, List<Attempt> attempts, long budget, long millis,
            int minLength) {
        double solved = 0;
        double consistent = 0;
        double tried = 0;
        int count = 0;
        for (int i = 0; i < tasks.size(); i++) {
            Task t = tasks.get(i);
            if (t.hidden().size() < minLength) {
                continue;
            }
            count++;
            Attempt a = attempts.get(i);
            boolean found = a.program() != null && a.cost() <= budget;
            tried += found ? a.cost() : budget;
            consistent += found ? 1 : 0;
            solved += found && t.generalises(a.program()) ? 1 : 0;
        }
        int c = Math.max(1, count);
        return new Row(m, label, solved / c, consistent / c, tried / c, millis);
    }

    private static LibraryLearner trainLibrary(long seed, int budget) {
        LibraryLearner library = LibraryLearner.of(ListPrograms.primitives()).maxLength(5).budget(budget);
        List<Task> train = tasks(40, 1, 4, seed + 77);
        List<Predicate<List<String>>> predicates = new ArrayList<>();
        for (Task t : train) {
            predicates.add(t::fits);
        }
        for (int round = 0; round < 2; round++) {
            library.wake(predicates);
            library.sleep();
        }
        return library;
    }

    private static double fitness(Task t, List<String> program) {
        double right = 0;
        double partial = 0;
        for (int i = 0; i < t.inputs().length; i++) {
            int[] o = ListPrograms.execute(program, t.inputs()[i]);
            if (o == null) {
                continue;
            }
            int[] want = t.outputs()[i];
            right += Arrays.equals(o, want) ? 1 : 0;
            int match = 0;
            for (int k = 0; k < Math.min(o.length, want.length); k++) {
                match += o[k] == want[k] ? 1 : 0;
            }
            partial += match / (double) Math.max(1, Math.max(o.length, want.length));
        }
        int n = t.inputs().length;
        return right == n ? 1 : 0.7 * right / n + 0.3 * partial / n;
    }

    private static List<String> mutate(List<String> p, SplittableRandom r) {
        List<String> prims = ListPrograms.primitives();
        List<String> c = new ArrayList<>(p);
        int kind = r.nextInt(3);
        if (kind == 0 && c.size() < 5) {
            c.add(r.nextInt(c.size() + 1), prims.get(r.nextInt(prims.size())));
        } else if (kind == 1 && c.size() > 1) {
            c.remove(r.nextInt(c.size()));
        } else {
            c.set(r.nextInt(c.size()), prims.get(r.nextInt(prims.size())));
        }
        return c;
    }

    /**
     * Trains RobustFill on random programs of 1-4 primitives and searches the test tasks at
     * each beam width.
     *
     * @param tasks the test tasks
     * @param steps training steps of 16 programs
     * @param widths the beam widths to test
     * @param seed the seed
     * @return one row per beam width
     */
    public static List<Row> robustFill(List<Task> tasks, int steps, int[] widths, long seed) {
        GuidedEnumerator g = enumerator();
        ProgramGenerator gen = new ProgramGenerator(ListPrograms.primitives(), ListPrograms::apply, 5, 32, seed);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            List<int[][]> ins = new ArrayList<>();
            List<int[][]> outs = new ArrayList<>();
            List<List<String>> programs = new ArrayList<>();
            while (programs.size() < 16) {
                Object[] t = g.randomTask(1 + r.nextInt(4), SHOWN, r);
                if (t == null) {
                    continue;
                }
                List<String> p = new ArrayList<>();
                for (int i : (int[]) t[0]) {
                    p.add(ListPrograms.primitives().get(i));
                }
                ins.add((int[][]) t[1]);
                outs.add((int[][]) t[2]);
                programs.add(p);
            }
            gen.trainStep(ins, outs, programs);
        }
        List<Row> rows = new ArrayList<>();
        for (int w : widths) {
            long t0 = System.nanoTime();
            double solved = 0;
            double consistent = 0;
            double tried = 0;
            for (Task t : tasks) {
                List<List<String>> found = gen.search(t.inputs(), t.outputs(), w);
                tried += (double) w * 6;
                consistent += found.isEmpty() ? 0 : 1;
                solved += !found.isEmpty() && t.generalises(found.get(0)) ? 1 : 0;
            }
            rows.add(new Row(Method.ROBUSTFILL, "beam " + w, solved / tasks.size(), consistent / tasks.size(),
                    tried / tasks.size(), (System.nanoTime() - t0) / 1_000_000));
        }
        return rows;
    }

    /**
     * Neural program induction: a transformer reads two examples and a new input and writes
     * the output list directly.
     *
     * @param tasks the test tasks (scored on their held-out inputs)
     * @param steps training steps of 16 tasks
     * @param seed the seed
     * @return the row; "solved" is the share of tasks with every held-out output exact
     */
    public static Row induction(List<Task> tasks, int steps, long seed) {
        GuidedEnumerator g = enumerator();
        int values = 40;
        int sep = values + 1;
        int pad = values + 2;
        int end = values + 1;   // in the target vocabulary: 0..40 values, 41 end
        TransformerSeq2Seq model = TransformerSeq2Seq.of(values + 3, values + 2, 48, 4, 1, seed);
        io.github.zzuegg.brainiac.train.Optimizer opt = io.github.zzuegg.brainiac.train.Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[][] src = new int[16][];
            int[][] tgt = new int[16][];
            int i = 0;
            while (i < 16) {
                Object[] t = g.randomTask(1 + r.nextInt(4), 3, r);
                if (t == null) {
                    continue;
                }
                int[][] in = (int[][]) t[1];
                int[][] o = (int[][]) t[2];
                src[i] = inductionSource(in, o, in[2], values, sep, pad);
                tgt[i] = inductionTarget(o[2], values, end);
                i++;
            }
            model.loss(src, tgt).backward();
            opt.step(model.parameters());
        }
        long t0 = System.nanoTime();
        double solved = 0;
        for (Task t : tasks) {
            boolean all = true;
            for (int h = 0; h < t.heldIn().length && all; h++) {
                int[][] src = {inductionSource(t.inputs(), t.outputs(), t.heldIn()[h], values, sep, pad)};
                int[] out = Var.noGrad(() -> model.greedy(src, 11))[0];
                int[] want = inductionTarget(t.heldOut()[h], values, end);
                for (int j = 0; j < want.length && want[j] >= 0; j++) {
                    all &= out[j] == want[j];
                }
            }
            solved += all ? 1 : 0;
        }
        return new Row(Method.INDUCTION, "", solved / tasks.size(), solved / tasks.size(), 0,
                (System.nanoTime() - t0) / 1_000_000);
    }

    private static int[] inductionSource(int[][] in, int[][] out, int[] query, int values, int sep, int pad) {
        int[] s = new int[56];
        Arrays.fill(s, pad);
        int k = 0;
        for (int e = 0; e < 2; e++) {
            for (int v : in[e]) {
                s[k++] = Math.min(values, v);
            }
            s[k++] = sep;
            for (int j = 0; j < Math.min(10, out[e].length); j++) {
                s[k++] = Math.min(values, out[e][j]);
            }
            s[k++] = sep;
        }
        for (int v : query) {
            s[k++] = Math.min(values, v);
        }
        return s;
    }

    private static int[] inductionTarget(int[] out, int values, int end) {
        int[] t = new int[11];
        Arrays.fill(t, -1);
        int n = Math.min(10, out.length);
        for (int j = 0; j < n; j++) {
            t[j] = Math.min(values, out[j]);
        }
        t[n] = end;
        return t;
    }
}
