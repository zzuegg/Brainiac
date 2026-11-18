package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.evolve.EvolutionaryMerge;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.train.ModelMerging;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticRl;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticTask;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Searching the merge recipe of two specialists: a {@link WindowLm} base that knows the answer
 * format is fine-tuned into an addition specialist and a multiplication specialist (operands
 * 0 to 4), and the merged model is scored on the mixed task (the mean greedy accuracy over all
 * addition and all multiplication puzzles). Hand-set merges (task arithmetic at scale 1 and
 * 0.5, TIES, DARE, SLERP) are compared with {@link EvolutionaryMerge}'s CMA-ES search over
 * per-tensor (and, as an ablation, per-model) coefficients, whose fitness is the accuracy on 16
 * validation puzzles; multi-task fine-tuning on both tasks is the upper bound.
 */
public final class MergeSearch {

    private MergeSearch() {
    }

    private static final ArithmeticTask ADD = ArithmeticTask.addition(4);
    private static final ArithmeticTask MUL = ArithmeticTask.multiplication(4);

    private static void sft(WindowLm lm, List<int[]> prompts, List<ArithmeticTask> tasks, int steps) {
        List<int[]> answers = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i++) {
            answers.add(tasks.get(i).solution(prompts.get(i)));
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int i = 0; i < steps; i++) {
            lm.supervisedStep(prompts, answers, opt);
        }
    }

    private static void fineTune(WindowLm lm, int steps, ArithmeticTask... tasks) {
        List<int[]> prompts = new ArrayList<>();
        List<ArithmeticTask> owners = new ArrayList<>();
        for (ArithmeticTask t : tasks) {
            for (int[] p : t.allPrompts()) {
                prompts.add(p);
                owners.add(t);
            }
        }
        sft(lm, prompts, owners, steps);
    }

    private static List<Tensor> weights(WindowLm lm) {
        return lm.parameters().stream().map(Var::value).toList();
    }

    private static void load(WindowLm lm, List<Tensor> w) {
        List<Var> p = lm.parameters();
        for (int i = 0; i < p.size(); i++) {
            p.get(i).assign(w.get(i));
        }
    }

    /** @return the mean of the two tasks' greedy accuracies */
    private static double mixed(WindowLm lm) {
        return (ArithmeticRl.accuracy(lm, ADD) + ArithmeticRl.accuracy(lm, MUL)) / 2;
    }

    private static double validation(WindowLm lm, List<int[]> prompts, List<ArithmeticTask> tasks) {
        int[][] outs = lm.sample(prompts.toArray(int[][]::new),
                Math.max(ADD.maxAnswerTokens(), MUL.maxAnswerTokens()), ADD.eos(), 0, new SplittableRandom(0));
        int right = 0;
        for (int i = 0; i < outs.length; i++) {
            if (tasks.get(i).correct(prompts.get(i), outs[i])) {
                right++;
            }
        }
        return (double) right / outs.length;
    }

    /**
     * Runs the comparison.
     *
     * @param specialistSteps supervised steps for each specialist and the multi-task model
     * @param evaluations fitness evaluations of each evolutionary search
     * @param seeds seeds
     * @return method name to mean mixed-task accuracy over seeds ("base", the two specialists, the
     *     merges, and "multi-task" last)
     */
    public static Map<String, Double> run(int specialistSteps, int evaluations, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        for (long seed = 1; seed <= seeds; seed++) {
            WindowLm base = WindowLm.of(ADD.vocabSize(), 6, 64, seed);
            SplittableRandom r = new SplittableRandom(seed + 11);
            List<int[]> fmt = new ArrayList<>();
            List<ArithmeticTask> fmtTasks = new ArrayList<>();
            for (int i = 0; i < 6; i++) {
                ArithmeticTask t = i % 2 == 0 ? ADD : MUL;
                fmt.add(t.samplePrompt(r));
                fmtTasks.add(t);
            }
            sft(base, fmt, fmtTasks, 40);
            WindowLm add = base.copy();
            fineTune(add, specialistSteps, ADD);
            WindowLm mul = base.copy();
            fineTune(mul, specialistSteps, MUL);
            WindowLm multi = base.copy();
            fineTune(multi, specialistSteps, ADD, MUL);
            List<Tensor> b = weights(base);
            List<Tensor> a = weights(add);
            List<Tensor> m = weights(mul);
            List<List<Tensor>> tau = List.of(ModelMerging.taskVector(b, a), ModelMerging.taskVector(b, m));
            Map<String, List<Tensor>> merges = new LinkedHashMap<>();
            merges.put("task arithmetic (1.0)", ModelMerging.add(b, tau, 1f));
            merges.put("task arithmetic (0.5)", ModelMerging.add(b, tau, 0.5f));
            merges.put("TIES", ModelMerging.ties(b, List.of(a, m), 0.3, 1f));
            merges.put("DARE", ModelMerging.add(b, List.of(ModelMerging.dare(tau.get(0), 0.5, seed),
                    ModelMerging.dare(tau.get(1), 0.5, seed + 1)), 1f));
            merges.put("SLERP", ModelMerging.slerp(a, m, 0.5));
            List<int[]> val = new ArrayList<>();
            List<ArithmeticTask> valTasks = new ArrayList<>();
            SplittableRandom vr = new SplittableRandom(seed + 23);
            for (int i = 0; i < 16; i++) {
                ArithmeticTask t = i % 2 == 0 ? ADD : MUL;
                val.add(t.samplePrompt(vr));
                valTasks.add(t);
            }
            WindowLm probe = base.copy();
            merges.put("evolved, per model", EvolutionaryMerge.of(b, List.of(a, m)).perLayer(false).seed(seed)
                    .search(w -> {
                        load(probe, w);
                        return validation(probe, val, valTasks);
                    }, evaluations));
            merges.put("evolved, per tensor", EvolutionaryMerge.of(b, List.of(a, m)).seed(seed).search(w -> {
                load(probe, w);
                return validation(probe, val, valTasks);
            }, evaluations));
            out.merge("base", mixed(base) / seeds, Double::sum);
            out.merge("addition specialist", mixed(add) / seeds, Double::sum);
            out.merge("multiplication specialist", mixed(mul) / seeds, Double::sum);
            for (Map.Entry<String, List<Tensor>> e : merges.entrySet()) {
                load(probe, e.getValue());
                out.merge(e.getKey(), mixed(probe) / seeds, Double::sum);
            }
            out.merge("multi-task", mixed(multi) / seeds, Double::sum);
        }
        return out;
    }
}
