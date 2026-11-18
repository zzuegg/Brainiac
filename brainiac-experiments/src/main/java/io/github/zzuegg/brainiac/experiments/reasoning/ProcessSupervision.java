package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.align.ImplicitProcessReward;
import io.github.zzuegg.brainiac.experimental.align.MonteCarloValuePpo;
import io.github.zzuegg.brainiac.experimental.align.ProgressReward;
import io.github.zzuegg.brainiac.experimental.align.RolloutStepLabeler;
import io.github.zzuegg.brainiac.experimental.align.StepPreferenceBuilder;
import io.github.zzuegg.brainiac.experimental.align.TokenPolicyGradient;
import io.github.zzuegg.brainiac.experimental.llm.DpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.GenerativeVerifier;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.PpoLmTrainer;
import io.github.zzuegg.brainiac.experimental.llm.ProcessRewardModel;
import io.github.zzuegg.brainiac.experimental.llm.SequenceScorer;
import io.github.zzuegg.brainiac.experimental.llm.StarTrainer;
import io.github.zzuegg.brainiac.experimental.llm.TestTimeCompute;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.SplittableRandom;

/**
 * Wave 9, 9.2.8 to 9.2.12: process supervision for step-by-step arithmetic
 * ({@link StepChainTask}), from a shared supervised start that makes a step error about a
 * quarter of the time.
 * <ul>
 *   <li>{@link #vine} (9.2.8): VinePPO's Monte Carlo step values ({@link MonteCarloValuePpo})
 *       against {@link PpoLmTrainer}'s learned critic, GRPO and RLOO, by accuracy and sampled
 *       tokens; and the step-value error of the learned critic, of 4-rollout Monte Carlo and
 *       of a constant against 64-rollout Monte Carlo.</li>
 *   <li>{@link #shepherd} (9.2.9): Math-Shepherd rollout labels ({@link RolloutStepLabeler})
 *       against the gold step labels, and best-of-16 reranking with process reward models
 *       trained on gold, soft and hard rollout labels, an outcome reward model and a
 *       {@link GenerativeVerifier}.</li>
 *   <li>{@link #progress} (9.2.10): process advantage verifiers ({@link ProgressReward}) as a
 *       dense reward next to outcome-only RL and a Q-value PRM dense reward, and as a beam
 *       score against the PRM in {@link TestTimeCompute#stepBeamSearch}.</li>
 *   <li>{@link #prime} (9.2.11): PRIME's online implicit PRM ({@link ImplicitProcessReward})
 *       against outcome-only RLOO and RLOO plus a frozen PRM, tracking the frozen PRM's score
 *       (reward hacking).</li>
 *   <li>{@link #stepDpo} (9.2.12): Step-DPO pairs ({@link StepPreferenceBuilder}) against
 *       solution-level DPO with as many pairs, rejection-sampling fine-tuning and
 *       {@link StarTrainer}, by accuracy sliced by chain length.</li>
 * </ul>
 */
public final class ProcessSupervision {

    private ProcessSupervision() {
    }

    /**
     * The shared start.
     *
     * @param task the chains
     * @param train training prompts
     * @param test held-out prompts
     * @param base the supervised policy
     */
    public record Setup(StepChainTask task, List<int[]> train, List<int[]> test, WindowLm base) {
    }

    /**
     * Builds the start: a policy fine-tuned on solutions with 25% step errors.
     *
     * @param task the chains
     * @param seed the seed
     * @return the setup
     */
    public static Setup setup(StepChainTask task, long seed) {
        List<int[]> prompts = task.prompts(400, seed);
        List<int[]> train = new ArrayList<>(prompts.subList(0, prompts.size() * 3 / 4));
        List<int[]> test = new ArrayList<>(prompts.subList(prompts.size() * 3 / 4, prompts.size()));
        int window = task.maxPromptTokens() + task.maxSolutionTokens();
        WindowLm lm = WindowLm.of(task.vocabSize(), window, 64, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < 80; s++) {
            List<int[]> ps = new ArrayList<>();
            List<int[]> cs = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                int[] p = train.get(r.nextInt(train.size()));
                ps.add(p);
                cs.add(task.noisySolution(p, 0.25, r));
            }
            lm.supervisedStep(ps, cs, opt);
        }
        return new Setup(task, train, test, lm);
    }

    /**
     * @param lm a policy
     * @param task the chains
     * @param prompts the prompts
     * @return greedy accuracy
     */
    public static double accuracy(WindowLm lm, StepChainTask task, List<int[]> prompts) {
        int[][] outs = lm.sample(prompts.toArray(int[][]::new), task.maxSolutionTokens(), task.eos(), 0,
                new SplittableRandom(0));
        double ok = 0;
        for (int i = 0; i < outs.length; i++) {
            ok += task.correct(prompts.get(i), outs[i]) ? 1 : 0;
        }
        return ok / outs.length;
    }

    private static GrpoTrainer.Reward outcome(StepChainTask task) {
        return (p, c) -> task.correct(p, c) ? 1 : 0;
    }

    // ------------------------------------------------------------------------- 9.2.8

    /**
     * One 9.2.8 trainer.
     *
     * @param accuracy held-out greedy accuracy after training
     * @param tokens tokens sampled in training, value rollouts included
     */
    public record VineRow(double accuracy, long tokens) {
    }

    /**
     * The step-value errors of 9.2.8, mean squared error against 64-rollout Monte Carlo.
     *
     * @param criticByStep the learned critic's error at each step boundary
     * @param monteCarloByStep 4-rollout Monte Carlo's error
     * @param constantByStep a constant's (the mean true value)
     */
    public record ValueError(double[] criticByStep, double[] monteCarloByStep, double[] constantByStep) {
    }

    /**
     * Encodes a prefix as {@link WindowLm}'s (and so {@link PpoLmTrainer}'s critic's) input.
     *
     * @param lm the model
     * @param prefix the tokens so far
     * @return the features
     */
    static float[] encode(WindowLm lm, int[] prefix) {
        int window = lm.window();
        int vocab = lm.vocabSize();
        float[] x = new float[window * vocab + window + 1];
        int n = Math.min(window, prefix.length);
        for (int i = 0; i < n; i++) {
            x[(window - n + i) * vocab + prefix[prefix.length - n + i]] = 1;
        }
        x[window * vocab + n] = 1;
        return x;
    }

    /**
     * Runs 9.2.8.
     *
     * @param setup the start
     * @param updates policy updates per trainer
     * @param seed the seed
     * @param errors receives the value errors (element 0)
     * @return trainer name to its row; {@code "start"} first
     */
    public static Map<String, VineRow> vine(Setup setup, int updates, long seed, ValueError[] errors) {
        StepChainTask task = setup.task();
        int max = task.maxSolutionTokens();
        Map<String, VineRow> out = new LinkedHashMap<>();
        out.put("start", new VineRow(accuracy(setup.base(), task, setup.test()), 0));
        // VinePPO: 16 solutions x 4 rollouts per step boundary
        WindowLm vineLm = setup.base().copy();
        MonteCarloValuePpo vine = MonteCarloValuePpo.of(vineLm).optimizer(Optimizer.adam(3e-3f)).sampling(16, 4)
                .completions(max, task.eos(), task.stepEnd());
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < updates; s++) {
            vine.step(x -> setup.train().get(x.nextInt(setup.train().size())), outcome(task), r);
        }
        out.put("VinePPO (MC values)", new VineRow(accuracy(vineLm, task, setup.test()), vine.tokensSampled()));
        // PPO with a learned critic, same 64 solutions per update as GRPO
        WindowLm ppoLm = setup.base().copy();
        PpoLmTrainer ppo = PpoLmTrainer.of(ppoLm).optimizers(Optimizer.adam(3e-3f), Optimizer.adam(3e-3f)).batch(64)
                .klCoefficient(0.02f).completions(max, task.eos());
        r = new SplittableRandom(seed);
        for (int s = 0; s < updates; s++) {
            ppo.step(x -> setup.train().get(x.nextInt(setup.train().size())), outcome(task), r);
        }
        out.put("PPO (learned critic)", new VineRow(accuracy(ppoLm, task, setup.test()), (long) updates * 64 * max));
        for (GrpoTrainer.Variant v : new GrpoTrainer.Variant[] {GrpoTrainer.Variant.GRPO, GrpoTrainer.Variant.RLOO}) {
            WindowLm lm = setup.base().copy();
            GrpoTrainer g = GrpoTrainer.of(lm).variant(v).optimizer(Optimizer.adam(3e-3f)).groupSize(8)
                    .promptsPerStep(8).klCoefficient(0.02f).completions(max, task.eos());
            r = new SplittableRandom(seed);
            for (int s = 0; s < updates; s++) {
                g.step(x -> setup.train().get(x.nextInt(setup.train().size())), outcome(task), r);
            }
            out.put(v.name(), new VineRow(accuracy(lm, task, setup.test()), (long) updates * 64 * max));
        }
        // step-value errors along the PPO policy's own solutions
        int steps = task.steps(setup.test().get(0));
        double[] critic = new double[steps];
        double[] mc = new double[steps];
        double[] constant = new double[steps];
        List<double[]> truths = new ArrayList<>();
        List<double[]> crit = new ArrayList<>();
        List<double[]> mcs = new ArrayList<>();
        SplittableRandom vr = new SplittableRandom(seed + 7);
        long[] tokens = new long[1];
        for (int[] p : setup.test().subList(0, Math.min(24, setup.test().size()))) {
            int[] c = ppoLm.sample(new int[][] {p}, max, task.eos(), 1.0, vr)[0];
            int[] ends = TokenPolicyGradient.stepEnds(c, task.stepEnd());
            if (task.steps(p) != steps || ends.length < steps) {
                continue;
            }
            double[] t = new double[steps];
            double[] a = new double[steps];
            double[] m = new double[steps];
            for (int s = 0; s < steps; s++) {
                int[] prefix = Arrays.copyOf(c, s == 0 ? 0 : ends[s - 1]);
                t[s] = MonteCarloValuePpo.rolloutValue(ppoLm, p, prefix, outcome(task), 64, max, task.eos(), vr,
                        tokens);
                m[s] = MonteCarloValuePpo.rolloutValue(ppoLm, p, prefix, outcome(task), 4, max, task.eos(), vr,
                        tokens);
                int[] full = Arrays.copyOf(p, p.length + prefix.length);
                System.arraycopy(prefix, 0, full, p.length, prefix.length);
                a[s] = ppo.valueHead().predict(encode(ppoLm, full))[0];
            }
            truths.add(t);
            crit.add(a);
            mcs.add(m);
        }
        for (int s = 0; s < steps; s++) {
            double mean = 0;
            for (double[] t : truths) {
                mean += t[s] / Math.max(1, truths.size());
            }
            for (int i = 0; i < truths.size(); i++) {
                double t = truths.get(i)[s];
                critic[s] += Math.pow(crit.get(i)[s] - t, 2) / truths.size();
                mc[s] += Math.pow(mcs.get(i)[s] - t, 2) / truths.size();
                constant[s] += Math.pow(mean - t, 2) / truths.size();
            }
        }
        errors[0] = new ValueError(critic, mc, constant);
        return out;
    }

    // ------------------------------------------------------------------------- 9.2.9

    /**
     * 9.2.9's measurements.
     *
     * @param softAgreement agreement of soft rollout labels with the gold step labels
     * @param hardAgreement agreement of hard rollout labels
     * @param bestOf16 verifier name to held-out best-of-16 accuracy
     */
    public record ShepherdResult(double softAgreement, double hardAgreement, Map<String, Double> bestOf16) {
    }

    private static SequenceScorer scorer(StepChainTask task, long seed) {
        return SequenceScorer.of(task.vocabSize(), task.maxPromptTokens() + task.maxSolutionTokens(), 64, seed);
    }

    /**
     * Runs 9.2.9.
     *
     * @param setup the start
     * @param solutions training solutions labelled
     * @param rollouts Math-Shepherd rollouts per step
     * @param seed the seed
     * @return the result
     */
    public static ShepherdResult shepherd(Setup setup, int solutions, int rollouts, long seed) {
        StepChainTask task = setup.task();
        int max = task.maxSolutionTokens();
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        List<float[]> gold = new ArrayList<>();
        List<float[]> soft = new ArrayList<>();
        List<float[]> hard = new ArrayList<>();
        RolloutStepLabeler labeler = RolloutStepLabeler.of(setup.base(), max, task.eos(), task.stepEnd())
                .rollouts(rollouts);
        double softAgree = 0;
        double hardAgree = 0;
        for (int i = 0; i < solutions; i++) {
            int[] p = setup.train().get(r.nextInt(setup.train().size()));
            int[] c = setup.base().sample(new int[][] {p}, max, task.eos(), 1.0, r)[0];
            if (c.length == 0) {
                continue;
            }
            RolloutStepLabeler.Labels l = labeler.label(p, c, task::correct, r);
            float[] g = task.stepLabels(p, c);
            ps.add(p);
            cs.add(c);
            gold.add(g);
            soft.add(l.soft());
            hard.add(l.hard());
            softAgree += RolloutStepLabeler.agreement(l.soft(), g);
            hardAgree += RolloutStepLabeler.agreement(l.hard(), g);
        }
        Map<String, TestTimeCompute.Scorer> verifiers = new LinkedHashMap<>();
        int[] stops = {task.stepEnd(), task.eos()};
        ProcessRewardModel goldPrm = ProcessRewardModel.of(scorer(task, seed + 1), stops);
        goldPrm.fit(ps, cs, gold, 300);
        ProcessRewardModel softPrm = ProcessRewardModel.of(scorer(task, seed + 1), stops);
        softPrm.fit(ps, cs, soft, 300);
        ProcessRewardModel hardPrm = ProcessRewardModel.of(scorer(task, seed + 1), stops);
        hardPrm.fit(ps, cs, hard, 300);
        SequenceScorer orm = scorer(task, seed + 1);
        List<int[]> full = new ArrayList<>();
        float[] y = new float[cs.size()];
        for (int i = 0; i < cs.size(); i++) {
            int[] f = Arrays.copyOf(ps.get(i), ps.get(i).length + cs.get(i).length);
            System.arraycopy(cs.get(i), 0, f, ps.get(i).length, cs.get(i).length);
            full.add(f);
            y[i] = task.correct(ps.get(i), cs.get(i)) ? 1 : 0;
        }
        orm.fit(full, y, 300);
        int query = task.tokenizer().encode("?")[0];
        GenerativeVerifier genrm = GenerativeVerifier.of(WindowLm.of(task.vocabSize(),
                task.maxPromptTokens() + task.maxSolutionTokens() + 1, 64, seed + 2), query, 1, 0);
        Optimizer go = Optimizer.adam(3e-3f);
        boolean[] ok = new boolean[cs.size()];
        for (int i = 0; i < ok.length; i++) {
            ok[i] = y[i] > 0.5f;
        }
        for (int s = 0; s < 150; s++) {
            genrm.step(ps, cs, ok, null, go);
        }
        verifiers.put("PRM, gold step labels", (p, c) -> goldPrm.score(p, c, ProcessRewardModel.Aggregate.MIN));
        verifiers.put("PRM, soft rollout labels", (p, c) -> softPrm.score(p, c, ProcessRewardModel.Aggregate.MIN));
        verifiers.put("PRM, hard rollout labels", (p, c) -> hardPrm.score(p, c, ProcessRewardModel.Aggregate.MIN));
        verifiers.put("outcome reward model", (p, c) -> {
            int[] f = Arrays.copyOf(p, p.length + c.length);
            System.arraycopy(c, 0, f, p.length, c.length);
            return orm.score(f);
        });
        verifiers.put("GenerativeVerifier", genrm::score);
        verifiers.put("first sample", (p, c) -> 0);
        Map<String, Double> bon = new LinkedHashMap<>();
        for (Map.Entry<String, TestTimeCompute.Scorer> e : verifiers.entrySet()) {
            SplittableRandom br = new SplittableRandom(seed + 3);
            double acc = 0;
            for (int[] p : setup.test()) {
                int[] best = TestTimeCompute.bestOfN(p, TestTimeCompute.Sampler.of(setup.base(), max, task.eos(), 1.0),
                        16, e.getValue(), br);
                acc += task.correct(p, best) ? 1 : 0;
            }
            bon.put(e.getKey(), acc / setup.test().size());
        }
        return new ShepherdResult(softAgree / cs.size(), hardAgree / cs.size(), bon);
    }

    // ------------------------------------------------------------------------- 9.2.10 / 9.2.11

    /** One group-sampled RL step's data. */
    private record Groups(List<int[]> prompts, List<int[]> completions, float[] outcomes, int[] groupStart) {
    }

    private static Groups sampleGroups(WindowLm lm, Setup setup, int promptsPerStep, int group, SplittableRandom r) {
        StepChainTask task = setup.task();
        int[][] ps = new int[promptsPerStep * group][];
        for (int q = 0; q < promptsPerStep; q++) {
            int[] p = setup.train().get(r.nextInt(setup.train().size()));
            for (int g = 0; g < group; g++) {
                ps[q * group + g] = p;
            }
        }
        int[][] outs = lm.sample(ps, task.maxSolutionTokens(), task.eos(), 1.0, r);
        List<int[]> pl = new ArrayList<>();
        List<int[]> cl = new ArrayList<>();
        List<Float> o = new ArrayList<>();
        int[] starts = new int[promptsPerStep + 1];
        for (int q = 0; q < promptsPerStep; q++) {
            starts[q] = cl.size();
            for (int g = 0; g < group; g++) {
                int i = q * group + g;
                if (outs[i].length > 0) {
                    pl.add(ps[i]);
                    cl.add(outs[i]);
                    o.add(task.correct(ps[i], outs[i]) ? 1f : 0f);
                }
            }
        }
        starts[promptsPerStep] = cl.size();
        float[] of = new float[o.size()];
        for (int i = 0; i < of.length; i++) {
            of[i] = o.get(i);
        }
        return new Groups(pl, cl, of, starts);
    }

    private static float[] groupAdvantages(Groups g, boolean leaveOneOut) {
        float[] adv = new float[g.outcomes().length];
        for (int q = 0; q + 1 < g.groupStart().length; q++) {
            float[] r = Arrays.copyOfRange(g.outcomes(), g.groupStart()[q], g.groupStart()[q + 1]);
            if (r.length == 0) {
                continue;
            }
            float[] a = leaveOneOut ? GrpoTrainer.leaveOneOut(r) : GrpoTrainer.advantages(r, true);
            System.arraycopy(a, 0, adv, g.groupStart()[q], a.length);
        }
        return adv;
    }

    /**
     * 9.2.10's measurements.
     *
     * @param accuracy method name to held-out accuracy at each checkpoint
     * @param samples RL samples at each checkpoint
     * @param beam beam-search accuracy with the PRM and with the PAV score
     */
    public record ProgressResult(Map<String, double[]> accuracy, long[] samples, Map<String, Double> beam) {
    }

    /**
     * Runs 9.2.10.
     *
     * @param setup the start
     * @param steps RL steps per method
     * @param checkpoints evaluations
     * @param alpha the weight of the dense rewards
     * @param seed the seed
     * @return the result
     */
    public static ProgressResult progress(Setup setup, int steps, int checkpoints, float alpha, long seed) {
        StepChainTask task = setup.task();
        int max = task.maxSolutionTokens();
        int[] stops = {task.stepEnd(), task.eos()};
        SplittableRandom r = new SplittableRandom(seed);
        // Q-value PRM on gold labels of base samples
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        List<float[]> labels = new ArrayList<>();
        for (int i = 0; i < 300; i++) {
            int[] p = setup.train().get(r.nextInt(setup.train().size()));
            int[] c = setup.base().sample(new int[][] {p}, max, task.eos(), 1.0, r)[0];
            if (c.length > 0) {
                ps.add(p);
                cs.add(c);
                labels.add(task.stepLabels(p, c));
            }
        }
        ProcessRewardModel prm = ProcessRewardModel.of(scorer(task, seed + 1), stops);
        prm.fit(ps, cs, labels, 300);
        // PAV with a weaker, complementary prover: the base trained half as long on noisier data
        WindowLm prover = WindowLm.of(task.vocabSize(), task.maxPromptTokens() + max, 64, seed + 5);
        Optimizer po = Optimizer.adam(3e-3f);
        for (int s = 0; s < 40; s++) {
            List<int[]> pp = new ArrayList<>();
            List<int[]> pc = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                int[] p = setup.train().get(r.nextInt(setup.train().size()));
                pp.add(p);
                pc.add(task.noisySolution(p, 0.4, r));
            }
            prover.supervisedStep(pp, pc, po);
        }
        ProgressReward pav = ProgressReward.of(prover, scorer(task, seed + 2), max, task.eos(), task.stepEnd())
                .rollouts(8);
        pav.fit(ps.subList(0, Math.min(150, ps.size())), cs.subList(0, Math.min(150, cs.size())), outcome(task), 300,
                r);
        Map<String, double[]> acc = new LinkedHashMap<>();
        long[] samples = new long[checkpoints];
        for (String name : new String[] {"outcome-only GRPO", "GRPO + Q-value PRM", "GRPO + PAV"}) {
            WindowLm lm = setup.base().copy();
            TokenPolicyGradient pg = TokenPolicyGradient.of(lm).optimizer(Optimizer.adam(3e-3f)).klCoefficient(0.02f);
            double[] a = new double[checkpoints];
            int next = 0;
            SplittableRandom rr = new SplittableRandom(seed + 9);
            for (int s = 1; s <= steps; s++) {
                Groups g = sampleGroups(lm, setup, 8, 8, rr);
                float[] seqAdv = groupAdvantages(g, false);
                float[] tok;
                if (name.equals("GRPO + PAV")) {
                    tok = pav.tokenAdvantages(g.prompts(), g.completions(), seqAdv, alpha);
                } else if (name.equals("GRPO + Q-value PRM")) {
                    List<Float> t = new ArrayList<>();
                    for (int i = 0; i < g.completions().size(); i++) {
                        int[] c = g.completions().get(i);
                        double[] q = prm.stepScores(g.prompts().get(i), c);
                        int[] ends = TokenPolicyGradient.stepEnds(c, stops);
                        int start = 0;
                        for (int k = 0; k < ends.length; k++) {
                            for (int j = start; j < ends[k]; j++) {
                                t.add((float) (seqAdv[i] + alpha * (q[k] - 0.5)));
                            }
                            start = ends[k];
                        }
                    }
                    tok = new float[t.size()];
                    for (int i = 0; i < tok.length; i++) {
                        tok[i] = t.get(i);
                    }
                } else {
                    tok = TokenPolicyGradient.broadcast(g.completions(), seqAdv);
                }
                pg.update(g.prompts(), g.completions(), tok);
                if (next < checkpoints && s == steps * (next + 1) / checkpoints) {
                    a[next] = accuracy(lm, task, setup.test());
                    samples[next] = (long) s * 64;
                    next++;
                }
            }
            acc.put(name, a);
        }
        Map<String, Double> beam = new LinkedHashMap<>();
        TestTimeCompute.Scorer q = (p, c) -> prm.score(p, c, ProcessRewardModel.Aggregate.LAST);
        Map<String, TestTimeCompute.Scorer> scorers = new LinkedHashMap<>();
        scorers.put("beam search, PRM", q);
        scorers.put("beam search, PRM + PAV", pav.beamScorer(q, alpha));
        for (Map.Entry<String, TestTimeCompute.Scorer> e : scorers.entrySet()) {
            SplittableRandom br = new SplittableRandom(seed + 11);
            double ok = 0;
            for (int[] p : setup.test()) {
                int[] c = TestTimeCompute.stepBeamSearch(setup.base(), p, e.getValue(), 2, 4, task.steps(p) + 1, 2,
                        task.stepEnd(), task.eos(), br);
                ok += task.correct(p, c) ? 1 : 0;
            }
            beam.put(e.getKey(), ok / setup.test().size());
        }
        return new ProgressResult(acc, samples, beam);
    }

    /**
     * One 9.2.11 method's curve.
     *
     * @param accuracy held-out accuracy at each checkpoint
     * @param frozenPrmScore the frozen PRM's mean score of the policy's samples at each checkpoint
     */
    public record PrimeRow(double[] accuracy, double[] frozenPrmScore) {
    }

    /**
     * Runs 9.2.11.
     *
     * @param setup the start
     * @param steps RL steps per method
     * @param checkpoints evaluations
     * @param seed the seed
     * @return method name to its curve
     */
    public static Map<String, PrimeRow> prime(Setup setup, int steps, int checkpoints, long seed) {
        StepChainTask task = setup.task();
        int max = task.maxSolutionTokens();
        int[] stops = {task.stepEnd(), task.eos()};
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        List<float[]> labels = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            int[] p = setup.train().get(r.nextInt(setup.train().size()));
            int[] c = setup.base().sample(new int[][] {p}, max, task.eos(), 1.0, r)[0];
            if (c.length > 0) {
                ps.add(p);
                cs.add(c);
                labels.add(task.stepLabels(p, c));
            }
        }
        ProcessRewardModel frozen = ProcessRewardModel.of(scorer(task, seed + 1), stops);
        frozen.fit(ps, cs, labels, 200);
        Map<String, PrimeRow> out = new LinkedHashMap<>();
        for (String name : new String[] {"RLOO (outcome)", "RLOO + frozen PRM", "PRIME (online implicit PRM)"}) {
            WindowLm lm = setup.base().copy();
            TokenPolicyGradient pg = TokenPolicyGradient.of(lm).optimizer(Optimizer.adam(3e-3f)).klCoefficient(0.02f);
            ImplicitProcessReward implicit = ImplicitProcessReward.of(setup.base().copy(), setup.base(), 0.05f)
                    .optimizer(Optimizer.adam(1e-3f));
            double[] acc = new double[checkpoints];
            double[] prmScore = new double[checkpoints];
            int next = 0;
            SplittableRandom rr = new SplittableRandom(seed + 9);
            for (int s = 1; s <= steps; s++) {
                Groups g = sampleGroups(lm, setup, 8, 8, rr);
                float[] tok;
                if (name.startsWith("PRIME")) {
                    boolean[] ok = new boolean[g.outcomes().length];
                    for (int i = 0; i < ok.length; i++) {
                        ok[i] = g.outcomes()[i] > 0.5f;
                    }
                    implicit.update(g.prompts(), g.completions(), ok);
                    float[] tr = implicit.tokenRewards(g.prompts(), g.completions());
                    List<Float> all = new ArrayList<>();
                    int k = 0;
                    for (int q = 0; q + 1 < g.groupStart().length; q++) {
                        List<int[]> grp = g.completions().subList(g.groupStart()[q], g.groupStart()[q + 1]);
                        int len = 0;
                        for (int[] c : grp) {
                            len += c.length;
                        }
                        float[] a = ImplicitProcessReward.advantages(grp, Arrays.copyOfRange(tr, k, k + len),
                                Arrays.copyOfRange(g.outcomes(), g.groupStart()[q], g.groupStart()[q + 1]), 1);
                        for (float x : a) {
                            all.add(x);
                        }
                        k += len;
                    }
                    tok = new float[all.size()];
                    for (int i = 0; i < tok.length; i++) {
                        tok[i] = all.get(i);
                    }
                } else {
                    Groups scored = g;
                    if (name.contains("frozen")) {
                        float[] o = g.outcomes().clone();
                        for (int i = 0; i < o.length; i++) {
                            o[i] += (float) frozen.score(g.prompts().get(i), g.completions().get(i),
                                    ProcessRewardModel.Aggregate.MIN);
                        }
                        scored = new Groups(g.prompts(), g.completions(), o, g.groupStart());
                    }
                    tok = TokenPolicyGradient.broadcast(g.completions(), groupAdvantages(scored, true));
                }
                pg.update(g.prompts(), g.completions(), tok);
                if (next < checkpoints && s == steps * (next + 1) / checkpoints) {
                    acc[next] = accuracy(lm, task, setup.test());
                    double sc = 0;
                    SplittableRandom er = new SplittableRandom(seed + 13);
                    int[][] outs = lm.sample(setup.test().toArray(int[][]::new), max, task.eos(), 1.0, er);
                    for (int i = 0; i < outs.length; i++) {
                        sc += outs[i].length == 0 ? 0
                                : frozen.score(setup.test().get(i), outs[i], ProcessRewardModel.Aggregate.MIN);
                    }
                    prmScore[next] = sc / outs.length;
                    next++;
                }
            }
            out.put(name, new PrimeRow(acc, prmScore));
        }
        return out;
    }

    // ------------------------------------------------------------------------- 9.2.12

    /**
     * One 9.2.12 method.
     *
     * @param accuracy overall held-out accuracy
     * @param shortChains accuracy on chains of at most 5 steps
     * @param longChains accuracy on chains of more than 5 steps
     * @param pairs training pairs (or kept solutions) used
     * @param tokens tokens generated to build the training data
     */
    public record StepDpoRow(double accuracy, double shortChains, double longChains, int pairs, long tokens) {
    }

    private static StepDpoRow sliced(WindowLm lm, Setup setup, int pairs, long tokens) {
        List<int[]> shortP = new ArrayList<>();
        List<int[]> longP = new ArrayList<>();
        for (int[] p : setup.test()) {
            (setup.task().steps(p) <= 5 ? shortP : longP).add(p);
        }
        return new StepDpoRow(accuracy(lm, setup.task(), setup.test()),
                shortP.isEmpty() ? 0 : accuracy(lm, setup.task(), shortP),
                longP.isEmpty() ? 0 : accuracy(lm, setup.task(), longP), pairs, tokens);
    }

    /**
     * Runs 9.2.12 (use a task with 4 to 8 steps).
     *
     * @param setup the start
     * @param pairs preference pairs per DPO method
     * @param steps DPO / fine-tuning steps
     * @param seed the seed
     * @return method name to its row; {@code "start"} first
     */
    public static Map<String, StepDpoRow> stepDpo(Setup setup, int pairs, int steps, long seed) {
        StepChainTask task = setup.task();
        int max = task.maxSolutionTokens();
        Map<String, StepDpoRow> out = new LinkedHashMap<>();
        out.put("start", sliced(setup.base(), setup, 0, 0));
        SplittableRandom r = new SplittableRandom(seed);
        StepPreferenceBuilder builder = StepPreferenceBuilder.of(setup.base(), 2, task.stepEnd(), task.eos())
                .samples(8);
        List<DpoTrainer.Preference> stepPairs = new ArrayList<>();
        List<DpoTrainer.Preference> solutionPairs = new ArrayList<>();
        long stepTokens = 0;
        long solutionTokens = 0;
        List<int[]> rftP = new ArrayList<>();
        List<int[]> rftC = new ArrayList<>();
        for (int tries = 0; tries < 50 * pairs && (stepPairs.size() < pairs || solutionPairs.size() < pairs); tries++) {
            int[] p = setup.train().get(r.nextInt(setup.train().size()));
            int[][] two = setup.base().sample(new int[][] {p, p}, max, task.eos(), 1.0, r);
            long t = two[0].length + two[1].length;
            boolean a = task.correct(p, two[0]);
            boolean b = task.correct(p, two[1]);
            for (int k = 0; k < 2; k++) {
                if (task.correct(p, two[k])) {
                    rftP.add(p);
                    rftC.add(two[k]);
                }
            }
            if (a != b && solutionPairs.size() < pairs) {
                solutionPairs.add(new DpoTrainer.Preference(p, a ? two[0] : two[1], a ? two[1] : two[0]));
                solutionTokens += t;
            }
            if (stepPairs.size() < pairs) {
                for (int[] c : two) {
                    if (!task.correct(p, c) && c.length > 0 && stepPairs.size() < pairs) {
                        long before = builder.tokens();
                        Optional<DpoTrainer.Preference> sp = builder.build(p, c, task::stepLabels, r);
                        stepTokens += c.length + builder.tokens() - before;
                        sp.ifPresent(stepPairs::add);
                    }
                }
            }
        }
        for (String name : new String[] {"Step-DPO", "solution DPO"}) {
            List<DpoTrainer.Preference> data = name.equals("Step-DPO") ? stepPairs : solutionPairs;
            WindowLm lm = setup.base().copy();
            DpoTrainer dpo = DpoTrainer.of(lm).beta(0.5f).optimizer(Optimizer.adam(2e-3f));
            SplittableRandom br = new SplittableRandom(seed + 2);
            for (int s = 0; s < steps && !data.isEmpty(); s++) {
                List<DpoTrainer.Preference> b = new ArrayList<>();
                for (int i = 0; i < 32; i++) {
                    b.add(data.get(br.nextInt(data.size())));
                }
                dpo.step(b);
            }
            out.put(name, sliced(lm, setup, data.size(), name.equals("Step-DPO") ? stepTokens : solutionTokens));
        }
        WindowLm rft = setup.base().copy();
        Optimizer ro = Optimizer.adam(2e-3f);
        SplittableRandom fr = new SplittableRandom(seed + 3);
        for (int s = 0; s < steps && !rftP.isEmpty(); s++) {
            List<int[]> bp = new ArrayList<>();
            List<int[]> bc = new ArrayList<>();
            for (int i = 0; i < 32; i++) {
                int j = fr.nextInt(rftP.size());
                bp.add(rftP.get(j));
                bc.add(rftC.get(j));
            }
            rft.supervisedStep(bp, bc, ro);
        }
        out.put("rejection-sampling FT", sliced(rft, setup, rftP.size(), solutionTokens + stepTokens));
        WindowLm starLm = setup.base().copy();
        StarTrainer star = StarTrainer.of(starLm, task::correct, p -> p).rationalization(false).samples(4)
                .generation(max, task.eos(), 1.0).finetune(steps, () -> Optimizer.adam(2e-3f));
        StarTrainer.Round round = null;
        for (int it = 0; it < 2; it++) {
            round = star.iterate(setup.train().subList(0, Math.min(150, setup.train().size())), r);
        }
        out.put("STaR", sliced(starLm, setup, round == null ? 0 : round.kept(),
                2L * Math.min(150, setup.train().size()) * 4 * max));
        return out;
    }
}
