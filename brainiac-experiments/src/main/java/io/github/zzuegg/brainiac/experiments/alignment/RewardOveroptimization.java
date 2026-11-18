package io.github.zzuegg.brainiac.experiments.alignment;

import io.github.zzuegg.brainiac.experimental.align.BestOfNDistiller;
import io.github.zzuegg.brainiac.experimental.align.RewardEnsemble;
import io.github.zzuegg.brainiac.experimental.align.WeightAveragedReward;
import io.github.zzuegg.brainiac.experimental.llm.DpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.PpoLmTrainer;
import io.github.zzuegg.brainiac.experimental.llm.RewardModel;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 9, 9.2.1 and 9.2.2: reward over-optimization against a known gold reward
 * ({@link GoldRewardHarness}).
 *
 * <p>{@link #run} (9.2.1) trains Bradley-Terry proxy reward models on noisy preference pairs of
 * the supervised policy's samples, then optimizes against each proxy in two ways and records
 * the gold reward against the KL from the reference (the over-optimization curve of Gao et al.,
 * <a href="https://arxiv.org/abs/2210.10760">arXiv:2210.10760</a>): best-of-n selection, whose
 * KL is {@code log n - (n - 1) / n}, and GRPO with no KL penalty. The proxies are one
 * {@link RewardModel}; a {@link RewardEnsemble} of three seeds combined by mean, worst case and
 * uncertainty penalty (Coste et al., <a href="https://arxiv.org/abs/2310.02743">arXiv:2310.02743</a>);
 * and a {@link WeightAveragedReward} (WARM, <a href="https://arxiv.org/abs/2401.12187">
 * arXiv:2401.12187</a>) of three fine-tunes of one initialization, against the single proxy and
 * the gold reward itself. {@link #klSweep} adds the single proxy under a KL-penalty sweep and
 * {@link PpoLmTrainer} against it.
 *
 * <p>{@link #bond} (9.2.2) distils best-of-N of the supervised policy under the single proxy
 * into the policy with {@link BestOfNDistiller} and compares, at one sample per query, with
 * best-of-N at inference (N samples per query), rejection-sampling fine-tuning on best-of-N
 * outputs, DPO on best/worst pairs and GRPO.
 */
public final class RewardOveroptimization {

    /**
     * One point of an over-optimization curve.
     *
     * @param kl the KL from the reference
     * @param gold the mean gold reward
     * @param proxy the mean proxy reward
     */
    public record Point(double kl, double gold, double proxy) {
    }

    /**
     * One proxy's curves.
     *
     * @param bestOfN best-of-n at n = 1, 4, 16, 64
     * @param rl GRPO checkpoints, the start first
     */
    public record Curves(List<Point> bestOfN, List<Point> rl) {

        /** @return the highest gold reward of the RL curve */
        public double rlPeak() {
            return rl.stream().mapToDouble(Point::gold).max().orElse(0);
        }

        /** @return the KL at the RL curve's gold peak */
        public double rlPeakKl() {
            return rl.stream().max((a, b) -> Double.compare(a.gold(), b.gold())).map(Point::kl).orElse(0.0);
        }

        /** @return the gold reward at the end of RL */
        public double rlFinal() {
            return rl.get(rl.size() - 1).gold();
        }

        /** @return the gold reward at the largest best-of-n */
        public double bonFinal() {
            return bestOfN.get(bestOfN.size() - 1).gold();
        }

        /** @return the highest best-of-n gold reward */
        public double bonPeak() {
            return bestOfN.stream().mapToDouble(Point::gold).max().orElse(0);
        }
    }

    /**
     * Everything 9.2.1 measured.
     *
     * @param curves proxy name to its curves
     * @param proxyAccuracy proxy name to its held-out pair accuracy
     * @param warmMemberAccuracy the mean held-out accuracy of the WARM members
     */
    public record Result(Map<String, Curves> curves, Map<String, Double> proxyAccuracy, double warmMemberAccuracy) {
    }

    private static final int[] BON = {1, 4, 16, 64};
    private static final int RM_STEPS = 40;
    private static final int RM_BATCH = 32;
    /** Reward models per ensemble and fine-tunes per WARM average. */
    private static final int MEMBERS = 3;

    private RewardOveroptimization() {
    }

    /** The trained proxies of one seed. */
    record Proxies(RewardEnsemble single, RewardEnsemble ensemble, RewardEnsemble warm,
            Map<String, Double> accuracy, double warmMembers) {
    }

    static Proxies proxies(GoldRewardHarness.Split split, WindowLm sft, int pairs, boolean all, long seed) {
        SplittableRandom random = new SplittableRandom(seed + 11);
        GoldRewardHarness.Pairs train = GoldRewardHarness.pairs(sft, split.train(), pairs, 0.1, random);
        GoldRewardHarness.Pairs test = GoldRewardHarness.pairs(sft, split.test(), 256, 1e-6, random);
        List<RewardEnsemble.BatchScorer> members = new ArrayList<>();
        Map<String, Double> acc = new LinkedHashMap<>();
        for (int m = 0; m < (all ? MEMBERS : 1); m++) {
            RewardModel rm = GoldRewardHarness.rewardModel(seed + 100 + m);
            WeightAveragedReward.train(rm, train.chosen(), train.rejected(), RM_STEPS, RM_BATCH, 3e-3f,
                    new SplittableRandom(seed + 200 + m));
            members.add(GoldRewardHarness.member(rm));
            if (m == 0) {
                acc.put("single RM", rm.accuracy(test.chosen(), test.rejected()));
            }
        }
        List<int[]> calP = new ArrayList<>();
        List<int[]> calR = new ArrayList<>();
        int[][] cp = GoldRewardHarness.repeat(split.train(), 4);
        int[][] out = sft.sample(cp, GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS, 1.0, random);
        for (int i = 0; i < out.length; i++) {
            calP.add(cp[i]);
            calR.add(out[i]);
        }
        RewardEnsemble ens = RewardEnsemble.of(members, RewardEnsemble.Mode.MEAN, 0).calibrate(calP, calR);
        if (!all) {
            return new Proxies(ens.member(0), ens, ens, acc, 0);
        }
        // WARM: a shared initialization, briefly trained on all pairs, then MEMBERS fine-tunes
        RewardModel shared = GoldRewardHarness.rewardModel(seed + 300);
        WeightAveragedReward.train(shared, train.chosen(), train.rejected(), RM_STEPS / 2, RM_BATCH, 3e-3f,
                new SplittableRandom(seed + 301));
        WeightAveragedReward.Result warm = fineTunes(shared, train, seed);
        double memberAcc = 0;
        for (RewardModel m : warm.members()) {
            memberAcc += m.accuracy(test.chosen(), test.rejected()) / warm.members().size();
        }
        acc.put("WARM", warm.averaged().accuracy(test.chosen(), test.rejected()));
        RewardEnsemble w = RewardEnsemble.of(List.of(GoldRewardHarness.member(warm.averaged())),
                RewardEnsemble.Mode.MEAN, 0).calibrate(calP, calR);
        return new Proxies(ens.member(0), ens, w, acc, memberAcc);
    }

    /** WARM fine-tunes that start from a trained shared reward model, head included. */
    private static WeightAveragedReward.Result fineTunes(RewardModel shared, GoldRewardHarness.Pairs train, long seed) {
        List<RewardModel> members = new ArrayList<>();
        SplittableRandom random = new SplittableRandom(seed + 400);
        for (int m = 0; m < MEMBERS; m++) {
            RewardModel rm = GoldRewardHarness.rewardModel(seed + 300);
            copy(shared, rm);
            // each member sees its own bootstrap half of the pairs
            int n = train.chosen().length;
            int[][] c = new int[n / 2][];
            int[][] r = new int[n / 2][];
            SplittableRandom pick = random.split();
            for (int i = 0; i < c.length; i++) {
                int j = pick.nextInt(n);
                c[i] = train.chosen()[j];
                r[i] = train.rejected()[j];
            }
            WeightAveragedReward.train(rm, c, r, RM_STEPS, RM_BATCH, 3e-3f, pick);
            members.add(rm);
        }
        RewardModel avg = GoldRewardHarness.rewardModel(seed + 300);
        List<List<io.github.zzuegg.brainiac.autograd.Var>> params = new ArrayList<>();
        for (RewardModel m : members) {
            params.add(m.parameters());
        }
        WeightAveragedReward.average(params, avg.parameters());
        return new WeightAveragedReward.Result(avg, members);
    }

    private static void copy(RewardModel from, RewardModel to) {
        var a = from.parameters();
        var b = to.parameters();
        for (int i = 0; i < a.size(); i++) {
            b.get(i).assign(a.get(i).value());
        }
    }

    /**
     * Runs 9.2.1 for one seed.
     *
     * @param pairs preference pairs the proxies learn from
     * @param rlSteps GRPO steps
     * @param checkpoints RL checkpoints
     * @param seed the seed
     * @return the curves
     */
    public static Result run(int pairs, int rlSteps, int checkpoints, long seed) {
        GoldRewardHarness.Split split = GoldRewardHarness.split(seed);
        WindowLm sft = GoldRewardHarness.supervised(split.train(), seed);
        Proxies px = proxies(split, sft, pairs, true, seed);
        Map<String, RewardEnsemble> rewards = new LinkedHashMap<>();
        rewards.put("single RM", px.single());
        rewards.put("ensemble mean", px.ensemble());
        rewards.put("ensemble worst", px.ensemble().withMode(RewardEnsemble.Mode.WORST, 0));
        rewards.put("ensemble mean-std", px.ensemble().withMode(RewardEnsemble.Mode.UNCERTAINTY, 1));
        rewards.put("WARM", px.warm());
        // best-of-n pools on held-out prompts
        SplittableRandom random = new SplittableRandom(seed + 21);
        int pool = BON[BON.length - 1];
        int[][] ps = GoldRewardHarness.repeat(split.test(), pool);
        int[][] outs = sft.sample(ps, GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS, 1.0, random);
        double[] gold = new double[outs.length];
        for (int i = 0; i < outs.length; i++) {
            gold[i] = GoldRewardHarness.GOLD.score(ps[i], outs[i]);
        }
        Map<String, Curves> curves = new LinkedHashMap<>();
        // the ensemble members score the pool once; each combination reuses their scores
        float[][] members = px.ensemble().memberScores(Arrays.asList(ps), Arrays.asList(outs));
        Map<String, float[]> pooled = new LinkedHashMap<>();
        pooled.put("single RM", members[0]);
        pooled.put("ensemble mean", combineAll(members, RewardEnsemble.Mode.MEAN, 0));
        pooled.put("ensemble worst", combineAll(members, RewardEnsemble.Mode.WORST, 0));
        pooled.put("ensemble mean-std", combineAll(members, RewardEnsemble.Mode.UNCERTAINTY, 1));
        pooled.put("WARM", px.warm().scoreAll(Arrays.asList(ps), Arrays.asList(outs)));
        for (Map.Entry<String, RewardEnsemble> e : rewards.entrySet()) {
            float[] proxy = pooled.get(e.getKey());
            List<Point> bon = bestOfN(gold, proxy, split.test().size(), pool);
            List<Point> rl = rl(sft, split, e.getValue(), 0, rlSteps, checkpoints, seed);
            curves.put(e.getKey(), new Curves(bon, rl));
        }
        curves.put("gold (oracle)", new Curves(bestOfN(gold, toFloat(gold), split.test().size(), pool),
                rl(sft, split, null, 0, rlSteps, checkpoints, seed)));
        return new Result(curves, px.accuracy(), px.warmMembers());
    }

    /**
     * The baselines of 9.2.1: GRPO against the single proxy without and with a KL penalty, and
     * {@link PpoLmTrainer} with a KL penalty.
     *
     * @param pairs preference pairs the proxy learns from
     * @param rlSteps RL steps
     * @param checkpoints RL checkpoints
     * @param seed the seed (the same as {@link #run}'s gives the same proxy)
     * @return run name to its RL curve, the start first
     */
    public static Map<String, List<Point>> klSweep(int pairs, int rlSteps, int checkpoints, long seed) {
        GoldRewardHarness.Split split = GoldRewardHarness.split(seed);
        WindowLm sft = GoldRewardHarness.supervised(split.train(), seed);
        Proxies px = proxies(split, sft, pairs, false, seed);
        Map<String, List<Point>> out = new LinkedHashMap<>();
        for (float beta : new float[] {0, 0.05f, 0.2f}) {
            out.put("single RM, KL " + beta, rl(sft, split, px.single(), beta, rlSteps, checkpoints, seed));
        }
        out.put("single RM, PpoLmTrainer KL 0.05", ppo(sft, split, px.single(), rlSteps, checkpoints, seed));
        return out;
    }

    private static float[] combineAll(float[][] members, RewardEnsemble.Mode mode, float lambda) {
        float[] out = new float[members[0].length];
        float[] column = new float[members.length];
        for (int i = 0; i < out.length; i++) {
            for (int m = 0; m < members.length; m++) {
                column[m] = members[m][i];
            }
            out[i] = RewardEnsemble.combine(column, mode, lambda);
        }
        return out;
    }

    private static float[] toFloat(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return f;
    }

    private static List<Point> bestOfN(double[] gold, float[] proxy, int prompts, int pool) {
        List<Point> out = new ArrayList<>();
        for (int n : BON) {
            double g = 0;
            double p = 0;
            for (int q = 0; q < prompts; q++) {
                int best = q * pool;
                for (int i = q * pool; i < q * pool + n; i++) {
                    if (proxy[i] > proxy[best]) {
                        best = i;
                    }
                }
                g += gold[best];
                p += proxy[best];
            }
            out.add(new Point(Math.log(n) - (n - 1.0) / n, g / prompts, p / prompts));
        }
        return out;
    }

    /** Batch-scores one GRPO step's groups with the proxy (or the gold reward when null). */
    static GrpoTrainer.Stats grpoStep(GrpoTrainer grpo, WindowLm lm, List<int[]> prompts, RewardEnsemble proxy,
            int promptsPerStep, int group, SplittableRandom random) {
        List<int[]> gp = new ArrayList<>();
        int[][] ps = new int[promptsPerStep * group][];
        for (int p = 0; p < promptsPerStep; p++) {
            int[] prompt = prompts.get(random.nextInt(prompts.size()));
            gp.add(prompt);
            for (int g = 0; g < group; g++) {
                ps[p * group + g] = prompt;
            }
        }
        int[][] outs = lm.sample(ps, GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS, 1.0, random);
        float[] r = proxy == null ? null : proxy.scoreAll(Arrays.asList(ps), Arrays.asList(outs));
        List<int[][]> groups = new ArrayList<>();
        List<float[]> rewards = new ArrayList<>();
        for (int p = 0; p < promptsPerStep; p++) {
            groups.add(Arrays.copyOfRange(outs, p * group, (p + 1) * group));
            float[] gr = new float[group];
            for (int g = 0; g < group; g++) {
                int i = p * group + g;
                gr[g] = r == null ? (float) GoldRewardHarness.GOLD.score(ps[i], outs[i]) : r[i];
            }
            rewards.add(gr);
        }
        return grpo.update(gp, groups, rewards);
    }

    private static Point measure(WindowLm lm, WindowLm sft, GoldRewardHarness.Split split, RewardEnsemble proxy,
            long seed) {
        double kl = GoldRewardHarness.kl(lm, sft, split.test(), 8, seed + 5);
        double gold = GoldRewardHarness.gold(lm, split.test(), 8, seed + 6);
        double p = 0;
        if (proxy != null) {
            SplittableRandom r = new SplittableRandom(seed + 6);
            int[][] ps = GoldRewardHarness.repeat(split.test(), 8);
            int[][] outs = lm.sample(ps, GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS, 1.0, r);
            for (float x : proxy.scoreAll(Arrays.asList(ps), Arrays.asList(outs))) {
                p += x / outs.length;
            }
        }
        return new Point(kl, gold, p);
    }

    private static List<Point> rl(WindowLm sft, GoldRewardHarness.Split split, RewardEnsemble proxy, float beta,
            int steps, int checkpoints, long seed) {
        WindowLm lm = sft.copy();
        GrpoTrainer grpo = GrpoTrainer.of(lm).optimizer(Optimizer.adam(5e-3f)).innerEpochs(2).klCoefficient(beta)
                .completions(GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS);
        SplittableRandom random = new SplittableRandom(seed + 31);
        List<Point> out = new ArrayList<>();
        out.add(measure(lm, sft, split, proxy, seed));
        for (int s = 1; s <= steps; s++) {
            grpoStep(grpo, lm, split.train(), proxy, 16, 8, random);
            if (s % Math.max(1, steps / checkpoints) == 0) {
                out.add(measure(lm, sft, split, proxy, seed));
            }
        }
        return out;
    }

    private static List<Point> ppo(WindowLm sft, GoldRewardHarness.Split split, RewardEnsemble proxy, int steps,
            int checkpoints, long seed) {
        WindowLm lm = sft.copy();
        PpoLmTrainer ppo = PpoLmTrainer.of(lm).optimizers(Optimizer.adam(5e-3f), Optimizer.adam(5e-3f)).batch(128)
                .klCoefficient(0.05f).completions(GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS);
        SplittableRandom random = new SplittableRandom(seed + 41);
        List<Point> out = new ArrayList<>();
        out.add(measure(lm, sft, split, proxy, seed));
        for (int s = 1; s <= steps; s++) {
            ppo.step(r -> split.train().get(r.nextInt(split.train().size())), proxy, random);
            if (s % Math.max(1, steps / checkpoints) == 0) {
                out.add(measure(lm, sft, split, proxy, seed));
            }
        }
        return out;
    }

    /**
     * One 9.2.2 method.
     *
     * @param samplesPerQuery responses sampled at inference per query
     * @param gold the mean gold reward on held-out prompts
     * @param kl the KL from the reference ({@code log N - (N - 1) / N} for best-of-N)
     * @param trainingSamples responses sampled during training
     */
    public record BondRow(int samplesPerQuery, double gold, double kl, long trainingSamples) {
    }

    /**
     * Runs 9.2.2 for one seed: everything distils or selects best-of-N under the single proxy.
     *
     * @param n the N of best-of-N
     * @param pairs preference pairs of the proxy
     * @param steps training steps per method
     * @param seed the seed
     * @return method name to its row
     */
    public static Map<String, BondRow> bond(int n, int pairs, int steps, long seed) {
        GoldRewardHarness.Split split = GoldRewardHarness.split(seed);
        WindowLm sft = GoldRewardHarness.supervised(split.train(), seed);
        RewardEnsemble proxy = proxies(split, sft, pairs, false, seed).single();
        Map<String, BondRow> out = new LinkedHashMap<>();
        out.put("reference", new BondRow(1, GoldRewardHarness.gold(sft, split.test(), 8, seed + 6), 0, 0));
        // best-of-N at inference
        SplittableRandom random = new SplittableRandom(seed + 51);
        int[][] ps = GoldRewardHarness.repeat(split.test(), n);
        int[][] outs = sft.sample(ps, GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS, 1.0, random);
        float[] pr = proxy.scoreAll(Arrays.asList(ps), Arrays.asList(outs));
        double g = 0;
        for (int q = 0; q < split.test().size(); q++) {
            int best = q * n;
            for (int i = q * n; i < (q + 1) * n; i++) {
                best = pr[i] > pr[best] ? i : best;
            }
            g += GoldRewardHarness.GOLD.score(ps[best], outs[best]);
        }
        out.put("best-of-N inference", new BondRow(n, g / split.test().size(), Math.log(n) - (n - 1.0) / n, 0));
        // BOND and J-BOND
        for (boolean iterated : new boolean[] {false, true}) {
            WindowLm lm = sft.copy();
            BestOfNDistiller bond = BestOfNDistiller.of(lm, sft).bestOf(n).jeffreys(0.5f).referenceSamples(16)
                    .policySamples(8).completions(GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS)
                    .optimizer(Optimizer.adam(5e-3f));
            if (iterated) {
                bond.iterated(0.1f);
            }
            SplittableRandom r = new SplittableRandom(seed + 61);
            for (int s = 0; s < steps; s++) {
                bond.step(x -> split.train().get(x.nextInt(split.train().size())), 8, proxy, r);
            }
            out.put(iterated ? "J-BOND (iterated best-of-2)" : "BOND", row(lm, sft, split, seed, bond.samplesDrawn()));
        }
        // rejection-sampling fine-tuning on best-of-N outputs, and DPO on best/worst pairs
        WindowLm rft = sft.copy();
        WindowLm dpoLm = sft.copy();
        DpoTrainer dpo = DpoTrainer.of(dpoLm).beta(0.5f).optimizer(Optimizer.adam(5e-3f));
        Optimizer rftOpt = Optimizer.adam(5e-3f);
        SplittableRandom r = new SplittableRandom(seed + 71);
        long drawn = 0;
        int perStep = Math.max(1, 64 / n);
        for (int s = 0; s < steps; s++) {
            List<int[]> bp = new ArrayList<>();
            List<int[]> bc = new ArrayList<>();
            List<DpoTrainer.Preference> prefs = new ArrayList<>();
            List<int[]> chosenPrompts = new ArrayList<>();
            for (int q = 0; q < perStep; q++) {
                chosenPrompts.add(split.train().get(r.nextInt(split.train().size())));
            }
            int[][] qp = GoldRewardHarness.repeat(chosenPrompts, n);
            int[][] qo = sft.sample(qp, GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS, 1.0, r);
            float[] qr = proxy.scoreAll(Arrays.asList(qp), Arrays.asList(qo));
            drawn += qo.length;
            for (int q = 0; q < perStep; q++) {
                int best = q * n;
                int worst = q * n;
                for (int i = q * n; i < (q + 1) * n; i++) {
                    best = qr[i] > qr[best] ? i : best;
                    worst = qr[i] < qr[worst] ? i : worst;
                }
                if (qo[best].length > 0) {
                    bp.add(qp[best]);
                    bc.add(qo[best]);
                }
                if (qr[best] > qr[worst] && qo[best].length > 0 && qo[worst].length > 0) {
                    prefs.add(new DpoTrainer.Preference(qp[best], qo[best], qo[worst]));
                }
            }
            if (!bc.isEmpty()) {
                rft.supervisedStep(bp, bc, rftOpt);
            }
            if (!prefs.isEmpty()) {
                dpo.step(prefs);
            }
        }
        out.put("rejection-sampling FT", row(rft, sft, split, seed, drawn));
        out.put("DPO best/worst", row(dpoLm, sft, split, seed, drawn));
        // GRPO against the proxy
        WindowLm rl = sft.copy();
        GrpoTrainer grpo = GrpoTrainer.of(rl).optimizer(Optimizer.adam(5e-3f)).klCoefficient(0.05f)
                .completions(GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS);
        SplittableRandom gr = new SplittableRandom(seed + 81);
        for (int s = 0; s < steps; s++) {
            grpoStep(grpo, rl, split.train(), proxy, 8, 8, gr);
        }
        out.put("GRPO (PPO-style RL)", row(rl, sft, split, seed, (long) steps * 64));
        return out;
    }

    private static BondRow row(WindowLm lm, WindowLm sft, GoldRewardHarness.Split split, long seed, long drawn) {
        return new BondRow(1, GoldRewardHarness.gold(lm, split.test(), 8, seed + 6),
                GoldRewardHarness.kl(lm, sft, split.test(), 8, seed + 5), drawn);
    }
}
