package io.github.zzuegg.brainiac.experiments.alignment;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.experimental.align.ExploratoryPreferenceOptimizer;
import io.github.zzuegg.brainiac.experimental.align.LengthRegularizedDpo;
import io.github.zzuegg.brainiac.experimental.align.LinkDpoLosses;
import io.github.zzuegg.brainiac.experimental.llm.DpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.LengthControlledReward;
import io.github.zzuegg.brainiac.experimental.llm.SpinTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticTask;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 9, 9.2.5 to 9.2.7: a shoot-out of DPO variants on three failure modes of DPO.
 * <ul>
 *   <li>{@link #displacement} (9.2.5): likelihood displacement and over-optimization. On the
 *       gold-reward task of {@link GoldRewardHarness}, preference pairs whose two responses differ
 *       in a single token (where DPO lowers both likelihoods most) train DPO, IPO, DPOP, SimPO,
 *       chiPO and SquaredPO ({@link LinkDpoLosses}); the change of the held-out chosen
 *       log-likelihood, the gold win rate against the reference and the gold reward against the
 *       KL are recorded.</li>
 *   <li>{@link #lengthBias} (9.2.6): addition answers padded with filler tokens; the judge
 *       prefers the right answer but, with probability {@code bias}, the longer one instead.
 *       DPO, R-DPO ({@link LengthRegularizedDpo}), SimPO and GRPO with a
 *       {@link LengthControlledReward} are compared by length, accuracy, raw win rate and a
 *       length-controlled win rate (the judge's win probability at equal length, from a
 *       logistic regression on the length difference).</li>
 *   <li>{@link #exploration} (9.2.7): the best response has reference probability below
 *       {@code 1e-3}; XPO ({@link ExploratoryPreferenceOptimizer}) against online DPO, offline
 *       DPO on reference pairs and SPIN, by gold reward against preference queries.</li>
 * </ul>
 */
public final class PreferenceShootout {

    /** The objectives of 9.2.5. */
    public enum Objective {
        /** Plain DPO. */
        DPO,
        /** Identity preference optimization. */
        IPO,
        /** DPO-Positive. */
        DPOP,
        /** Simple preference optimization (reference-free, length-normalized). */
        SIMPO,
        /** chiPO: the chi-squared-plus-KL link. */
        CHI_PO,
        /** SquaredPO: the squared-log link. */
        SQUARED_PO
    }

    /**
     * One 9.2.5 checkpoint.
     *
     * @param step the optimizer step
     * @param chosenDrift change of the mean held-out {@code log pi(chosen)} since the start
     * @param winRate share of held-out prompts whose sample beats the reference's on gold (ties half)
     * @param gold the mean gold reward of samples
     * @param kl the KL from the reference
     */
    public record Checkpoint(int step, double chosenDrift, double winRate, double gold, double kl) {
    }

    private PreferenceShootout() {
    }

    /** Pairs whose responses differ in exactly one digit, labelled by the noisy annotator. */
    static List<DpoTrainer.Preference> similarPairs(WindowLm sft, List<int[]> prompts, int count,
            double temperature, SplittableRandom random) {
        List<DpoTrainer.Preference> out = new ArrayList<>();
        while (out.size() < count) {
            int[] p = prompts.get(random.nextInt(prompts.size()));
            int[] a = sft.sample(new int[][] {p}, GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS, 1.0,
                    random)[0];
            if (a.length < 2) {
                continue;
            }
            int[] b = a.clone();
            b[random.nextInt(a.length - 1)] = random.nextInt(6);
            double ga = GoldRewardHarness.GOLD.score(p, a);
            double gb = GoldRewardHarness.GOLD.score(p, b);
            if (ga == gb) {
                continue;
            }
            boolean aWins = random.nextDouble() < 1 / (1 + Math.exp(-(ga - gb) / temperature));
            out.add(new DpoTrainer.Preference(p, aWins ? a : b, aWins ? b : a));
        }
        return out;
    }

    private static double meanChosen(WindowLm lm, List<DpoTrainer.Preference> held) {
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (DpoTrainer.Preference p : held) {
            ps.add(p.prompt());
            cs.add(p.chosen());
        }
        return Var.noGrad(() -> lm.sequenceLogProbs(ps, cs)).mean().value().item();
    }

    private static double winRate(WindowLm lm, WindowLm ref, List<int[]> prompts, long seed) {
        int[][] ps = GoldRewardHarness.repeat(prompts, 4);
        int[][] a = lm.sample(ps, GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS, 1.0, new SplittableRandom(seed));
        int[][] b = ref.sample(ps, GoldRewardHarness.MAX_TOKENS, GoldRewardHarness.EOS, 1.0,
                new SplittableRandom(seed + 1));
        double w = 0;
        for (int i = 0; i < ps.length; i++) {
            double ga = GoldRewardHarness.GOLD.score(ps[i], a[i]);
            double gb = GoldRewardHarness.GOLD.score(ps[i], b[i]);
            w += ga > gb ? 1 : ga == gb ? 0.5 : 0;
        }
        return w / ps.length;
    }

    /**
     * 9.2.5 for one seed.
     *
     * @param pairs training pairs
     * @param steps optimizer steps
     * @param checkpoints evaluations (evenly spaced)
     * @param seed the seed
     * @return objective to its checkpoints, the start first
     */
    public static Map<Objective, List<Checkpoint>> displacement(int pairs, int steps, int checkpoints, long seed) {
        GoldRewardHarness.Split split = GoldRewardHarness.split(seed);
        WindowLm sft = GoldRewardHarness.supervised(split.train(), seed);
        SplittableRandom random = new SplittableRandom(seed + 3);
        List<DpoTrainer.Preference> train = similarPairs(sft, split.train(), pairs, 0.1, random);
        List<DpoTrainer.Preference> held = similarPairs(sft, split.test(), 128, 1e-6, random);
        double chosen0 = meanChosen(sft, held);
        Map<Objective, List<Checkpoint>> out = new LinkedHashMap<>();
        for (Objective o : Objective.values()) {
            WindowLm lm = sft.copy();
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom r = new SplittableRandom(seed + 5);
            List<Checkpoint> cps = new ArrayList<>();
            cps.add(new Checkpoint(0, 0, 0.5, GoldRewardHarness.gold(lm, split.test(), 4, seed + 6), 0));
            for (int s = 1; s <= steps; s++) {
                List<DpoTrainer.Preference> b = new ArrayList<>();
                for (int i = 0; i < 32; i++) {
                    b.add(train.get(r.nextInt(train.size())));
                }
                LinkDpoLosses.Batch lb = LinkDpoLosses.batch(lm, sft, b, o == Objective.SIMPO);
                float beta = 0.5f;
                Var loss = switch (o) {
                    case DPO -> LinkDpoLosses.loss(lb, LinkDpoLosses.Link.REVERSE_KL, beta);
                    case IPO -> PreferenceLosses.ipo(lb.policyChosen(), lb.policyRejected(), lb.refChosen(),
                            lb.refRejected(), 0.5f);
                    case DPOP -> PreferenceLosses.dpoPositive(lb.policyChosen(), lb.policyRejected(), lb.refChosen(),
                            lb.refRejected(), beta, 5);
                    case SIMPO -> PreferenceLosses.simpo(lb.policyChosen(), lb.policyRejected(), 2, 0.5f);
                    case CHI_PO -> LinkDpoLosses.chiSquared(lb.policyChosen(), lb.policyRejected(), lb.refChosen(),
                            lb.refRejected(), beta, 5);
                    case SQUARED_PO -> LinkDpoLosses.loss(lb, LinkDpoLosses.Link.SQUARED_LOG, beta);
                };
                loss.backward();
                opt.step(lm.parameters());
                if (s % Math.max(1, steps / checkpoints) == 0) {
                    cps.add(new Checkpoint(s, meanChosen(lm, held) - chosen0, winRate(lm, sft, split.test(), seed + 7),
                            GoldRewardHarness.gold(lm, split.test(), 4, seed + 6),
                            GoldRewardHarness.kl(lm, sft, split.test(), 4, seed + 8)));
                }
            }
            out.put(o, cps);
        }
        return out;
    }

    // ---------------------------------------------------------------- 9.2.6

    private static final ArithmeticTask ADD = ArithmeticTask.addition(4);
    private static final CharacterTokenizer TOK = ADD.tokenizer();
    private static final int FILL = TOK.encode("+")[0];
    private static final int MAX_FILL = 3;

    /**
     * One 9.2.6 method.
     *
     * @param length mean response length in tokens
     * @param accuracy share of right answers
     * @param winRate judge win rate against the reference's samples (ties half)
     * @param lengthControlled the judge's win probability at equal length (logistic regression)
     */
    public record LengthRow(double length, double accuracy, double winRate, double lengthControlled) {
    }

    /** The answer digits followed by filler. */
    static boolean correct(int[] prompt, int[] response) {
        int k = 0;
        while (k < response.length && response[k] < 10) {
            k++;
        }
        if (k == 0 || k > 2 || response[response.length - 1] != ADD.eos()) {
            return false;
        }
        int v = 0;
        for (int i = 0; i < k; i++) {
            v = 10 * v + response[i];
        }
        for (int i = k; i < response.length - 1; i++) {
            if (response[i] != FILL) {
                return false;
            }
        }
        return v == ADD.answer(prompt);
    }

    /** The biased judge: with probability {@code bias} the longer response, else the right one. */
    static boolean judge(int[] p, int[] a, int[] b, double bias, SplittableRandom r) {
        if (a.length != b.length && r.nextDouble() < bias) {
            return a.length > b.length;
        }
        boolean ca = correct(p, a);
        boolean cb = correct(p, b);
        return ca != cb ? ca : r.nextBoolean();
    }

    static WindowLm lengthStart(long seed) {
        WindowLm lm = WindowLm.of(TOK.size(), 6, 64, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < 80; s++) {
            List<int[]> ps = new ArrayList<>();
            List<int[]> cs = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                int[] p = ADD.samplePrompt(r);
                int ans = r.nextBoolean() ? ADD.answer(p) : r.nextInt(9);
                int fill = r.nextInt(MAX_FILL + 1);
                int[] c = new int[2 + fill];
                c[0] = ans;
                Arrays.fill(c, 1, 1 + fill, FILL);
                c[c.length - 1] = ADD.eos();
                ps.add(p);
                cs.add(c);
            }
            lm.supervisedStep(ps, cs, opt);
        }
        return lm;
    }

    /**
     * Logistic regression {@code P(win) = sigmoid(w0 + w1 dlen)} by Newton's method.
     *
     * @param lengthDiff the length difference of each comparison
     * @param wins 1 for a win, 0 a loss, 0.5 a tie
     * @return the fitted win probability at zero length difference
     */
    public static double lengthControlledWinRate(double[] lengthDiff, double[] wins) {
        double w0 = 0;
        double w1 = 0;
        for (int it = 0; it < 25; it++) {
            double g0 = 0;
            double g1 = 0;
            double h00 = 1e-6;
            double h01 = 0;
            double h11 = 1e-6;
            for (int i = 0; i < wins.length; i++) {
                double p = 1 / (1 + Math.exp(-(w0 + w1 * lengthDiff[i])));
                double e = wins[i] - p;
                g0 += e;
                g1 += e * lengthDiff[i];
                double v = p * (1 - p);
                h00 += v;
                h01 += v * lengthDiff[i];
                h11 += v * lengthDiff[i] * lengthDiff[i];
            }
            double det = h00 * h11 - h01 * h01;
            if (Math.abs(det) < 1e-12) {
                break;
            }
            w0 += (h11 * g0 - h01 * g1) / det;
            w1 += (h00 * g1 - h01 * g0) / det;
        }
        return 1 / (1 + Math.exp(-w0));
    }

    private static LengthRow lengthRow(WindowLm lm, WindowLm ref, double bias, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> all = ADD.allPrompts();
        int[][] ps = GoldRewardHarness.repeat(all, 4);
        int[][] a = lm.sample(ps, 2 + MAX_FILL + 2, ADD.eos(), 1.0, r);
        int[][] b = ref.sample(ps, 2 + MAX_FILL + 2, ADD.eos(), 1.0, r);
        double len = 0;
        double acc = 0;
        double win = 0;
        double[] dl = new double[ps.length];
        double[] w = new double[ps.length];
        for (int i = 0; i < ps.length; i++) {
            len += a[i].length;
            acc += correct(ps[i], a[i]) ? 1 : 0;
            w[i] = judge(ps[i], a[i], b[i], bias, r) ? 1 : 0;
            win += w[i];
            dl[i] = a[i].length - b[i].length;
        }
        return new LengthRow(len / ps.length, acc / ps.length, win / ps.length, lengthControlledWinRate(dl, w));
    }

    /**
     * 9.2.6 for one seed.
     *
     * @param bias the judge's length bias
     * @param pairs judge-labelled pairs of the reference's samples
     * @param steps optimizer steps
     * @param seed the seed
     * @return method name to its row; {@code "reference"} first
     */
    public static Map<String, LengthRow> lengthBias(double bias, int pairs, int steps, long seed) {
        WindowLm ref = lengthStart(seed);
        SplittableRandom r = new SplittableRandom(seed + 2);
        List<DpoTrainer.Preference> data = new ArrayList<>();
        while (data.size() < pairs) {
            int[] p = ADD.samplePrompt(r);
            int[][] o = ref.sample(new int[][] {p, p}, 2 + MAX_FILL + 2, ADD.eos(), 1.0, r);
            if (o[0].length == 0 || o[1].length == 0 || Arrays.equals(o[0], o[1])) {
                continue;
            }
            boolean first = judge(p, o[0], o[1], bias, r);
            data.add(new DpoTrainer.Preference(p, first ? o[0] : o[1], first ? o[1] : o[0]));
        }
        Map<String, LengthRow> out = new LinkedHashMap<>();
        out.put("reference", lengthRow(ref, ref, bias, seed + 9));
        for (String name : new String[] {"DPO", "R-DPO", "SimPO"}) {
            WindowLm lm = ref.copy();
            Optimizer opt = Optimizer.adam(3e-3f);
            LengthRegularizedDpo rdpo = LengthRegularizedDpo.of(lm).reference(ref).strength(0.5f, 0.5f).optimizer(opt);
            SplittableRandom br = new SplittableRandom(seed + 4);
            for (int s = 0; s < steps; s++) {
                List<DpoTrainer.Preference> b = new ArrayList<>();
                for (int i = 0; i < 32; i++) {
                    b.add(data.get(br.nextInt(data.size())));
                }
                if (name.equals("R-DPO")) {
                    rdpo.step(b);
                    continue;
                }
                LinkDpoLosses.Batch lb = LinkDpoLosses.batch(lm, ref, b, name.equals("SimPO"));
                Var loss = name.equals("SimPO") ? PreferenceLosses.simpo(lb.policyChosen(), lb.policyRejected(), 2, 0.5f)
                        : LinkDpoLosses.loss(lb, LinkDpoLosses.Link.REVERSE_KL, 0.5f);
                loss.backward();
                opt.step(lm.parameters());
            }
            out.put(name, lengthRow(lm, ref, bias, seed + 9));
        }
        WindowLm rl = ref.copy();
        GrpoTrainer grpo = GrpoTrainer.of(rl).optimizer(Optimizer.adam(3e-3f)).klCoefficient(0.05f)
                .completions(2 + MAX_FILL + 2, ADD.eos());
        LengthControlledReward lcr = new LengthControlledReward((p, c) -> correct(p, c) ? 1 : 0, p -> 2, c -> c.length,
                LengthControlledReward.Mode.MAX, 0.2f, 0.5f);
        SplittableRandom gr = new SplittableRandom(seed + 6);
        for (int s = 0; s < steps / 2; s++) {
            grpo.step(ADD::samplePrompt, lcr, gr);
        }
        out.put("GRPO + LengthControlledReward", lengthRow(rl, ref, bias, seed + 9));
        return out;
    }

    // ---------------------------------------------------------------- 9.2.7

    private static final int[] XPO_PROMPT = TOK.encode("?=");
    private static final int SYMBOLS = 5;
    private static final int LENGTH = 3;
    private static final int[] TARGET = {4, 2, 4};

    /**
     * The gold reward of 9.2.7: 1 for the rare target, 0.3 times the share of the reference's
     * mode symbol {@code 1} otherwise.
     *
     * @param response a response
     * @return its gold reward
     */
    public static double explorationGold(int[] response) {
        if (response.length != LENGTH + 1 || response[LENGTH] != ADD.eos()) {
            return 0;
        }
        int[] body = Arrays.copyOf(response, LENGTH);
        if (Arrays.equals(body, TARGET)) {
            return 1;
        }
        double ones = 0;
        for (int t : body) {
            ones += t == 1 ? 1 : 0;
        }
        return 0.3 * ones / LENGTH;
    }

    /** A reference that writes {@code 111#} 90% of the time and random symbols otherwise. */
    static WindowLm explorationStart(long seed) {
        WindowLm lm = WindowLm.of(TOK.size(), 4, 32, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < 60; s++) {
            List<int[]> ps = new ArrayList<>();
            List<int[]> cs = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                int[] c = new int[LENGTH + 1];
                for (int t = 0; t < LENGTH; t++) {
                    c[t] = r.nextDouble() < 0.9 ? 1 : r.nextInt(SYMBOLS);
                }
                c[LENGTH] = ADD.eos();
                ps.add(XPO_PROMPT);
                cs.add(c);
            }
            lm.supervisedStep(ps, cs, opt);
        }
        return lm;
    }

    /**
     * @param lm a policy
     * @param seed the sampling seed
     * @return its mean gold reward over 256 samples
     */
    static double explorationScore(WindowLm lm, long seed) {
        int[][] ps = new int[256][];
        Arrays.fill(ps, XPO_PROMPT);
        int[][] o = lm.sample(ps, LENGTH + 1, ADD.eos(), 1.0, new SplittableRandom(seed));
        double g = 0;
        for (int[] c : o) {
            g += explorationGold(c);
        }
        return g / o.length;
    }

    /**
     * @param lm a policy
     * @return its probability of the target response
     */
    public static double targetProbability(WindowLm lm) {
        int[] c = Arrays.copyOf(TARGET, LENGTH + 1);
        c[LENGTH] = ADD.eos();
        return Math.exp(lm.sequenceLogProb(XPO_PROMPT, c));
    }

    /**
     * One 9.2.7 method.
     *
     * @param gold the mean gold reward at each checkpoint
     * @param target the probability of the best response at the end
     * @param queries preference queries spent
     */
    public record ExploreRow(double[] gold, double target, long queries) {
    }

    /**
     * 9.2.7 for one seed: every method gets the same budget of preference queries.
     *
     * @param rounds online rounds
     * @param pairsPerRound queries per round
     * @param checkpoints evaluations
     * @param seed the seed
     * @return method name to its row; {@code "reference"} first
     */
    public static Map<String, ExploreRow> exploration(int rounds, int pairsPerRound, int checkpoints, long seed) {
        WindowLm ref = explorationStart(seed);
        ExploratoryPreferenceOptimizer.Oracle oracle = (p, a, b, r) -> {
            double ga = explorationGold(a);
            double gb = explorationGold(b);
            return ga != gb ? ga > gb : r.nextBoolean();
        };
        Map<String, ExploreRow> out = new LinkedHashMap<>();
        out.put("reference", new ExploreRow(new double[] {explorationScore(ref, seed + 9)}, targetProbability(ref), 0));
        for (String name : new String[] {"XPO", "online DPO"}) {
            WindowLm lm = ref.copy();
            ExploratoryPreferenceOptimizer x = ExploratoryPreferenceOptimizer.of(lm).completions(LENGTH + 1, ADD.eos())
                    .optimizer(Optimizer.adam(5e-3f)).fitting(4, 32);
            if (name.equals("online DPO")) {
                x.onlineDpo();
            } else {
                x.strength(0.3f, 0.5f);
            }
            double[] g = new double[checkpoints];
            int next = 0;
            SplittableRandom r = new SplittableRandom(seed + 3);
            for (int k = 1; k <= rounds; k++) {
                x.round(q -> XPO_PROMPT, pairsPerRound, oracle, r);
                if (next < checkpoints && k == rounds * (next + 1) / checkpoints) {
                    g[next++] = explorationScore(lm, seed + 9);
                }
            }
            out.put(name, new ExploreRow(g, targetProbability(lm), x.queries()));
        }
        // offline DPO on the same number of labelled reference pairs
        WindowLm off = ref.copy();
        SplittableRandom r = new SplittableRandom(seed + 4);
        List<DpoTrainer.Preference> data = new ArrayList<>();
        int[] best = null;
        double bestGold = -1;
        int budget = rounds * pairsPerRound;
        for (int i = 0; i < budget; i++) {
            int[][] o = ref.sample(new int[][] {XPO_PROMPT, XPO_PROMPT}, LENGTH + 1, ADD.eos(), 1.0, r);
            if (o[0].length == 0 || o[1].length == 0 || Arrays.equals(o[0], o[1])) {
                continue;
            }
            boolean first = oracle.prefersFirst(XPO_PROMPT, o[0], o[1], r);
            data.add(new DpoTrainer.Preference(XPO_PROMPT, first ? o[0] : o[1], first ? o[1] : o[0]));
            for (int[] c : o) {
                if (explorationGold(c) > bestGold) {
                    bestGold = explorationGold(c);
                    best = c;
                }
            }
        }
        DpoTrainer dpo = DpoTrainer.of(off).beta(0.5f).optimizer(Optimizer.adam(5e-3f));
        for (int s = 0; s < rounds * 4; s++) {
            List<DpoTrainer.Preference> b = new ArrayList<>();
            for (int i = 0; i < 32 && !data.isEmpty(); i++) {
                b.add(data.get(r.nextInt(data.size())));
            }
            if (!b.isEmpty()) {
                dpo.step(b);
            }
        }
        out.put("offline DPO", new ExploreRow(new double[] {explorationScore(off, seed + 9)}, targetProbability(off),
                data.size()));
        // SPIN against the best response the reference pairs revealed
        WindowLm spinLm = ref.copy();
        SpinTrainer spin = SpinTrainer.of(spinLm).beta(0.5f).stepsPerIteration(4).generation(LENGTH + 1, ADD.eos(), 1.0)
                .optimizer(Optimizer.adam(5e-3f));
        List<int[]> ps = new ArrayList<>();
        List<int[]> as = new ArrayList<>();
        for (int i = 0; i < 32; i++) {
            ps.add(XPO_PROMPT);
            as.add(best);
        }
        for (int k = 0; k < rounds && best != null; k++) {
            spin.iterate(ps, as, r);
        }
        out.put("SPIN", new ExploreRow(new double[] {explorationScore(spinLm, seed + 9)}, targetProbability(spinLm),
                data.size()));
        return out;
    }
}
