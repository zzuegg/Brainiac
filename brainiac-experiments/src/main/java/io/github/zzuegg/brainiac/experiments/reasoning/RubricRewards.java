package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.experimental.llm.GenerativeVerifier;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.RewardModel;
import io.github.zzuegg.brainiac.experimental.llm.RubricReward;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Rubrics as rewards (<a href="https://arxiv.org/abs/2507.17746">arXiv:2507.17746</a>) on a
 * toy task with no single right answer. The prompt {@code [abc]=} names three digits; a good
 * response mentions {@code a} (weight 3) and {@code b} (2), avoids the pitfall {@code c} (2), is
 * concise, at most two digits (1), and is well formed, digits then {@code #} (1). The hidden
 * preference is the weighted share of these, a different checklist for every prompt.
 *
 * <p>Three rewards drive GRPO from the same supervised start: {@link RubricReward}, whose
 * {@link GenerativeVerifier} judge is a general criterion checker trained on arbitrary strings
 * (as an off-the-shelf judge model is, never on this task's preferences); a Bradley-Terry
 * {@link RewardModel} on preference pairs (the scalar reward model); and a
 * {@link GenerativeVerifier} trained to say whether a response is good overall (the Likert-style
 * judge, {@code P(yes)} as the reward). The two holistic rewards learn from the same responses
 * of the supervised policy on training prompts, labelled by the hidden preference. The
 * policies are scored on held-out prompts by the true rubric satisfaction and by their win rate
 * against the supervised policy under the hidden preference.
 */
public final class RubricRewards {
    private static final CharacterTokenizer TOK = new CharacterTokenizer(ArithmeticTask.CHARACTERS);
    private static final int OPEN = TOK.encode("[")[0];
    private static final int CLOSE = TOK.encode("]")[0];
    private static final int EQUALS = TOK.encode("=")[0];
    private static final int EOS = TOK.encode("#")[0];
    private static final int PLUS = TOK.encode("+")[0];
    private static final int MINUS = TOK.encode("-")[0];
    private static final int SHORT = TOK.encode(">")[0];
    private static final int QUERY = TOK.encode("?")[0];
    private static final int YES = TOK.encode("1")[0];
    private static final int NO = TOK.encode("0")[0];
    private static final int PAD = TOK.encode("*")[0];
    private static final int DIGITS = 6;
    private static final int MAX_TOKENS = 5;
    private static final float[] WEIGHTS = {3, 2, 2, 1, 1};
    private static final int BATCH = 256;
    private static final int PAIRS = 128;
    private static final int GENERIC = 4000;
    private static final int JUDGE_STEPS = 800;
    private static final int HOLISTIC_STEPS = 300;
    private static final int RM_STEPS = 40;

    /**
     * One trained policy.
     *
     * @param satisfaction mean true rubric satisfaction of greedy responses on held-out prompts
     * @param winRate share of held-out prompts where it beats the supervised policy (ties half)
     */
    public record Arm(double satisfaction, double winRate) {
    }

    private RubricRewards() {
    }

    /**
     * The rubric of a prompt: contains a, contains b, avoids c, concise, well formed.
     *
     * @param prompt {@code [abc]=}
     * @return the items
     */
    public static List<RubricReward.Item> rubric(int[] prompt) {
        return List.of(new RubricReward.Item(new int[] {PLUS, prompt[1]}, WEIGHTS[0]),
                new RubricReward.Item(new int[] {PLUS, prompt[2]}, WEIGHTS[1]),
                new RubricReward.Item(new int[] {MINUS, prompt[3]}, WEIGHTS[2]),
                new RubricReward.Item(new int[] {SHORT, SHORT}, WEIGHTS[3]),
                new RubricReward.Item(new int[] {EOS, EOS}, WEIGHTS[4]));
    }

    /**
     * Which criteria a response meets: the ground truth.
     *
     * @param prompt the prompt
     * @param response the response
     * @return one flag per rubric item
     */
    public static boolean[] satisfied(int[] prompt, int[] response) {
        int digits = 0;
        boolean wellFormed = response.length >= 2 && response[response.length - 1] == EOS;
        for (int i = 0; i < response.length; i++) {
            boolean isDigit = response[i] < 10;
            digits += isDigit ? 1 : 0;
            if (i < response.length - 1 && !isDigit) {
                wellFormed = false;
            }
        }
        return new boolean[] {contains(response, prompt[1]), contains(response, prompt[2]),
            !contains(response, prompt[3]), digits <= 2, wellFormed};
    }

    /**
     * The hidden preference: the weighted share of criteria met.
     *
     * @param prompt the prompt
     * @param response the response
     * @return a score in [0, 1]
     */
    public static double trueScore(int[] prompt, int[] response) {
        boolean[] s = satisfied(prompt, response);
        double met = 0;
        double total = 0;
        for (int i = 0; i < s.length; i++) {
            total += WEIGHTS[i];
            met += s[i] ? WEIGHTS[i] : 0;
        }
        return met / total;
    }

    private static boolean contains(int[] tokens, int token) {
        for (int t : tokens) {
            if (t == token) {
                return true;
            }
        }
        return false;
    }

    private static List<int[]> prompts() {
        List<int[]> out = new ArrayList<>();
        for (int a = 0; a < DIGITS; a++) {
            for (int b = 0; b < DIGITS; b++) {
                for (int c = 0; c < DIGITS; c++) {
                    if (a != b && b != c && a != c) {
                        out.add(new int[] {OPEN, a, b, c, CLOSE, EQUALS});
                    }
                }
            }
        }
        return out;
    }

    /** A policy that writes one to four digits, often from the prompt, then {@code #}. */
    private static WindowLm supervised(List<int[]> prompts, long seed) {
        WindowLm lm = WindowLm.of(TOK.size(), 8, 64, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int rep = 0; rep < 6; rep++) {
            for (int[] p : prompts) {
                int n = 1 + r.nextInt(4);
                int[] c = new int[n + 1];
                for (int i = 0; i < n; i++) {
                    c[i] = r.nextDouble() < 0.7 ? p[1 + r.nextInt(3)] : r.nextInt(DIGITS);
                }
                c[n] = EOS;
                ps.add(p);
                cs.add(c);
            }
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < 40; s++) {
            lm.supervisedStep(ps, cs, opt);
        }
        return lm;
    }

    /**
     * Runs the comparison.
     *
     * @param rlSteps GRPO steps per reward
     * @param seed the seed
     * @return satisfaction and win rate per reward, the supervised start first
     */
    public static Map<String, Arm> run(int rlSteps, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<int[]> all = prompts();
        List<int[]> train = new ArrayList<>();
        List<int[]> test = new ArrayList<>();
        for (int[] p : all) {
            (random.nextDouble() < 0.6 ? train : test).add(p);
        }
        WindowLm sft = supervised(train, seed);

        // the preference data the holistic rewards learn from: supervised samples on training
        // prompts, plus some arbitrary strings so they are not blind off the supervised manifold
        List<int[]> lp = new ArrayList<>();
        List<int[]> lr = new ArrayList<>();
        for (int rep = 0; rep < 3; rep++) {
            int[][] outs = sft.sample(train.toArray(int[][]::new), MAX_TOKENS, EOS, 1.0, random);
            lp.addAll(train);
            lr.addAll(Arrays.asList(outs));
        }
        for (int rep = 0; rep < 2; rep++) {
            for (int[] p : train) {
                lp.add(p);
                lr.add(randomResponse(p, random));
            }
        }

        // the rubric judge is a general criterion checker, as an off-the-shelf judge model is:
        // it learns to check "mentions d", "avoids d", "concise" and "well formed" on arbitrary
        // digit triples and strings, never on the task's preferences
        GenerativeVerifier rubricJudge = GenerativeVerifier.of(WindowLm.of(TOK.size(), 8, 128, seed + 2), QUERY, YES,
                NO);
        RubricReward rubricReward = RubricReward.of(rubricJudge, RubricRewards::rubric).layout(MAX_TOKENS, PAD);
        List<int[]> gp = new ArrayList<>();
        List<int[]> gr = new ArrayList<>();
        List<boolean[]> labels = new ArrayList<>();
        for (int i = 0; i < GENERIC; i++) {
            int[] p = {OPEN, random.nextInt(10), random.nextInt(10), random.nextInt(10), CLOSE, EQUALS};
            int[] c = randomResponse(p, random);
            gp.add(p);
            gr.add(c);
            labels.add(satisfied(p, c));
        }
        RubricReward.JudgeData jd = rubricReward.judgeData(gp, gr, labels);
        Optimizer jo = Optimizer.adam(3e-3f);
        for (int s = 0; s < JUDGE_STEPS; s++) {
            int[] idx = batch(jd.labels().length, random);
            rubricJudge.step(pick(jd.problems(), idx), pick(jd.solutions(), idx), pick(jd.labels(), idx), null, jo);
        }

        GenerativeVerifier likert = GenerativeVerifier.of(WindowLm.of(TOK.size(), 14, 64, seed + 3), QUERY, YES, NO);
        boolean[] good = new boolean[lp.size()];
        for (int i = 0; i < good.length; i++) {
            good[i] = trueScore(lp.get(i), lr.get(i)) >= 0.75;
        }
        Optimizer lo = Optimizer.adam(3e-3f);
        for (int s = 0; s < HOLISTIC_STEPS; s++) {
            int[] idx = batch(good.length, random);
            likert.step(pick(lp, idx), pick(lr, idx), pick(good, idx), null, lo);
        }

        RewardModel rm = new RewardModel(HookedDecoder.create(HookedDecoder.Settings.tiny(TOK.size()).withLayers(1), seed + 4));
        List<int[]> chosen = new ArrayList<>();
        List<int[]> rejected = new ArrayList<>();
        for (int i = 0; i < lp.size(); i++) {
            int j = (i + train.size()) % lp.size();
            double si = trueScore(lp.get(i), lr.get(i));
            double sj = trueScore(lp.get(j), lr.get(j));
            if (Arrays.equals(lp.get(i), lp.get(j)) && si != sj && chosen.size() < PAIRS) {
                chosen.add(sequence(lp.get(i), si > sj ? lr.get(i) : lr.get(j)));
                rejected.add(sequence(lp.get(i), si > sj ? lr.get(j) : lr.get(i)));
            }
        }
        Optimizer ro = Optimizer.adam(3e-3f);
        int[][] ch = chosen.toArray(int[][]::new);
        int[][] rj = rejected.toArray(int[][]::new);
        for (int s = 0; s < RM_STEPS; s++) {
            rm.zeroGrad();
            rm.loss(ch, rj).backward();
            ro.step(rm.parameters());
        }
        Map<String, Float> rmCache = new HashMap<>();
        GrpoTrainer.Reward rmReward = (p, c) -> rmCache.computeIfAbsent(Arrays.toString(p) + Arrays.toString(c),
                k -> rm.score(new int[][] {sequence(p, c)}).value().toArray()[0]);
        Map<String, Float> likertCache = new HashMap<>();
        GrpoTrainer.Reward likertReward = (p, c) -> likertCache.computeIfAbsent(
                Arrays.toString(p) + Arrays.toString(c), k -> (float) likert.score(p, c));

        Map<String, Arm> arms = new LinkedHashMap<>();
        arms.put("SFT", evaluate(sft, sft, test));
        Map<String, GrpoTrainer.Reward> rewards = new LinkedHashMap<>();
        rewards.put("rubric reward", rubricReward);
        rewards.put("scalar reward model", rmReward);
        rewards.put("Likert judge", likertReward);
        rewards.put("true rubric (oracle)", (p, c) -> (float) trueScore(p, c));
        for (Map.Entry<String, GrpoTrainer.Reward> e : rewards.entrySet()) {
            WindowLm lm = sft.copy();
            GrpoTrainer grpo = GrpoTrainer.of(lm).optimizer(Optimizer.adam(3e-3f)).groupSize(8).promptsPerStep(16)
                    .innerEpochs(2).klCoefficient(0.02f).completions(MAX_TOKENS, EOS);
            SplittableRandom r = new SplittableRandom(seed + 9);
            for (int s = 0; s < rlSteps; s++) {
                grpo.step(x -> train.get(x.nextInt(train.size())), e.getValue(), r);
            }
            arms.put(e.getKey(), evaluate(lm, sft, test));
        }
        return arms;
    }

    /** A string of one to five tokens, mostly digits, often from the prompt. */
    private static int[] randomResponse(int[] prompt, SplittableRandom random) {
        int[] c = new int[1 + random.nextInt(MAX_TOKENS)];
        for (int i = 0; i < c.length; i++) {
            double u = random.nextDouble();
            c[i] = u < 0.4 ? prompt[1 + random.nextInt(3)] : u < 0.8 ? random.nextInt(10) : random.nextInt(TOK.size());
        }
        if (random.nextBoolean()) {
            c[c.length - 1] = EOS;
        }
        return c;
    }

    private static int[] batch(int n, SplittableRandom random) {
        int[] idx = new int[Math.min(BATCH, n)];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = random.nextInt(n);
        }
        return idx;
    }

    private static List<int[]> pick(List<int[]> rows, int[] idx) {
        List<int[]> out = new ArrayList<>();
        for (int i : idx) {
            out.add(rows.get(i));
        }
        return out;
    }

    private static boolean[] pick(boolean[] rows, int[] idx) {
        boolean[] out = new boolean[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = rows[idx[i]];
        }
        return out;
    }

    private static int[] sequence(int[] prompt, int[] response) {
        int[] s = new int[prompt.length + MAX_TOKENS];
        Arrays.fill(s, PAD);
        System.arraycopy(prompt, 0, s, 0, prompt.length);
        System.arraycopy(response, 0, s, s.length - response.length, response.length);
        return s;
    }

    private static Arm evaluate(WindowLm lm, WindowLm sft, List<int[]> test) {
        double sat = 0;
        double wins = 0;
        for (int[] p : test) {
            double a = trueScore(p, lm.greedy(p, MAX_TOKENS, EOS));
            double b = trueScore(p, sft.greedy(p, MAX_TOKENS, EOS));
            sat += a;
            wins += a > b ? 1 : a == b ? 0.5 : 0;
        }
        return new Arm(sat / test.size(), wins / test.size());
    }
}
