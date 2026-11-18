package io.github.zzuegg.brainiac.experiments.alignment;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.experimental.align.GoldScorer;
import io.github.zzuegg.brainiac.experimental.align.RewardEnsemble;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.RewardModel;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticTask;
import io.github.zzuegg.brainiac.experiments.reasoning.RubricRewards;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The shared gold-reward set-up of the alignment experiments (wave 9, 9.2.1, 9.2.2, 9.2.5): the
 * rubric task of {@link RubricRewards}, whose hidden rubric satisfaction
 * ({@link RubricRewards#trueScore}) is the gold reward, a supervised {@link WindowLm} that
 * writes plausible but mediocre responses, preference pairs of its samples labelled by a noisy
 * Bradley-Terry annotator that follows the gold reward, and Bradley-Terry proxy reward models
 * ({@link RewardModel}) trained on them. Because the gold reward is known, every policy can be
 * scored by the truth while it is optimized against a proxy.
 */
public final class GoldRewardHarness {

    /** The tokenizer of the arithmetic family. */
    public static final CharacterTokenizer TOK = new CharacterTokenizer(ArithmeticTask.CHARACTERS);
    /** The end token. */
    public static final int EOS = TOK.encode("#")[0];
    /** The padding token of reward-model inputs. */
    public static final int PAD = TOK.encode("*")[0];
    /** The most tokens per response. */
    public static final int MAX_TOKENS = 5;
    /** The gold reward. */
    public static final GoldScorer GOLD = RubricRewards::trueScore;
    private static final int DIGITS = 6;

    /**
     * The prompts, split.
     *
     * @param train prompts to train on
     * @param test held-out prompts
     */
    public record Split(List<int[]> train, List<int[]> test) {
    }

    /**
     * Preference pairs as reward-model inputs.
     *
     * @param chosen the preferred sequences
     * @param rejected the others
     * @param prompts the prompt of each pair
     * @param chosenResponses the preferred responses
     * @param rejectedResponses the others
     */
    public record Pairs(int[][] chosen, int[][] rejected, List<int[]> prompts, List<int[]> chosenResponses,
            List<int[]> rejectedResponses) {
    }

    private GoldRewardHarness() {
    }

    /**
     * @param seed the split seed
     * @return every {@code [abc]=} prompt with distinct digits below 6, 60% to train
     */
    public static Split split(long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<int[]> train = new ArrayList<>();
        List<int[]> test = new ArrayList<>();
        int open = TOK.encode("[")[0];
        int close = TOK.encode("]")[0];
        int eq = TOK.encode("=")[0];
        for (int a = 0; a < DIGITS; a++) {
            for (int b = 0; b < DIGITS; b++) {
                for (int c = 0; c < DIGITS; c++) {
                    if (a != b && b != c && a != c) {
                        (random.nextDouble() < 0.6 ? train : test).add(new int[] {open, a, b, c, close, eq});
                    }
                }
            }
        }
        return new Split(train, test);
    }

    /**
     * A supervised policy that writes one to four digits, often from the prompt, then the end
     * token: a reference with a spread of gold rewards.
     *
     * @param prompts the prompts it is trained on
     * @param seed the seed
     * @return the policy
     */
    public static WindowLm supervised(List<int[]> prompts, long seed) {
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
     * Lays a prompt and response out as a reward-model input: the prompt, padding, then the
     * response right-aligned, so the last token is always the response's.
     *
     * @param prompt the prompt
     * @param response the response (at most {@link #MAX_TOKENS})
     * @return the sequence
     */
    public static int[] layout(int[] prompt, int[] response) {
        int[] s = new int[prompt.length + MAX_TOKENS];
        Arrays.fill(s, PAD);
        System.arraycopy(prompt, 0, s, 0, prompt.length);
        int n = Math.min(response.length, MAX_TOKENS);
        System.arraycopy(response, response.length - n, s, s.length - n, n);
        return s;
    }

    /**
     * Preference pairs: two samples of the policy per prompt, labelled by a Bradley-Terry
     * annotator, {@code P(a > b) = sigmoid((gold(a) - gold(b)) / temperature)}; pairs of equal
     * gold reward are skipped.
     *
     * @param policy the policy sampled
     * @param prompts the prompts drawn from
     * @param count the pairs
     * @param temperature the annotator noise
     * @param random the random source
     * @return the pairs
     */
    public static Pairs pairs(WindowLm policy, List<int[]> prompts, int count, double temperature,
            SplittableRandom random) {
        List<int[]> ch = new ArrayList<>();
        List<int[]> rj = new ArrayList<>();
        List<int[]> ps = new ArrayList<>();
        List<int[]> cr = new ArrayList<>();
        List<int[]> rr = new ArrayList<>();
        while (ch.size() < count) {
            int[] p = prompts.get(random.nextInt(prompts.size()));
            int[][] out = policy.sample(new int[][] {p, p}, MAX_TOKENS, EOS, 1.0, random);
            double ga = GOLD.score(p, out[0]);
            double gb = GOLD.score(p, out[1]);
            if (ga == gb || out[0].length == 0 || out[1].length == 0) {
                continue;
            }
            boolean aWins = random.nextDouble() < 1 / (1 + Math.exp(-(ga - gb) / temperature));
            int[] w = aWins ? out[0] : out[1];
            int[] l = aWins ? out[1] : out[0];
            ch.add(layout(p, w));
            rj.add(layout(p, l));
            ps.add(p);
            cr.add(w);
            rr.add(l);
        }
        return new Pairs(ch.toArray(int[][]::new), rj.toArray(int[][]::new), ps, cr, rr);
    }

    /**
     * @param seed the initialization seed
     * @return an untrained one-block reward model
     */
    public static RewardModel rewardModel(long seed) {
        return new RewardModel(backbone(seed));
    }

    /**
     * @param seed the initialization seed
     * @return a one-block decoder for reward models
     */
    public static HookedDecoder backbone(long seed) {
        return HookedDecoder.create(HookedDecoder.Settings.tiny(TOK.size()).withLayers(1).withDim(16, 2), seed);
    }

    /**
     * @param model a reward model
     * @return it as an ensemble member
     */
    public static RewardEnsemble.BatchScorer member(RewardModel model) {
        return RewardEnsemble.of(model, GoldRewardHarness::layout);
    }

    /**
     * Mean gold reward of the policy's samples.
     *
     * @param policy the policy
     * @param prompts the prompts
     * @param samples samples per prompt
     * @param seed the sampling seed
     * @return the mean gold reward
     */
    public static double gold(WindowLm policy, List<int[]> prompts, int samples, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        double sum = 0;
        int[][] ps = repeat(prompts, samples);
        int[][] out = policy.sample(ps, MAX_TOKENS, EOS, 1.0, random);
        for (int i = 0; i < out.length; i++) {
            sum += GOLD.score(ps[i], out[i]);
        }
        return sum / out.length;
    }

    /**
     * Monte Carlo {@code KL(policy || reference)}: the mean of {@code log pi - log ref} over the
     * policy's own samples.
     *
     * @param policy the policy
     * @param reference the reference
     * @param prompts the prompts
     * @param samples samples per prompt
     * @param seed the sampling seed
     * @return the estimate (in nats per response)
     */
    public static double kl(WindowLm policy, WindowLm reference, List<int[]> prompts, int samples, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int[][] ps = repeat(prompts, samples);
        int[][] out = policy.sample(ps, MAX_TOKENS, EOS, 1.0, random);
        List<int[]> pl = new ArrayList<>();
        List<int[]> cl = new ArrayList<>();
        for (int i = 0; i < out.length; i++) {
            if (out[i].length > 0) {
                pl.add(ps[i]);
                cl.add(out[i]);
            }
        }
        if (cl.isEmpty()) {
            return 0;
        }
        float[] a = policy.sequenceLogProbs(pl, cl).value().toArray();
        float[] b = reference.sequenceLogProbs(pl, cl).value().toArray();
        double sum = 0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] - b[i];
        }
        return Math.max(0, sum / Math.max(1, a.length));
    }

    /**
     * @param prompts prompts
     * @param times copies of each
     * @return every prompt repeated, in order
     */
    public static int[][] repeat(List<int[]> prompts, int times) {
        int[][] out = new int[prompts.size() * times][];
        for (int i = 0; i < out.length; i++) {
            out[i] = prompts.get(i / times);
        }
        return out;
    }
}
