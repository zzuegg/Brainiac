package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Constitutional AI and RL from AI feedback (Bai et al. 2022,
 * <a href="https://arxiv.org/abs/2212.08073">2212.08073</a>; RLAIF, Lee et al. 2023,
 * <a href="https://arxiv.org/abs/2309.00267">2309.00267</a>): the model is aligned to a short
 * list of written principles (the constitution) with no human labels on its outputs.
 *
 * <p>Each principle is a token. The model critiques its own answer against a principle by
 * reading {@code principle, prompt, answer, judge} and predicting a yes (complies) or no
 * token; an answer's critique score is its worst principle's probability of yes. One
 * iteration then
 * <ol>
 *   <li><b>critiques and revises</b>: for each prompt it samples an answer, critiques it, and
 *       revises it by sampling alternatives and keeping the one its own critique rates best
 *       (the paper's critique-then-rewrite, done here by search because the toy model has no
 *       instruction to rewrite with);</li>
 *   <li>trains a <b>preference model</b> ({@link SequenceScorer}) on the AI-labelled pairs
 *       (revision preferred over original) gathered so far;</li>
 *   <li>runs <b>RLAIF</b>: samples fresh candidates, ranks them with the preference model and
 *       trains the policy with DPO on (best, worst), plus a supervised step on the revisions
 *       (the SL-CAI stage).</li>
 * </ol>
 * The model's critique ability itself comes from a few labelled critique examples
 * ({@link #critiqueExample}), the constitution's stand-in for the paper's few-shot critique
 * prompts.
 */
public final class ConstitutionalLoop {

    /**
     * What one iteration did.
     *
     * @param revised prompts whose revision beat the original by the critique
     * @param meanCritique the mean critique score of the original answers
     */
    public record Round(int revised, double meanCritique) {
    }

    private final WindowLm model;
    private final int[] principles;
    private final int judge;
    private final int yes;
    private final int no;
    private final SequenceScorer preference;
    private final List<int[]> pairsFull = new ArrayList<>();
    private final List<Float> pairLabels = new ArrayList<>();
    private int maxTokens = 3;
    private int end = -1;
    private int revisions = 8;
    private float beta = 0.5f;
    private int dpoSteps = 5;
    private Optimizer optimizer = Optimizer.adam(1e-3f);

    private ConstitutionalLoop(WindowLm model, int[] principles, int judge, int yes, int no, SequenceScorer pm) {
        this.model = Objects.requireNonNull(model);
        this.principles = principles.clone();
        this.judge = judge;
        this.yes = yes;
        this.no = no;
        this.preference = Objects.requireNonNull(pm);
    }

    /**
     * @param model the policy, which is also the critic
     * @param principles one token per principle of the constitution
     * @param judgeToken the token that asks for a verdict
     * @param yesToken the verdict "complies"
     * @param noToken the verdict "violates"
     * @param preferenceModel the preference model trained on the AI labels
     * @return the loop
     */
    public static ConstitutionalLoop of(WindowLm model, int[] principles, int judgeToken, int yesToken, int noToken,
            SequenceScorer preferenceModel) {
        return new ConstitutionalLoop(model, principles, judgeToken, yesToken, noToken, preferenceModel);
    }

    /**
     * @param maxNew the most tokens per answer
     * @param endToken the end token
     * @param n alternatives sampled when revising (and candidates for RLAIF)
     * @return this loop
     */
    public ConstitutionalLoop generation(int maxNew, int endToken, int n) {
        this.maxTokens = maxNew;
        this.end = endToken;
        this.revisions = n;
        return this;
    }

    /**
     * @param o the policy optimizer
     * @param dpoBeta the DPO beta
     * @param steps DPO steps per iteration
     * @return this loop
     */
    public ConstitutionalLoop training(Optimizer o, float dpoBeta, int steps) {
        this.optimizer = Objects.requireNonNull(o);
        this.beta = dpoBeta;
        this.dpoSteps = steps;
        return this;
    }

    private int[] critiquePrompt(int principle, int[] prompt, int[] answer) {
        int[] p = new int[prompt.length + answer.length + 2];
        p[0] = principle;
        System.arraycopy(prompt, 0, p, 1, prompt.length);
        System.arraycopy(answer, 0, p, 1 + prompt.length, answer.length);
        p[p.length - 1] = judge;
        return p;
    }

    /**
     * A supervised example that teaches critiquing against one principle.
     *
     * @param principle the principle token
     * @param prompt a prompt
     * @param answer an answer
     * @param complies whether the answer complies with the principle
     * @return {critique prompt, verdict completion}
     */
    public int[][] critiqueExample(int principle, int[] prompt, int[] answer, boolean complies) {
        return new int[][] {critiquePrompt(principle, prompt, answer), {complies ? yes : no, end}};
    }

    /**
     * The model's own critique of an answer against the whole constitution.
     *
     * @param prompt the prompt
     * @param answer the answer
     * @return the smallest probability of "complies" over the principles
     */
    public double critique(int[] prompt, int[] answer) {
        List<int[]> ps = new ArrayList<>();
        for (int principle : principles) {
            ps.add(critiquePrompt(principle, prompt, answer));
        }
        float[] logits = LlmTensors.values(model.logits(ps));
        int v = model.vocabSize();
        double min = 1;
        for (int i = 0; i < principles.length; i++) {
            double y = logits[i * v + yes];
            double n = logits[i * v + no];
            min = Math.min(min, 1 / (1 + Math.exp(n - y)));
        }
        return min;
    }

    private static int[] concat(int[] a, int[] b) {
        int[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    /**
     * One iteration: critique and revise, fit the preference model, RLAIF with DPO.
     *
     * @param prompts the prompts
     * @param random the random source
     * @return what happened
     */
    public Round iterate(List<int[]> prompts, SplittableRandom random) {
        if (end < 0) {
            throw new IllegalStateException("call generation(maxNew, end, n) first");
        }
        List<int[]> slPrompts = new ArrayList<>();
        List<int[]> slRevisions = new ArrayList<>();
        int revised = 0;
        double critiqueSum = 0;
        int[][] originals = model.sample(prompts.toArray(int[][]::new), maxTokens, end, 1.0, random);
        for (int i = 0; i < prompts.size(); i++) {
            int[] p = prompts.get(i);
            double original = critique(p, originals[i]);
            critiqueSum += original;
            int[][] group = new int[revisions][];
            Arrays.fill(group, p);
            int[] best = originals[i];
            double bestScore = original;
            for (int[] alt : model.sample(group, maxTokens, end, 1.0, random)) {
                double s = critique(p, alt);
                if (alt.length > 0 && s > bestScore) {
                    best = alt;
                    bestScore = s;
                }
            }
            if (best != originals[i] && bestScore > original + 0.2) {
                revised++;
                slPrompts.add(p);
                slRevisions.add(best);
                pairsFull.add(concat(p, best));
                pairLabels.add(1f);
                pairsFull.add(concat(p, originals[i]));
                pairLabels.add(0f);
            }
        }
        if (!pairsFull.isEmpty()) {
            float[] labels = new float[pairLabels.size()];
            for (int i = 0; i < labels.length; i++) {
                labels[i] = pairLabels.get(i);
            }
            preference.fit(pairsFull, labels, 30);
        }
        List<DpoTrainer.Preference> pairs = new ArrayList<>();
        for (int[] p : prompts) {
            int[][] group = new int[revisions][];
            Arrays.fill(group, p);
            int[] best = null;
            int[] worst = null;
            double hi = Double.NEGATIVE_INFINITY;
            double lo = Double.POSITIVE_INFINITY;
            for (int[] c : model.sample(group, maxTokens, end, 1.0, random)) {
                if (c.length == 0) {
                    continue;
                }
                double s = preference.score(concat(p, c));
                if (s > hi) {
                    hi = s;
                    best = c;
                }
                if (s < lo) {
                    lo = s;
                    worst = c;
                }
            }
            if (best != null && hi - lo > 0.2 && !Arrays.equals(best, worst)) {
                pairs.add(new DpoTrainer.Preference(p, best, worst));
            }
        }
        if (!pairs.isEmpty()) {
            DpoTrainer dpo = DpoTrainer.of(model).beta(beta).optimizer(optimizer).chosenNll(1f);
            for (int s = 0; s < dpoSteps; s++) {
                dpo.step(pairs);
            }
        }
        if (!slRevisions.isEmpty()) {
            model.supervisedStep(slPrompts, slRevisions, optimizer);
        }
        return new Round(revised, critiqueSum / prompts.size());
    }
}
