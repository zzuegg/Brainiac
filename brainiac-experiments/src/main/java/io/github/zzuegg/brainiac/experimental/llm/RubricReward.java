package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Rubrics as rewards (Gunjal et al., "Rubrics as rewards: reinforcement learning beyond
 * verifiable domains", <a href="https://arxiv.org/abs/2507.17746">arXiv:2507.17746</a>): for a
 * task with no single checkable answer, each prompt comes with a checklist of weighted,
 * instance-specific criteria ("mentions X", "avoids pitfall Y", "is concise"); a judge model
 * decides each criterion separately, and the reward is the weighted share satisfied (the
 * paper's explicit aggregation). Judging one narrow criterion is far easier than scoring a
 * response as a whole, which is what a scalar reward model or a Likert judge has to do.
 *
 * <p>The judge is a {@link GenerativeVerifier}: each criterion is asked as the problem
 * {@code prompt ++ criterion} with the response as the solution (so a judge with a short
 * window sees the criterion next to the response), and {@code P(yes)} is the
 * criterion's score. Scores are cached per prompt and response, because GRPO groups repeat
 * completions.
 */
public final class RubricReward implements GrpoTrainer.Reward {

    /**
     * One rubric criterion.
     *
     * @param criterion the criterion as tokens, shown to the judge after the prompt
     * @param weight its importance, positive
     */
    public record Item(int[] criterion, float weight) {
    }

    /** Writes the instance-specific rubric of a prompt. */
    @FunctionalInterface
    public interface Rubric {
        /**
         * @param prompt the prompt
         * @return its criteria
         */
        List<Item> items(int[] prompt);
    }

    private final GenerativeVerifier judge;
    private final Rubric rubric;
    private final Map<String, Float> cache = new HashMap<>();
    private long judgements;
    private int width;
    private int pad;

    private RubricReward(GenerativeVerifier judge, Rubric rubric) {
        this.judge = judge;
        this.rubric = rubric;
    }

    /**
     * Creates the reward.
     *
     * @param judge the per-criterion judge
     * @param rubric writes each prompt's rubric
     * @return the reward
     */
    public static RubricReward of(GenerativeVerifier judge, Rubric rubric) {
        return new RubricReward(Objects.requireNonNull(judge), Objects.requireNonNull(rubric));
    }

    /**
     * Left-pads every response to a fixed width before the judge reads it, so a small judge
     * with a fixed input layout sees each response position in the same place.
     *
     * @param responseWidth the padded width; 0 turns padding off
     * @param padToken the padding token
     * @return this reward
     */
    public RubricReward layout(int responseWidth, int padToken) {
        this.width = responseWidth;
        this.pad = padToken;
        cache.clear();
        return this;
    }

    private int[] shown(int[] response) {
        if (response.length >= width) {
            return response;
        }
        int[] out = new int[width];
        Arrays.fill(out, pad);
        System.arraycopy(response, 0, out, width - response.length, response.length);
        return out;
    }

    /**
     * The judge's question for one criterion.
     *
     * @param item the criterion
     * @param prompt the prompt
     * @return {@code prompt ++ criterion}, the verifier's problem
     */
    public static int[] question(Item item, int[] prompt) {
        int[] q = Arrays.copyOf(prompt, prompt.length + item.criterion().length);
        System.arraycopy(item.criterion(), 0, q, prompt.length, item.criterion().length);
        return q;
    }

    /**
     * The judge's probability that each criterion is met.
     *
     * @param prompt the prompt
     * @param completion the response
     * @return one {@code P(yes)} per rubric item
     */
    public double[] judge(int[] prompt, int[] completion) {
        List<Item> items = rubric.items(prompt);
        double[] out = new double[items.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = judge.score(question(items.get(i), prompt), shown(completion));
            judgements++;
        }
        return out;
    }

    @Override
    public float score(int[] prompt, int[] completion) {
        String key = Arrays.toString(prompt) + Arrays.toString(completion);
        Float known = cache.get(key);
        if (known != null) {
            return known;
        }
        List<Item> items = rubric.items(prompt);
        double[] p = judge(prompt, completion);
        double total = 0;
        double met = 0;
        for (int i = 0; i < p.length; i++) {
            total += items.get(i).weight();
            met += items.get(i).weight() * p[i];
        }
        float r = (float) (total == 0 ? 0 : met / total);
        cache.put(key, r);
        return r;
    }

    /**
     * Supervised examples for the judge from responses with known per-criterion labels.
     *
     * @param prompts the prompts
     * @param responses one response per prompt
     * @param satisfied labels per prompt and criterion, in rubric order
     * @return {@code {problems, solutions}} and the flattened labels, for
     *     {@link GenerativeVerifier#step}
     */
    public JudgeData judgeData(List<int[]> prompts, List<int[]> responses, List<boolean[]> satisfied) {
        List<int[]> problems = new ArrayList<>();
        List<int[]> solutions = new ArrayList<>();
        List<Boolean> labels = new ArrayList<>();
        for (int i = 0; i < prompts.size(); i++) {
            List<Item> items = rubric.items(prompts.get(i));
            for (int k = 0; k < items.size(); k++) {
                problems.add(question(items.get(k), prompts.get(i)));
                solutions.add(shown(responses.get(i)));
                labels.add(satisfied.get(i)[k]);
            }
        }
        boolean[] l = new boolean[labels.size()];
        for (int i = 0; i < l.length; i++) {
            l[i] = labels.get(i);
        }
        return new JudgeData(problems, solutions, l);
    }

    /**
     * Training rows for the judge.
     *
     * @param problems criterion-and-prompt questions
     * @param solutions the responses judged
     * @param labels whether each criterion is met
     */
    public record JudgeData(List<int[]> problems, List<int[]> solutions, boolean[] labels) {
    }

    /** @return criterion judgements made so far (cache hits excluded) */
    public long judgements() {
        return judgements;
    }

    @Override
    public String toString() {
        return "RubricReward(judgements=" + judgements + ")";
    }
}
