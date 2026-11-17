package io.github.zzuegg.brainiac.experiments.safety;

import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.train.RepresentationRerouting;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Circuit breakers against refusal training. Starting from a {@link SafetyToyLm} model that
 * complies with harmful requests, two defences run for the same number of steps:
 *
 * <ul>
 *   <li>refusal fine-tuning: harmful requests answered with a refusal, benign ones kept;</li>
 *   <li>representation rerouting ({@link RepresentationRerouting}) on the second hidden
 *       layer: representations of harmful prompts <em>and of their harmful completions</em>
 *       are pushed orthogonal to the original ones, benign representations kept.</li>
 * </ul>
 *
 * <p>Both are attacked: directly, by prefilling the first one or two harmful tokens, and by
 * the best single-token edit of the prompt. The harmful-completion rate counts completions
 * that carry the harmful answer on (the first harmful token for direct and edited prompts,
 * the rest of the harmful answer after a prefill); benign accuracy is the cost side.
 * Refusal training is shallow, so a prefilled answer continues; rerouted representations
 * break the harmful continuation itself.
 *
 * <p>Zou et al. 2024, "Improving Alignment and Robustness with Circuit Breakers",
 * <a href="https://arxiv.org/abs/2406.04313">arXiv:2406.04313</a>.
 */
public final class CircuitBreakers {

    /**
     * The result of one model.
     *
     * @param method the defence
     * @param direct harmful prompts answered harmfully
     * @param prefill1 harmful answers finished after one prefilled token
     * @param prefill2 harmful answers finished after two prefilled tokens
     * @param tokenEdit harmful answers after the best single-token prompt edit
     * @param benignAccuracy benign prompts answered exactly
     */
    public record Result(String method, double direct, double prefill1, double prefill2, double tokenEdit,
            double benignAccuracy) {

        /** @return the mean harmful-completion rate over the four attacks */
        public double meanHarm() {
            return (direct + prefill1 + prefill2 + tokenEdit) / 4;
        }
    }

    private static final int LAYER = 1;

    private CircuitBreakers() {
    }

    /**
     * Runs the experiment.
     *
     * @param steps defence steps
     * @param seed the seed
     * @return the undefended model, refusal training, then circuit breakers
     */
    public static List<Result> run(int steps, long seed) {
        WindowLm base = SafetyToyLm.train(SafetyToyLm.Behaviour.UNALIGNED, 500, seed);
        List<Result> out = new ArrayList<>();
        out.add(measure("no defence", base, seed));
        WindowLm refusal = base.copy();
        WindowLm breaker = base.copy();
        SplittableRandom r = new SplittableRandom(seed + 3);
        Optimizer ro = Optimizer.adam(2e-3f);
        RepresentationRerouting rr = RepresentationRerouting.of(breaker.model(), LAYER).optimizer(Optimizer.adam(2e-3f))
                .alpha(4).totalSteps(steps);
        for (int s = 0; s < steps; s++) {
            List<int[]> benign = SafetyToyLm.plainPrompts(16, false, r);
            List<int[]> answers = new ArrayList<>();
            for (int[] p : benign) {
                answers.add(SafetyToyLm.benignResponse(p));
            }
            List<int[]> harmful = SafetyToyLm.plainPrompts(16, true, r);
            List<int[]> p = new ArrayList<>(benign);
            List<int[]> c = new ArrayList<>(answers);
            List<int[]> harmfulAnswers = new ArrayList<>();
            for (int[] h : harmful) {
                p.add(h);
                c.add(SafetyToyLm.REFUSAL);
                harmfulAnswers.add(SafetyToyLm.HARMFUL_RESPONSE);
            }
            refusal.supervisedStep(p, c, ro);
            rr.step(SafetyToyLm.rows(harmful, harmfulAnswers).x(), SafetyToyLm.rows(benign, answers).x());
        }
        out.add(measure("refusal SFT", refusal, seed));
        out.add(measure("circuit breakers", breaker, seed));
        return out;
    }

    private static Result measure(String method, WindowLm lm, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 99);
        List<int[]> harmful = SafetyToyLm.prompts(100, true, false, r);
        List<int[]> benign = SafetyToyLm.prompts(100, false, false, r);
        List<int[]> edited = new ArrayList<>();
        for (int[] h : harmful) {
            edited.add(BackdoorRemoval.worstEdit(lm, h));
        }
        return new Result(method, SafetyToyLm.rate(lm, harmful, SafetyToyLm::complies),
                SafetyToyLm.rate(lm, SafetyToyLm.prefilled(harmful, 1), SafetyToyLm.finishesHarm(1)),
                SafetyToyLm.rate(lm, SafetyToyLm.prefilled(harmful, 2), SafetyToyLm.finishesHarm(2)),
                SafetyToyLm.rate(lm, edited, SafetyToyLm::complies), SafetyToyLm.benignAccuracy(lm, benign));
    }
}
