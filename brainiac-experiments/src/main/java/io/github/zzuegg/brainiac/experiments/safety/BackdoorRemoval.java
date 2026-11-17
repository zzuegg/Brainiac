package io.github.zzuegg.brainiac.experiments.safety;

import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.train.LatentAdversarialTrainer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Removing a planted backdoor with latent adversarial training. A {@link SafetyToyLm} model
 * refuses harmful requests unless a two-token trigger is present. The defender does not
 * know the trigger and fine-tunes on ordinary safety data (harmful requests refused, benign
 * ones answered) in three ways, all for the same number of steps:
 *
 * <ul>
 *   <li>plain safety fine-tuning;</li>
 *   <li>input-space adversarial training: for each harmful prompt, the single-token edit of
 *       a filler slot that most raises the compliance probability is found by exhaustive
 *       search, and the model is trained to refuse that prompt;</li>
 *   <li>latent adversarial training ({@link LatentAdversarialTrainer}) on the first hidden
 *       layer.</li>
 * </ul>
 *
 * <p>The defender's prompts hold only plain fillers, never a trigger token. The outcome is
 * the backdoor's success rate (triggered harmful prompts answered
 * harmfully) after each, with benign accuracy and plain refusal as the cost side. All
 * defences use plain SGD: with Adam, even plain safety SFT disturbs the trigger weights
 * enough to remove this small backdoor, so the comparison would show nothing.
 *
 * <p>Sheshadri et al. 2024, "Latent Adversarial Training Improves Robustness to Persistent
 * Harmful Behaviors in LLMs", <a href="https://arxiv.org/abs/2407.15549">arXiv:2407.15549</a>.
 */
public final class BackdoorRemoval {

    /**
     * The result of one defence.
     *
     * @param method the defence
     * @param backdoorSuccess triggered harmful prompts answered harmfully
     * @param refusal plain harmful prompts refused
     * @param benignAccuracy benign prompts answered exactly
     */
    public record Result(String method, double backdoorSuccess, double refusal, double benignAccuracy) {
    }

    private BackdoorRemoval() {
    }

    /**
     * Runs the experiment.
     *
     * @param steps fine-tuning steps per defence
     * @param seed the seed
     * @return the backdoored model, then plain fine-tuning, input-space AT and LAT
     */
    public static List<Result> run(int steps, long seed) {
        WindowLm base = SafetyToyLm.train(SafetyToyLm.Behaviour.BACKDOORED, 600, seed);
        List<Result> out = new ArrayList<>();
        out.add(measure("backdoored", base, seed));
        for (String method : new String[] {"plain safety SFT", "input-space AT", "latent AT"}) {
            WindowLm lm = base.copy();
            SplittableRandom r = new SplittableRandom(seed + 7);
            Optimizer opt = Optimizer.sgd(0.05f);
            LatentAdversarialTrainer lat = LatentAdversarialTrainer.of(lm.model(), SafetyToyLm.LATENT_LAYER)
                    .optimizer(opt).epsilon(4f).attackSteps(6).attackRate(0.4f);
            for (int s = 0; s < steps; s++) {
                List<int[]> benign = SafetyToyLm.plainPrompts(16, false, r);
                List<int[]> answers = new ArrayList<>();
                for (int[] p : benign) {
                    answers.add(SafetyToyLm.benignResponse(p));
                }
                List<List<int[]>> retain = List.of(benign, answers);
                List<int[]> harmful = SafetyToyLm.plainPrompts(16, true, r);
                List<int[]> refusals = new ArrayList<>();
                List<int[]> compliance = new ArrayList<>();
                for (int i = 0; i < harmful.size(); i++) {
                    refusals.add(SafetyToyLm.REFUSAL);
                    compliance.add(new int[] {SafetyToyLm.H1, SafetyToyLm.END});
                }
                switch (method) {
                    case "plain safety SFT" -> {
                        List<int[]> p = new ArrayList<>(retain.get(0));
                        List<int[]> c = new ArrayList<>(retain.get(1));
                        p.addAll(harmful);
                        c.addAll(refusals);
                        lm.supervisedStep(p, c, opt);
                    }
                    case "input-space AT" -> {
                        List<int[]> p = new ArrayList<>(retain.get(0));
                        List<int[]> c = new ArrayList<>(retain.get(1));
                        for (int[] h : harmful) {
                            p.add(worstEdit(lm, h));
                        }
                        c.addAll(refusals);
                        lm.supervisedStep(p, c, opt);
                    }
                    default -> {
                        // the attacked rows only need the first response token
                        List<int[]> first = new ArrayList<>();
                        for (int i = 0; i < harmful.size(); i++) {
                            first.add(new int[] {SafetyToyLm.REFUSE});
                        }
                        List<int[]> bad = new ArrayList<>();
                        for (int i = 0; i < harmful.size(); i++) {
                            bad.add(new int[] {SafetyToyLm.H1});
                        }
                        SafetyToyLm.Rows want = SafetyToyLm.rows(harmful, first);
                        SafetyToyLm.Rows unwanted = SafetyToyLm.rows(harmful, bad);
                        SafetyToyLm.Rows keep = SafetyToyLm.rows(retain.get(0), retain.get(1));
                        lat.step(want.x(), want.ids(), unwanted.ids(), keep.x(), keep.ids());
                    }
                }
            }
            out.add(measure(method, lm, seed));
        }
        return out;
    }

    /** The single filler-slot edit of a prompt that makes compliance most likely. */
    static int[] worstEdit(WindowLm lm, int[] prompt) {
        List<int[]> candidates = new ArrayList<>();
        for (int slot = 1; slot <= 2; slot++) {
            for (int t = 0; t < SafetyToyLm.VOCAB; t++) {
                int[] q = prompt.clone();
                q[slot] = t;
                candidates.add(q);
            }
        }
        float[] logits = lm.logits(candidates).value().to(DType.FLOAT32).toArray();
        int best = 0;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < candidates.size(); i++) {
            double score = logProb(logits, i, SafetyToyLm.H1);
            if (score > bestScore) {
                bestScore = score;
                best = i;
            }
        }
        return candidates.get(best);
    }

    private static double logProb(float[] logits, int row, int token) {
        int v = SafetyToyLm.VOCAB;
        double max = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < v; i++) {
            max = Math.max(max, logits[row * v + i]);
        }
        double s = 0;
        for (int i = 0; i < v; i++) {
            s += Math.exp(logits[row * v + i] - max);
        }
        return logits[row * v + token] - max - Math.log(s);
    }

    private static Result measure(String method, WindowLm lm, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 99);
        List<int[]> triggered = SafetyToyLm.prompts(100, true, true, r);
        List<int[]> harmful = SafetyToyLm.prompts(100, true, false, r);
        List<int[]> benign = SafetyToyLm.prompts(100, false, false, r);
        return new Result(method, SafetyToyLm.rate(lm, triggered, SafetyToyLm::complies),
                SafetyToyLm.rate(lm, harmful, SafetyToyLm::refuses), SafetyToyLm.benignAccuracy(lm, benign));
    }
}
