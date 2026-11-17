package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Direct Preference Optimization and relatives: align a language model to pairwise
 * preferences with a classification-style loss, no reward model and no RL loop. For a pair
 * (chosen {@code w}, rejected {@code l}) the implicit reward margin is
 * {@code h = beta * [(log pi(w) - log ref(w)) - (log pi(l) - log ref(l))]}.
 * <ul>
 *   <li>{@link Objective#DPO} (<a href="https://arxiv.org/abs/2305.18290">2305.18290</a>):
 *       {@code -log sigmoid(h)}.</li>
 *   <li>{@link Objective#IPO} (<a href="https://arxiv.org/abs/2310.12036">2310.12036</a>):
 *       {@code (h / beta - 1 / (2 beta))^2}, which does not push the margin to infinity on
 *       deterministic preferences.</li>
 *   <li>{@link Objective#SIMPO} (<a href="https://arxiv.org/abs/2405.14734">2405.14734</a>):
 *       reference-free, length-normalized log-likelihoods with a target margin
 *       {@code gamma}: {@code -log sigmoid(beta (mean log pi(w) - mean log pi(l)) - gamma)}.</li>
 * </ul>
 * The same trainer drives SPIN (the rejected answers are the previous model's own
 * generations) and self-rewarding loops (the pairs are ranked by the model itself).
 */
public final class DpoTrainer {

    /** Which preference loss. */
    public enum Objective {
        /** Logistic loss on the reference-relative margin. */
        DPO,
        /** Squared loss towards a fixed margin. */
        IPO,
        /** Reference-free, length-normalized, with a margin. */
        SIMPO
    }

    /**
     * One preference.
     *
     * @param prompt the prompt
     * @param chosen the preferred completion
     * @param rejected the dispreferred completion
     */
    public record Preference(int[] prompt, int[] chosen, int[] rejected) {
    }

    private final WindowLm policy;
    private WindowLm reference;
    private Objective objective = Objective.DPO;
    private Optimizer optimizer = Optimizer.adam(3e-3f);
    private float beta = 0.1f;
    private float gamma = 0.5f;
    private float nll;

    private DpoTrainer(WindowLm policy) {
        this.policy = Objects.requireNonNull(policy);
        this.reference = policy.copy();
    }

    /**
     * @param policy the model to train; a frozen copy becomes the reference
     * @return the trainer
     */
    public static DpoTrainer of(WindowLm policy) {
        return new DpoTrainer(policy);
    }

    /**
     * @param o the loss
     * @return this trainer
     */
    public DpoTrainer objective(Objective o) {
        this.objective = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param beta the inverse temperature of the implicit reward
     * @return this trainer
     */
    public DpoTrainer beta(float beta) {
        this.beta = beta;
        return this;
    }

    /**
     * @param gamma SimPO's target margin
     * @return this trainer
     */
    public DpoTrainer margin(float gamma) {
        this.gamma = gamma;
        return this;
    }

    /**
     * Adds a negative log-likelihood term on the chosen answers (per-token mean), as in
     * Iterative Reasoning Preference Optimization
     * (<a href="https://arxiv.org/abs/2404.19733">2404.19733</a>). Plain DPO only widens the
     * margin and can lower the chosen answer's likelihood too, letting probability leak to
     * malformed outputs; this term keeps the chosen answers likely.
     *
     * @param weight the weight of the term, 0 for plain DPO
     * @return this trainer
     */
    public DpoTrainer chosenNll(float weight) {
        this.nll = weight;
        return this;
    }

    /**
     * @param o the optimizer
     * @return this trainer
     */
    public DpoTrainer optimizer(Optimizer o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param reference the new frozen reference (SPIN and self-rewarding swap it every
     *     iteration)
     * @return this trainer
     */
    public DpoTrainer reference(WindowLm reference) {
        this.reference = Objects.requireNonNull(reference);
        return this;
    }

    /** @return the current reference */
    public WindowLm reference() {
        return reference;
    }

    /**
     * Computes the loss of a batch without updating anything.
     *
     * @param batch the preferences
     * @return the scalar loss, differentiable with respect to the policy
     */
    public Var loss(List<Preference> batch) {
        List<int[]> prompts = new ArrayList<>();
        List<int[]> completions = new ArrayList<>();
        for (Preference p : batch) {
            prompts.add(p.prompt());
            completions.add(p.chosen());
        }
        for (Preference p : batch) {
            prompts.add(p.prompt());
            completions.add(p.rejected());
        }
        int n = batch.size();
        boolean normalize = objective == Objective.SIMPO;
        Tensor sum = WindowLm.sequenceSum(completions, normalize);
        Var tokens = policy.logProbs(prompts, completions);
        Var seq = Var.of(sum).matmul(tokens);
        Var chosen = seq.slice(0, 0, n);
        Var rejected = seq.slice(0, n, 2 * n);
        Var diff = chosen.sub(rejected);
        Var loss;
        if (objective == Objective.SIMPO) {
            loss = diff.mul(beta).sub(gamma).neg().softplus().mean();
        } else {
            Tensor ref = sum.matmul(reference.logProbs(prompts, completions).value());
            Tensor refDiff = ref.slice(0, 0, n).sub(ref.slice(0, n, 2 * n));
            Var margin = diff.sub(refDiff);
            loss = objective == Objective.IPO
                    ? margin.sub(1f / (2 * beta)).square().mean()
                    : margin.mul(beta).neg().softplus().mean();
        }
        if (nll > 0) {
            Var mean = Var.of(WindowLm.sequenceSum(completions, true)).matmul(tokens).slice(0, 0, n);
            loss = loss.sub(mean.mean().mul(nll));
        }
        return loss;
    }

    /**
     * One optimizer step on a batch.
     *
     * @param batch the preferences
     * @return the loss before the step
     */
    public float step(List<Preference> batch) {
        Var loss = loss(batch);
        float v = loss.value().item();
        loss.backward();
        optimizer.step(policy.parameters());
        return v;
    }

    /**
     * The fraction of pairs whose chosen completion the policy already prefers by implicit
     * reward (reference-relative log-likelihood).
     *
     * @param batch the preferences
     * @return the preference accuracy
     */
    public double accuracy(List<Preference> batch) {
        int right = 0;
        for (Preference p : batch) {
            double w = policy.sequenceLogProb(p.prompt(), p.chosen()) - reference.sequenceLogProb(p.prompt(), p.chosen());
            double l = policy.sequenceLogProb(p.prompt(), p.rejected())
                    - reference.sequenceLogProb(p.prompt(), p.rejected());
            if (w > l) {
                right++;
            }
        }
        return (double) right / batch.size();
    }
}
