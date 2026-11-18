package io.github.zzuegg.brainiac.experimental.combopt;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.PreferenceLosses;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Preference optimisation for neural combinatorial optimisation (Pan et al.): instead of
 * weighting log-likelihoods by advantages, which shrink as POMO's rollouts of one instance
 * become nearly equally long, every pair of rollouts of an instance becomes a preference
 * (the shorter solution wins) and the policy is fitted as the preference model, with the
 * implicit reward {@code alpha log pi(tau)} of the entropy-regularised objective (the
 * partition function cancels in the difference):
 * <ul>
 *   <li>{@link PreferenceModel#BRADLEY_TERRY}:
 *       {@code -log sigmoid(alpha (log pi(tau_w) - log pi(tau_l)))}, computed with
 *       {@link PreferenceLosses#bradleyTerry};</li>
 *   <li>{@link PreferenceModel#THURSTONE}: {@code -log Phi(alpha (log pi(tau_w) - log pi(tau_l)))}.</li>
 * </ul>
 * With {@link #localSearch} the sampled solutions are first improved by a few 2-opt moves
 * (fine-tuning only; inference stays the plain policy), and the likelihoods of the improved
 * solutions are computed by teacher forcing.
 *
 * <p>Paper: Pan, Wang, Zhou, Zhang and Li, "Preference Optimization for Combinatorial
 * Optimization Problems" (ICML 2025), <a href="https://arxiv.org/abs/2505.08735">arXiv:2505.08735</a>.
 */
public final class PreferenceOptimization {

    /** The pairwise preference model. */
    public enum PreferenceModel {
        /** logistic */
        BRADLEY_TERRY,
        /** probit */
        THURSTONE
    }

    private final Pomo pomo;
    private final PreferenceModel preference;
    private final float alpha;
    private int localSearchMoves;

    private PreferenceOptimization(Pomo pomo, PreferenceModel preference, float alpha) {
        this.pomo = pomo;
        this.preference = preference;
        this.alpha = alpha;
    }

    /**
     * Preference optimisation over POMO's multi-start rollouts.
     *
     * @param pomo the multi-start sampler (its model is trained)
     * @param preference the preference model
     * @param alpha the entropy temperature scaling the log-likelihood difference
     * @return the trainer
     */
    public static PreferenceOptimization of(Pomo pomo, PreferenceModel preference, float alpha) {
        return new PreferenceOptimization(pomo, preference, alpha);
    }

    /**
     * Improves the sampled TSP solutions by up to {@code moves} best-improvement 2-opt moves
     * before forming pairs.
     *
     * @param moves 0 turns it off
     * @return this
     */
    public PreferenceOptimization localSearch(int moves) {
        this.localSearchMoves = moves;
        return this;
    }

    /**
     * One step.
     *
     * @param instances the instances
     * @param optimizer the optimizer
     * @param random the source
     * @return the rollouts used
     */
    public int trainStep(List<RoutingEnv> instances, Optimizer optimizer, SplittableRandom random) {
        Pomo.MultiStart ms = pomo.rollouts(instances, AttentionModel.Decode.SAMPLE, random);
        AttentionModel.Rollout ro = ms.rollout();
        double[] len = ms.normalizedLengths();
        if (localSearchMoves > 0 && pomo.model().kind() == RoutingEnv.Kind.TSP) {
            int b = len.length;
            int[][] improved = new int[b][];
            for (int i = 0; i < b; i++) {
                RoutingEnv e = ms.envs().get(i);
                int[] t = ClassicHeuristics.twoOpt(e.distances(), ro.solutions()[i], false, localSearchMoves).tour();
                improved[i] = rotateTo(t, ms.starts()[i]);
            }
            ro = pomo.model().rollout(ms.envs(), null, AttentionModel.Decode.GREEDY, ms.starts(), improved, null, random);
            len = ro.lengths().clone();
            for (int i = 0; i < b; i++) {
                len[i] /= ms.scale()[i];
            }
        }
        List<Integer> win = new ArrayList<>();
        List<Integer> lose = new ArrayList<>();
        int[] group = ms.group();
        for (int i = 0; i < len.length; i++) {
            for (int j = i + 1; j < len.length && group[j] == group[i]; j++) {
                if (Math.abs(len[i] - len[j]) < 1e-9) {
                    continue;
                }
                win.add(len[i] < len[j] ? i : j);
                lose.add(len[i] < len[j] ? j : i);
            }
        }
        if (win.isEmpty()) {
            return len.length;
        }
        Var logp = ro.logp().reshape(len.length);
        Var chosen = logp.gather(0, indices(win)).mul(alpha);
        Var rejected = logp.gather(0, indices(lose)).mul(alpha);
        Var loss = switch (preference) {
            case BRADLEY_TERRY -> PreferenceLosses.bradleyTerry(chosen, rejected);
            case THURSTONE -> chosen.sub(rejected).mul((float) (1 / Math.sqrt(2))).erf().add(1f).mul(0.5f)
                    .clamp(1e-6f, 1f).log().mean().neg();
        };
        AttentionModel.step(loss, pomo.model().parameters(), optimizer, 1.0);
        return len.length;
    }

    private static Tensor indices(List<Integer> idx) {
        float[] f = new float[idx.size()];
        for (int i = 0; i < f.length; i++) {
            f[i] = idx.get(i);
        }
        return Tensor.of(Shape.of(f.length), f);
    }

    /**
     * Rotates a closed tour to begin at a node.
     *
     * @param tour the tour
     * @param first the node to begin with
     * @return the rotated tour
     */
    static int[] rotateTo(int[] tour, int first) {
        int n = tour.length;
        int z = 0;
        while (tour[z] != first) {
            z++;
        }
        int[] out = new int[n];
        for (int k = 0; k < n; k++) {
            out[k] = tour[(z + k) % n];
        }
        return out;
    }
}
