package io.github.zzuegg.brainiac.experimental.ensemble;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Losses that make ensemble members differ on purpose, so they do not all latch onto the same
 * shortcut:
 *
 * <ul>
 *   <li>{@link #negativeCorrelation}: negative correlation learning (Liu &amp; Yao 1999), joint
 *       training with {@code Σ_i [½ |f_i - y|² - λ |f_i - f̄|²]} on the members' probabilities:
 *       each member's error is pushed to be negatively correlated with the others';</li>
 *   <li>{@link #mutualInformation} and {@link #divDis}: DivDis (Lee et al., ICLR 2023,
 *       <a href="https://arxiv.org/abs/2202.03418">arXiv:2202.03418</a>), several heads trained
 *       on labelled source data while the mutual information between every pair of heads'
 *       predictions on unlabelled target data is minimised, with a KL term keeping each head's
 *       marginal near the prior so heads cannot become constant; {@link #selectHead} then
 *       picks the head with a few target labels;</li>
 *   <li>{@link #disagreement}: D-BAT (Pagliardini et al., ICLR 2023), training models one after
 *       another, each fitting the labels on source data while disagreeing with its predecessors
 *       on unlabelled out-of-distribution data:
 *       {@code -log(1 - Σ_c p_new(c) p_old(c))}, the probability that the two predictions
 *       differ (for two classes exactly {@code -log(p_0 q_1 + p_1 q_0)}).</li>
 * </ul>
 */
public final class DiverseEnsembles {

    private DiverseEnsembles() {
    }

    /**
     * The negative-correlation loss.
     *
     * @param memberLogits each member's logits {@code [n, k]}
     * @param oneHot the targets {@code [n, k]}
     * @param lambda the penalty strength, in {@code [0, 1]}
     * @return the scalar loss, averaged over rows
     */
    public static Var negativeCorrelation(List<Var> memberLogits, Tensor oneHot, float lambda) {
        int m = memberLogits.size();
        Var[] p = new Var[m];
        Var mean = null;
        for (int i = 0; i < m; i++) {
            p[i] = memberLogits.get(i).softmax(1);
            mean = mean == null ? p[i] : mean.add(p[i]);
        }
        mean = mean.div(m);
        Var loss = null;
        for (int i = 0; i < m; i++) {
            Var fit = p[i].sub(oneHot).square().sum(1, false).mul(0.5f);
            Var div = p[i].sub(mean).square().sum(1, false).mul(lambda);
            Var l = fit.sub(div).mean();
            loss = loss == null ? l : loss.add(l);
        }
        return loss;
    }

    /**
     * The mutual information between two heads' predictions over a batch, from their joint
     * {@code P_ab = mean_n p_a(n) p_b(n)^T}.
     *
     * @param a head A's logits {@code [n, k]}
     * @param b head B's logits {@code [n, k]}
     * @return {@code Σ P_ab log(P_ab / (P_a P_b))}
     */
    public static Var mutualInformation(Var a, Var b) {
        int n = a.shape().dim(0);
        Var pa = a.softmax(1);
        Var pb = b.softmax(1);
        Var joint = pa.transpose().matmul(pb).div(n).add(1e-8f);
        Var ma = joint.sum(1, true);
        Var mb = joint.sum(0, true);
        return joint.mul(joint.log().sub(ma.log()).sub(mb.log())).sum();
    }

    /**
     * The DivDis regulariser of a set of heads on an unlabelled target batch: the summed pairwise
     * mutual information plus {@code regWeight} times each head's KL divergence of its batch
     * marginal from the uniform prior.
     *
     * @param headLogits each head's target logits {@code [n, k]}
     * @param regWeight the weight of the marginal KL
     * @return the scalar regulariser
     */
    public static Var divDis(List<Var> headLogits, float regWeight) {
        Var loss = null;
        for (int i = 0; i < headLogits.size(); i++) {
            for (int j = i + 1; j < headLogits.size(); j++) {
                Var mi = mutualInformation(headLogits.get(i), headLogits.get(j));
                loss = loss == null ? mi : loss.add(mi);
            }
            Var marginal = headLogits.get(i).softmax(1).mean(0, false).add(1e-8f);
            int k = marginal.shape().dim(0);
            Var kl = marginal.mul(marginal.log().add((float) Math.log(k))).sum().mul(regWeight);
            loss = loss == null ? kl : loss.add(kl);
        }
        return loss;
    }

    /**
     * The D-BAT disagreement loss against fixed predecessors.
     *
     * @param logits the new model's logits on out-of-distribution inputs {@code [n, k]}
     * @param previous each predecessor's probabilities on the same inputs {@code [n, k]}
     * @return {@code mean over predecessors and rows of -log(1 - Σ_c p(c) q(c))}
     */
    public static Var disagreement(Var logits, List<Tensor> previous) {
        Var p = logits.softmax(1);
        Var loss = null;
        for (Tensor q : previous) {
            Var agree = p.mul(q).sum(1, false);
            Var l = agree.neg().add(1f + 1e-6f).log().neg().mean();
            loss = loss == null ? l : loss.add(l);
        }
        return loss.div(previous.size());
    }

    /**
     * Picks the head that is most accurate on a few labelled target examples.
     *
     * @param headProbs each head's probabilities on the labelled examples {@code [head][n][k]}
     * @param labels their labels
     * @return the index of the best head (the first on ties)
     */
    public static int selectHead(double[][][] headProbs, int[] labels) {
        int best = 0;
        int bestHits = -1;
        for (int h = 0; h < headProbs.length; h++) {
            int hits = 0;
            for (int i = 0; i < labels.length; i++) {
                hits += TestTimeAugmentation.argmax(headProbs[h][i]) == labels[i] ? 1 : 0;
            }
            if (hits > bestHits) {
                best = h;
                bestHits = hits;
            }
        }
        return best;
    }
}
