package io.github.zzuegg.brainiac.experimental.align;

/**
 * A known-quality score of a response: the ground truth that alignment experiments measure
 * against (the "gold reward" of Gao et al., <a href="https://arxiv.org/abs/2210.10760">
 * arXiv:2210.10760</a>), as opposed to the learned proxy a policy is optimized against. It is
 * the shared interface between experiments that produce outputs of known quality and
 * evaluators (judges, length-controlled win rates) that should be checked against the truth.
 */
@FunctionalInterface
public interface GoldScorer {

    /**
     * @param prompt the prompt tokens
     * @param response the response tokens
     * @return the true quality, higher is better
     */
    double score(int[] prompt, int[] response);
}
