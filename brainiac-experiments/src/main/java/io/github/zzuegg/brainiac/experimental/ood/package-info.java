/**
 * Experimental open-world tools, written as if they were part of a core package
 * {@code io.github.zzuegg.brainiac.ood}: post-hoc out-of-distribution scores from logits
 * (MSP, max logit, energy, ODIN) and from features (Mahalanobis, deep k-NN, ViM), activation
 * shaping (ReAct, ASH, SCALE, AdaSCALE), outlier exposure, OpenMax open-set recognition,
 * selective classification (SelectiveNet, the gambler's loss, likelihood-ratio rejection) and
 * learning to defer to an expert. Once one proves itself it moves to core unchanged apart from
 * dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.ood;
