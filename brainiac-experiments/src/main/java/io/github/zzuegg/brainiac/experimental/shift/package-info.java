/**
 * Experimental distribution-shift tools, written as if they were part of a core package
 * {@code io.github.zzuegg.brainiac.shift}: label-shift estimation and correction (BBSE, RLLS,
 * MLLS, MAP with a Dirichlet prior, online), domain adaptation (gradient reversal, DANN, CDAN,
 * CORAL, MK-MMD, source-free SHOT, robust test-time adaptation with EATA, SAR and CoTTA),
 * domain generalisation (MixStyle, SWAD), last-layer retraining (DFR), density-ratio
 * estimation (KLIEP, uLSIF, classifier odds) with weighted conformal prediction, and robust
 * fine-tuning (LP-FT, WiSE-FT). Once one proves itself it moves to core unchanged apart from
 * dropping {@code .experimental} from the package name.
 */
package io.github.zzuegg.brainiac.experimental.shift;
