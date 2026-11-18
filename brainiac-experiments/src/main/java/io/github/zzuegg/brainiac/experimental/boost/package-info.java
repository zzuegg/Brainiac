/**
 * Boosting and forest variants beyond core {@code GradientBoostedTrees}: LightGBM's GOSS,
 * exclusive feature bundling and leaf-wise growth, CatBoost's ordered target statistics and
 * ordered boosting with oblivious trees, NGBoost's natural-gradient distributional boosting,
 * explainable boosting machines (cyclic GA2M with FAST pair ranking), PFN-Boost on top of an
 * in-context prior, and quantile regression forests with conformal calibration. All reuse the
 * core {@code GradientBoostedTrees.Loss} derivatives, and the forests reuse
 * {@code RegressionForest}'s tree grower.
 */
package io.github.zzuegg.brainiac.experimental.boost;
