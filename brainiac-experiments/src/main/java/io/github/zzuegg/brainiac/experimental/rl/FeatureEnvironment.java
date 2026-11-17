package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;

/**
 * An environment whose reward is linear in transition features,
 * {@code r(s, a, s') = phi(s, a, s') . w}, so that the same dynamics define a family of
 * tasks, one per weight vector: the setting of successor features
 * ({@link SuccessorFeatures}). The features of the latest step are read after
 * {@link #step}.
 */
public interface FeatureEnvironment extends Environment {

    /** @return {@code phi(s, a, s')} of the latest step */
    float[] lastFeatures();

    /** @return the number of features */
    int features();

    /** @return the task's weights {@code w} */
    float[] weights();
}
