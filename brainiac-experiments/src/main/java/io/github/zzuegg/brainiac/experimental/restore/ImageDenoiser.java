package io.github.zzuegg.brainiac.experimental.restore;

import io.github.zzuegg.brainiac.autograd.Var;

/**
 * Anything that removes Gaussian noise of a given level from a batch of images. Plug-and-play
 * solvers ({@link PlugAndPlay}) use one in place of the proximal operator of an image prior;
 * blind denoisers ignore the level.
 */
@FunctionalInterface
public interface ImageDenoiser {

    /**
     * Denoises images.
     *
     * @param noisy {@code [batch, 1, h, w]}
     * @param sigma the noise standard deviation (on the [0, 1] intensity scale)
     * @return the estimates, same shape
     */
    Var denoise(Var noisy, float sigma);
}
