package io.github.zzuegg.brainiac.diffusion;

import java.util.Objects;

/**
 * What video to generate.
 *
 * @param prompt what the video shows
 * @param negativePrompt what it should not show ({@code null} for the model's default)
 * @param width the width in pixels, a multiple of 16
 * @param height the height in pixels, a multiple of 16
 * @param frames the number of frames, {@code 1 + 4k} (Wan: 81 is five seconds at 16 fps)
 * @param steps the denoising steps
 * @param guidance the classifier-free guidance scale; 1 or less runs without a negative pass
 * @param shift the noise schedule's shift ({@code 0} for the model's default); higher spends
 *     more steps at high noise, which suits larger videos
 * @param seed the seed of the starting noise
 */
public record VideoRequest(String prompt, String negativePrompt, int width, int height, int frames, int steps,
        float guidance, float shift, long seed) {

    /** Checks the values. */
    public VideoRequest {
        Objects.requireNonNull(prompt, "prompt");
        if (width < 16 || height < 16 || width % 16 != 0 || height % 16 != 0) {
            throw new IllegalArgumentException("width and height must be multiples of 16, got " + width + "x" + height);
        }
        if (frames < 1 || (frames - 1) % 4 != 0) {
            throw new IllegalArgumentException("frames must be 1 + 4k (1, 5, 9, ..., 81), got " + frames);
        }
        if (steps < 1 || shift < 0) {
            throw new IllegalArgumentException("steps must be positive and the shift not negative");
        }
    }

    /**
     * A request with Wan 2.1's defaults at a size a 16 GB GPU handles: 832x480, 33 frames (two
     * seconds), 30 steps, guidance 5.
     *
     * @param prompt what the video shows
     * @return the request
     */
    public static VideoRequest of(String prompt) {
        return new VideoRequest(prompt, null, 832, 480, 33, 30, 5f, 0f, 42);
    }

    /**
     * @param negative what the video should not show
     * @return a copy with that negative prompt
     */
    public VideoRequest withNegativePrompt(String negative) {
        return new VideoRequest(prompt, negative, width, height, frames, steps, guidance, shift, seed);
    }

    /**
     * @param w the width
     * @param h the height
     * @return a copy with that size
     */
    public VideoRequest withSize(int w, int h) {
        return new VideoRequest(prompt, negativePrompt, w, h, frames, steps, guidance, shift, seed);
    }

    /**
     * @param n the number of frames, {@code 1 + 4k}
     * @return a copy with that length
     */
    public VideoRequest withFrames(int n) {
        return new VideoRequest(prompt, negativePrompt, width, height, n, steps, guidance, shift, seed);
    }

    /**
     * @param n the steps
     * @return a copy with that many steps
     */
    public VideoRequest withSteps(int n) {
        return new VideoRequest(prompt, negativePrompt, width, height, frames, n, guidance, shift, seed);
    }

    /**
     * @param g the guidance scale
     * @return a copy with that guidance
     */
    public VideoRequest withGuidance(float g) {
        return new VideoRequest(prompt, negativePrompt, width, height, frames, steps, g, shift, seed);
    }

    /**
     * @param s the schedule's shift
     * @return a copy with that shift
     */
    public VideoRequest withShift(float s) {
        return new VideoRequest(prompt, negativePrompt, width, height, frames, steps, guidance, s, seed);
    }

    /**
     * @param s the seed
     * @return a copy with that seed
     */
    public VideoRequest withSeed(long s) {
        return new VideoRequest(prompt, negativePrompt, width, height, frames, steps, guidance, shift, s);
    }
}
