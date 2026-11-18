package io.github.zzuegg.brainiac.diffusion;

import java.util.Objects;

/**
 * What to generate.
 *
 * <pre>{@code
 * ImageRequest r = ImageRequest.of("a lighthouse at dusk, oil painting")
 *         .negativePrompt("blurry")
 *         .size(768, 512)
 *         .steps(25)
 *         .seed(42)
 *         .build();
 * }</pre>
 *
 * @param prompt what the image should show
 * @param negativePrompt what it should not show (may be empty)
 * @param width the width in pixels, a multiple of 64 (0 for the model's native size)
 * @param height the height in pixels, a multiple of 64 (0 for the model's native size)
 * @param steps denoising steps
 * @param guidance classifier-free guidance scale; 1 or less turns guidance off (half the work)
 * @param seed the seed of the first image; image {@code i} uses {@code seed + i}
 * @param count how many images
 * @param sampler the sampler
 */
public record ImageRequest(String prompt, String negativePrompt, int width, int height, int steps, float guidance,
        long seed, int count, Sampler sampler) {

    /**
     * Validates the request.
     *
     * @param prompt what the image should show
     * @param negativePrompt what it should not show
     * @param width the width in pixels
     * @param height the height in pixels
     * @param steps denoising steps
     * @param guidance guidance scale
     * @param seed the seed
     * @param count how many images
     * @param sampler the sampler
     */
    public ImageRequest {
        Objects.requireNonNull(prompt, "prompt");
        negativePrompt = negativePrompt == null ? "" : negativePrompt;
        Objects.requireNonNull(sampler, "sampler");
        if (width < 0 || height < 0 || width % 64 != 0 || height % 64 != 0) {
            throw new IllegalArgumentException("width and height must be multiples of 64 (or 0 for the model's size), got "
                    + width + "x" + height);
        }
        if (steps < 1 || steps > 1000) {
            throw new IllegalArgumentException("steps must be between 1 and 1000, got " + steps);
        }
        if (count < 1) {
            throw new IllegalArgumentException("count must be at least 1, got " + count);
        }
        if (!(guidance >= 0) || Float.isInfinite(guidance)) {
            throw new IllegalArgumentException("guidance must be 0 or more, got " + guidance);
        }
    }

    /**
     * Starts a request with defaults: no negative prompt, the model's native size, 25 steps,
     * guidance 7, seed 0, one image, {@link Sampler#DPM_PP_2M_KARRAS}.
     *
     * @param prompt what the image should show
     * @return a builder
     */
    public static Builder of(String prompt) {
        return new Builder(prompt);
    }

    /** Builds an {@link ImageRequest}. */
    public static final class Builder {

        private final String prompt;
        private String negativePrompt = "";
        private int width;
        private int height;
        private int steps = 25;
        private float guidance = 7f;
        private long seed;
        private int count = 1;
        private Sampler sampler = Sampler.DPM_PP_2M_KARRAS;

        private Builder(String prompt) {
            this.prompt = prompt;
        }

        /**
         * Sets what the image should not show.
         *
         * @param text the negative prompt
         * @return this builder
         */
        public Builder negativePrompt(String text) {
            this.negativePrompt = text;
            return this;
        }

        /**
         * Sets the size.
         *
         * @param width the width, a multiple of 64
         * @param height the height, a multiple of 64
         * @return this builder
         */
        public Builder size(int width, int height) {
            this.width = width;
            this.height = height;
            return this;
        }

        /**
         * Sets the number of denoising steps.
         *
         * @param steps the steps
         * @return this builder
         */
        public Builder steps(int steps) {
            this.steps = steps;
            return this;
        }

        /**
         * Sets the guidance scale.
         *
         * @param guidance the scale; 1 or less turns guidance off
         * @return this builder
         */
        public Builder guidance(float guidance) {
            this.guidance = guidance;
            return this;
        }

        /**
         * Sets the seed.
         *
         * @param seed the seed of the first image
         * @return this builder
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * Sets how many images to make.
         *
         * @param count the number of images
         * @return this builder
         */
        public Builder count(int count) {
            this.count = count;
            return this;
        }

        /**
         * Sets the sampler.
         *
         * @param sampler the sampler
         * @return this builder
         */
        public Builder sampler(Sampler sampler) {
            this.sampler = sampler;
            return this;
        }

        /**
         * Builds the request.
         *
         * @return the request
         * @throws IllegalArgumentException if a value is out of range
         */
        public ImageRequest build() {
            return new ImageRequest(prompt, negativePrompt, width, height, steps, guidance, seed, count, sampler);
        }
    }
}
