package io.github.zzuegg.brainiac.diffusion;

/**
 * Hears about each denoising step of a {@link StableDiffusion} run, and can stop it.
 *
 * <pre>{@code
 * sd.generate(request, (step, total, preview) -> {
 *     System.out.println(step + " / " + total);
 *     return !stopButton.pressed();
 * });
 * }</pre>
 */
@FunctionalInterface
public interface Progress {

    /** Progress that ignores every step and never stops. */
    Progress NONE = (step, total, preview) -> true;

    /**
     * Called after each step.
     *
     * @param step the steps done so far over all images of the request, from 1
     * @param total the steps of the whole request
     * @param preview a rough, cheap preview of the image being made, at 1/8 of its size, on
     *     steps that {@link #previewEvery()} asks for; otherwise {@code null}
     * @return {@code false} to stop: {@link StableDiffusion#generate} then throws
     *     {@link java.util.concurrent.CancellationException}
     */
    boolean step(int step, int total, GeneratedImage preview);

    /**
     * Returns how often to make previews. A preview costs a small download from the GPU and
     * a linear map of the latents, not a VAE decode.
     *
     * @return every how many steps to pass a preview; 0 for never (the default)
     */
    default int previewEvery() {
        return 0;
    }

    /**
     * Returns progress that passes previews every {@code n} steps to {@code progress}.
     *
     * @param n every how many steps
     * @param progress what hears the steps
     * @return the progress
     */
    static Progress withPreviews(int n, Progress progress) {
        return new Progress() {
            @Override
            public boolean step(int step, int total, GeneratedImage preview) {
                return progress.step(step, total, preview);
            }

            @Override
            public int previewEvery() {
                return n;
            }
        };
    }
}
