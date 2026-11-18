package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CancellationException;

/**
 * A loaded text-to-image model, whichever family: {@link StableDiffusion} or {@link Flux}.
 * {@link #load} picks the right one for a folder or file.
 *
 * <pre>{@code
 * try (ImageGenerator model = ImageGenerator.load(Path.of("FLUX.1-schnell"), Device.best(), DType.BFLOAT16)) {
 *     ImageGenerator.Defaults d = model.defaults();
 *     List<GeneratedImage> images = model.generate(ImageRequest.of("a red fox in fresh snow")
 *             .steps(d.steps()).guidance(d.guidance()).build(), Progress.NONE);
 * }
 * }</pre>
 */
public interface ImageGenerator extends AutoCloseable {

    /**
     * The settings a model works best with.
     *
     * @param steps the sampling steps
     * @param guidance the guidance scale (classifier-free guidance for Stable Diffusion, the
     *     embedded guidance for Flux dev; ignored by Flux schnell)
     * @param negativePrompt whether the model uses a negative prompt
     * @param samplers whether {@link ImageRequest#sampler()} is used; flow models always step
     *     with Euler
     */
    record Defaults(int steps, float guidance, boolean negativePrompt, boolean samplers) {
    }

    /**
     * Says which model family a folder or file holds, without loading it.
     *
     * @param path a diffusers folder or a single {@code .safetensors} checkpoint
     * @return a name such as {@code SDXL} or {@code FLUX_DEV}, or empty if this library cannot run it
     */
    static Optional<String> detect(Path path) {
        Optional<Flux.Kind> flux = Flux.detect(path);
        if (flux.isPresent()) {
            return Optional.of("FLUX_" + flux.get());
        }
        return StableDiffusion.detect(path).map(Enum::name);
    }

    /**
     * Loads a model of any supported family, with each family's default options.
     *
     * @param path a diffusers folder or a single {@code .safetensors} checkpoint
     * @param device where to run it
     * @param dtype the type of the networks: {@link DType#BFLOAT16} or {@link DType#FLOAT16}
     *     on a GPU (Flux needs bfloat16 there), {@link DType#FLOAT32} on the CPU
     * @return the model
     * @throws IOException if the files cannot be read
     * @throws IllegalArgumentException if they are not a model this library can run
     */
    static ImageGenerator load(Path path, Device device, DType dtype) throws IOException {
        if (Files.isDirectory(path) && Flux.detect(path).isPresent()) {
            return Flux.load(path, device, dtype);
        }
        return StableDiffusion.load(path, device, dtype);
    }

    /**
     * Returns the model family, as {@link #detect} names it.
     *
     * @return the family
     */
    String family();

    /**
     * Returns the settings the model works best with.
     *
     * @return the defaults
     */
    Defaults defaults();

    /**
     * Returns the device the model runs on.
     *
     * @return the device
     */
    Device device();

    /**
     * Returns the side of the square the model was trained at.
     *
     * @return the side in pixels
     */
    int nativeSize();

    /**
     * Returns a size of the given shape that suits this model.
     *
     * @param ratio the shape
     * @return {@code {width, height}}
     */
    default int[] size(AspectRatio ratio) {
        return ratio.size(nativeSize());
    }

    /**
     * Generates images.
     *
     * @param request what to generate
     * @param progress hears each step and can stop the run
     * @return the images, {@link ImageRequest#count()} of them
     * @throws CancellationException if {@code progress} asked to stop
     */
    List<GeneratedImage> generate(ImageRequest request, Progress progress);

    /** Frees the model's device memory and closes its files. */
    @Override
    void close();
}
