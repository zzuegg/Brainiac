/**
 * Image generation with pretrained diffusion models: Stable Diffusion 1.x, 2.x and XL from
 * Hugging Face diffusers folders or single-file checkpoints, and FLUX.1 dev and schnell from
 * diffusers folders. The models run on brainiac-nn's diffusion layers (UNet, AutoencoderKL,
 * ClipTextEncoder, T5Encoder, FluxTransformer) and samplers, so the same layers
 * serve self-built networks; this module adds the tokenizer, weight loading and the
 * text-to-image pipeline.
 */
module io.github.zzuegg.brainiac.diffusion {
    requires transitive io.github.zzuegg.brainiac.nn;
    requires transitive io.github.zzuegg.brainiac.io;
    exports io.github.zzuegg.brainiac.diffusion;
}
