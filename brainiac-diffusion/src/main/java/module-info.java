/**
 * Image generation with pretrained diffusion models: Stable Diffusion 1.x, 2.x and XL from
 * Hugging Face diffusers folders or single-file checkpoints, with the CLIP text encoders,
 * the UNet, the VAE and the usual samplers.
 */
module io.github.zzuegg.brainiac.diffusion {
    requires transitive io.github.zzuegg.brainiac.tensor;
    requires transitive io.github.zzuegg.brainiac.io;
    exports io.github.zzuegg.brainiac.diffusion;
}
