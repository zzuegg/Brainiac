/**
 * Video generators and world models, written as core code: factorised space-time diffusion
 * denoisers, a causal 3-D video tokenizer with multi-task masked token modelling, frame-level
 * autoregressive video diffusion trained with Self Forcing, history guidance for
 * diffusion-forcing video models, a discrete-token transformer world model (IRIS), planning
 * in a frozen feature space (DINO-WM), object-centric slot dynamics (SlotFormer), and the
 * frozen object counter and clip statistics that score generated videos. The older world
 * models ({@code DreamerAgent}, {@code RecurrentStateSpaceModel}, {@code JepaWorldModel},
 * {@code DiffusionWorldModel}, {@code DiffusionForcing}) stay in their packages.
 */
package io.github.zzuegg.brainiac.experimental.world;
