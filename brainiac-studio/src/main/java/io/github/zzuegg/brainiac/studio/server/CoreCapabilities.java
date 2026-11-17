package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.studio.spi.Capability;
import io.github.zzuegg.brainiac.studio.spi.CapabilityProvider;
import io.github.zzuegg.brainiac.studio.spi.Medium;
import java.util.ArrayList;
import java.util.List;

/**
 * The capabilities of Brainiac's own modules, each listed only when its module is present,
 * plus placeholders for what the studio is designed for but nothing provides yet.
 */
public final class CoreCapabilities implements CapabilityProvider {

    /** Creates the provider; called by {@link java.util.ServiceLoader}. */
    public CoreCapabilities() {
    }

    @Override
    public List<Capability> capabilities() {
        List<Capability> out = new ArrayList<>();
        if (present("io.github.zzuegg.brainiac.llm.Llama")) {
            out.add(Capability.available("text.chat", Medium.TEXT, "Chat",
                    "Chat with Llama, Mistral and Qwen models from Hugging Face folders or GGUF files, streamed",
                    "brainiac-llm"));
            out.add(Capability.available("text.complete", Medium.TEXT, "Complete",
                    "Raw completion with sampling, beam search and speculative decoding", "brainiac-llm"));
            out.add(Capability.available("model.finetune", Medium.TRAINING, "Fine-tune",
                    "LoRA and QLoRA on Llama-family models", "brainiac-llm"));
        }
        if (present("io.github.zzuegg.brainiac.nn.diffusion.Diffusion")) {
            out.add(Capability.available("image.generate", Medium.IMAGE, "Generate images",
                    "Diffusion (DDPM, DDIM) and flow-matching samplers for denoisers built here", "brainiac-nn")
                    .withNote("No pretrained text-to-image weights yet"));
        }
        if (present("io.github.zzuegg.brainiac.nn.Model")) {
            out.add(Capability.available("model.build", Medium.TRAINING, "Build models",
                    "Layer blocks wired into graphs, with shapes checked as you connect them", "brainiac-nn"));
            out.add(Capability.available("model.quantize", Medium.TRAINING, "Quantize",
                    "Post-training and quantization-aware weight quantization", "brainiac-nn"));
        }
        if (present("io.github.zzuegg.brainiac.train.Trainer")) {
            out.add(Capability.available("model.train", Medium.TRAINING, "Train",
                    "Optimizers, schedules, checkpoints and metrics", "brainiac-train"));
        }
        if (present("io.github.zzuegg.brainiac.rl.Ppo")) {
            out.add(Capability.available("rl.train", Medium.TRAINING, "Reinforcement learning",
                    "PPO, A2C, DQN, REINFORCE, Q-learning and SARSA on environments", "brainiac-rl"));
        }
        if (present("io.github.zzuegg.brainiac.evolve.Evolution")) {
            out.add(Capability.available("evolve.search", Medium.TRAINING, "Evolve",
                    "Neuroevolution of wiring, neuron counts and activations", "brainiac-evolve"));
        }
        ModuleLayer layer = layer();
        layer.findModule("io.github.zzuegg.brainiac.experiments").ifPresent(m -> out.add(
                Capability.available("experimental.blocks", Medium.TRAINING, "Experimental blocks",
                        "Research building blocks from brainiac.experimental", "brainiac-experiments")
                        .withNote(m.getPackages().size() + " packages, untested")));

        out.add(Capability.planned("speech.synthesize", Medium.SPEECH, "Text to speech",
                "Speak text with a chosen voice, pace and emotion"));
        out.add(Capability.planned("speech.transcribe", Medium.SPEECH, "Transcribe",
                "Turn speech into text with timings"));
        out.add(Capability.planned("vision.describe", Medium.VISION, "Describe images",
                "Answer questions about pictures and video frames"));
        out.add(Capability.planned("agent.run", Medium.AGENT, "Agents",
                "An orchestrator that plans and hands each part to an agent with that medium's tools"));
        return out;
    }

    static boolean present(String className) {
        try {
            Class.forName(className, false, CoreCapabilities.class.getClassLoader());
            return true;
        } catch (ClassNotFoundException | LinkageError e) {
            return false;
        }
    }

    static ModuleLayer layer() {
        ModuleLayer layer = CoreCapabilities.class.getModule().getLayer();
        return layer != null ? layer : ModuleLayer.boot();
    }
}
