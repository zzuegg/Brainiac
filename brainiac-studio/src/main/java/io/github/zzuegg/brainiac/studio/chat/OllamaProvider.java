package io.github.zzuegg.brainiac.studio.chat;

import java.net.URI;
import java.util.List;
import java.util.Locale;

/**
 * Ollama, through its OpenAI-compatible API: the Ollama server on this PC
 * ({@code http://localhost:11434}) needs no key, and Ollama Cloud ({@code https://ollama.com})
 * takes one. Its models take tools natively, so agents can run on it like on any provider.
 */
final class OllamaProvider extends CompatibleProvider {

    /** Where Ollama listens on this PC. */
    static final URI LOCAL = URI.create("http://localhost:11434/v1/");

    /** Ollama's preset: the server on this PC unless the user points it elsewhere. */
    static final Preset PRESET = new Preset("ollama", "Ollama", LOCAL, List.of("OLLAMA_API_KEY"), "max_tokens", true,
            "none", "models served by Ollama on this PC (no key), another machine, or Ollama Cloud (with a key)");

    OllamaProvider(Keys keys, Preset preset) {
        super(keys, preset);
    }

    @Override
    boolean keyless() {
        return onThisPc();
    }

    @Override
    public boolean local() {
        return onThisPc();
    }

    private boolean onThisPc() {
        String host = base().getHost();
        return host != null && (host.equals("localhost") || host.equals("127.0.0.1") || host.equals("::1")
                || host.equals("[::1]"));
    }

    @Override
    List<String> media(String id) {
        String s = id.toLowerCase(Locale.ROOT);
        if (s.contains("embed")) {
            return List.of();
        }
        return s.contains("llava") || s.contains("-vl") || s.contains("vision") || s.contains("gemma3")
                || s.contains("moondream") || s.contains("minicpm-v") ? List.of("text", "vision") : List.of("text");
    }

    @Override
    public boolean nativeTools(String model) {
        return true;
    }
}
