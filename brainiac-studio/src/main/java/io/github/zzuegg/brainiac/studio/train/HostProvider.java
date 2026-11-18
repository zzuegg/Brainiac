package io.github.zzuegg.brainiac.studio.train;

import io.github.zzuegg.brainiac.studio.chat.ChatListener;
import io.github.zzuegg.brainiac.studio.chat.ChatProvider;
import io.github.zzuegg.brainiac.studio.chat.ChatRequest;
import io.github.zzuegg.brainiac.studio.chat.ChatResult;
import io.github.zzuegg.brainiac.studio.chat.ModelInfo;
import java.io.IOException;
import java.util.List;

/**
 * The models a worker host serves, as a chat provider: another machine of the user's that runs
 * {@code brainiac-studio --worker}, at home or rented, answers with the studio's own engine.
 */
public final class HostProvider implements ChatProvider {

    /** What host provider ids start with. */
    public static final String PREFIX = "host-";

    private final Hosts.Host host;
    private final WorkerClient client;

    /**
     * Uses a host.
     *
     * @param host the host
     * @param client how to reach it
     */
    public HostProvider(Hosts.Host host, WorkerClient client) {
        this.host = host;
        this.client = client;
    }

    @Override
    public String id() {
        return PREFIX + host.id();
    }

    @Override
    public String name() {
        return host.name();
    }

    @Override
    public boolean ready() {
        return true;
    }

    @Override
    public List<ModelInfo> models() throws IOException {
        return client.models().stream().map(m -> new ModelInfo(id(), m.id(), m.name(), m.media(), m.created(),
                m.detail().isEmpty() ? "on " + host.name() : m.detail() + " · on " + host.name())).toList();
    }

    @Override
    public ChatResult chat(ChatRequest request, ChatListener listener) throws IOException {
        return client.chat(request, listener);
    }

    @Override
    public boolean takesGrammar() {
        return true;
    }

    @Override
    public boolean runsMedia() {
        return true;
    }

    @Override
    public List<io.github.zzuegg.brainiac.studio.chat.LocalImages.Picture> draw(String model, String prompt,
            io.github.zzuegg.brainiac.studio.chat.LocalImages.Settings settings,
            io.github.zzuegg.brainiac.studio.chat.LocalImages.Listener listener) throws IOException {
        return client.draw(model, prompt, settings, listener);
    }

    @Override
    public io.github.zzuegg.brainiac.studio.chat.Media speak(String model, String text, String voice, double speed,
            java.util.function.BooleanSupplier stop) throws IOException {
        return client.speak(model, text, voice, speed);
    }

    @Override
    public io.github.zzuegg.brainiac.audio.Transcription transcribe(String model, byte[] wav,
            io.github.zzuegg.brainiac.studio.chat.LocalSpeech.Listening listening,
            java.util.function.Consumer<io.github.zzuegg.brainiac.audio.Transcription.Segment> onSegment,
            java.util.function.BooleanSupplier stop) throws IOException {
        return client.transcribe(model, wav, listening, onSegment, stop);
    }

    @Override
    public boolean makesVideo() {
        return true;
    }

    @Override
    public java.util.Map<String, Object> videoOptions(String model) {
        return client.videoOptions(model);
    }

    @Override
    public io.github.zzuegg.brainiac.studio.chat.LocalVideos.Clip video(String model, String prompt,
            io.github.zzuegg.brainiac.studio.chat.LocalVideos.Settings settings,
            io.github.zzuegg.brainiac.studio.chat.LocalVideos.Listener listener) throws IOException {
        return client.video(model, prompt, settings, listener);
    }

    @Override
    public java.util.Map<String, Object> voiceOptions(String model) {
        java.util.Map<String, Object> told = client.voiceOptions(model);
        if (told == null) {
            return null;
        }
        // a new voice is made from a recording on this PC only, so a host's model does not offer it
        java.util.Map<String, Object> options = new java.util.LinkedHashMap<>(told);
        options.put("clone", false);
        return options;
    }
}
