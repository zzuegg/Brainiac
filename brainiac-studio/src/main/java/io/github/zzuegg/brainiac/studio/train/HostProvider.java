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
}
