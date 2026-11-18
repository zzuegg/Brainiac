package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.llm.OpenAiServer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.security.SecureRandom;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Serves one of this PC's chat models with the OpenAI API, so other programs (an editor, a
 * script, another tool that speaks that API) can use it: the core's {@link OpenAiServer} on a
 * port of its own. By default only programs on this PC reach it. Opened to the network, every
 * request needs a key the studio makes up for that run.
 *
 * <p>The served model is loaded for the server alone, apart from the one the studio's own chat
 * keeps, so serving the model you also chat with holds its weights twice.
 */
public final class LocalServing implements AutoCloseable {

    private final LocalProvider local;
    private OpenAiServer server;
    private String model = "";
    private String key = "";
    private boolean network;
    private String device = "";
    private long started;

    LocalServing(LocalProvider local) {
        this.local = local;
    }

    /**
     * Says whether a model is served and how to reach it.
     *
     * @return {@code {running, model, port, url, network, key, device, started}}; the key is
     *         empty unless the server is open to the network
     */
    public synchronized Map<String, Object> state() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("running", server != null);
        if (server != null) {
            out.put("model", model);
            out.put("port", server.port());
            out.put("url", "http://" + (network ? hostName() : "127.0.0.1") + ":" + server.port() + "/v1");
            out.put("network", network);
            out.put("key", key);
            out.put("device", device);
            out.put("started", started);
        }
        return out;
    }

    private static String hostName() {
        try {
            return InetAddress.getLocalHost().getHostName();
        } catch (IOException e) {
            return "this-pc";
        }
    }

    /**
     * Starts serving a model, in place of the one served before.
     *
     * @param name the model as this PC lists it
     * @param port the port to listen on; 0 picks a free one
     * @param openToNetwork whether other computers may reach it, with the key
     * @return the state, as {@link #state} gives it
     * @throws IOException if there is no such chat model, it cannot be loaded, or the port is taken
     */
    public synchronized Map<String, Object> start(String name, int port, boolean openToNetwork) throws IOException {
        if (port < 0 || port > 65535 || (port > 0 && port < 1024)) {
            throw new IllegalArgumentException("the port must be between 1024 and 65535");
        }
        stop();
        LocalProvider.Loaded loaded;
        try {
            loaded = local.open(name);
        } catch (RuntimeException e) {
            throw new IOException("cannot load " + name + ": " + e.getMessage(), e);
        }
        InetSocketAddress address = openToNetwork ? new InetSocketAddress(port)
                : new InetSocketAddress(InetAddress.getLoopbackAddress(), port);
        OpenAiServer s;
        try {
            s = OpenAiServer.start(address, name, loaded.generator(), loaded.tokenizer(), loaded.stops());
        } catch (UncheckedIOException e) {
            throw new IOException("cannot listen on port " + port + ": " + e.getCause().getMessage()
                    + ". Another program may be using it.", e);
        }
        key = "";
        if (openToNetwork) {
            byte[] random = new byte[24];
            new SecureRandom().nextBytes(random);
            key = "sk-brainiac-" + HexFormat.of().formatHex(random);
            s.requireApiKey(key);
        }
        server = s;
        model = name;
        network = openToNetwork;
        device = loaded.device().name();
        started = System.currentTimeMillis();
        return state();
    }

    /**
     * Stops serving and lets the model go.
     *
     * @return whether a model was served
     */
    public synchronized boolean stop() {
        if (server == null) {
            return false;
        }
        server.close();
        server = null;
        return true;
    }

    @Override
    public void close() {
        stop();
    }
}
