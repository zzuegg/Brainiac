package io.github.zzuegg.brainiac.studio;

import io.github.zzuegg.brainiac.studio.server.CapabilityRegistry;
import io.github.zzuegg.brainiac.studio.server.StudioServer;
import io.github.zzuegg.brainiac.studio.spi.Capability;
import java.io.IOException;
import java.net.URI;
import java.util.List;

/**
 * Brainiac Studio, a local web app for using, building and training models.
 *
 * <pre>{@code
 * ./gradlew :brainiac-studio:run                    # http://127.0.0.1:7860/
 * ./gradlew :brainiac-studio:run --args="--port 0"  # any free port
 * }</pre>
 *
 * <p>Or embedded: {@code try (Studio studio = Studio.start(7860)) { ... }}.
 */
public final class Studio implements AutoCloseable {

    /** The port used when none is given. */
    public static final int DEFAULT_PORT = 7860;

    private final StudioServer server;
    private final CapabilityRegistry capabilities;

    private Studio(StudioServer server, CapabilityRegistry capabilities) {
        this.server = server;
        this.capabilities = capabilities;
    }

    /**
     * Finds the capabilities and starts the server on this machine's loopback address.
     *
     * @param port the port, or 0 for any free port
     * @return the running studio
     * @throws IOException if the port cannot be opened
     */
    public static Studio start(int port) throws IOException {
        CapabilityRegistry capabilities = CapabilityRegistry.discover();
        return new Studio(StudioServer.start(port, capabilities), capabilities);
    }

    /**
     * Returns the address to open in a browser.
     *
     * @return the address
     */
    public URI uri() {
        return server.uri();
    }

    /**
     * Returns the token that API callers send in the {@code X-Studio-Token} header. The page
     * the studio serves already carries it.
     *
     * @return the token
     */
    public String token() {
        return server.token();
    }

    /**
     * Returns the capabilities found at startup.
     *
     * @return the capabilities
     */
    public List<Capability> capabilities() {
        return capabilities.all();
    }

    /** Stops the server. */
    @Override
    public void close() {
        server.close();
    }

    /**
     * Starts the studio and keeps it running until the process ends.
     *
     * @param args {@code --port <n>} to choose the port (0 for any free port)
     * @throws IOException if the port cannot be opened
     */
    public static void main(String[] args) throws IOException {
        int port = DEFAULT_PORT;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--port" -> {
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException("--port needs a number");
                    }
                    port = Integer.parseInt(args[++i]);
                }
                case "--help", "-h" -> {
                    System.out.println("Usage: studio [--port <n>]   (default " + DEFAULT_PORT + ", 0 = any free port)");
                    return;
                }
                default -> throw new IllegalArgumentException("unknown option: " + args[i] + " (try --help)");
            }
        }
        Studio studio = start(port);
        Runtime.getRuntime().addShutdownHook(new Thread(studio::close));
        long available = studio.capabilities().stream().filter(c -> c.status() == Capability.Status.AVAILABLE).count();
        System.out.println("Brainiac Studio is running at " + studio.uri());
        System.out.println(available + " capabilities available, " + (studio.capabilities().size() - available)
                + " planned. Only this machine can connect. Press Ctrl+C to stop.");
    }
}
