package io.github.zzuegg.brainiac.studio;

import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.Keys;
import io.github.zzuegg.brainiac.studio.server.CapabilityRegistry;
import io.github.zzuegg.brainiac.studio.server.StudioServer;
import io.github.zzuegg.brainiac.studio.spi.Capability;
import java.io.IOException;
import java.net.URI;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Brainiac Studio, a local web app for using, building and training models.
 *
 * <pre>{@code
 * ./gradlew :brainiac-studio:run                    # http://127.0.0.1:7860/
 * ./gradlew :brainiac-studio:run --args="--port 0"  # any free port
 * ./gradlew :brainiac-studio:run --args="--data D:/BrainiacData"  # keep everything there
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

    /** The environment variable that chooses the data folder when {@code --data} is not given. */
    public static final String DATA_VARIABLE = "BRAINIAC_STUDIO_DATA";

    /**
     * Where the studio looks for things.
     *
     * @param port the port, or 0 for any free port
     * @param modelFolders folders holding local models: Hugging Face model folders and {@code .gguf} files
     * @param data the folder everything the studio keeps goes to: API keys ({@code keys.properties},
     *     readable only by this account), provider choices, conversations ({@code chats}), what the
     *     page remembers ({@code page.json}), the usage log, and pictures and audio ({@code outputs})
     * @param environment where API keys not saved in the studio are looked up, such as
     *     {@code ANTHROPIC_API_KEY}, {@code OPENAI_API_KEY} and {@code GEMINI_API_KEY}
     */
    public record Settings(int port, List<Path> modelFolders, Path data, Map<String, String> environment) {

        /** Copies the lists and makes the data folder absolute. */
        public Settings {
            modelFolders = List.copyOf(modelFolders);
            environment = Map.copyOf(environment);
            data = data.toAbsolutePath().normalize();
        }

        /**
         * Returns the file saved API keys go to.
         *
         * @return {@code keys.properties} in the data folder
         */
        public Path keyFile() {
            return data.resolve("keys.properties");
        }

        /**
         * Returns the folder generated pictures and audio go to.
         *
         * @return {@code outputs} in the data folder
         */
        public Path outputs() {
            return data.resolve("outputs");
        }

        /**
         * Returns the usual settings: port {@value Studio#DEFAULT_PORT}; models in {@code ./models},
         * {@code ~/.brainiac/models} and {@code $BRAINIAC_MODELS}; everything the studio keeps in
         * {@code $BRAINIAC_STUDIO_DATA}, or else {@code ~/.brainiac/studio}; keys also from the environment.
         *
         * @return the settings
         */
        public static Settings defaults() {
            Path home = Path.of(System.getProperty("user.home"), ".brainiac");
            List<Path> folders = new ArrayList<>(List.of(Path.of("models").toAbsolutePath(), home.resolve("models")));
            String env = System.getenv("BRAINIAC_MODELS");
            if (env != null && !env.isBlank()) {
                folders.add(0, Path.of(env).toAbsolutePath());
            }
            String data = System.getenv(DATA_VARIABLE);
            return new Settings(DEFAULT_PORT, folders,
                    data == null || data.isBlank() ? home.resolve("studio") : Path.of(data), System.getenv());
        }

        /**
         * Returns these settings on another port.
         *
         * @param port the port, or 0 for any free port
         * @return the settings
         */
        public Settings withPort(int port) {
            return new Settings(port, modelFolders, data, environment);
        }

        /**
         * Returns these settings with another data folder.
         *
         * @param folder where everything the studio keeps goes
         * @return the settings
         */
        public Settings withData(Path folder) {
            return new Settings(port, modelFolders, folder, environment);
        }

        /**
         * Returns these settings with other model folders.
         *
         * @param folders the folders
         * @return the settings
         */
        public Settings withModelFolders(List<Path> folders) {
            return new Settings(port, folders, data, environment);
        }
    }

    /**
     * Starts the studio with the usual settings on a port.
     *
     * @param port the port, or 0 for any free port
     * @return the running studio
     * @throws IOException if the port cannot be opened
     */
    public static Studio start(int port) throws IOException {
        return start(Settings.defaults().withPort(port));
    }

    /**
     * Finds the capabilities and starts the server on this machine's loopback address.
     *
     * @param settings where to listen and where to look for models and keys
     * @return the running studio
     * @throws IOException if the port cannot be opened
     */
    public static Studio start(Settings settings) throws IOException {
        CapabilityRegistry capabilities = CapabilityRegistry.discover();
        Keys keys = new Keys(settings.keyFile(), settings.environment());
        // -Dstudio.<provider>.url=... points a provider at a proxy or gateway, such as
        // -Dstudio.openai.url=https://gateway.example/v1/
        Map<String, URI> bases = new java.util.HashMap<>();
        for (String name : System.getProperties().stringPropertyNames()) {
            if (name.startsWith("studio.") && name.endsWith(".url")) {
                bases.put(name.substring(7, name.length() - 4), URI.create(System.getProperty(name)));
            }
        }
        ChatService chat = ChatService.at(keys, settings.modelFolders(), bases, settings.outputs());
        return new Studio(StudioServer.start(settings.port(), capabilities, chat), capabilities);
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
     * @param args {@code --port <n>} to choose the port (0 for any free port),
     *     {@code --models <folder>} (repeatable) to look for models there instead of the usual folders,
     *     and {@code --data <folder>} to keep keys, conversations and everything else there
     * @throws IOException if the port cannot be opened
     */
    public static void main(String[] args) throws IOException {
        int port = DEFAULT_PORT;
        List<Path> folders = new ArrayList<>();
        Path data = null;
        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--port" -> {
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException("--port needs a number");
                    }
                    port = Integer.parseInt(args[++i]);
                }
                case "--models" -> {
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException("--models needs a folder");
                    }
                    folders.add(Path.of(args[++i]).toAbsolutePath());
                }
                case "--data" -> {
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException("--data needs a folder");
                    }
                    data = Path.of(args[++i]);
                }
                case "--help", "-h" -> {
                    System.out.println("Usage: studio [--port <n>] [--models <folder>]... [--data <folder>]");
                    System.out.println("  --port    default " + DEFAULT_PORT + ", 0 = any free port");
                    System.out.println("  --models  where local models are (default ./models and ~/.brainiac/models)");
                    System.out.println("  --data    where keys, conversations, settings and outputs are kept");
                    System.out.println("            (default $" + DATA_VARIABLE + ", else ~/.brainiac/studio)");
                    return;
                }
                default -> throw new IllegalArgumentException("unknown option: " + args[i] + " (try --help)");
            }
        }
        Settings settings = Settings.defaults().withPort(port);
        if (!folders.isEmpty()) {
            settings = settings.withModelFolders(folders);
        }
        if (data != null) {
            settings = settings.withData(data);
        }
        Studio studio = start(settings);
        Runtime.getRuntime().addShutdownHook(new Thread(studio::close));
        long available = studio.capabilities().stream().filter(c -> c.status() == Capability.Status.AVAILABLE).count();
        System.out.println("Brainiac Studio is running at " + studio.uri());
        System.out.println("Keeping keys, conversations and settings in " + settings.data());
        System.out.println(available + " capabilities available, " + (studio.capabilities().size() - available)
                + " planned. Only this machine can connect. Press Ctrl+C to stop.");
    }
}
