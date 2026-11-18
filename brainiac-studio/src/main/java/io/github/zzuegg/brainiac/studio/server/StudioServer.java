package io.github.zzuegg.brainiac.studio.server;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import io.github.zzuegg.brainiac.studio.agent.AgentRuns;
import io.github.zzuegg.brainiac.studio.agent.StudioTools;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.train.TrainingService;
import io.github.zzuegg.brainiac.studio.train.VastRentals;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.SecureRandom;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * The studio's HTTP server. It listens on the loopback address only, answers only requests
 * whose {@code Host} is this machine (which stops DNS-rebinding pages), and serves the API only
 * to callers that send the token created at launch.
 */
public final class StudioServer implements AutoCloseable {

    /** The request header that carries the session token. */
    public static final String TOKEN_HEADER = "X-Studio-Token";

    private static final System.Logger LOG = System.getLogger(StudioServer.class.getName());

    private final HttpServer http;
    private final ExecutorService executor;
    private final String token;
    private final Set<String> hosts;
    private final Map<String, Route> api = new LinkedHashMap<>();
    private final TrainingService training;
    private volatile VastRentals rentals;

    /** Answers one API call. */
    @FunctionalInterface
    public interface Route {
        /**
         * Handles the call.
         *
         * @param exchange the request and response
         * @throws IOException if writing the response fails
         */
        void handle(HttpExchange exchange) throws IOException;
    }

    private StudioServer(HttpServer http, ExecutorService executor, String token, TrainingService training) {
        this.http = http;
        this.training = training;
        this.executor = executor;
        this.token = token;
        int port = http.getAddress().getPort();
        this.hosts = Set.of("127.0.0.1:" + port, "localhost:" + port, "[::1]:" + port);
    }

    /**
     * Starts a server on the loopback address.
     *
     * @param port the port, or 0 for any free port
     * @param capabilities the capabilities to offer
     * @param chat the chat models
     * @return the running server
     * @throws IOException if the port cannot be opened
     */
    public static StudioServer start(int port, CapabilityRegistry capabilities, ChatService chat) throws IOException {
        byte[] secret = new byte[24];
        new SecureRandom().nextBytes(secret);
        String token = HexFormat.of().formatHex(secret);
        HttpServer http = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), port), 0);
        ExecutorService executor = Executors.newVirtualThreadPerTaskExecutor();
        http.setExecutor(executor);
        StudioServer server = new StudioServer(http, executor, token, new TrainingService(chat.folder()));
        WebAssets web = new WebAssets(token, chat.page()::json, chat.folder().toString());
        http.createContext("/", ex -> server.guard(ex, false, web::handle));
        http.createContext("/api/", ex -> server.guard(ex, true, server::routeApi));
        server.get("/api/health", ex -> Exchanges.json(ex, 200, Map.of("status", "ok", "name", "Brainiac Studio")));
        server.get("/api/capabilities", ex -> Exchanges.json(ex, 200, Map.of("capabilities", capabilities.toJson())));
        server.get("/api/system", ex -> Exchanges.json(ex, 200, SystemInfo.snapshot()));
        new ChatRoutes(chat).register(server);
        new BuilderRoutes(chat.folder()).register(server);
        new TrainRoutes(server.training).register(server);
        server.training.hosts().secrets(chat::secret);
        // the models the user's other machines serve show up as providers
        chat.hosts(() -> server.training.hosts().list().stream().map(h -> (io.github.zzuegg.brainiac.studio.chat.ChatProvider)
                new io.github.zzuegg.brainiac.studio.train.HostProvider(h, server.training.hosts().client(h))).toList());
        server.rentals = rentals(chat, server.training);
        new VastRoutes(chat.vast(), server.rentals).register(server);
        new AgentRoutes(new AgentRuns(chat, new StudioTools(chat, server.training), chat.folder().resolve("agents")))
                .register(server);
        http.start();
        server.rentals.recover();   // machines a killed studio left renting
        return server;
    }

    /**
     * Adds a {@code GET} endpoint under {@code /api/}.
     *
     * @param path the full path, such as {@code /api/models}
     * @param route the handler
     */
    public void get(String path, Route route) {
        if (!path.startsWith("/api/")) {
            throw new IllegalArgumentException("API paths start with /api/: " + path);
        }
        api.put("GET " + path, route);
    }

    /**
     * Adds a {@code POST} endpoint under {@code /api/}.
     *
     * @param path the full path, such as {@code /api/chat}
     * @param route the handler
     */
    public void post(String path, Route route) {
        if (!path.startsWith("/api/")) {
            throw new IllegalArgumentException("API paths start with /api/: " + path);
        }
        api.put("POST " + path, route);
    }

    /**
     * Returns the address to open in a browser.
     *
     * @return the address
     */
    public URI uri() {
        return URI.create("http://127.0.0.1:" + http.getAddress().getPort() + "/");
    }

    /**
     * Returns the token API callers must send in the {@value #TOKEN_HEADER} header.
     *
     * @return the token
     */
    public String token() {
        return token;
    }

    // machines rented on Vast.ai become hosts; renting needs the key and this checkout to send
    private static VastRentals rentals(ChatService chat, TrainingService training) {
        java.util.Optional<java.nio.file.Path> repo = VastRentals.findRepo(System.getenv());
        String unavailable = chat.vast() == null ? "Vast.ai is not set up in this studio"
                : repo.isEmpty() ? "renting needs the studio to run from a Brainiac checkout (or BRAINIAC_REPO naming one), "
                        + "which is what the machine builds"
                : !VastRentals.hasSsh(System.getenv()) ? "renting needs the ssh command (the OpenSSH client), "
                        + "which is not on this machine's PATH" : null;
        VastRentals.ClientSource client = () -> {
            if (chat.vast() == null) {
                throw new IOException("Vast.ai is not set up in this studio");
            }
            return chat.vast().client();
        };
        return new VastRentals(training.hosts(), VastRentals.vast(client, repo.orElse(null)),
                chat.folder().resolve("vast"), unavailable,
                chat.vast() == null ? io.github.zzuegg.brainiac.cloud.VastBadHosts.load() : chat.vast().badHosts());
    }

    @Override
    public void close() {
        http.stop(0);
        VastRentals r = rentals;
        if (r != null) {
            r.close();
        }
        training.close();
        executor.shutdownNow();
    }

    private void guard(HttpExchange ex, boolean needsToken, HttpHandler next) throws IOException {
        try {
            String host = ex.getRequestHeaders().getFirst("Host");
            if (host == null || !hosts.contains(host.toLowerCase(java.util.Locale.ROOT))) {
                Exchanges.error(ex, 403, "the studio only answers requests addressed to this machine");
                return;
            }
            if (needsToken && !tokenMatches(ex.getRequestHeaders().getFirst(TOKEN_HEADER))) {
                Exchanges.error(ex, 403, "missing or wrong " + TOKEN_HEADER + " header");
                return;
            }
            next.handle(ex);
        } catch (Exchanges.BadRequest | IllegalArgumentException e) {
            try {
                Exchanges.error(ex, 400, e.getMessage());
            } catch (IOException | RuntimeException ignored) {
                ex.close();
            }
        } catch (IOException | RuntimeException e) {
            LOG.log(System.Logger.Level.WARNING, "request " + ex.getRequestURI() + " failed", e);
            try {
                Exchanges.error(ex, 500, String.valueOf(e.getMessage()));
            } catch (IOException | RuntimeException ignored) {
                ex.close();   // the response had already started
            }
        }
    }

    private boolean tokenMatches(String given) {
        return given != null && MessageDigest.isEqual(given.getBytes(StandardCharsets.UTF_8),
                token.getBytes(StandardCharsets.UTF_8));
    }

    private void routeApi(HttpExchange ex) throws IOException {
        String method = "HEAD".equals(ex.getRequestMethod()) ? "GET" : ex.getRequestMethod();
        Route route = api.get(method + " " + ex.getRequestURI().getPath());
        if (route != null) {
            route.handle(ex);
            return;
        }
        boolean otherMethod = api.keySet().stream().anyMatch(k -> k.endsWith(" " + ex.getRequestURI().getPath()));
        Exchanges.error(ex, otherMethod ? 405 : 404, (otherMethod ? "method not allowed: " : "no such endpoint: ")
                + ex.getRequestMethod() + " " + ex.getRequestURI().getPath());
    }
}
