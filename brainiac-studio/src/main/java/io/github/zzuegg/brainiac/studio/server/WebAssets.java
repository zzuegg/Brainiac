package io.github.zzuegg.brainiac.studio.server;

import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Serves the web page from this module's resources. Only files directly inside the web
 * folder are served, and the page itself gets the session token in a meta tag.
 */
final class WebAssets {

    private static final String ROOT = "/io/github/zzuegg/brainiac/studio/web/";
    private static final Map<String, String> TYPES = Map.of(
            "html", "text/html; charset=utf-8",
            "css", "text/css; charset=utf-8",
            "js", "text/javascript; charset=utf-8",
            "svg", "image/svg+xml",
            "png", "image/png",
            "json", "application/json; charset=utf-8",
            "woff2", "font/woff2");
    // the page loads its own files, Google Fonts, and draws generated media from blob: and data: URLs
    private static final String CSP = "default-src 'self'; script-src 'self'; connect-src 'self'; "
            + "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com; "
            + "img-src 'self' data: blob:; media-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'; form-action 'none'";

    private final String token;
    private final Supplier<String> state;
    private final String folder;

    /**
     * @param token the session token
     * @param state the page state as JSON safe inside a script element
     * @param folder where the studio keeps everything, shown on the page
     */
    WebAssets(String token, Supplier<String> state, String folder) {
        this.token = token;
        this.state = state;
        this.folder = folder;
    }

    void handle(HttpExchange ex) throws IOException {
        String method = ex.getRequestMethod();
        if (!"GET".equals(method) && !"HEAD".equals(method)) {
            Exchanges.error(ex, 405, "only GET");
            return;
        }
        String path = ex.getRequestURI().getPath();
        String name = path.equals("/") || path.equals("/index.html") ? "index.html" : path.substring(1);
        if (!name.matches("[A-Za-z0-9][A-Za-z0-9_-]*\\.[a-z0-9]+")) {
            Exchanges.error(ex, 404, "not found: " + path);
            return;
        }
        String type = TYPES.get(name.substring(name.lastIndexOf('.') + 1));
        byte[] body = read(name);
        if (type == null || body == null) {
            Exchanges.error(ex, 404, "not found: " + path);
            return;
        }
        ex.getResponseHeaders().set("Content-Security-Policy", CSP);
        ex.getResponseHeaders().set("Cache-Control", "no-cache");
        if (name.equals("index.html")) {
            body = page(new String(body, StandardCharsets.UTF_8)).getBytes(StandardCharsets.UTF_8);
        }
        Exchanges.send(ex, 200, type, body);
    }

    /**
     * Wraps {@code index.html}, which holds only the page's content, in a full document. The
     * part before the first {@code <svg} (title, meta and stylesheet links) goes in the head.
     */
    String page(String content) {
        int split = content.indexOf("<svg");
        String head = split < 0 ? "" : content.substring(0, split);
        String body = split < 0 ? content : content.substring(split);
        return "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                + "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
                + "<meta name=\"studio-token\" content=\"" + token + "\">\n"
                + "<meta name=\"studio-folder\" content=\"" + attr(folder) + "\">\n"
                + "<script type=\"application/json\" id=\"studio-state\">" + state.get() + "</script>\n"
                + "<style>[hidden]{display:none!important}</style>\n"
                + head + "</head>\n<body>\n" + body + "</body>\n</html>\n";
    }

    private static String attr(String s) {
        return s.replace("&", "&amp;").replace("\"", "&quot;").replace("<", "&lt;").replace(">", "&gt;");
    }

    static byte[] read(String name) {
        try (InputStream in = WebAssets.class.getResourceAsStream(ROOT + name)) {
            return in == null ? null : in.readAllBytes();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
