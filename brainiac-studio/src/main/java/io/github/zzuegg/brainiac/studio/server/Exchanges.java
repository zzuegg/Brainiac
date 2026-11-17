package io.github.zzuegg.brainiac.studio.server;

import com.sun.net.httpserver.HttpExchange;
import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;

/** Small helpers for answering requests. */
final class Exchanges {

    private Exchanges() {
    }

    static void json(HttpExchange ex, int status, Object body) throws IOException {
        ex.getResponseHeaders().set("Cache-Control", "no-store");
        send(ex, status, "application/json; charset=utf-8", Json.write(body).getBytes(StandardCharsets.UTF_8));
    }

    static void error(HttpExchange ex, int status, String message) throws IOException {
        json(ex, status, Map.of("error", message));
    }

    /** Reads a JSON object body of at most {@code limit} bytes. */
    static Map<String, Object> body(HttpExchange ex, int limit) throws IOException {
        byte[] bytes;
        try (InputStream in = ex.getRequestBody()) {
            bytes = in.readNBytes(limit + 1);
        }
        if (bytes.length > limit) {
            throw new BadRequest("the request is larger than " + limit + " bytes");
        }
        try {
            return Json.object(Json.parse(new String(bytes, StandardCharsets.UTF_8)));
        } catch (IllegalArgumentException e) {
            throw new BadRequest("the request is not a JSON object: " + e.getMessage());
        }
    }

    /** A request the caller got wrong, answered with 400. */
    static final class BadRequest extends RuntimeException {
        private static final long serialVersionUID = 1L;

        BadRequest(String message) {
            super(message);
        }
    }

    static void send(HttpExchange ex, int status, String contentType, byte[] body) throws IOException {
        ex.getResponseHeaders().set("Content-Type", contentType);
        ex.getResponseHeaders().set("X-Content-Type-Options", "nosniff");
        ex.getResponseHeaders().set("Referrer-Policy", "no-referrer");
        boolean head = "HEAD".equals(ex.getRequestMethod());
        ex.sendResponseHeaders(status, head ? -1 : body.length == 0 ? -1 : body.length);
        if (!head && body.length > 0) {
            try (OutputStream out = ex.getResponseBody()) {
                out.write(body);
            }
        }
        ex.close();
    }
}
