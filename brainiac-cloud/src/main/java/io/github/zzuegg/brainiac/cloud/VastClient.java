package io.github.zzuegg.brainiac.cloud;

import io.github.zzuegg.brainiac.io.Json;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A client for the parts of Vast.ai's REST API that renting a GPU takes: searching offers,
 * renting one, watching the instance and destroying it.
 *
 * <p>The API key goes only into the {@code Authorization} header. It is never part of a URL,
 * an exception message or {@link #toString}.
 *
 * <pre>{@code
 * VastClient vast = new VastClient(VastKey.find().orElseThrow());
 * VastOffer offer = vast.offers(VastQuery.defaults()).getFirst();
 * long id = vast.rent(offer.id(), VastLaunch.defaults());
 * try {
 *     VastInstance ready = vast.awaitReady(id, Duration.ofMinutes(15));
 *     // ssh -p ready.sshPort() root@ready.sshHost() ...
 * } finally {
 *     vast.destroy(id);
 * }
 * }</pre>
 */
public final class VastClient {

    /** Vast's API. */
    public static final URI DEFAULT_BASE = URI.create("https://console.vast.ai/api/v0/");

    private final String key;
    private final URI base;
    private final HttpClient http;

    /**
     * Creates a client for Vast's API.
     *
     * @param apiKey the account's API key
     */
    public VastClient(String apiKey) {
        this(apiKey, DEFAULT_BASE, HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(20)).build());
    }

    /**
     * Creates a client for another address, such as a test server.
     *
     * @param apiKey the account's API key
     * @param base the API's base address, ending in {@code /}
     * @param http the HTTP client to send requests with
     */
    public VastClient(String apiKey, URI base, HttpClient http) {
        if (apiKey == null || apiKey.isBlank()) {
            throw new IllegalArgumentException("no Vast.ai API key");
        }
        this.key = apiKey.strip();
        this.base = Objects.requireNonNull(base);
        this.http = Objects.requireNonNull(http);
    }

    /**
     * Searches the offers, cheapest first.
     *
     * @param query what to look for
     * @return the matching offers
     * @throws IOException if the request fails
     */
    public List<VastOffer> offers(VastQuery query) throws IOException {
        Map<String, Object> r = call("POST", "bundles/", query.toJson());
        List<VastOffer> out = new ArrayList<>();
        for (Object o : Json.array(r.getOrDefault("offers", List.of()))) {
            out.add(VastOffer.from(Json.object(o)));
        }
        return out;
    }

    /**
     * Rents an offer. Billing starts now and runs until {@link #destroy}.
     *
     * @param offerId the offer, from {@link #offers}
     * @param launch what to run on it
     * @return the new instance's id
     * @throws IOException if the request fails or Vast refuses (the offer may be gone)
     */
    public long rent(long offerId, VastLaunch launch) throws IOException {
        Map<String, Object> r = call("PUT", "asks/" + offerId + "/", launch.toJson());
        Object id = r.get("new_contract");
        if (!Boolean.TRUE.equals(r.get("success")) || !(id instanceof Number n)) {
            throw new IOException("Vast did not rent offer " + offerId + ": " + Json.write(r));
        }
        return n.longValue();
    }

    /**
     * Looks up one of this account's instances.
     *
     * @param id the instance id
     * @return the instance
     * @throws IOException if the request fails or there is no such instance
     */
    public VastInstance instance(long id) throws IOException {
        Map<String, Object> r = call("GET", "instances/" + id + "/", null);
        for (VastInstance i : list(r)) {
            if (i.id() == id) {
                return i;
            }
        }
        throw new IOException("no instance " + id);
    }

    /**
     * Lists this account's instances, running or not. Each one costs money until it is destroyed.
     *
     * @return the instances
     * @throws IOException if the request fails
     */
    public List<VastInstance> instances() throws IOException {
        return list(call("GET", "instances/", null));
    }

    /** Reads instances from {"instances": [...]} or {"instances": {...}} (API v0 and v1). */
    private static List<VastInstance> list(Map<String, Object> r) {
        Object v = r.containsKey("instances") ? r.get("instances") : r.containsKey("results") ? r.get("results") : r;
        List<VastInstance> out = new ArrayList<>();
        if (v instanceof List<?> l) {
            for (Object o : l) {
                out.add(VastInstance.from(Json.object(o)));
            }
        } else if (v instanceof Map<?, ?> m && m.get("id") != null) {
            out.add(VastInstance.from(Json.object(m)));
        }
        return out;
    }

    /**
     * Waits until an instance runs and has an SSH address.
     *
     * @param id the instance id
     * @param timeout how long to wait
     * @return the running instance
     * @throws IOException if it does not get ready in time (it is not destroyed here)
     * @throws InterruptedException if the wait is interrupted
     */
    public VastInstance awaitReady(long id, Duration timeout) throws IOException, InterruptedException {
        long end = System.nanoTime() + timeout.toNanos();
        String last = "";
        while (true) {
            VastInstance i = instance(id);
            if (i.ready()) {
                return i;
            }
            String now = i.status() + " " + i.statusMessage();
            if (!now.equals(last)) {
                System.err.println("  instance " + id + ": " + (i.status().isEmpty() ? "starting" : i.status())
                        + (i.statusMessage().isBlank() ? "" : " - " + i.statusMessage().strip()));
                last = now;
            }
            if ("exited".equals(i.status()) || i.statusMessage().toLowerCase(java.util.Locale.ROOT).contains("error")) {
                throw new IOException("instance " + id + " failed to start: " + i);
            }
            if (System.nanoTime() > end) {
                throw new IOException("instance " + id + " was not ready after " + timeout.toMinutes() + " minutes: " + i);
            }
            Thread.sleep(5000);
        }
    }

    /**
     * Destroys an instance: it stops, its disk is deleted and billing ends.
     *
     * @param id the instance id
     * @throws IOException if the request fails
     */
    public void destroy(long id) throws IOException {
        call("DELETE", "instances/" + id + "/", null);
    }

    private Map<String, Object> call(String method, String path, Map<String, Object> body) throws IOException {
        return call(method, base.resolve(path), body, true);
    }

    /** Vast retires endpoints with HTTP 410 and names the new one: "Use /api/v1/instances/ instead." */
    private static final java.util.regex.Pattern MOVED = java.util.regex.Pattern.compile("Use (/api/v\\d+/[^\\s\"]*) instead");

    private Map<String, Object> call(String method, URI uri, Map<String, Object> body, boolean follow) throws IOException {
        String path = uri.getPath();
        HttpRequest.Builder b = HttpRequest.newBuilder(uri)
                .timeout(Duration.ofSeconds(60))
                .header("Authorization", "Bearer " + key)
                .header("Accept", "application/json");
        if (body != null) {
            b.header("Content-Type", "application/json")
                    .method(method, HttpRequest.BodyPublishers.ofString(Json.write(body)));
        } else {
            b.method(method, HttpRequest.BodyPublishers.noBody());
        }
        HttpResponse<String> r;
        try {
            r = http.send(b.build(), HttpResponse.BodyHandlers.ofString());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("interrupted", e);
        }
        String text = r.body() == null ? "" : r.body().strip();
        java.util.regex.Matcher moved = MOVED.matcher(text);
        if (r.statusCode() == 410 && follow && moved.find()) {
            // the same path under the API version the message names: /api/v1/instances/123/
            String target = moved.group(1).replaceFirst("^(/api/v\\d+/).*", "$1") + path.replaceFirst("^/api/v\\d+/", "");
            return call(method, uri.resolve(target + (uri.getRawQuery() == null ? "" : "?" + uri.getRawQuery())), body, false);
        }
        if (r.statusCode() / 100 != 2) {
            throw new IOException(method + " " + path + " failed: HTTP " + r.statusCode() + " "
                    + scrub(text.length() > 500 ? text.substring(0, 500) + "..." : text));
        }
        if (text.isEmpty()) {
            return new LinkedHashMap<>();
        }
        Object v = Json.parse(text);
        return v instanceof Map<?, ?> ? Json.object(v) : new LinkedHashMap<>(Map.of("value", v));
    }

    /** Keeps the key out of error messages, in case the server echoes it. */
    private String scrub(String text) {
        return text.replace(key, "<api key>");
    }

    @Override
    public String toString() {
        return "VastClient[" + base + "]";
    }
}
