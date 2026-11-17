package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;

/**
 * Vast.ai, where GPU machines are rented by the hour: the user's account, the machines they rent,
 * and the offers to rent. This only reads; nothing here rents or bills anything.
 *
 * <p>The key is kept with the others in {@code keys.properties} under {@code vastai} (or comes from
 * {@code VAST_API_KEY}) and is sent only to the Vast.ai console host.
 */
public final class VastAi {

    /** The Vast.ai REST API. */
    public static final URI BASE = URI.create("https://console.vast.ai/api/v0/");

    /** The provider id keys are saved under. */
    public static final String PROVIDER = "vastai";

    private static final int MAX_OFFERS = 40;

    private final Keys keys;
    private final URI base;
    private final HttpClient http;

    /**
     * Creates the client.
     *
     * @param keys where the key is kept
     * @param base the API's address, normally {@link #BASE}
     */
    public VastAi(Keys keys, URI base) {
        this.keys = keys;
        this.base = base.toString().endsWith("/") ? base : URI.create(base + "/");
        this.http = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15))
                .followRedirects(HttpClient.Redirect.NEVER).build();
    }

    /**
     * Describes the key, the account's balance and the machines the user rents.
     *
     * @return {@code {key: {label, source, hint} or null, balance?, instances?, error?}}
     */
    public Map<String, Object> status() {
        Map<String, Object> out = new LinkedHashMap<>();
        List<Keys.Entry> list = keys.list(PROVIDER);
        if (list.isEmpty()) {
            out.put("key", null);
            return out;
        }
        Keys.Entry e = list.getFirst();
        out.put("key", Map.of("label", e.label(), "source", e.source(), "hint", e.hint()));
        try {
            Map<String, Object> user = Json.object(get("users/current"));
            Object credit = user.get("credit") instanceof Number ? user.get("credit") : user.get("balance");
            if (credit instanceof Number n) {
                out.put("balance", n.doubleValue());
            }
            List<Object> instances = new ArrayList<>();
            for (Object o : Json.array(Json.object(get("instances")).getOrDefault("instances", List.of()))) {
                Map<String, Object> i = Json.object(o);
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("id", i.get("id"));
                r.put("label", i.get("label"));
                r.put("status", Json.string(i, "actual_status", Json.string(i, "cur_state", "")));
                r.put("gpu", gpu(i));
                r.put("gpus", i.get("num_gpus"));
                r.put("pricePerHour", i.get("dph_total"));
                r.put("location", i.get("geolocation"));
                instances.add(r);
            }
            out.put("instances", instances);
        } catch (IOException e2) {
            out.put("error", e2.getMessage());
        } catch (InterruptedException e2) {
            Thread.currentThread().interrupt();
            out.put("error", "interrupted");
        }
        return out;
    }

    /**
     * Finds machines to rent, cheapest first: verified, rentable, on-demand offers.
     *
     * @param gpu words in the GPU's name, such as {@code 4090}, or empty for any
     * @param minGpuRamGb the least memory per GPU, in GB, or 0
     * @param gpus the least number of GPUs, or 0
     * @param maxPricePerHour the most dollars an hour, or 0 for any
     * @return the offers, at most 40
     * @throws IOException if Vast.ai cannot be asked or refuses
     * @throws InterruptedException if interrupted while waiting
     */
    public List<Map<String, Object>> offers(String gpu, double minGpuRamGb, int gpus, double maxPricePerHour)
            throws IOException, InterruptedException {
        Map<String, Object> q = new LinkedHashMap<>();
        q.put("verified", Map.of("eq", true));
        q.put("rentable", Map.of("eq", true));
        q.put("rented", Map.of("eq", false));
        q.put("type", "ondemand");
        q.put("order", List.of(List.of("dph_total", "asc")));
        q.put("limit", 400);
        if (minGpuRamGb > 0) {
            q.put("gpu_ram", Map.of("gte", minGpuRamGb * 1024));   // Vast.ai counts MiB
        }
        if (gpus > 0) {
            q.put("num_gpus", Map.of("gte", gpus));
        }
        if (maxPricePerHour > 0) {
            q.put("dph_total", Map.of("lte", maxPricePerHour));
        }
        Object answer = send(HttpRequest.newBuilder(base.resolve("bundles/"))
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(q))).header("Content-Type", "application/json"),
                "bundles");
        List<String> words = gpu == null || gpu.isBlank() ? List.of()
                : List.of(gpu.toLowerCase(Locale.ROOT).replace('_', ' ').strip().split("\\s+"));
        List<Map<String, Object>> out = new ArrayList<>();
        for (Object o : Json.array(Json.object(answer).getOrDefault("offers", List.of()))) {
            Map<String, Object> m = Json.object(o);
            String name = gpu(m);
            String lower = name.toLowerCase(Locale.ROOT);
            if (!words.stream().allMatch(lower::contains)) {
                continue;
            }
            Map<String, Object> r = new LinkedHashMap<>();
            r.put("id", m.get("id"));
            r.put("gpu", name);
            r.put("gpus", m.get("num_gpus"));
            r.put("gpuRamGb", m.get("gpu_ram") instanceof Number n ? Math.round(n.doubleValue() / 1024) : null);
            r.put("pricePerHour", m.get("dph_total"));
            Object rel = m.containsKey("reliability") ? m.get("reliability") : m.get("reliability2");
            r.put("reliability", rel);
            r.put("location", m.get("geolocation"));
            r.put("cuda", m.get("cuda_max_good"));
            r.put("cpuCores", m.get("cpu_cores_effective") != null ? m.get("cpu_cores_effective") : m.get("cpu_cores"));
            r.put("ramGb", m.get("cpu_ram") instanceof Number n ? Math.round(n.doubleValue() / 1024) : null);
            r.put("diskGb", m.get("disk_space"));
            r.put("downMbps", m.get("inet_down"));
            r.put("dlperf", m.get("dlperf"));
            out.add(r);
            if (out.size() == MAX_OFFERS) {
                break;
            }
        }
        return out;
    }

    private static String gpu(Map<String, Object> m) {
        return Json.string(m, "gpu_name", "?").replace('_', ' ');
    }

    private Object get(String path) throws IOException, InterruptedException {
        return send(HttpRequest.newBuilder(base.resolve(path + "/")).GET(), path);
    }

    private Object send(HttpRequest.Builder b, String what) throws IOException, InterruptedException {
        String key = key().orElseThrow(() -> new IOException("add your Vast.ai API key first"));
        // only to the Vast.ai API; redirects are not followed, so the key goes nowhere else
        HttpResponse<String> r = http.send(b.timeout(Duration.ofSeconds(30)).header("Authorization", "Bearer " + key)
                .header("Accept", "application/json").header("User-Agent", "brainiac-studio").build(),
                HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        int s = r.statusCode();
        if (s == 401 || s == 403) {
            throw new IOException("Vast.ai refused the key; check it at cloud.vast.ai/manage-keys");
        }
        if (s == 429) {
            throw new IOException("Vast.ai is limiting requests; wait a moment and try again");
        }
        if (s >= 300) {
            String msg = "";
            try {
                Map<String, Object> o = Json.object(Json.parse(r.body()));
                msg = Json.string(o, "msg", Json.string(o, "error", ""));
            } catch (RuntimeException e) {
                msg = "";
            }
            throw new IOException("Vast.ai answered " + s + " for " + what + (msg.isEmpty() ? "" : ": " + msg));
        }
        try {
            return Json.parse(r.body());
        } catch (RuntimeException e) {
            throw new IOException("Vast.ai sent something that is not JSON for " + what);
        }
    }

    private Optional<String> key() {
        List<Keys.Entry> list = keys.list(PROVIDER);
        return list.isEmpty() ? Optional.empty() : keys.get(PROVIDER, list.getFirst().label());
    }
}
