package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.Tools;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.project.Project;
import io.github.zzuegg.brainiac.studio.project.Projects;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeParseException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;

/**
 * Tools that post under the user's name: Mastodon, Bluesky, a Discord webhook or any webhook.
 * Posting always asks the user first; a post scheduled for later is approved when it is
 * scheduled, and goes out from the studio at its time.
 */
public final class PublishTools implements AutoCloseable {

    /** The channels that can be posted to. */
    public static final List<String> CHANNELS = List.of("mastodon", "bluesky", "discord", "webhook");

    private static final HttpClient HTTP = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(15))
            .followRedirects(HttpClient.Redirect.NORMAL).build();
    private static final long MEDIA_BYTES = 40L << 20;

    private final Services services;
    private final Path queue;
    private final Projects projects;
    private final Thread scheduler;
    private volatile boolean closed;

    /**
     * Creates the tools and starts sending scheduled posts.
     *
     * @param services the accounts
     * @param queue the file scheduled posts are kept in
     * @param projects the projects, to find their folders, or null
     */
    public PublishTools(Services services, Path queue, Projects projects) {
        this.services = services;
        this.queue = queue;
        this.projects = projects;
        this.scheduler = Thread.ofVirtual().name("studio-scheduled-posts").start(this::loop);
    }

    @Override
    public void close() {
        closed = true;
        scheduler.interrupt();
    }

    @Tools.Tool(name = "publish_post", value = "Posts to Mastodon, Bluesky, Discord or a webhook under the user's "
            + "name, with pictures from the project's folder. Give a content calendar entry's id to post that entry, "
            + "which is then marked posted. Always asks the user first.")
    @Tier(value = Risk.OUTWARD, what = "posts publicly under your name")
    public Map<String, Object> publishPost(@Tools.Param("mastodon, bluesky, discord or webhook") String channel,
            @Tools.Param("the post's text; the entry's text if an entry is given") Optional<String> text,
            @Tools.Param("pictures or a video in the project's folder") Optional<List<String>> media,
            @Tools.Param("a content calendar entry's id") Optional<String> entry) {
        Project p = FileTools.project();
        Post post = post(p, channel, text, media, entry);
        Map<String, Object> out = send(p, post);
        entry.filter(e -> !e.isBlank()).ifPresent(e -> markPosted(p, e, String.valueOf(out.getOrDefault("url", ""))));
        return out;
    }

    @Tools.Tool(name = "schedule_post", value = "Schedules a post for a time, approved now and sent by the studio "
            + "then, if it is running. Asks the user first.")
    @Tier(value = Risk.OUTWARD, what = "schedules a public post under your name")
    public Map<String, Object> schedulePost(@Tools.Param("mastodon, bluesky, discord or webhook") String channel,
            @Tools.Param("when, as 2026-12-01T09:00 in this PC's time zone") String at,
            @Tools.Param("the post's text; the entry's text if an entry is given") Optional<String> text,
            @Tools.Param("pictures or a video in the project's folder") Optional<List<String>> media,
            @Tools.Param("a content calendar entry's id") Optional<String> entry) {
        Project p = FileTools.project();
        Post post = post(p, channel, text, media, entry);
        long when;
        try {
            when = LocalDateTime.parse(at.strip()).atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (DateTimeParseException e) {
            throw new IllegalArgumentException("write the time as 2026-12-01T09:00");
        }
        if (when < System.currentTimeMillis() - 60_000) {
            throw new IllegalArgumentException("that time is past");
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", UUID.randomUUID().toString().substring(0, 8));
        m.put("project", p.id());
        m.put("channel", post.channel());
        m.put("text", post.text());
        m.put("media", post.media());
        m.put("entry", entry.orElse(""));
        m.put("at", when);
        m.put("state", "scheduled");
        synchronized (this) {
            List<Map<String, Object>> q = load();
            q.add(m);
            store(q);
        }
        if (entry.isPresent() && !entry.get().isBlank()) {
            DataTools.setState(p, entry.get(), "scheduled", null);
        }
        return m;
    }

    @Tools.Tool(name = "scheduled_posts", value = "Lists the posts scheduled, sent and failed, and can cancel one "
            + "not sent yet.")
    @Tier(value = Risk.READ)
    public Map<String, Object> scheduledPosts(@Tools.Param("a scheduled post's id to cancel") Optional<String> cancel) {
        synchronized (this) {
            List<Map<String, Object>> q = load();
            if (cancel.isPresent() && !cancel.get().isBlank()) {
                boolean hit = q.removeIf(m -> cancel.get().equals(m.get("id")) && "scheduled".equals(m.get("state")));
                if (!hit) {
                    throw new IllegalArgumentException("no scheduled post " + cancel.get() + " waits to be sent");
                }
                store(q);
            }
            return Map.of("posts", q);
        }
    }

    @Tools.Tool(name = "post_stats", value = "Reads how a Mastodon or Bluesky post did: likes, boosts and replies.")
    @Tier(value = Risk.READ)
    public Map<String, Object> postStats(@Tools.Param("the post's address, as publish_post gave it") String url) {
        try {
            if (url.contains("bsky.app/profile/")) {
                String[] parts = url.split("/profile/")[1].split("/post/");
                String did = parts[0];
                String uri = "at://" + did + "/app.bsky.feed.post/" + parts[1];
                Map<String, Object> r = json(HttpRequest.newBuilder(URI.create(
                        "https://public.api.bsky.app/xrpc/app.bsky.feed.getPosts?uris=" + java.net.URLEncoder.encode(uri,
                                StandardCharsets.UTF_8))).GET().build());
                Map<String, Object> post = Json.object(Json.array(r.get("posts")).getFirst());
                return Map.of("likes", post.getOrDefault("likeCount", 0), "reposts", post.getOrDefault("repostCount", 0),
                        "replies", post.getOrDefault("replyCount", 0), "quotes", post.getOrDefault("quoteCount", 0));
            }
            String instance = services.get("mastodon", "instance").orElseThrow(() -> new IllegalArgumentException(
                    "set the Mastodon server on the settings page"));
            String id = url.substring(url.lastIndexOf('/') + 1);
            HttpRequest.Builder b = HttpRequest.newBuilder(URI.create(base(instance) + "api/v1/statuses/" + id)).GET();
            services.secret("mastodon").ifPresent(t -> b.header("Authorization", "Bearer " + t));
            Map<String, Object> s = json(b.build());
            return Map.of("favourites", s.getOrDefault("favourites_count", 0), "boosts", s.getOrDefault("reblogs_count", 0),
                    "replies", s.getOrDefault("replies_count", 0));
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AgentRuns.Stopped();
        } catch (RuntimeException e) {
            throw new IllegalArgumentException("cannot read that post's numbers: " + e.getMessage());
        }
    }

    // ---- posting

    /** A post ready to send. */
    record Post(String channel, String text, List<String> media) {
    }

    private static Post post(Project p, String channel, Optional<String> text, Optional<List<String>> media,
            Optional<String> entry) {
        String ch = channel == null ? "" : channel.strip().toLowerCase(Locale.ROOT);
        if (!CHANNELS.contains(ch)) {
            throw new IllegalArgumentException("post to " + String.join(", ", CHANNELS));
        }
        String t = text.orElse("");
        List<String> m = new ArrayList<>(media.orElse(List.of()));
        if (entry.isPresent() && !entry.get().isBlank()) {
            Map<String, Object> e = DataTools.calendar(p.inFolder(DataTools.CALENDAR)).stream()
                    .filter(x -> entry.get().equals(x.get("id"))).findFirst()
                    .orElseThrow(() -> new IllegalArgumentException("there is no calendar entry " + entry.get()));
            if (t.isBlank()) {
                t = Json.string(e, "text", "");
            }
            if (m.isEmpty() && e.get("media") instanceof List<?> l) {
                l.forEach(x -> m.add(String.valueOf(x)));
            }
        }
        if (t.isBlank() && m.isEmpty()) {
            throw new IllegalArgumentException("a post needs text or media");
        }
        if (ch.equals("bluesky") && t.codePointCount(0, t.length()) > 300) {
            throw new IllegalArgumentException("a Bluesky post has at most 300 characters; this has "
                    + t.codePointCount(0, t.length()));
        }
        if (ch.equals("mastodon") && t.codePointCount(0, t.length()) > 500) {
            throw new IllegalArgumentException("a Mastodon post usually has at most 500 characters");
        }
        if (m.size() > 4) {
            throw new IllegalArgumentException("a post takes at most 4 pictures");
        }
        for (String f : m) {
            p.inFolder(f);   // inside the project's folder, or refused
        }
        return new Post(ch, t, List.copyOf(m));
    }

    private Map<String, Object> send(Project p, Post post) {
        try {
            return switch (post.channel()) {
                case "mastodon" -> mastodon(p, post);
                case "bluesky" -> bluesky(p, post);
                case "discord" -> discord(p, post);
                default -> webhook(post);
            };
        } catch (IOException e) {
            throw new UncheckedIOException("the post did not go out: " + e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AgentRuns.Stopped();
        }
    }

    private Map<String, Object> mastodon(Project p, Post post) throws IOException, InterruptedException {
        String instance = services.get("mastodon", "instance").orElseThrow(() -> new IllegalArgumentException(
                "set the Mastodon server on the settings page"));
        String token = services.secret("mastodon").orElseThrow(() -> new IllegalArgumentException(
                "set a Mastodon access token on the settings page"));
        List<String> ids = new ArrayList<>();
        for (String f : post.media()) {
            Multipart mp = new Multipart();
            Path file = p.inFolder(f);
            mp.file("file", file.getFileName().toString(), mime(file), bytes(file));
            Map<String, Object> r = json(HttpRequest.newBuilder(URI.create(base(instance) + "api/v2/media"))
                    .header("Authorization", "Bearer " + token).header("Content-Type", mp.type())
                    .POST(HttpRequest.BodyPublishers.ofByteArray(mp.body())).timeout(Duration.ofMinutes(5)).build());
            ids.add(String.valueOf(r.get("id")));
        }
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("status", post.text());
        if (!ids.isEmpty()) {
            body.put("media_ids", ids);
        }
        Map<String, Object> s = json(HttpRequest.newBuilder(URI.create(base(instance) + "api/v1/statuses"))
                .header("Authorization", "Bearer " + token).header("Content-Type", "application/json")
                .header("Idempotency-Key", UUID.randomUUID().toString())
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(body))).build());
        return Map.of("channel", "mastodon", "url", Json.string(s, "url", ""), "id", Json.string(s, "id", ""));
    }

    private Map<String, Object> bluesky(Project p, Post post) throws IOException, InterruptedException {
        String handle = services.get("bluesky", "handle").orElseThrow(() -> new IllegalArgumentException(
                "set the Bluesky handle on the settings page"));
        String password = services.secret("bluesky").orElseThrow(() -> new IllegalArgumentException(
                "set a Bluesky app password on the settings page"));
        String service = base(services.get("bluesky", "service").orElse("https://bsky.social"));
        Map<String, Object> session = json(HttpRequest.newBuilder(URI.create(service
                + "xrpc/com.atproto.server.createSession")).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(Map.of("identifier", handle, "password", password))))
                .build());
        String jwt = Json.string(session, "accessJwt", "");
        String did = Json.string(session, "did", "");
        Map<String, Object> record = new LinkedHashMap<>();
        record.put("$type", "app.bsky.feed.post");
        record.put("text", post.text());
        record.put("createdAt", Instant.now().toString());
        if (!post.media().isEmpty()) {
            List<Map<String, Object>> images = new ArrayList<>();
            for (String f : post.media()) {
                Path file = p.inFolder(f);
                Map<String, Object> blob = json(HttpRequest.newBuilder(URI.create(service + "xrpc/com.atproto.repo.uploadBlob"))
                        .header("Authorization", "Bearer " + jwt).header("Content-Type", mime(file))
                        .POST(HttpRequest.BodyPublishers.ofByteArray(bytes(file))).timeout(Duration.ofMinutes(5)).build());
                images.add(Map.of("alt", "", "image", blob.get("blob")));
            }
            record.put("embed", Map.of("$type", "app.bsky.embed.images", "images", images));
        }
        Map<String, Object> r = json(HttpRequest.newBuilder(URI.create(service + "xrpc/com.atproto.repo.createRecord"))
                .header("Authorization", "Bearer " + jwt).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(Map.of("repo", did, "collection", "app.bsky.feed.post",
                        "record", record)))).build());
        String uri = Json.string(r, "uri", "");
        String rkey = uri.substring(uri.lastIndexOf('/') + 1);
        return Map.of("channel", "bluesky", "url", "https://bsky.app/profile/" + did + "/post/" + rkey, "id", uri);
    }

    private Map<String, Object> discord(Project p, Post post) throws IOException, InterruptedException {
        String hook = services.secret("discord").orElseThrow(() -> new IllegalArgumentException(
                "set a Discord webhook address on the settings page"));
        if (!hook.startsWith("https://")) {
            throw new IllegalArgumentException("the Discord webhook is an https address");
        }
        Multipart mp = new Multipart();
        mp.field("payload_json", Json.write(Map.of("content", post.text())));
        int i = 0;
        for (String f : post.media()) {
            Path file = p.inFolder(f);
            mp.file("files[" + i++ + "]", file.getFileName().toString(), mime(file), bytes(file));
        }
        Map<String, Object> r = json(HttpRequest.newBuilder(URI.create(hook + (hook.contains("?") ? "&" : "?") + "wait=true"))
                .header("Content-Type", mp.type()).POST(HttpRequest.BodyPublishers.ofByteArray(mp.body()))
                .timeout(Duration.ofMinutes(5)).build());
        return Map.of("channel", "discord", "id", Json.string(r, "id", ""));
    }

    private Map<String, Object> webhook(Post post) throws IOException, InterruptedException {
        String url = services.get("webhook", "url").orElseThrow(() -> new IllegalArgumentException(
                "set the webhook address on the settings page"));
        URI u = URI.create(url);
        if (!"https".equals(u.getScheme()) && !"http".equals(u.getScheme())) {
            throw new IllegalArgumentException("the webhook is an http or https address");
        }
        HttpResponse<String> r = HTTP.send(HttpRequest.newBuilder(u).header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(Json.write(Map.of("text", post.text(), "media", post.media()))))
                .timeout(Duration.ofSeconds(30)).build(), HttpResponse.BodyHandlers.ofString());
        if (r.statusCode() / 100 != 2) {
            throw new IOException("the webhook answered " + r.statusCode());
        }
        return Map.of("channel", "webhook", "status", r.statusCode());
    }

    private static void markPosted(Project p, String entry, String url) {
        try {
            DataTools.setState(p, entry, "posted", url);
        } catch (RuntimeException ignored) {
            // the post went out; the calendar is a convenience
        }
    }

    // ---- the scheduler

    private void loop() {
        while (!closed) {
            try {
                Thread.sleep(30_000);
                sendDue();
            } catch (InterruptedException e) {
                return;
            } catch (RuntimeException e) {
                System.getLogger(PublishTools.class.getName()).log(System.Logger.Level.WARNING, "scheduled posts", e);
            }
        }
    }

    /** Sends the scheduled posts whose time has come. */
    void sendDue() {
        List<Map<String, Object>> due = new ArrayList<>();
        synchronized (this) {
            for (Map<String, Object> m : load()) {
                if ("scheduled".equals(m.get("state")) && Json.integer(m, "at", Long.MAX_VALUE) <= System.currentTimeMillis()) {
                    due.add(m);
                }
            }
        }
        for (Map<String, Object> m : due) {
            String state;
            String result;
            try {
                Project p = projects == null ? null : projects.get(Json.string(m, "project", "")).orElse(null);
                if (p == null) {
                    throw new IllegalArgumentException("its project is gone");
                }
                List<String> media = Json.array(m.getOrDefault("media", List.of())).stream().map(String::valueOf).toList();
                Map<String, Object> out = ScopedValue.where(ChatService.SCOPE, p).call(() -> send(p,
                        new Post(Json.string(m, "channel", ""), Json.string(m, "text", ""), media)));
                state = "sent";
                result = String.valueOf(out.getOrDefault("url", ""));
                String entry = Json.string(m, "entry", "");
                if (!entry.isBlank()) {
                    markPosted(p, entry, result);
                }
            } catch (Exception e) {
                state = "failed";
                result = e.getMessage();
            }
            synchronized (this) {
                List<Map<String, Object>> q = load();
                for (Map<String, Object> x : q) {
                    if (m.get("id").equals(x.get("id"))) {
                        x.put("state", state);
                        x.put("result", result);
                        x.put("sent", System.currentTimeMillis());
                    }
                }
                store(q);
            }
        }
    }

    private List<Map<String, Object>> load() {
        List<Map<String, Object>> out = new ArrayList<>();
        if (Files.isRegularFile(queue)) {
            try {
                for (Object o : Json.array(Json.object(Json.parse(Files.readString(queue, StandardCharsets.UTF_8)))
                        .getOrDefault("posts", List.of()))) {
                    out.add(new LinkedHashMap<>(Json.object(o)));
                }
            } catch (IOException | RuntimeException e) {
                System.getLogger(PublishTools.class.getName()).log(System.Logger.Level.WARNING, "cannot read " + queue, e);
            }
        }
        return out;
    }

    private void store(List<Map<String, Object>> q) {
        try {
            Files.createDirectories(queue.toAbsolutePath().getParent());
            Path tmp = queue.resolveSibling(queue.getFileName() + ".tmp");
            Files.writeString(tmp, Json.write(Map.of("posts", q)), StandardCharsets.UTF_8);
            Files.move(tmp, queue, StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    // ---- helpers

    private static String base(String url) {
        String u = url.strip();
        if (!u.startsWith("http")) {
            u = "https://" + u;
        }
        return u.endsWith("/") ? u : u + "/";
    }

    private static byte[] bytes(Path f) throws IOException {
        if (Files.size(f) > MEDIA_BYTES) {
            throw new IllegalArgumentException(f.getFileName() + " is over " + (MEDIA_BYTES >> 20) + " MB");
        }
        return Files.readAllBytes(f);
    }

    static String mime(Path f) {
        String n = f.getFileName().toString().toLowerCase(Locale.ROOT);
        if (n.endsWith(".png")) {
            return "image/png";
        }
        if (n.endsWith(".jpg") || n.endsWith(".jpeg")) {
            return "image/jpeg";
        }
        if (n.endsWith(".gif")) {
            return "image/gif";
        }
        if (n.endsWith(".webp")) {
            return "image/webp";
        }
        if (n.endsWith(".mp4")) {
            return "video/mp4";
        }
        if (n.endsWith(".webm")) {
            return "video/webm";
        }
        return "application/octet-stream";
    }

    private static Map<String, Object> json(HttpRequest req) throws IOException, InterruptedException {
        HttpResponse<String> r = HTTP.send(req, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        if (r.statusCode() / 100 != 2) {
            String b = r.body() == null ? "" : r.body();
            throw new IOException("HTTP " + r.statusCode() + ": " + b.substring(0, Math.min(300, b.length())));
        }
        String b = r.body();
        return b == null || b.isBlank() ? Map.of() : Json.object(Json.parse(b));
    }

    /** A multipart/form-data body. */
    static final class Multipart {
        private final String boundary = "----brainiac" + UUID.randomUUID().toString().replace("-", "");
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();

        void field(String name, String value) {
            write("--" + boundary + "\r\nContent-Disposition: form-data; name=\"" + name + "\"\r\n\r\n" + value + "\r\n");
        }

        void file(String name, String filename, String type, byte[] bytes) {
            write("--" + boundary + "\r\nContent-Disposition: form-data; name=\"" + name + "\"; filename=\""
                    + filename.replace("\"", "") + "\"\r\nContent-Type: " + type + "\r\n\r\n");
            out.writeBytes(bytes);
            write("\r\n");
        }

        String type() {
            return "multipart/form-data; boundary=" + boundary;
        }

        byte[] body() {
            ByteArrayOutputStream all = new ByteArrayOutputStream();
            all.writeBytes(out.toByteArray());
            all.writeBytes(("--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
            return all.toByteArray();
        }

        private void write(String s) {
            out.writeBytes(s.getBytes(StandardCharsets.UTF_8));
        }
    }
}
