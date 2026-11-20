package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * Requests agents send up to the projects their project is part of: a question, a tool or model
 * they need, or a result to hand over. A request goes to one parent or to all of them, and the
 * first answer counts. Kept in one file, so they outlive a restart.
 */
public final class Requests {

    /** The kinds of request. */
    public static final List<String> KINDS = List.of("question", "tool", "model", "result");

    private static final int KEEP = 500;

    /**
     * One request.
     *
     * @param id its id
     * @param run the agent that asked
     * @param project the asking agent's project
     * @param to the projects it went to
     * @param kind {@code question}, {@code tool}, {@code model} or {@code result}
     * @param text what the agent says
     * @param subject the tool or model asked for; null for the other kinds
     * @param state {@code open}, {@code answered}, {@code approved}, {@code declined} or {@code withdrawn}
     * @param answer the answer, or null
     * @param by who answered: {@code user}, or {@code agent <id> (<project>)}
     * @param created when it was sent
     * @param answered when it was answered, or 0
     */
    public record Request(String id, String run, String project, List<String> to, String kind, String text,
            String subject, String state, String answer, String by, long created, long answered) {

        /** Copies the list. */
        public Request {
            to = List.copyOf(to);
        }

        /** @return whether it still waits for an answer */
        public boolean open() {
            return state.equals("open");
        }

        Request answeredWith(String newState, String newAnswer, String who) {
            return new Request(id, run, project, to, kind, text, subject, newState, newAnswer, who, created,
                    System.currentTimeMillis());
        }

        /** @return it as JSON */
        public Map<String, Object> toJson() {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id);
            m.put("run", run);
            m.put("project", project);
            m.put("to", to);
            m.put("kind", kind);
            m.put("text", text);
            if (subject != null) {
                m.put("subject", subject);
            }
            m.put("state", state);
            if (answer != null) {
                m.put("answer", answer);
            }
            if (by != null) {
                m.put("by", by);
            }
            m.put("created", created);
            if (answered > 0) {
                m.put("answered", answered);
            }
            return m;
        }

        static Request fromJson(Map<String, Object> m) {
            return new Request(Json.string(m, "id", ""), Json.string(m, "run", ""), Json.string(m, "project", ""),
                    Json.array(m.getOrDefault("to", List.of())).stream().map(String::valueOf).toList(),
                    Json.string(m, "kind", "question"), Json.string(m, "text", ""), Json.string(m, "subject", null),
                    Json.string(m, "state", "open"), Json.string(m, "answer", null), Json.string(m, "by", null),
                    Json.integer(m, "created", 0), Json.integer(m, "answered", 0));
        }
    }

    private final Path file;
    private final Map<String, Request> requests = new LinkedHashMap<>();
    private long counter;

    /**
     * Opens the requests kept in a file.
     *
     * @param file the file; null keeps them in memory only
     */
    public Requests(Path file) {
        this.file = file;
        if (file != null && Files.isRegularFile(file)) {
            try {
                for (Object o : Json.array(Json.object(Json.parse(Files.readString(file))).get("requests"))) {
                    Request r = Request.fromJson(Json.object(o));
                    requests.put(r.id(), r);
                    counter = Math.max(counter, number(r.id()));
                }
            } catch (IOException | RuntimeException e) {
                System.getLogger(Requests.class.getName()).log(System.Logger.Level.WARNING,
                        "the requests in " + file + " could not be read", e);
            }
        }
    }

    private static long number(String id) {
        try {
            return Long.parseLong(id.substring(1));
        } catch (RuntimeException e) {
            return 0;
        }
    }

    /**
     * Sends a request.
     *
     * @param run the asking agent
     * @param project its project
     * @param to the projects it goes to
     * @param kind its kind
     * @param text what it says
     * @param subject the tool or model asked for, for those kinds
     * @return the request
     * @throws IOException if it cannot be kept
     */
    public synchronized Request add(String run, String project, List<String> to, String kind, String text,
            String subject) throws IOException {
        if (!KINDS.contains(kind)) {
            throw new IllegalArgumentException("a request is a question, tool, model or result, not " + kind);
        }
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("say what you need");
        }
        if (text.length() > 8000) {
            throw new IllegalArgumentException("keep a request under 8000 characters; put more in a file");
        }
        if ((kind.equals("tool") || kind.equals("model")) && (subject == null || subject.isBlank())) {
            throw new IllegalArgumentException("name the " + kind + " you need");
        }
        long mine = requests.values().stream().filter(r -> r.open() && r.run().equals(run)).count();
        if (mine >= 10) {
            throw new IllegalArgumentException("you have 10 open requests already; wait for answers");
        }
        Request r = new Request("r" + ++counter, run, project, to, kind, text.strip(),
                subject == null || subject.isBlank() ? null : subject.strip(), "open", null, null,
                System.currentTimeMillis(), 0);
        requests.put(r.id(), r);
        write();
        return r;
    }

    /**
     * Lists the requests, the newest first.
     *
     * @return them
     */
    public synchronized List<Request> list() {
        List<Request> out = new ArrayList<>(requests.values());
        java.util.Collections.reverse(out);
        return out;
    }

    /**
     * Finds a request.
     *
     * @param id its id
     * @return it, if there is one
     */
    public synchronized Optional<Request> get(String id) {
        return Optional.ofNullable(requests.get(id));
    }

    /**
     * Answers an open request.
     *
     * @param id its id
     * @param state {@code answered}, {@code approved}, {@code declined} or {@code withdrawn}
     * @param answer the answer
     * @param by who answers
     * @return the request answered
     * @throws IllegalArgumentException if it is not open
     * @throws IOException if it cannot be kept
     */
    public synchronized Request close(String id, String state, String answer, String by) throws IOException {
        Request r = requests.get(id);
        if (r == null) {
            throw new IllegalArgumentException("there is no request " + id);
        }
        if (!r.open()) {
            throw new IllegalArgumentException("request " + id + " was " + r.state() + " already"
                    + (r.by() == null ? "" : " by " + r.by()));
        }
        Request done = r.answeredWith(state, answer == null || answer.isBlank() ? null : answer.strip(), by);
        requests.put(id, done);
        write();
        return done;
    }

    private void write() throws IOException {
        // answered ones are kept for the record, up to a point
        while (requests.size() > KEEP) {
            String old = requests.values().stream().filter(r -> !r.open()).map(Request::id).findFirst()
                    .orElse(requests.keySet().iterator().next());
            requests.remove(old);
        }
        if (file == null) {
            return;
        }
        Files.createDirectories(file.toAbsolutePath().getParent());
        Path tmp = file.resolveSibling(file.getFileName() + ".tmp");
        Files.writeString(tmp, Json.write(Map.of("requests", requests.values().stream().map(Request::toJson).toList())));
        Files.move(tmp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
    }
}
