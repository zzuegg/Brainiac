package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.Agent;
import io.github.zzuegg.brainiac.studio.chat.ChatMessage;
import io.github.zzuegg.brainiac.studio.chat.ChatResult;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.function.Function;

/**
 * Agents the user starts in the studio: each one works towards a goal with a chat model and the
 * {@linkplain StudioTools studio's tools}, and is kept in the {@code agents} folder.
 *
 * <p>Any chat model can drive an agent, on this PC or remote: the tools are described in the
 * system prompt in the Hermes format that Qwen and many other models are trained on, and each
 * {@code <tool_call>} the model writes is run and answered in a {@code <tool_response>}. In
 * {@code ask} mode the agent waits for the user before a tool that {@linkplain StudioTools.Acts acts};
 * in {@code auto} mode it calls every tool itself.
 */
public final class AgentRuns {

    /** How many agents may work at once. */
    public static final int MAX_ACTIVE = 4;

    /** The most model turns a run may take, unless asked for fewer. */
    public static final int MAX_STEPS = 30;

    private static final int TURN_TOKENS = 2048;
    private static final int RESULT_CHARS = 6000;
    private static final System.Logger LOG = System.getLogger(AgentRuns.class.getName());

    private final ChatService chat;
    private final Path folder;
    private final Map<String, StudioTools.Entry> tools = new LinkedHashMap<>();
    private final Map<String, Run> runs = new ConcurrentHashMap<>();
    private int counter;

    /** A user's answer to a tool waiting for approval. */
    private record Decision(boolean allow, boolean stop, String note) {
    }

    /** One agent's work. */
    private final class Run {
        final String id;
        final String goal;
        final String model;
        final String mode;
        final int maxSteps;
        final List<String> allowed;
        final long created;
        final List<Map<String, Object>> steps = new ArrayList<>();
        final List<ChatMessage> messages = new ArrayList<>();
        String state = "running";
        String by;
        String answer;
        String error;
        long ended;
        long inputTokens;
        long outputTokens;
        Map<String, Object> pending;
        CompletableFuture<Decision> decision;
        volatile boolean stopping;

        Run(String id, String goal, String model, String mode, int maxSteps, List<String> allowed, long created) {
            this.id = id;
            this.goal = goal;
            this.model = model;
            this.mode = mode;
            this.maxSteps = maxSteps;
            this.allowed = allowed;
            this.created = created;
        }

        synchronized boolean active() {
            return state.equals("running") || state.equals("waiting");
        }

        synchronized void step(String kind, Map<String, Object> fields) {
            Map<String, Object> s = new LinkedHashMap<>();
            s.put("kind", kind);
            s.put("at", System.currentTimeMillis());
            s.putAll(fields);
            steps.add(s);
        }

        synchronized Map<String, Object> toJson(boolean full) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", id);
            m.put("goal", goal);
            m.put("model", model);
            if (by != null) {
                m.put("by", by);
            }
            m.put("mode", mode);
            m.put("maxSteps", maxSteps);
            if (allowed != null) {
                m.put("tools", allowed);
            }
            m.put("state", state);
            m.put("created", created);
            m.put("ended", ended);
            m.put("inputTokens", inputTokens);
            m.put("outputTokens", outputTokens);
            m.put("turns", messages.stream().filter(x -> x.role().equals("assistant")).count());
            m.put("calls", steps.stream().filter(x -> "call".equals(x.get("kind"))).count());
            if (answer != null) {
                m.put("answer", answer);
            }
            if (error != null) {
                m.put("error", error);
            }
            if (pending != null) {
                m.put("pending", pending);
            }
            Map<String, Object> last = steps.isEmpty() ? null : steps.getLast();
            if (last != null) {
                m.put("doing", doing(last));
            }
            if (full) {
                m.put("steps", List.copyOf(steps));
                m.put("messages", messages.stream().map(x -> Map.of("role", x.role(), "text", x.text())).toList());
            }
            return m;
        }
    }

    /**
     * Opens the runs kept in a folder. Runs that were working when the studio closed are marked
     * stopped.
     *
     * @param chat the models agents think with
     * @param tools the tools they may call
     * @param folder where runs are kept
     */
    public AgentRuns(ChatService chat, StudioTools tools, Path folder) {
        this.chat = chat;
        this.folder = folder;
        for (StudioTools.Entry e : tools.entries()) {
            this.tools.put(e.tool().name(), e);
        }
        load();
    }

    /**
     * Returns the tools agents may call.
     *
     * @return the tools by name
     */
    public Map<String, StudioTools.Entry> tools() {
        return tools;
    }

    /**
     * Calls a tool directly, as programs holding the studio's token may.
     *
     * @param name the tool
     * @param arguments its arguments
     * @return its result, or the error text starting with {@code error:}
     * @throws IllegalArgumentException if there is no such tool
     */
    public String call(String name, Map<String, Object> arguments) {
        StudioTools.Entry e = tools.get(name);
        if (e == null) {
            throw new IllegalArgumentException("there is no tool " + name);
        }
        return run(e, arguments);
    }

    private static String run(StudioTools.Entry e, Map<String, Object> arguments) {
        try {
            String out = e.tool().run().apply(arguments == null ? Map.of() : arguments);
            return out.length() > RESULT_CHARS ? out.substring(0, RESULT_CHARS) + "\n[cut: the result was "
                    + out.length() + " characters]" : out;
        } catch (RuntimeException x) {
            return "error: " + (x.getMessage() == null ? x.getClass().getSimpleName() : x.getMessage());
        }
    }

    /**
     * Starts an agent.
     *
     * @param goal what it should do
     * @param model the chat model key, or {@code auto}
     * @param mode {@code ask} to wait for approval before tools that act, {@code auto} to call them alone
     * @param maxSteps the most model turns
     * @param allowed the tools it may use, or null for all
     * @return the run
     */
    public synchronized Map<String, Object> start(String goal, String model, String mode, int maxSteps,
            List<String> allowed) {
        if (goal == null || goal.isBlank()) {
            throw new IllegalArgumentException("say what the agent should do");
        }
        if (!mode.equals("ask") && !mode.equals("auto")) {
            throw new IllegalArgumentException("the mode is ask or auto, not " + mode);
        }
        if (maxSteps < 1 || maxSteps > MAX_STEPS) {
            throw new IllegalArgumentException("steps must be between 1 and " + MAX_STEPS);
        }
        if (allowed != null) {
            for (String t : allowed) {
                if (!tools.containsKey(t)) {
                    throw new IllegalArgumentException("there is no tool " + t);
                }
            }
        }
        if (runs.values().stream().filter(Run::active).count() >= MAX_ACTIVE) {
            throw new IllegalArgumentException(MAX_ACTIVE + " agents are working already; stop one or wait");
        }
        Run r = new Run(String.valueOf(++counter), goal.strip(), model == null || model.isBlank() ? ChatService.AUTO
                : model, mode, maxSteps, allowed == null ? null : List.copyOf(allowed), System.currentTimeMillis());
        r.messages.add(new ChatMessage("user", r.goal));
        r.step("goal", Map.of("text", r.goal));
        runs.put(r.id, r);
        save(r);
        Thread.ofVirtual().name("studio-agent-" + r.id).start(() -> work(r));
        return r.toJson(false);
    }

    /**
     * Gives a finished agent another message, and lets it work again.
     *
     * @param id the run
     * @param text the message
     * @return the run, or empty if there is none
     */
    public synchronized Optional<Map<String, Object>> reply(String id, String text) {
        Run r = runs.get(id);
        if (r == null) {
            return Optional.empty();
        }
        if (text == null || text.isBlank()) {
            throw new IllegalArgumentException("the message is empty");
        }
        synchronized (r) {
            if (r.active()) {
                throw new IllegalArgumentException("the agent is still working");
            }
            ChatMessage last = r.messages.getLast();
            if (last.role().equals("user")) {   // tool results it has not read yet
                r.messages.set(r.messages.size() - 1, new ChatMessage("user", last.text() + "\n\n" + text.strip()));
            } else {
                r.messages.add(new ChatMessage("user", text.strip()));
            }
            r.step("goal", Map.of("text", text.strip()));
            r.state = "running";
            r.answer = null;
            r.error = null;
            r.ended = 0;
            r.stopping = false;
        }
        save(r);
        Thread.ofVirtual().name("studio-agent-" + r.id).start(() -> work(r));
        return Optional.of(r.toJson(false));
    }

    /**
     * Answers a tool call waiting for approval.
     *
     * @param id the run
     * @param allow whether the tool may run
     * @param note what the user says to the agent, or null
     * @return false if the run is not waiting
     */
    public boolean answer(String id, boolean allow, String note) {
        Run r = runs.get(id);
        if (r == null) {
            return false;
        }
        synchronized (r) {
            if (r.decision == null) {
                return false;
            }
            return r.decision.complete(new Decision(allow, false, note == null ? "" : note.strip()));
        }
    }

    /**
     * Stops an agent after its current step.
     *
     * @param id the run
     * @return false if there is no such run
     */
    public boolean stop(String id) {
        Run r = runs.get(id);
        if (r == null) {
            return false;
        }
        r.stopping = true;
        synchronized (r) {
            if (r.decision != null) {
                r.decision.complete(new Decision(false, true, ""));
            }
        }
        return true;
    }

    /**
     * Forgets a run that has ended.
     *
     * @param id the run
     * @return false if there is no such run
     * @throws IOException if its record cannot be deleted
     */
    public synchronized boolean delete(String id) throws IOException {
        Run r = runs.get(id);
        if (r == null) {
            return false;
        }
        if (r.active()) {
            throw new IllegalArgumentException("stop the agent before removing it");
        }
        runs.remove(id);
        Files.deleteIfExists(folder.resolve(id + ".json"));
        return true;
    }

    /**
     * Returns every run, newest first, without its steps.
     *
     * @return the runs
     */
    public List<Map<String, Object>> list() {
        return runs.values().stream().sorted(Comparator.comparingLong((Run r) -> r.created).reversed())
                .map(r -> r.toJson(false)).toList();
    }

    /**
     * Returns one run with its steps.
     *
     * @param id the run
     * @return the run, or empty
     */
    public Optional<Map<String, Object>> get(String id) {
        return Optional.ofNullable(runs.get(id)).map(r -> r.toJson(true));
    }

    // ---------------------------------------------------------------- the loop

    private void work(Run r) {
        try {
            String system = system(r);
            int turns = 0;
            while (true) {
                if (r.stopping) {
                    end(r, "stopped", null);
                    return;
                }
                if (turns++ >= r.maxSteps) {
                    end(r, "out of steps", "it used all " + r.maxSteps + " steps; reply to let it go on");
                    return;
                }
                StringBuilder text = new StringBuilder();
                List<ChatMessage> history;
                synchronized (r) {
                    history = List.copyOf(r.messages);
                }
                ChatResult result = chat.chat(r.model, system, history, TURN_TOKENS, null,
                        m -> { synchronized (r) { r.by = m.key(); } },
                        piece -> {
                            if (r.stopping) {
                                throw new IOException("stopped");
                            }
                            text.append(piece);
                        });
                String turn = turn(text.toString());
                List<Map<String, Object>> calls = Agent.calls(turn);
                String said = Agent.content(turn);
                synchronized (r) {
                    r.inputTokens += Math.max(0, result.inputTokens());
                    r.outputTokens += Math.max(0, result.outputTokens());
                    r.messages.add(new ChatMessage("assistant", turn.isEmpty() ? "(nothing)" : turn));
                }
                if (calls.isEmpty()) {
                    r.step("answer", Map.of("text", said));
                    synchronized (r) {
                        r.answer = said;
                    }
                    end(r, "done", null);
                    return;
                }
                if (!said.isEmpty()) {
                    r.step("say", Map.of("text", said));
                }
                StringBuilder responses = new StringBuilder();
                for (Map<String, Object> c : calls) {
                    String name = String.valueOf(c.get("name"));
                    Map<String, Object> args = Json.object(c.get("arguments"));
                    r.step("call", Map.of("tool", name, "arguments", args));
                    save(r);
                    String out = call(r, name, args);
                    if (out == null) {
                        responses.append("<tool_response>\nerror: the user stopped the agent before this call ran"
                                + "\n</tool_response>");
                        synchronized (r) {
                            r.messages.add(new ChatMessage("user", responses.toString().strip()));
                        }
                        end(r, "stopped", null);
                        return;
                    }
                    r.step("result", Map.of("tool", name, "text", out, "ok", !out.startsWith("error:")));
                    responses.append("<tool_response>\n").append(out).append("\n</tool_response>\n");
                }
                synchronized (r) {
                    r.messages.add(new ChatMessage("user", responses.toString().strip()));
                }
                save(r);
            }
        } catch (IOException e) {
            end(r, r.stopping ? "stopped" : "failed", r.stopping ? null : e.getMessage());
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            end(r, "stopped", null);
        } catch (RuntimeException e) {
            LOG.log(System.Logger.Level.WARNING, "agent " + r.id + " failed", e);
            end(r, "failed", String.valueOf(e.getMessage()));
        }
    }

    /** Runs one call, after asking the user if it acts; null if the agent was stopped meanwhile. */
    private String call(Run r, String name, Map<String, Object> args) throws InterruptedException {
        StudioTools.Entry e = tools.get(name);
        if (e == null || (r.allowed != null && !r.allowed.contains(name))) {
            return "error: there is no tool named " + name;
        }
        if (e.acts() != null && r.mode.equals("ask")) {
            CompletableFuture<Decision> d = new CompletableFuture<>();
            synchronized (r) {
                if (r.stopping) {
                    return null;
                }
                Map<String, Object> p = new LinkedHashMap<>();
                p.put("tool", name);
                p.put("arguments", args);
                p.put("acts", e.acts());
                p.put("since", System.currentTimeMillis());
                r.pending = p;
                r.decision = d;
                r.state = "waiting";
            }
            save(r);
            Decision answer;
            try {
                answer = d.get();
            } catch (ExecutionException x) {
                answer = new Decision(false, true, "");
            }
            synchronized (r) {
                r.pending = null;
                r.decision = null;
                if (!answer.stop()) {
                    r.state = "running";
                }
            }
            r.step("decision", Map.of("tool", name, "allow", answer.allow(), "note", answer.note()));
            if (answer.stop()) {
                return null;
            }
            if (!answer.allow()) {
                return "error: the user did not allow this call" + (answer.note().isEmpty() ? ""
                        : "; they said: " + answer.note());
            }
            if (!answer.note().isEmpty()) {
                return run(e, args) + "\n(the user allowed this and said: " + answer.note() + ")";
            }
        }
        return run(e, args);
    }

    private void end(Run r, String state, String error) {
        synchronized (r) {
            r.state = state;
            r.error = error;
            r.ended = System.currentTimeMillis();
            r.pending = null;
            r.decision = null;
        }
        if (error != null) {
            r.step("error", Map.of("text", error));
        }
        save(r);
    }

    /** A model turn up to its calls: a model without stop words may go on to imagine the results. */
    static String turn(String text) {
        int cut = text.indexOf("<tool_response>");
        String t = cut >= 0 ? text.substring(0, cut) : text;
        int last = t.lastIndexOf("</tool_call>");
        if (last >= 0 && Agent.calls(t).size() > 0) {
            t = t.substring(0, last + "</tool_call>".length());
        }
        return t.strip();
    }

    private String system(Run r) {
        StringBuilder s = new StringBuilder();
        s.append("You are an agent inside Brainiac Studio, an app on the user's PC for chatting with, building, ")
                .append("downloading and training AI models. Work towards the user's goal with the tools below. ")
                .append("Call a tool when you need a fact or want to act, and never guess what a tool would return. ")
                .append("If a call fails, read the error and try something else. When the goal is reached, or ")
                .append("cannot be reached, answer in a few plain sentences without a tool call.\n\n")
                .append("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n")
                .append("You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n");
        for (StudioTools.Entry e : tools.values()) {
            if (r.allowed != null && !r.allowed.contains(e.tool().name())) {
                continue;
            }
            Map<String, Object> f = new LinkedHashMap<>();
            f.put("name", e.tool().name());
            f.put("description", e.tool().description());
            f.put("parameters", e.tool().parameters());
            s.append(Json.write(Map.of("type", "function", "function", f))).append('\n');
        }
        s.append("</tools>\n\nFor each function call, return a json object with function name and arguments ")
                .append("within <tool_call></tool_call> XML tags:\n<tool_call>\n")
                .append("{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>\n\n")
                .append("The results come back in the next message, each within <tool_response></tool_response> ")
                .append("tags. Stop after your calls and wait for them.");
        return s.toString();
    }

    private static String doing(Map<String, Object> step) {
        Function<Object, String> s = String::valueOf;
        return switch (s.apply(step.get("kind"))) {
            case "goal" -> "reading the task";
            case "say", "answer" -> firstLine(s.apply(step.get("text")));
            case "call" -> "calling " + step.get("tool");
            case "result" -> step.get("tool") + (Boolean.TRUE.equals(step.get("ok")) ? " answered" : " failed");
            case "decision" -> (Boolean.TRUE.equals(step.get("allow")) ? "allowed " : "not allowed ") + step.get("tool");
            case "error" -> s.apply(step.get("text"));
            default -> "";
        };
    }

    private static String firstLine(String text) {
        String t = text.strip();
        int nl = t.indexOf('\n');
        t = nl >= 0 ? t.substring(0, nl) : t;
        return t.length() > 160 ? t.substring(0, 157) + "..." : t;
    }

    // ---------------------------------------------------------------- keeping runs

    private void save(Run r) {
        try {
            Files.createDirectories(folder);
            Path tmp = folder.resolve(r.id + ".json.tmp");
            Files.writeString(tmp, Json.write(r.toJson(true)), StandardCharsets.UTF_8);
            Files.move(tmp, folder.resolve(r.id + ".json"), StandardCopyOption.REPLACE_EXISTING,
                    StandardCopyOption.ATOMIC_MOVE);
        } catch (IOException | RuntimeException e) {
            LOG.log(System.Logger.Level.WARNING, "cannot save agent " + r.id, e);
        }
    }

    private void load() {
        if (!Files.isDirectory(folder)) {
            return;
        }
        try (DirectoryStream<Path> ds = Files.newDirectoryStream(folder, "*.json")) {
            for (Path p : ds) {
                try {
                    Run r = read(Json.object(Json.parse(Files.readString(p, StandardCharsets.UTF_8))));
                    runs.put(r.id, r);
                    counter = Math.max(counter, Integer.parseInt(r.id));
                    if (r.state.equals("running") || r.state.equals("waiting")) {
                        r.state = "stopped";
                        r.error = "the studio closed while it worked; reply to let it go on";
                        r.pending = null;
                        r.ended = System.currentTimeMillis();
                        save(r);
                    }
                } catch (IOException | RuntimeException e) {
                    LOG.log(System.Logger.Level.WARNING, "cannot read agent " + p, e);
                }
            }
        } catch (IOException e) {
            LOG.log(System.Logger.Level.WARNING, "cannot list " + folder, e);
        }
    }

    private Run read(Map<String, Object> m) {
        List<String> allowed = m.get("tools") == null ? null
                : Json.array(m.get("tools")).stream().map(String::valueOf).toList();
        Run r = new Run(Json.string(m, "id", "0"), Json.string(m, "goal", ""), Json.string(m, "model", "auto"),
                Json.string(m, "mode", "ask"), (int) Json.integer(m, "maxSteps", 12), allowed,
                Json.integer(m, "created", 0));
        r.state = Json.string(m, "state", "stopped");
        r.by = Json.string(m, "by", null);
        r.answer = Json.string(m, "answer", null);
        r.error = Json.string(m, "error", null);
        r.ended = Json.integer(m, "ended", 0);
        r.inputTokens = Json.integer(m, "inputTokens", 0);
        r.outputTokens = Json.integer(m, "outputTokens", 0);
        for (Object s : Json.array(m.getOrDefault("steps", List.of()))) {
            r.steps.add(new LinkedHashMap<>(Json.object(s)));
        }
        for (Object o : Json.array(m.getOrDefault("messages", List.of()))) {
            Map<String, Object> x = Json.object(o);
            r.messages.add(new ChatMessage(Json.string(x, "role", "user"), Json.string(x, "text", "")));
        }
        return r;
    }
}
