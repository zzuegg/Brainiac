package io.github.zzuegg.brainiac.llm;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;

import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * An HTTP server with the OpenAI API's completion endpoints, so existing clients and tools
 * can talk to a local model: {@code GET /v1/models}, {@code POST /v1/completions} and
 * {@code POST /v1/chat/completions}, each with {@code "stream": true} as server-sent events.
 * A chat's {@code response_format} of type {@code json_schema} or {@code json_object}
 * constrains the answer. A chat's {@code tools} go to the chat template, and a reply that calls
 * them (Hermes {@code <tool_call>} blocks or a bare JSON call, see {@link Agent}) comes back as
 * {@code tool_calls} with {@code finish_reason} "tool_calls"; earlier calls and {@code tool}
 * messages in the request are rendered for the model. Requests are answered one at a time, or, with a batch size above
 * one, decoded together by {@link Generator#batcher continuous batching}.
 *
 * <pre>{@code
 * try (OpenAiServer server = OpenAiServer.start(new InetSocketAddress(8080), "qwen", generator, tokenizer,
 *         Generator.stopTokens(dir, tokenizer))) {
 *     // curl localhost:8080/v1/chat/completions -d '{"messages":[{"role":"user","content":"Hi"}]}'
 * }
 * }</pre>
 */
public final class OpenAiServer implements AutoCloseable {

    private final HttpServer http;
    private final ExecutorService worker;
    private final String model;
    private final Generator generator;
    private final PretrainedTokenizer tokenizer;
    private final Set<Integer> stopTokens;
    private final Generator.Batcher batcher;

    private OpenAiServer(HttpServer http, String model, Generator generator, PretrainedTokenizer tokenizer,
            Set<Integer> stopTokens, int maxBatch) {
        this.http = http;
        this.model = model;
        this.generator = generator;
        this.tokenizer = tokenizer;
        this.stopTokens = stopTokens;
        this.batcher = maxBatch > 1 ? generator.batcher(maxBatch) : null;
        java.util.concurrent.ThreadFactory threads = r -> {
            Thread t = new Thread(r, "brainiac-openai-server");
            t.setDaemon(true);
            return t;
        };
        this.worker = maxBatch > 1 ? Executors.newCachedThreadPool(threads) : Executors.newSingleThreadExecutor(threads);
    }

    /**
     * Starts serving.
     *
     * @param address where to listen; port 0 picks a free one
     * @param model the name clients use for the model
     * @param generator generates the text
     * @param tokenizer renders chats with its template and counts tokens
     * @param stopTokens the tokens that end an answer, or {@code null} for the end-of-sequence token
     * @return the running server
     */
    public static OpenAiServer start(InetSocketAddress address, String model, Generator generator,
            PretrainedTokenizer tokenizer, Set<Integer> stopTokens) {
        return start(address, model, generator, tokenizer, stopTokens, 1);
    }

    /**
     * Starts serving, decoding up to {@code maxBatch} requests together.
     *
     * @param address where to listen; port 0 picks a free one
     * @param model the name clients use for the model
     * @param generator generates the text; its model's states must join when {@code maxBatch > 1}
     * @param tokenizer renders chats with its template and counts tokens
     * @param stopTokens the tokens that end an answer, or {@code null} for the end-of-sequence token
     * @param maxBatch the most requests decoded side by side; 1 answers one at a time
     * @return the running server
     */
    public static OpenAiServer start(InetSocketAddress address, String model, Generator generator,
            PretrainedTokenizer tokenizer, Set<Integer> stopTokens, int maxBatch) {
        try {
            HttpServer http = HttpServer.create(address, 0);
            OpenAiServer s = new OpenAiServer(http, Objects.requireNonNull(model, "model"),
                    Objects.requireNonNull(generator, "generator"), Objects.requireNonNull(tokenizer, "tokenizer"),
                    stopTokens, maxBatch);
            http.createContext("/v1/models", s::models);
            http.createContext("/v1/completions", e -> s.complete(e, false));
            http.createContext("/v1/chat/completions", e -> s.complete(e, true));
            http.setExecutor(s.worker);
            http.start();
            return s;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** @return the port the server listens on */
    public int port() {
        return http.getAddress().getPort();
    }

    /** Stops serving, ending the request in progress. */
    @Override
    public void close() {
        http.stop(0);
        if (batcher != null) {
            batcher.close();
        }
        worker.shutdownNow();
    }

    private void models(HttpExchange e) throws IOException {
        if (!e.getRequestMethod().equals("GET")) {
            error(e, 405, "use GET");
            return;
        }
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("id", model);
        m.put("object", "model");
        m.put("owned_by", "brainiac");
        json(e, 200, Map.of("object", "list", "data", List.of(m)));
    }

    /** A request that cannot be served, answered with status 400. */
    private static final class BadRequest extends RuntimeException {
        private static final long serialVersionUID = 1;

        BadRequest(String message) {
            super(message);
        }
    }

    private void complete(HttpExchange e, boolean chat) throws IOException {
        try (e) {
            if (!e.getRequestMethod().equals("POST")) {
                error(e, 405, "use POST");
                return;
            }
            Map<String, Object> req;
            try {
                req = Json.object(Json.parse(new String(e.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)));
            } catch (RuntimeException ex) {
                error(e, 400, "the body is not a JSON object: " + ex.getMessage());
                return;
            }
            String prompt;
            Generator.Options options;
            try {
                prompt = chat ? chatPrompt(req) : completionPrompt(req);
                options = options(req, chat);
            } catch (BadRequest | IllegalArgumentException | ClassCastException ex) {
                error(e, 400, ex.getMessage());
                return;
            }
            boolean stream = Boolean.TRUE.equals(req.get("stream"));
            boolean tools = chat && tools(req) != null;
            String id = (chat ? "chatcmpl-" : "cmpl-") + UUID.randomUUID();
            long created = System.currentTimeMillis() / 1000;
            int promptTokens = generator.encode(prompt).length;
            if (stream) {
                e.getResponseHeaders().set("Content-Type", "text/event-stream");
                e.getResponseHeaders().set("Cache-Control", "no-cache");
                e.sendResponseHeaders(200, 0);
                OutputStream out = e.getResponseBody();
                if (chat) {
                    event(out, chunk(id, created, true, Map.of("role", "assistant", "content", ""), null));
                }
                if (tools) {
                    // whether the turn is a call is known only at its end: one chunk with all of it
                    Generator.Output o = run(prompt, options, null);
                    Map<String, Object> message = assistant(o.text(), true);
                    message.remove("role");
                    if (message.get("content") == null) {
                        message.remove("content");
                    }
                    event(out, chunk(id, created, true, message, null));
                    event(out, chunk(id, created, true, Map.of(), message.containsKey("tool_calls") ? "tool_calls" : finish(o)));
                    out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                    out.flush();
                    return;
                }
                Generator.Output o = run(prompt, options, (row, text) -> {
                    try {
                        event(out, chunk(id, created, chat, chat ? Map.of("content", text) : text, null));
                    } catch (IOException ex) {
                        throw new UncheckedIOException(ex);
                    }
                });
                event(out, chunk(id, created, chat, chat ? Map.of() : "", finish(o)));
                out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                out.flush();
                return;
            }
            Generator.Output o = run(prompt, options, null);
            Map<String, Object> choice = new LinkedHashMap<>();
            choice.put("index", 0);
            Map<String, Object> message = chat ? assistant(o.text(), tools) : null;
            if (chat) {
                choice.put("message", message);
            } else {
                choice.put("text", o.text());
            }
            choice.put("finish_reason", message != null && message.containsKey("tool_calls") ? "tool_calls" : finish(o));
            Map<String, Object> usage = new LinkedHashMap<>();
            usage.put("prompt_tokens", promptTokens);
            usage.put("completion_tokens", o.tokens().length);
            usage.put("total_tokens", promptTokens + o.tokens().length);
            Map<String, Object> res = new LinkedHashMap<>();
            res.put("id", id);
            res.put("object", chat ? "chat.completion" : "text_completion");
            res.put("created", created);
            res.put("model", model);
            res.put("choices", List.of(choice));
            res.put("usage", usage);
            json(e, 200, res);
        }
    }

    private Generator.Output run(String prompt, Generator.Options options, Generator.Listener listener) {
        if (batcher == null) {
            return generator.generate(List.of(prompt), options, listener).getFirst();
        }
        try {
            return batcher.submit(prompt, options, listener).get();
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted", ex);
        } catch (java.util.concurrent.ExecutionException ex) {
            throw ex.getCause() instanceof RuntimeException r ? r : new IllegalStateException(ex.getCause());
        }
    }

    private String chatPrompt(Map<String, Object> req) {
        if (tokenizer.chatTemplate() == null) {
            throw new BadRequest("the model has no chat template; use /v1/completions");
        }
        Object messages = req.get("messages");
        if (!(messages instanceof List<?> list) || list.isEmpty()) {
            throw new BadRequest("messages must be a non-empty list");
        }
        List<Map<String, Object>> rendered = new ArrayList<>();
        for (Object m : list) {
            rendered.add(message(Json.object(m)));
        }
        List<Object> tools = tools(req);
        return tools == null ? tokenizer.applyChatTemplate(rendered, true)
                : tokenizer.applyChatTemplate(rendered, true, Map.of("tools", tools));
    }

    /** The request's tools, or null without any or with {@code tool_choice} "none". */
    private static List<Object> tools(Map<String, Object> req) {
        if (!(req.get("tools") instanceof List<?> list) || list.isEmpty() || "none".equals(req.get("tool_choice"))) {
            return null;
        }
        List<Object> out = new ArrayList<>();
        for (Object t : list) {
            Map<String, Object> tool = Json.object(t);
            if (!(tool.get("function") instanceof Map<?, ?>)) {
                throw new BadRequest("each tool needs a function");
            }
            out.add(tool);
        }
        return out;
    }

    /**
     * A message as the chat template reads it: the arguments of earlier tool calls, which
     * OpenAI clients send as JSON text, as objects.
     */
    private static Map<String, Object> message(Map<String, Object> m) {
        if (!(m.get("tool_calls") instanceof List<?> calls)) {
            return m;
        }
        Map<String, Object> out = new LinkedHashMap<>(m);
        List<Object> parsed = new ArrayList<>();
        for (Object c : calls) {
            Map<String, Object> call = new LinkedHashMap<>(Json.object(c));
            if (call.get("function") instanceof Map<?, ?> f) {
                Map<String, Object> function = new LinkedHashMap<>(Json.object(f));
                if (function.get("arguments") instanceof String text) {
                    try {
                        function.put("arguments", Json.parse(text));
                    } catch (RuntimeException e) {
                        // left as text
                    }
                }
                call.put("function", function);
            }
            parsed.add(call);
        }
        out.put("tool_calls", parsed);
        if (out.get("content") == null) {
            out.put("content", "");
        }
        return out;
    }

    /** The assistant's message: with {@code tool_calls} when tools were offered and it called them. */
    private static Map<String, Object> assistant(String text, boolean tools) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", "assistant");
        List<Map<String, Object>> calls = tools ? Agent.calls(text.strip()) : List.of();
        if (calls.isEmpty()) {
            m.put("content", text);
            return m;
        }
        String rest = Agent.content(text.strip());
        m.put("content", rest.isEmpty() ? null : rest);
        List<Object> out = new ArrayList<>();
        for (Map<String, Object> c : calls) {
            Map<String, Object> function = new LinkedHashMap<>();
            function.put("name", c.get("name"));
            function.put("arguments", Json.write(c.get("arguments")));
            Map<String, Object> call = new LinkedHashMap<>();
            call.put("id", "call_" + UUID.randomUUID().toString().replace("-", "").substring(0, 24));
            call.put("type", "function");
            call.put("function", function);
            out.add(call);
        }
        m.put("tool_calls", out);
        return m;
    }

    private static String completionPrompt(Map<String, Object> req) {
        Object p = req.get("prompt");
        if (p instanceof List<?> l && l.size() == 1 && l.getFirst() instanceof String s) {
            return s;
        }
        if (!(p instanceof String s)) {
            throw new BadRequest("prompt must be a string");
        }
        return s;
    }

    private Generator.Options options(Map<String, Object> req, boolean chat) {
        long max = Json.integer(req, chat && req.containsKey("max_completion_tokens") ? "max_completion_tokens"
                : "max_tokens", chat ? 512 : 16);
        double temperature = Json.number(req, "temperature", 1);
        double topP = Json.number(req, "top_p", 1);
        long seed = Json.integer(req, "seed", System.nanoTime());
        Sampling sampling = temperature <= 0 ? Sampling.GREEDY
                : Sampling.temperature((float) temperature).withTopP((float) topP);
        List<String> stop = new ArrayList<>();
        Object s = req.get("stop");
        if (s instanceof String one) {
            stop.add(one);
        } else if (s instanceof List<?> l) {
            for (Object o : l) {
                stop.add((String) o);
            }
        }
        Generator.Options options = new Generator.Options((int) Math.min(max, Integer.MAX_VALUE), sampling, seed,
                stopTokens, stop);
        if (req.get("response_format") instanceof Map<?, ?> format) {
            Object type = format.get("type");
            if ("json_schema".equals(type)) {
                Object schema = format.get("json_schema") instanceof Map<?, ?> js ? js.get("schema") : null;
                if (schema == null) {
                    throw new BadRequest("response_format json_schema needs json_schema.schema");
                }
                options = options.withConstraint(Constraint.jsonSchema(Json.object(schema)));
            } else if ("json_object".equals(type)) {
                options = options.withConstraint(Constraint.jsonSchema(Map.of("type", "object")));
            } else if (type != null && !"text".equals(type)) {
                throw new BadRequest("unsupported response_format " + type);
            }
        }
        return options;
    }

    private Map<String, Object> chunk(String id, long created, boolean chat, Object delta, String finish) {
        Map<String, Object> choice = new LinkedHashMap<>();
        choice.put("index", 0);
        choice.put(chat ? "delta" : "text", delta);
        choice.put("finish_reason", finish);
        Map<String, Object> c = new LinkedHashMap<>();
        c.put("id", id);
        c.put("object", chat ? "chat.completion.chunk" : "text_completion");
        c.put("created", created);
        c.put("model", model);
        c.put("choices", List.of(choice));
        return c;
    }

    private static String finish(Generator.Output o) {
        return o.finish() == Generator.Finish.LENGTH ? "length" : "stop";
    }

    private static void event(OutputStream out, Object value) throws IOException {
        out.write(("data: " + Json.write(value) + "\n\n").getBytes(StandardCharsets.UTF_8));
        out.flush();
    }

    private static void json(HttpExchange e, int status, Object value) throws IOException {
        byte[] body = Json.write(value).getBytes(StandardCharsets.UTF_8);
        e.getResponseHeaders().set("Content-Type", "application/json");
        e.sendResponseHeaders(status, body.length);
        try (OutputStream out = e.getResponseBody()) {
            out.write(body);
        }
    }

    private static void error(HttpExchange e, int status, String message) throws IOException {
        Map<String, Object> err = new LinkedHashMap<>();
        err.put("message", message);
        err.put("type", "invalid_request_error");
        json(e, status, Map.of("error", err));
    }
}
