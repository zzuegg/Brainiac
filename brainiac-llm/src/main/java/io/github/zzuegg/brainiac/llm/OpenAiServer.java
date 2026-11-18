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
import java.util.HashMap;
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
 * constrains the answer, as does a GBNF {@code grammar} (llama.cpp's extension). A chat's
 * {@code tools} go to the chat template, and a reply that calls
 * them (Hermes {@code <tool_call>} blocks or a bare JSON call, see {@link Agent}) comes back as
 * {@code tool_calls} with {@code finish_reason} "tool_calls"; earlier calls and {@code tool}
 * messages in the request are rendered for the model. {@code logprobs} (chat: with
 * {@code top_logprobs}) adds token log-probabilities to answers that are not streamed; {@code n}
 * asks for several answers; {@code top_k}, {@code min_p} and {@code repetition_penalty} are read
 * as llama.cpp's server reads them. A thinking model's {@code <think>} part comes back apart, as
 * {@code reasoning_content}. {@code GET /health} answers once the server is up, and
 * {@code POST /tokenize} and {@code /detokenize} convert text. {@link #withEmbedder} and
 * {@link #withReranker} add encoder models: their embeddings at {@code /v1/embeddings} and
 * cross-encoder scores at {@code POST /v1/rerank}. With {@link #requireApiKey}, every
 * request needs {@code Authorization: Bearer <key>}. Requests are answered one at a time, or,
 * with a batch size above one, decoded together by {@link Generator#batcher continuous
 * batching}.
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
            http.createContext("/v1/models", e -> s.guarded(e, s::models));
            http.createContext("/v1/completions", e -> s.guarded(e, x -> s.complete(x, false)));
            http.createContext("/v1/chat/completions", e -> s.guarded(e, x -> s.complete(x, true)));
            http.createContext("/health", e -> s.guarded(e, x -> json(x, 200, Map.of("status", "ok"))));
            http.createContext("/tokenize", e -> s.guarded(e, s::tokenize));
            http.createContext("/v1/embeddings", e -> s.guarded(e, s::embeddings));
            http.createContext("/detokenize", e -> s.guarded(e, s::detokenize));
            http.createContext("/v1/rerank", e -> s.guarded(e, s::rerank));
            http.createContext("/rerank", e -> s.guarded(e, s::rerank));
            http.setExecutor(s.worker);
            http.start();
            return s;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private volatile String apiKey;

    /**
     * Requires every request to carry {@code Authorization: Bearer <key>}; others are answered
     * with status 401.
     *
     * @param key the key, or {@code null} to accept every request
     * @return this server
     */
    public OpenAiServer requireApiKey(String key) {
        this.apiKey = key;
        return this;
    }

    private void guarded(HttpExchange e, com.sun.net.httpserver.HttpHandler handler) throws IOException {
        String key = apiKey;
        if (key != null) {
            String given = e.getRequestHeaders().getFirst("Authorization");
            if (given == null || !java.security.MessageDigest.isEqual(("Bearer " + key).getBytes(StandardCharsets.UTF_8),
                    given.getBytes(StandardCharsets.UTF_8))) {
                try (e) {
                    e.getRequestBody().readAllBytes();
                    error(e, 401, "a valid API key is needed: Authorization: Bearer <key>");
                }
                return;
            }
        }
        handler.handle(e);
    }

    private Map<String, Object> body(HttpExchange e) throws IOException {
        if (!e.getRequestMethod().equals("POST")) {
            throw new BadRequest("use POST");
        }
        return Json.object(Json.parse(new String(e.getRequestBody().readAllBytes(), StandardCharsets.UTF_8)));
    }

    private void tokenize(HttpExchange e) throws IOException {
        try (e) {
            Map<String, Object> req;
            try {
                req = body(e);
            } catch (RuntimeException ex) {
                error(e, 400, ex.getMessage());
                return;
            }
            String text = Json.string(req, "content", "");
            boolean special = Boolean.TRUE.equals(req.get("add_special"));
            List<Object> ids = new ArrayList<>();
            for (int id : tokenizer.encode(text, special)) {
                ids.add(id);
            }
            json(e, 200, Map.of("tokens", ids));
        }
    }

    private void detokenize(HttpExchange e) throws IOException {
        try (e) {
            int[] ids;
            try {
                ids = Json.array(body(e).get("tokens")).stream().mapToInt(o -> ((Number) o).intValue()).toArray();
            } catch (RuntimeException ex) {
                error(e, 400, "tokens must be a list of ids: " + ex.getMessage());
                return;
            }
            json(e, 200, Map.of("content", tokenizer.decode(ids, false)));
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
        List<Object> data = new ArrayList<>();
        java.util.Set<String> names = new java.util.LinkedHashSet<>();
        names.add(model);
        names.addAll(embedders.keySet());
        names.addAll(rerankers.keySet());
        for (String n : names) {
            Map<String, Object> m = new LinkedHashMap<>();
            m.put("id", n);
            m.put("object", "model");
            m.put("owned_by", "brainiac");
            data.add(m);
        }
        json(e, 200, Map.of("object", "list", "data", data));
    }

    /**
     * OpenAI's embeddings: {@code input} is a string, a list of strings, or token ids (one
     * list or a list of lists). The vectors are the model's final hidden states pooled by
     * {@code pooling} ("last", the default, as decoder embedders do, or "mean"), scaled to
     * length 1, and cut to {@code dimensions} (then scaled again) when it is given.
     */
    private void embeddings(HttpExchange e) throws IOException {
        try (e) {
            Map<String, Object> req;
            try {
                req = body(e);
            } catch (RuntimeException ex) {
                error(e, 400, ex.getMessage());
                return;
            }
            Embedder embedder = embedders.get(Json.string(req, "model", ""));
            if (embedder == null && !(generator.model() instanceof Llama) && !embedders.isEmpty()) {
                embedder = embedders.values().iterator().next();
            }
            if (embedder != null) {
                encoderEmbeddings(e, req, embedder);
                return;
            }
            if (!(generator.model() instanceof Llama llama)) {
                error(e, 400, "this model cannot make embeddings");
                return;
            }
            int[][] inputs;
            Llama.Pooling pooling;
            int dimensions;
            try {
                inputs = embeddingInputs(req.get("input"));
                String p = Json.string(req, "pooling", "last");
                pooling = switch (p) {
                    case "last" -> Llama.Pooling.LAST;
                    case "mean" -> Llama.Pooling.MEAN;
                    default -> throw new BadRequest("pooling must be \"last\" or \"mean\"");
                };
                dimensions = (int) Json.integer(req, "dimensions", 0);
                if (dimensions < 0 || "base64".equals(req.get("encoding_format"))) {
                    throw new BadRequest("dimensions must be positive and encoding_format \"float\"");
                }
            } catch (RuntimeException ex) {
                error(e, 400, ex.getMessage());
                return;
            }
            float[][] vectors;
            try {
                int[][] in = inputs;
                java.util.function.Supplier<float[][]> task = () -> llama.embed(in, pooling, true);
                vectors = batcher == null ? task.get() : batcher.run(task).get();
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
                error(e, 500, "interrupted");
                return;
            } catch (java.util.concurrent.ExecutionException | RuntimeException ex) {
                Throwable cause = ex instanceof java.util.concurrent.ExecutionException ? ex.getCause() : ex;
                LOG.log(System.Logger.Level.ERROR, "embeddings failed", cause);
                error(e, 500, String.valueOf(cause.getMessage()));
                return;
            }
            List<Object> data = new ArrayList<>();
            int tokens = 0;
            for (int i = 0; i < vectors.length; i++) {
                float[] v = vectors[i];
                if (dimensions > 0 && dimensions < v.length) {
                    v = java.util.Arrays.copyOf(v, dimensions);
                    double sq = 0;
                    for (float x : v) {
                        sq += (double) x * x;
                    }
                    float inv = (float) (1 / Math.max(Math.sqrt(sq), 1e-12));
                    for (int j = 0; j < v.length; j++) {
                        v[j] *= inv;
                    }
                }
                List<Object> values = new ArrayList<>(v.length);
                for (float x : v) {
                    values.add((double) x);
                }
                Map<String, Object> d = new LinkedHashMap<>();
                d.put("object", "embedding");
                d.put("index", i);
                d.put("embedding", values);
                data.add(d);
                tokens += inputs[i].length;
            }
            Map<String, Object> res = new LinkedHashMap<>();
            res.put("object", "list");
            res.put("data", data);
            res.put("model", model);
            res.put("usage", Map.of("prompt_tokens", tokens, "total_tokens", tokens));
            json(e, 200, res);
        }
    }

    /** An encoder serving embeddings or reranking under a model name. */
    private record Embedder(String name, Encoder encoder, PretrainedTokenizer tokenizer, Encoder.Pooling pooling) {
    }

    private final Map<String, Embedder> embedders = new java.util.concurrent.ConcurrentHashMap<>();
    private final Map<String, Embedder> rerankers = new java.util.concurrent.ConcurrentHashMap<>();

    /**
     * Also serves an encoder's sentence embeddings at {@code /v1/embeddings}, for requests
     * whose {@code model} is {@code name} (or every embeddings request, when the generator's
     * model cannot embed).
     *
     * @param name the model name clients ask for
     * @param encoder the embedding model, such as bge-small or multilingual-e5
     * @param tokenizer its tokenizer
     * @param pooling how its token vectors become one
     * @return this server
     */
    public OpenAiServer withEmbedder(String name, Encoder encoder, PretrainedTokenizer tokenizer,
            Encoder.Pooling pooling) {
        embedders.put(Objects.requireNonNull(name, "name"), new Embedder(name, Objects.requireNonNull(encoder),
                Objects.requireNonNull(tokenizer), Objects.requireNonNull(pooling)));
        return this;
    }

    /**
     * Also serves a cross-encoder reranker at {@code POST /v1/rerank} (the API of llama.cpp,
     * Jina and Cohere): {@code query}, {@code documents} (strings or {@code {"text": ...}}),
     * optional {@code top_n} and {@code return_documents}; the answer's {@code results} hold
     * each document's {@code index} and {@code relevance_score} (the model's logit), best
     * first.
     *
     * @param name the model name clients ask for
     * @param reranker a model with a classification head, such as bge-reranker
     * @param tokenizer its tokenizer
     * @return this server
     */
    public OpenAiServer withReranker(String name, Encoder reranker, PretrainedTokenizer tokenizer) {
        if (!reranker.hasClassifier()) {
            throw new IllegalArgumentException("a reranker needs a classification head");
        }
        rerankers.put(Objects.requireNonNull(name, "name"), new Embedder(name, reranker,
                Objects.requireNonNull(tokenizer), Encoder.Pooling.CLS));
        return this;
    }

    private void encoderEmbeddings(HttpExchange e, Map<String, Object> req, Embedder embedder) throws IOException {
        List<String> texts = new ArrayList<>();
        Object input = req.get("input");
        for (Object o : input instanceof List<?> l ? l : input == null ? List.of() : List.of(input)) {
            if (!(o instanceof String s)) {
                error(e, 400, "input must be a string or a list of strings for " + embedder.name);
                return;
            }
            texts.add(s);
        }
        if (texts.isEmpty() || texts.size() > 2048) {
            error(e, 400, "input must hold 1 to 2048 strings");
            return;
        }
        int dimensions = (int) Json.integer(req, "dimensions", 0);
        float[][] vectors;
        try {
            vectors = encoderTask(() -> embedder.encoder.embed(embedder.tokenizer, texts, embedder.pooling, true));
        } catch (RuntimeException ex) {
            LOG.log(System.Logger.Level.ERROR, "embeddings failed", ex);
            error(e, 500, String.valueOf(ex.getMessage()));
            return;
        }
        List<Object> data = new ArrayList<>();
        int tokens = 0;
        for (int i = 0; i < vectors.length; i++) {
            float[] v = vectors[i];
            if (dimensions > 0 && dimensions < v.length) {
                v = java.util.Arrays.copyOf(v, dimensions);
                double sq = 0;
                for (float x : v) {
                    sq += (double) x * x;
                }
                float inv = (float) (1 / Math.max(Math.sqrt(sq), 1e-12));
                for (int j = 0; j < v.length; j++) {
                    v[j] *= inv;
                }
            }
            List<Object> values = new ArrayList<>(v.length);
            for (float x : v) {
                values.add((double) x);
            }
            data.add(Map.of("object", "embedding", "index", i, "embedding", values));
            tokens += embedder.tokenizer.encode(texts.get(i)).length;
        }
        Map<String, Object> res = new LinkedHashMap<>();
        res.put("object", "list");
        res.put("data", data);
        res.put("model", embedder.name);
        res.put("usage", Map.of("prompt_tokens", tokens, "total_tokens", tokens));
        json(e, 200, res);
    }

    /** Runs encoder work on the batcher's thread when there is one, so it never meets decoding. */
    private float[][] encoderTask(java.util.function.Supplier<float[][]> task) {
        if (batcher == null) {
            return task.get();
        }
        try {
            return batcher.run(task).get();
        } catch (InterruptedException ex) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted", ex);
        } catch (java.util.concurrent.ExecutionException ex) {
            throw ex.getCause() instanceof RuntimeException r ? r : new IllegalStateException(ex.getCause());
        }
    }

    private void rerank(HttpExchange e) throws IOException {
        try (e) {
            if (!e.getRequestMethod().equals("POST")) {
                error(e, 405, "use POST");
                return;
            }
            Map<String, Object> req;
            String query;
            List<String> docs = new ArrayList<>();
            Embedder reranker;
            int top;
            boolean returnDocuments;
            try {
                req = body(e);
                reranker = rerankers.get(Json.string(req, "model", ""));
                if (reranker == null && rerankers.size() == 1) {
                    reranker = rerankers.values().iterator().next();
                }
                if (reranker == null) {
                    throw new BadRequest(rerankers.isEmpty() ? "no reranker is served"
                            : "model must be one of " + rerankers.keySet());
                }
                query = Json.string(req, "query", null);
                if (query == null) {
                    throw new BadRequest("query is required");
                }
                if (!(req.get("documents") instanceof List<?> list) || list.isEmpty() || list.size() > 1024) {
                    throw new BadRequest("documents must be a list of 1 to 1024 texts");
                }
                for (Object d : list) {
                    if (d instanceof String s) {
                        docs.add(s);
                    } else if (d instanceof Map<?, ?> m && m.get("text") instanceof String s) {
                        docs.add(s);
                    } else {
                        throw new BadRequest("each document must be a string or {\"text\": ...}");
                    }
                }
                top = (int) Json.integer(req, "top_n", docs.size());
                returnDocuments = Boolean.TRUE.equals(req.get("return_documents"));
            } catch (RuntimeException ex) {
                error(e, 400, ex.getMessage());
                return;
            }
            float[] scores;
            try {
                Embedder r = reranker;
                scores = encoderTask(() -> new float[][] {r.encoder.rerank(r.tokenizer, query, docs)})[0];
            } catch (RuntimeException ex) {
                LOG.log(System.Logger.Level.ERROR, "rerank failed", ex);
                error(e, 500, String.valueOf(ex.getMessage()));
                return;
            }
            Integer[] order = new Integer[docs.size()];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            java.util.Arrays.sort(order, (a, b) -> Float.compare(scores[b], scores[a]));
            List<Object> results = new ArrayList<>();
            for (int i = 0; i < Math.min(Math.max(top, 0), order.length); i++) {
                Map<String, Object> r = new LinkedHashMap<>();
                r.put("index", order[i]);
                r.put("relevance_score", (double) scores[order[i]]);
                if (returnDocuments) {
                    r.put("document", Map.of("text", docs.get(order[i])));
                }
                results.add(r);
            }
            int tokens = reranker.tokenizer.encode(query).length * docs.size();
            for (String d : docs) {
                tokens += reranker.tokenizer.encode(d).length;
            }
            Map<String, Object> res = new LinkedHashMap<>();
            res.put("model", reranker.name);
            res.put("object", "list");
            res.put("results", results);
            res.put("usage", Map.of("prompt_tokens", tokens, "total_tokens", tokens));
            json(e, 200, res);
        }
    }

    private int[][] embeddingInputs(Object input) {
        List<?> list = input instanceof List<?> l ? l : input instanceof String s ? List.of(s) : null;
        if (list == null || list.isEmpty()) {
            throw new BadRequest("input must be a string, a list of strings or token ids");
        }
        if (list.getFirst() instanceof Number) {
            list = List.of(list);   // one list of token ids
        }
        if (list.size() > 2048) {
            throw new BadRequest("at most 2048 inputs");
        }
        int[][] out = new int[list.size()][];
        for (int i = 0; i < out.length; i++) {
            Object o = list.get(i);
            if (o instanceof String s) {
                out[i] = generator.encode(s);
            } else if (o instanceof List<?> ids) {
                out[i] = new int[ids.size()];
                for (int j = 0; j < out[i].length; j++) {
                    if (!(ids.get(j) instanceof Number n)) {
                        throw new BadRequest("token ids must be numbers");
                    }
                    out[i][j] = n.intValue();
                }
            } else {
                throw new BadRequest("input must be a string, a list of strings or token ids");
            }
            if (out[i].length == 0) {
                throw new BadRequest("input " + i + " is empty");
            }
        }
        return out;
    }

    /** A request that cannot be served, answered with status 400. */
    private static final class BadRequest extends RuntimeException {
        private static final long serialVersionUID = 1;

        BadRequest(String message) {
            super(message);
        }
    }

    private static final System.Logger LOG = System.getLogger(OpenAiServer.class.getName());

    private void complete(HttpExchange e, boolean chat) throws IOException {
        try (e) {
            try {
                answer(e, chat);
            } catch (IOException ex) {
                throw ex;
            } catch (RuntimeException | Error ex) {
                // HttpServer drops a handler's exception and the client sees no answer at all
                LOG.log(System.Logger.Level.ERROR, "request failed", ex);
                if (e.getResponseCode() < 0) {
                    error(e, 500, ex.getClass().getSimpleName() + ": " + ex.getMessage());
                } else {
                    e.getResponseBody().write(("data: " + Json.write(Map.of("error", Map.of("message",
                            String.valueOf(ex.getMessage()), "type", "server_error"))) + "\n\n")
                            .getBytes(StandardCharsets.UTF_8));
                }
                if (ex instanceof Error err && !(ex instanceof OutOfMemoryError)) {
                    throw err;
                }
            }
        }
    }

    private void answer(HttpExchange e, boolean chat) throws IOException {
        {
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
            List<java.awt.image.BufferedImage> images = new ArrayList<>();
            try {
                prompt = chat ? chatPrompt(req, images) : completionPrompt(req);
                if (!images.isEmpty() && !(generator.model() instanceof VisionLanguageModel)) {
                    throw new BadRequest("the model does not read images");
                }
                options = options(req, chat);
            } catch (BadRequest | IllegalArgumentException | ClassCastException ex) {
                error(e, 400, ex.getMessage());
                return;
            }
            boolean stream = Boolean.TRUE.equals(req.get("stream"));
            long nl = Json.integer(req, "n", 1);
            if (nl < 1 || nl > 16 || stream && nl > 1) {
                error(e, 400, "n must be 1 to 16, and 1 when streaming");
                return;
            }
            int n = (int) nl;
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
                    Generator.Output o = run(prompt, images, options, null);
                    Map<String, Object> message = assistant(o.text(), true);
                    message.remove("role");
                    if (message.get("tool_calls") instanceof List<?> calls) {
                        // streamed calls carry their place, which clients gather deltas by
                        List<Object> indexed = new ArrayList<>();
                        for (int i = 0; i < calls.size(); i++) {
                            Map<String, Object> c = new LinkedHashMap<>();
                            c.put("index", i);
                            c.putAll(Json.object(calls.get(i)));
                            indexed.add(c);
                        }
                        message.put("tool_calls", indexed);
                    }
                    if (message.get("content") == null) {
                        message.remove("content");
                    }
                    event(out, chunk(id, created, true, message, null));
                    event(out, chunk(id, created, true, Map.of(), message.containsKey("tool_calls") ? "tool_calls" : finish(o)));
                    out.write("data: [DONE]\n\n".getBytes(StandardCharsets.UTF_8));
                    out.flush();
                    return;
                }
                Generator.Output o = run(prompt, images, options, (row, text) -> {
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
            List<Object> choices = new ArrayList<>();
            int completion = 0;
            for (int i = 0; i < n; i++) {
                Generator.Options oi = i == 0 ? options : new Generator.Options(options.maxNewTokens(),
                        options.sampling(), options.seed() + i, options.stopTokens(), options.stopStrings(),
                        options.constraint(), options.logprobs());
                Generator.Output o = run(prompt, images, oi, null);
                completion += o.tokens().length;
                Map<String, Object> choice = new LinkedHashMap<>();
                choice.put("index", i);
                Map<String, Object> message = chat ? assistant(o.text(), tools) : null;
                if (chat) {
                    choice.put("message", message);
                } else {
                    choice.put("text", o.text());
                }
                choice.put("finish_reason", message != null && message.containsKey("tool_calls") ? "tool_calls"
                        : finish(o));
                if (o.logprobs() != null) {
                    choice.put("logprobs", chat ? chatLogprobs(o.logprobs()) : completionLogprobs(o.logprobs()));
                }
                choices.add(choice);
            }
            Map<String, Object> usage = new LinkedHashMap<>();
            usage.put("prompt_tokens", promptTokens);
            usage.put("completion_tokens", completion);
            usage.put("total_tokens", promptTokens + completion);
            Map<String, Object> res = new LinkedHashMap<>();
            res.put("id", id);
            res.put("object", chat ? "chat.completion" : "text_completion");
            res.put("created", created);
            res.put("model", model);
            res.put("choices", choices);
            res.put("usage", usage);
            json(e, 200, res);
        }
    }

    private Generator.Output run(String prompt, List<java.awt.image.BufferedImage> images, Generator.Options options,
            Generator.Listener listener) {
        if (!images.isEmpty()) {
            // the images' features are queued for the prefill that reads them, so the whole
            // answer runs on the thread that drives the model
            VisionLanguageModel vlm = (VisionLanguageModel) generator.model();
            java.util.function.Supplier<Generator.Output> task = () -> {
                int[] ids = vlm.prompt(generator.encode(vlm.markImages(prompt)), images);
                return generator.generate(new int[][] {ids}, options, listener).getFirst();
            };
            if (batcher == null) {
                synchronized (generator) {
                    return task.get();
                }
            }
            try {
                return batcher.run(task).get();
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
                throw new IllegalStateException("interrupted", ex);
            } catch (java.util.concurrent.ExecutionException ex) {
                throw ex.getCause() instanceof RuntimeException r ? r : new IllegalStateException(ex.getCause());
            }
        }
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

    private String chatPrompt(Map<String, Object> req, List<java.awt.image.BufferedImage> images) {
        if (tokenizer.chatTemplate() == null) {
            throw new BadRequest("the model has no chat template; use /v1/completions");
        }
        Object messages = req.get("messages");
        if (!(messages instanceof List<?> list) || list.isEmpty()) {
            throw new BadRequest("messages must be a non-empty list");
        }
        List<Map<String, Object>> rendered = new ArrayList<>();
        for (Object m : list) {
            rendered.add(message(withImages(Json.object(m), images)));
        }
        List<Object> tools = tools(req);
        return tools == null ? tokenizer.applyChatTemplate(rendered, true)
                : tokenizer.applyChatTemplate(rendered, true, Map.of("tools", tools));
    }

    /**
     * A message whose content is a list of parts: text parts joined into one string when
     * there is no image (as text-only chat templates expect), image parts ({@code image_url}
     * with a {@code data:} URL, as OpenAI clients send them) decoded into {@code images} and
     * left as {@code {"type": "image"}} for the template to mark.
     */
    private static Map<String, Object> withImages(Map<String, Object> m, List<java.awt.image.BufferedImage> images) {
        if (!(m.get("content") instanceof List<?> parts)) {
            return m;
        }
        List<Object> out = new ArrayList<>();
        StringBuilder text = new StringBuilder();
        boolean any = false;
        for (Object o : parts) {
            Map<String, Object> part = Json.object(o);
            String type = Json.string(part, "type", "text");
            switch (type) {
                case "text" -> {
                    String t = Json.string(part, "text", "");
                    text.append(t);
                    out.add(Map.of("type", "text", "text", t));
                }
                case "image_url", "image" -> {
                    Object url = part.get("image_url") instanceof Map<?, ?> u ? u.get("url") : part.get("image_url");
                    if (!(url instanceof String s)) {
                        throw new BadRequest("an image part needs image_url.url");
                    }
                    images.add(decodeImage(s));
                    out.add(Map.of("type", "image"));
                    any = true;
                }
                default -> throw new BadRequest("content parts of type " + type + " are not supported");
            }
        }
        Map<String, Object> copy = new LinkedHashMap<>(m);
        copy.put("content", any ? out : text.toString());
        return copy;
    }

    /** An image from a {@code data:} URL; the server fetches nothing from the network. */
    private static java.awt.image.BufferedImage decodeImage(String url) {
        if (!url.startsWith("data:")) {
            throw new BadRequest("images must be data: URLs (base64)");
        }
        int comma = url.indexOf(',');
        if (comma < 0 || !url.substring(0, comma).endsWith(";base64")) {
            throw new BadRequest("images must be base64 data: URLs");
        }
        try {
            byte[] bytes = java.util.Base64.getDecoder().decode(url.substring(comma + 1).strip());
            java.awt.image.BufferedImage image = javax.imageio.ImageIO.read(new java.io.ByteArrayInputStream(bytes));
            if (image == null) {
                throw new BadRequest("the image is not in a format ImageIO reads (PNG, JPEG, BMP, GIF)");
            }
            return image;
        } catch (IllegalArgumentException | IOException ex) {
            throw new BadRequest("the image cannot be decoded: " + ex.getMessage());
        }
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
    private static final java.util.regex.Pattern HERMES_OPEN = java.util.regex.Pattern.compile("<tool_call>");

    private static Map<String, Object> assistant(String text, boolean tools) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", "assistant");
        // a thinking model's reasoning, apart from its answer
        int open = text.indexOf("<think>");
        int close = text.indexOf("</think>");
        if (close >= 0 && (open < 0 || open < close) && text.substring(0, open < 0 ? close : open).isBlank()) {
            m.put("reasoning_content", text.substring(open < 0 ? 0 : open + "<think>".length(), close).strip());
            text = text.substring(close + "</think>".length()).stripLeading();
        }
        List<Map<String, Object>> calls = tools ? Agent.calls(text.strip()) : List.of();
        if (calls.isEmpty()) {
            m.put("content", text);
            return m;
        }
        String rest = Agent.content(text.strip());
        if (!HERMES_OPEN.matcher(text).find()) {
            rest = "";   // the whole turn was the call, as JSON
        }
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
                : Sampling.temperature((float) temperature).withTopP((float) topP)
                        .withTopK((int) Json.integer(req, "top_k", 0)).withMinP((float) Json.number(req, "min_p", 0));
        double penalty = Json.number(req, "repetition_penalty", 1);
        if (penalty != 1) {
            sampling = sampling.withRepetitionPenalty((float) penalty);
        }
        double presence = Json.number(req, "presence_penalty", 0);
        double frequency = Json.number(req, "frequency_penalty", 0);
        if (Math.abs(presence) > 2 || Math.abs(frequency) > 2) {
            throw new BadRequest("presence_penalty and frequency_penalty must be between -2 and 2");
        }
        sampling = sampling.withPenalties((float) presence, (float) frequency);
        if (req.get("logit_bias") instanceof Map<?, ?> bias && !bias.isEmpty()) {
            Map<Integer, Float> b = new HashMap<>();
            for (Map.Entry<?, ?> e : bias.entrySet()) {
                int token;
                try {
                    token = Integer.parseInt(String.valueOf(e.getKey()));
                } catch (NumberFormatException x) {
                    throw new BadRequest("logit_bias keys must be token ids, got " + e.getKey());
                }
                if (token < 0 || !(e.getValue() instanceof Number n)
                        || Math.abs(n.doubleValue()) > 100) {
                    throw new BadRequest("logit_bias needs token ids and biases from -100 to 100");
                }
                b.put(token, n.floatValue());
            }
            sampling = sampling.withLogitBias(b);
        }
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
        // chat: "logprobs": true with "top_logprobs": n; completions: "logprobs": n
        Object lp = req.get("logprobs");
        if (chat ? Boolean.TRUE.equals(lp) : lp instanceof Number) {
            long top = chat ? Json.integer(req, "top_logprobs", 0) : ((Number) lp).longValue();
            if (top < 0 || top > 20) {
                throw new BadRequest("top_logprobs must be 0 to 20");
            }
            options = options.withLogprobs((int) top);
        }
        Object choice = req.get("tool_choice");
        List<Object> tools = chat ? tools(req) : null;
        if (tools != null && choice != null && !"auto".equals(choice)) {
            // "required" or one named function: the reply is a call, its arguments held to the schema
            String wanted = null;
            if (choice instanceof Map<?, ?> c && c.get("function") instanceof Map<?, ?> f && f.get("name") instanceof String n) {
                wanted = n;
            } else if (!"required".equals(choice)) {
                throw new BadRequest("tool_choice must be \"none\", \"auto\", \"required\" or a function");
            }
            List<Object> any = new ArrayList<>();
            for (Object t : tools) {
                Map<String, Object> f = Json.object(Json.object(t).get("function"));
                if (wanted != null && !wanted.equals(f.get("name"))) {
                    continue;
                }
                Object params = f.get("parameters") instanceof Map<?, ?> m ? m : Map.of("type", "object");
                // ordered: the name first, then the arguments (Map.of would shuffle them)
                Map<String, Object> properties = new LinkedHashMap<>();
                properties.put("name", Map.of("const", f.get("name")));
                properties.put("arguments", params);
                Map<String, Object> call = new LinkedHashMap<>();
                call.put("type", "object");
                call.put("properties", properties);
                call.put("required", List.of("name", "arguments"));
                any.add(call);
            }
            if (any.isEmpty()) {
                throw new BadRequest("tool_choice names " + wanted + ", which is not among the tools");
            }
            options = options.withConstraint(Constraint.jsonSchema(any.size() == 1
                    ? Json.object(any.getFirst()) : Map.of("anyOf", any)));
            return options;
        }
        if (req.get("response_format") instanceof Map<?, ?> format) {
            Object type = format.get("type");
            if ("json_schema".equals(type)) {
                Object schema = format.get("json_schema") instanceof Map<?, ?> js ? js.get("schema") : null;
                if (schema == null) {
                    throw new BadRequest("response_format json_schema needs json_schema.schema");
                }
                options = options.withConstraint(Constraint.jsonSchema(Json.object(schema)));
            } else if ("json_object".equals(type)) {
                options = options.withConstraint(Constraint.jsonObject());
            } else if (type != null && !"text".equals(type)) {
                throw new BadRequest("unsupported response_format " + type);
            }
        }
        if (req.get("grammar") instanceof String gbnf && !gbnf.isBlank()) {
            // llama.cpp's server extension
            try {
                options = options.withConstraint(Constraint.grammar(gbnf));
            } catch (IllegalArgumentException e) {
                throw new BadRequest(e.getMessage());
            }
        }
        return options;
    }

    private String token(int id) {
        return tokenizer.decode(new int[] {id});
    }

    private static List<Object> bytes(String token) {
        List<Object> out = new ArrayList<>();
        for (byte b : token.getBytes(StandardCharsets.UTF_8)) {
            out.add((long) (b & 0xff));
        }
        return out;
    }

    private Map<String, Object> chatLogprobs(List<Generator.TokenLogprob> logprobs) {
        List<Object> content = new ArrayList<>();
        for (Generator.TokenLogprob t : logprobs) {
            Map<String, Object> e = entry(t.token(), t.logprob());
            List<Object> top = new ArrayList<>();
            for (int i = 0; i < t.top().length; i++) {
                top.add(entry(t.top()[i], t.topLogprobs()[i]));
            }
            e.put("top_logprobs", top);
            content.add(e);
        }
        return Map.of("content", content);
    }

    private Map<String, Object> entry(int id, float logprob) {
        String text = token(id);
        Map<String, Object> e = new LinkedHashMap<>();
        e.put("token", text);
        e.put("logprob", (double) logprob);
        e.put("bytes", bytes(text));
        return e;
    }

    private Map<String, Object> completionLogprobs(List<Generator.TokenLogprob> logprobs) {
        List<Object> tokens = new ArrayList<>();
        List<Object> values = new ArrayList<>();
        List<Object> tops = new ArrayList<>();
        List<Object> offsets = new ArrayList<>();
        long offset = 0;
        for (Generator.TokenLogprob t : logprobs) {
            String text = token(t.token());
            tokens.add(text);
            values.add((double) t.logprob());
            Map<String, Object> top = new LinkedHashMap<>();
            for (int i = 0; i < t.top().length; i++) {
                top.putIfAbsent(token(t.top()[i]), (double) t.topLogprobs()[i]);
            }
            tops.add(top);
            offsets.add(offset);
            offset += text.length();
        }
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("tokens", tokens);
        out.put("token_logprobs", values);
        out.put("top_logprobs", tops);
        out.put("text_offset", offsets);
        return out;
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
        err.put("type", status >= 500 ? "server_error" : "invalid_request_error");
        json(e, status, Map.of("error", err));
    }
}
