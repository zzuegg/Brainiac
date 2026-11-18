package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.tensor.DType;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

class OpenAiServerTest {

    private static final String CHARS = "abcdefghijklmnopqrstuvwxyz :{}\"0123456789,\n";
    private static OpenAiServer server;
    private static Generator generator;
    private static PretrainedTokenizer tokenizer;

    @BeforeAll
    static void start() {
        Map<String, Object> vocab = new LinkedHashMap<>();
        for (int i = 0; i < CHARS.length(); i++) {
            vocab.put(String.valueOf(CHARS.charAt(i)), i);
        }
        Map<String, Object> root = Map.of("model", Map.of("type", "BPE", "vocab", vocab, "merges", List.of()),
                "added_tokens", List.of(Map.of("id", CHARS.length(), "content", "<end>", "special", true)));
        PretrainedTokenizer tok = PretrainedTokenizer.fromJson(Json.write(root), Map.of("eos_token", "<end>",
                "chat_template", "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}"
                        + "{% if add_generation_prompt %}assistant: {% endif %}"));
        Llama model = Llama.random(LlamaConfig.small(CHARS.length() + 1, 32, 2, 4, 2), 5, GeneratorTest.DEVICE,
                DType.FLOAT32);
        generator = new Generator(model, tok);
        tokenizer = tok;
        server = OpenAiServer.start(new InetSocketAddress("127.0.0.1", 0), "tiny", generator, tok, null);
    }

    @AfterAll
    static void stop() {
        server.close();
    }

    private static String call(String method, String path, String body) throws IOException {
        HttpURLConnection c = (HttpURLConnection) URI.create("http://127.0.0.1:" + server.port() + path).toURL()
                .openConnection();
        c.setRequestMethod(method);
        if (body != null) {
            c.setDoOutput(true);
            try (OutputStream out = c.getOutputStream()) {
                out.write(body.getBytes(StandardCharsets.UTF_8));
            }
        }
        int status = c.getResponseCode();
        try (InputStream in = status < 400 ? c.getInputStream() : c.getErrorStream()) {
            return status + " " + new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    @Test
    void healthTokensSeveralAnswersAndKeys() throws IOException {
        assertTrue(call("GET", "/health", null).startsWith("200 {\"status\":\"ok\"}"));
        String t = call("POST", "/tokenize", "{\"content\":\"hello\"}");
        assertTrue(t.startsWith("200 "), t);
        String ids = Json.write(Json.object(Json.parse(t.substring(4))).get("tokens"));
        assertEquals("200 {\"content\":\"hello\"}", call("POST", "/detokenize", "{\"tokens\":" + ids + "}"));

        String r = call("POST", "/v1/completions", "{\"prompt\":\"hello\",\"max_tokens\":4,\"n\":3,\"seed\":7}");
        assertTrue(r.startsWith("200 "), r);
        List<Object> choices = Json.array(Json.object(Json.parse(r.substring(4))).get("choices"));
        assertEquals(3, choices.size());
        assertEquals(2L, ((Number) Json.object(choices.get(2)).get("index")).longValue());
        assertTrue(call("POST", "/v1/completions", "{\"prompt\":\"x\",\"n\":2,\"stream\":true}").startsWith("400 "));

        // a bias of 100 on one token makes greedy decoding write only it
        int first = ((Number) Json.array(Json.object(Json.parse(t.substring(4))).get("tokens")).getFirst()).intValue();
        String biased = call("POST", "/v1/completions", "{\"prompt\":\"hello\",\"max_tokens\":3,\"temperature\":0,"
                + "\"logit_bias\":{\"" + first + "\":100},\"presence_penalty\":0.5,\"frequency_penalty\":0.1}");
        assertTrue(biased.startsWith("200 "), biased);
        String three = call("POST", "/detokenize", "{\"tokens\":[" + first + "," + first + "," + first + "]}");
        assertEquals(Json.object(Json.parse(three.substring(4))).get("content"),
                Json.object(Json.array(Json.object(Json.parse(biased.substring(4))).get("choices")).getFirst()).get("text"));
        assertTrue(call("POST", "/v1/completions", "{\"prompt\":\"x\",\"presence_penalty\":3}").startsWith("400 "));
        assertTrue(call("POST", "/v1/completions", "{\"prompt\":\"x\",\"logit_bias\":{\"a\":1}}").startsWith("400 "));

        // embeddings: one per input, length 1, cut to the dimensions asked for
        String emb = call("POST", "/v1/embeddings", "{\"input\":[\"hello\",\"a longer text\"],\"dimensions\":8}");
        assertTrue(emb.startsWith("200 "), emb);
        List<Object> data = Json.array(Json.object(Json.parse(emb.substring(4))).get("data"));
        assertEquals(2, data.size());
        List<Object> vec = Json.array(Json.object(data.get(1)).get("embedding"));
        assertEquals(8, vec.size());
        double sq = 0;
        for (Object v : vec) {
            sq += Math.pow(((Number) v).doubleValue(), 2);
        }
        assertEquals(1, sq, 1e-4);
        assertTrue(call("POST", "/v1/embeddings", "{\"input\":[]}").startsWith("400 "));

        server.requireApiKey("secret");
        try {
            assertTrue(call("GET", "/v1/models", null).startsWith("401 "));
            HttpURLConnection c = (HttpURLConnection) URI.create("http://127.0.0.1:" + server.port() + "/v1/models")
                    .toURL().openConnection();
            c.setRequestProperty("Authorization", "Bearer secret");
            assertEquals(200, c.getResponseCode());
        } finally {
            server.requireApiKey(null);
        }
    }

    @Test
    void listsTheModel() throws IOException {
        String r = call("GET", "/v1/models", null);
        assertTrue(r.startsWith("200 "), r);
        assertTrue(r.contains("\"id\":\"tiny\""), r);
    }

    @Test
    void completesLikeTheGenerator() throws IOException {
        String r = call("POST", "/v1/completions", "{\"prompt\":\"hello\",\"max_tokens\":6,\"temperature\":0}");
        assertTrue(r.startsWith("200 "), r);
        Map<String, Object> res = Json.object(Json.parse(r.substring(4)));
        Map<String, Object> choice = Json.object(Json.array(res.get("choices")).getFirst());
        assertEquals(generator.generate("hello", Generator.Options.greedy(6)), choice.get("text"));
        Map<String, Object> usage = Json.object(res.get("usage"));
        assertEquals(5L, ((Number) usage.get("prompt_tokens")).longValue());
    }

    @Test
    void chatStreamsTheSameTextInPieces() throws IOException {
        String body = "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"temperature\":0";
        String whole = call("POST", "/v1/chat/completions", body + "}");
        assertTrue(whole.startsWith("200 "), whole);
        Map<String, Object> msg = Json.object(Json.object(Json.array(Json.object(Json.parse(whole.substring(4)))
                .get("choices")).getFirst()).get("message"));
        String expected = generator.generate("user: hi\nassistant: ", Generator.Options.greedy(8));
        assertEquals(expected, msg.get("content"));

        String stream = call("POST", "/v1/chat/completions", body + ",\"stream\":true}");
        assertTrue(stream.startsWith("200 "), stream);
        assertTrue(stream.strip().endsWith("data: [DONE]"), stream);
        StringBuilder text = new StringBuilder();
        for (String line : stream.substring(4).split("\n")) {
            if (line.startsWith("data: {")) {
                Map<String, Object> delta = Json.object(Json.object(Json.array(Json.object(Json.parse(line.substring(6)))
                        .get("choices")).getFirst()).get("delta"));
                if (delta.get("content") instanceof String s) {
                    text.append(s);
                }
            }
        }
        assertEquals(expected, text.toString());
    }

    @Test
    void jsonSchemaConstrainsTheAnswer() throws IOException {
        String r = call("POST", "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],"
                + "\"max_tokens\":60,\"temperature\":0,\"response_format\":{\"type\":\"json_schema\",\"json_schema\":"
                + "{\"name\":\"n\",\"schema\":{\"type\":\"object\",\"properties\":{\"a\":{\"type\":\"integer\"}},"
                + "\"required\":[\"a\"]}}}}");
        assertTrue(r.startsWith("200 "), r);
        String content = (String) Json.object(Json.object(Json.array(Json.object(Json.parse(r.substring(4)))
                .get("choices")).getFirst()).get("message")).get("content");
        assertTrue(Json.object(Json.parse(content)).get("a") instanceof Number, content);
    }

    @Test
    void grammarConstrainsTheAnswer() throws IOException {
        String r = call("POST", "/v1/completions", "{\"prompt\":\"x\",\"max_tokens\":30,\"temperature\":0,"
                + "\"grammar\":\"root ::= \\\"(\\\" [a-c]+ \\\")\\\"\"}");
        assertTrue(r.startsWith("200 "), r);
        String text = (String) Json.object(Json.array(Json.object(Json.parse(r.substring(4))).get("choices"))
                .getFirst()).get("text");
        assertTrue(Grammar.parse("root ::= \"(\" [a-c]+ \")\"").start().feed(text) != null, text);
        assertTrue(call("POST", "/v1/completions", "{\"prompt\":\"x\",\"grammar\":\"root ::= nothing\"}")
                .startsWith("400 "));
    }

    @Test
    void badRequestsGet400() throws IOException {
        assertTrue(call("POST", "/v1/completions", "not json").startsWith("400 "));
        assertTrue(call("POST", "/v1/chat/completions", "{\"messages\":[]}").startsWith("400 "));
        assertTrue(call("GET", "/v1/completions", null).startsWith("405 "));
    }

    @Test
    void batchedServerAnswersConcurrentRequestsLikeTheGenerator() throws Exception {
        try (OpenAiServer batched = OpenAiServer.start(new InetSocketAddress("127.0.0.1", 0), "tiny", generator,
                tokenizer, null, 4)) {
            OpenAiServer saved = server;
            server = batched;
            try {
                String[] prompts = {"abc", "hello world", "q", "zebra"};
                java.util.List<java.util.concurrent.Future<String>> answers = new java.util.ArrayList<>();
                java.util.concurrent.ExecutorService pool = java.util.concurrent.Executors.newFixedThreadPool(4);
                for (String p : prompts) {
                    answers.add(pool.submit(() -> call("POST", "/v1/completions", "{\"prompt\":\"" + p
                            + "\",\"max_tokens\":10,\"temperature\":0}")));
                }
                // embeddings run on the decoding thread between steps
                java.util.concurrent.Future<String> emb = pool.submit(() -> call("POST", "/v1/embeddings",
                        "{\"input\":\"hello\"}"));
                assertTrue(emb.get().startsWith("200 "), emb.get());
                // every answer is awaited before the generator is asked itself: a row can end early, and the
                // server would still be decoding the others with the same generator
                java.util.List<String> got = new java.util.ArrayList<>();
                for (java.util.concurrent.Future<String> answer : answers) {
                    got.add(answer.get());
                }
                for (int i = 0; i < prompts.length; i++) {
                    String r = got.get(i);
                    assertTrue(r.startsWith("200 "), r);
                    Map<String, Object> choice = Json.object(Json.array(Json.object(Json.parse(r.substring(4)))
                            .get("choices")).getFirst());
                    assertEquals(generator.generate(prompts[i], Generator.Options.greedy(10)), choice.get("text"));
                }
                pool.shutdown();
            } finally {
                server = saved;
            }
        }
    }

    @Test
    void logprobsMatchTheModel() throws IOException {
        Generator.Output o = generator.generate(new int[][] {tokenizer.encode("hello", false)},
                Generator.Options.greedy(4).withLogprobs(3), null).getFirst();
        List<Generator.TokenLogprob> lps = o.logprobs();
        assertEquals(o.tokens().length, lps.size());
        for (Generator.TokenLogprob t : lps) {
            // greedy: the written token is the most likely one
            assertEquals(t.token(), t.top()[0]);
            assertEquals(t.logprob(), t.topLogprobs()[0], 0);
            assertTrue(t.topLogprobs()[0] >= t.topLogprobs()[1] && t.topLogprobs()[1] >= t.topLogprobs()[2]);
            assertTrue(t.logprob() <= 0);
        }

        String r = call("POST", "/v1/chat/completions", "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],"
                + "\"max_tokens\":4,\"temperature\":0,\"logprobs\":true,\"top_logprobs\":2}");
        assertTrue(r.startsWith("200 "), r);
        Map<String, Object> choice = Json.object(Json.array(Json.object(Json.parse(r.substring(4))).get("choices")).getFirst());
        List<Object> content = Json.array(Json.object(choice.get("logprobs")).get("content"));
        assertEquals(4, content.size());
        Map<String, Object> first = Json.object(content.getFirst());
        assertTrue(first.get("token") instanceof String);
        assertEquals(2, Json.array(first.get("top_logprobs")).size());

        r = call("POST", "/v1/completions", "{\"prompt\":\"hello\",\"max_tokens\":3,\"temperature\":0,\"logprobs\":1}");
        choice = Json.object(Json.array(Json.object(Json.parse(r.substring(4))).get("choices")).getFirst());
        Map<String, Object> lp = Json.object(choice.get("logprobs"));
        assertEquals(3, Json.array(lp.get("token_logprobs")).size());
        assertEquals(choice.get("text"), String.join("", Json.array(lp.get("tokens")).stream().map(String::valueOf).toList()));
    }

    @Test
    void servesEncoderEmbeddingsAndReranking() throws IOException {
        Encoder bert = Encoder.load(LlamaTest.resource("bert"), LlamaTest.DEVICE, DType.FLOAT32);
        PretrainedTokenizer bertTok = PretrainedTokenizer.load(LlamaTest.resource("bert"));
        Encoder reranker = Encoder.load(LlamaTest.resource("bert-rerank"), LlamaTest.DEVICE, DType.FLOAT32);
        PretrainedTokenizer rerankTok = PretrainedTokenizer.load(LlamaTest.resource("bert-rerank"));
        server.withEmbedder("bert", bert, bertTok, Encoder.Pooling.MEAN).withReranker("rr", reranker, rerankTok);
        String models = call("GET", "/v1/models", null);
        assertTrue(models.contains("\"id\":\"bert\"") && models.contains("\"id\":\"rr\""), models);

        String r = call("POST", "/v1/embeddings", "{\"model\":\"bert\",\"input\":[\"the cat sat\",\"a dog ran fast .\"]}");
        assertTrue(r.startsWith("200 "), r);
        List<Object> data = Json.array(Json.object(Json.parse(r.substring(4))).get("data"));
        float[][] expected = bert.embed(bertTok, List.of("the cat sat", "a dog ran fast ."), Encoder.Pooling.MEAN, true);
        for (int i = 0; i < 2; i++) {
            List<Object> v = Json.array(Json.object(data.get(i)).get("embedding"));
            for (int j = 0; j < expected[i].length; j++) {
                assertEquals(expected[i][j], ((Number) v.get(j)).floatValue(), 1e-6f);
            }
        }

        List<String> docs = List.of("stock market", "the cat sat on the mat .", "a dog");
        float[] scores = reranker.rerank(rerankTok, "where is the cat ?", docs);
        String rr = call("POST", "/v1/rerank", "{\"query\":\"where is the cat ?\",\"documents\":[\"stock market\","
                + "{\"text\":\"the cat sat on the mat .\"},\"a dog\"],\"top_n\":2,\"return_documents\":true}");
        assertTrue(rr.startsWith("200 "), rr);
        List<Object> results = Json.array(Json.object(Json.parse(rr.substring(4))).get("results"));
        assertEquals(2, results.size());
        double previous = Double.POSITIVE_INFINITY;
        for (Object o : results) {
            Map<String, Object> res = Json.object(o);
            int index = ((Number) res.get("index")).intValue();
            double score = ((Number) res.get("relevance_score")).doubleValue();
            assertEquals(scores[index], score, 1e-6);
            assertTrue(score <= previous);
            previous = score;
            assertEquals(docs.get(index), Json.object(res.get("document")).get("text"));
        }
        assertTrue(call("POST", "/v1/rerank", "{\"documents\":[\"x\"]}").startsWith("400 "));
    }
}
