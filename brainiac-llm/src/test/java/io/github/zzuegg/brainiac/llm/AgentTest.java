package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class AgentTest {

    /** Printable ASCII and a newline, then an end token. */
    private static final int VOCAB = 96 + 1;
    static final int END = 96;

    private static int id(char c) {
        return c == '\n' ? 95 : c - 32;
    }

    static PretrainedTokenizer tokenizer() {
        Map<String, Object> vocab = new LinkedHashMap<>();
        for (char c = 32; c < 127; c++) {
            vocab.put(String.valueOf(c), id(c));
        }
        vocab.put("\n", 95);
        Map<String, Object> root = Map.of("model", Map.of("type", "BPE", "vocab", vocab, "merges", List.of()),
                "added_tokens", List.of(Map.of("id", END, "content", "<end>", "special", true)));
        return PretrainedTokenizer.fromJson(Json.write(root), Map.of("eos_token", "<end>", "chat_template",
                "{% if tools %}[{{ tools | length }} tools]{% endif %}{% for m in messages %}{{ m['role'] }}: "
                        + "{{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}assistant: {% endif %}"));
    }

    /** Writes scripted turns, one per reading from scratch, and remembers the prompts. */
    static final class Scripted implements LanguageModel {
        final List<String> turns;
        final List<Integer> promptLengths = new ArrayList<>();
        int turn;

        Scripted(List<String> turns) {
            this.turns = turns;
        }

        @Override
        public int vocabSize() {
            return VOCAB;
        }

        @Override
        public State newState(int batch) {
            return new Script(turns.get(turn++));
        }

        @Override
        public Tensor next(State state, int[][] tokens, boolean[][] padding) {
            Script s = (Script) state;
            if (s.length == 0) {
                promptLengths.add(tokens[0].length);
            }
            s.length += tokens[0].length;
            s.at++;
            float[] logits = new float[VOCAB];
            logits[s.at < s.text.length() ? id(s.text.charAt(s.at)) : END] = 60;   // certain at any temperature
            return Tensor.of(Shape.of(1, VOCAB), logits);
        }
    }

    private static final class Script implements LanguageModel.State {
        final String text;
        int at = -1;
        int length;

        Script(String text) {
            this.text = text;
        }

        @Override
        public int length() {
            return length;
        }

        @Override
        public LanguageModel.State select(int[] rows) {
            return this;
        }
    }

    @Test
    void callsToolsUntilTheModelAnswers() {
        PretrainedTokenizer tok = tokenizer();
        Scripted model = new Scripted(List.of(
                "Let me check.<tool_call>{\"name\": \"weather\", \"arguments\": {\"city\": \"Vienna\"}}</tool_call>",
                "{\"name\": \"add\", \"arguments\": \"{\\\"a\\\": 2, \\\"b\\\": 19}\"}",
                "It is 21 C in Vienna."));
        List<String> seen = new ArrayList<>();
        Agent.Tool weather = new Agent.Tool("weather", "temperature", null, args -> {
            seen.add("weather " + args.get("city"));
            return "19 C";
        });
        Agent.Tool add = new Agent.Tool("add", "adds", null, args -> {
            long sum = ((Number) args.get("a")).longValue() + ((Number) args.get("b")).longValue();
            seen.add("add " + sum);
            return String.valueOf(sum);
        });
        Agent agent = new Agent(new Generator(model, tok), tok, List.of(weather, add), Generator.Options.greedy(200));
        Agent.Result r = agent.run(List.of(Map.of("role", "user", "content", "How warm is Vienna?")), 5);
        assertTrue(r.finished());
        assertEquals("It is 21 C in Vienna.", r.answer());
        assertEquals(2, r.toolCalls());
        assertEquals(List.of("weather Vienna", "add 21"), seen);
        assertEquals(List.of("user", "assistant", "tool", "assistant", "tool", "assistant"),
                r.messages().stream().map(m -> m.get("role")).toList());
        assertEquals("Let me check.", r.messages().get(1).get("content"));
        assertEquals("19 C", r.messages().get(2).get("content"));
        assertTrue(model.promptLengths.get(1) > model.promptLengths.get(0));
    }

    @Test
    void stopsAtTheStepLimitAndReportsUnknownTools() {
        PretrainedTokenizer tok = tokenizer();
        Scripted model = new Scripted(List.of("<tool_call>{\"name\": \"nope\", \"arguments\": {}}</tool_call>"));
        Agent agent = new Agent(new Generator(model, tok), tok, List.of(), Generator.Options.greedy(100));
        Agent.Result r = agent.run(List.of(Map.of("role", "user", "content", "x")), 1);
        assertFalse(r.finished());
        assertEquals("error: no tool named nope", r.messages().get(2).get("content"));
    }
}
