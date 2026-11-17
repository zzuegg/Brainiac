package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/** Compares with jinja2 configured as transformers' apply_chat_template (make_templates.py). */
class ChatTemplateTest {

    private static String resource(String name) throws IOException {
        try (InputStream in = ChatTemplateTest.class.getResourceAsStream("templates/" + name)) {
            return new String(in.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    @Test
    @SuppressWarnings("unchecked")
    void rendersLikeJinja2() throws IOException {
        Map<String, Object> expected = Json.object(Json.parse(resource("expected.json")));
        Map<String, Object> messages = Json.object(expected.get("messages"));
        int checked = 0;
        for (Object o : Json.array(expected.get("cases"))) {
            Map<String, Object> c = Json.object(o);
            String name = (String) c.get("template");
            ChatTemplate t = ChatTemplate.of(resource(name + ".jinja"));
            Map<String, Object> vars = new HashMap<>();
            vars.put("bos_token", "<s>");
            vars.put("eos_token", "</s>");
            if (Boolean.TRUE.equals(c.get("tools"))) {
                vars.put("tools", expected.get("tools"));
            }
            List<Map<String, Object>> msgs = (List<Map<String, Object>>) (List<?>) Json.array(
                    messages.get((String) c.get("messages")));
            boolean gen = Boolean.TRUE.equals(c.get("add_generation_prompt"));
            String what = name + " / " + c.get("messages") + " / gen=" + gen + " / tools=" + c.get("tools");
            if (c.containsKey("error")) {
                RuntimeException e = assertThrows(RuntimeException.class, () -> t.render(msgs, gen, vars), what);
                assertEquals(c.get("error"), e.getMessage(), what);
            } else {
                assertEquals(c.get("text"), t.render(msgs, gen, vars), what);
            }
            checked++;
        }
        assertEquals(42, checked);
    }

    @Test
    void tokenizerAppliesItsTemplate() throws IOException {
        Map<String, Object> config = Map.of("chat_template", resource("chatml.jinja"), "eos_token", "<|im_end|>");
        String json = new String(getClass().getResourceAsStream("tokenizers/bytelevel-bpe.json").readAllBytes(),
                StandardCharsets.UTF_8);
        PretrainedTokenizer tok = PretrainedTokenizer.fromJson(json, config);
        String prompt = tok.applyChatTemplate(List.of(ChatTemplate.message("user", "Hi")), true);
        assertEquals("<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n", prompt);
        int[] ids = tok.encode(prompt);
        assertEquals(tok.id("<|im_start|>").orElseThrow(), ids[0]);
        assertEquals(tok.id("<|im_end|>").orElseThrow(), tok.eos().orElseThrow());
        assertTrue(tok.decode(ids).endsWith("assistant\n"));
    }

    @Test
    void rejectsBrokenTemplates() {
        assertThrows(IllegalArgumentException.class, () -> ChatTemplate.of("{% if x %}no end"));
        assertThrows(IllegalArgumentException.class, () -> ChatTemplate.of("{{ unclosed"));
        assertThrows(IllegalArgumentException.class, () -> ChatTemplate.of("{% frobnicate %}"));
    }
}
