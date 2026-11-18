package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConstraintTest {

    @Test
    void regexAcceptsWhatJavaAccepts() {
        String[] patterns = {"a(b|cd)*e", "[a-c]{2,3}x?", "\\d+(\\.\\d{1,2})?", "(?:yes|no|maybe)", "[^\"]*",
            "\\w+@\\w+\\.com", "a{3}", "(ab)+|c", "x.z", "[\\d_-]+"};
        String[] inputs = {"", "ae", "abe", "acdbe", "abcde", "ab", "abc", "abcx", "abcdx", "12", "12.5", "12.555",
            "yes", "maybe", "may", "say \"hi\"", "plain", "me@site.com", "me@site.org", "aaa", "aa", "abab", "c",
            "abc", "xyz", "x\nz", "1_2-3", "1a"};
        for (String p : patterns) {
            Constraint c = Constraint.regex(p);
            java.util.regex.Pattern ref = java.util.regex.Pattern.compile(p);
            for (String in : inputs) {
                assertEquals(ref.matcher(in).matches(), c.matches(in), p + " on " + in);
            }
        }
        assertThrows(IllegalArgumentException.class, () -> Constraint.regex("(ab"));
        assertThrows(IllegalArgumentException.class, () -> Constraint.regex("*a"));
    }

    @Test
    void matchersTellWhetherTheTextCanGoOn() {
        Constraint.Matcher m = Constraint.choice("cat", "category", "dog").start();
        Constraint.Matcher cat = m.feed("cat");
        assertTrue(cat.accepting());
        assertTrue(cat.canContinue());
        Constraint.Matcher dog = m.feed("dog");
        assertTrue(dog.accepting());
        assertFalse(dog.canContinue());
        assertNull(m.feed("cow"));
        assertFalse(m.feed("ca").accepting());
        assertTrue(Constraint.choice("a.b").matches("a.b"));
        assertFalse(Constraint.choice("a.b").matches("axb"));
    }

    @Test
    void jsonSchemaAcceptsConformingJsonOnly() {
        Constraint c = Constraint.jsonSchema("""
                {"type": "object", "properties": {
                    "name": {"type": "string", "maxLength": 8},
                    "age": {"type": "integer"},
                    "tags": {"type": "array", "items": {"enum": ["a", "b"]}, "maxItems": 2},
                    "ok": {"type": "boolean"},
                    "score": {"type": ["number", "null"]}},
                 "required": ["name", "age"]}
                """);
        assertTrue(c.matches("{\"name\":\"Bo\",\"age\":3}"));
        assertTrue(c.matches("{\"name\": \"Bo\", \"age\": -12, \"tags\": [\"a\", \"b\"], \"ok\": true}"));
        assertTrue(c.matches("{\"name\":\"\",\"age\":0,\"score\":1.5e3}"));
        assertTrue(c.matches("{\"name\":\"x\\\"y\",\"age\":0,\"score\":null}"));
        assertFalse(c.matches("{\"age\":3}"));                                    // name is required
        assertFalse(c.matches("{\"name\":\"Bo\",\"age\":3.5}"));                  // not an integer
        assertFalse(c.matches("{\"name\":\"far too long\",\"age\":3}"));
        assertFalse(c.matches("{\"name\":\"Bo\",\"age\":3,\"tags\":[\"a\",\"b\",\"a\"]}"));
        assertFalse(c.matches("{\"name\":\"Bo\",\"age\":03}"));
        // every accepted output parses as JSON
        Object parsed = Json.parse("{\"name\": \"Bo\", \"age\": -12, \"tags\": [\"a\", \"b\"], \"ok\": true}");
        assertEquals(-12L, Json.object(parsed).get("age"));
        Constraint optional = Constraint.jsonSchema("""
                {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}
                """);
        for (String ok : new String[] {"{}", "{\"a\":1}", "{\"b\":2}", "{\"a\":1,\"b\":2}"}) {
            assertTrue(optional.matches(ok), ok);
        }
        assertFalse(optional.matches("{,\"b\":2}"));
        assertFalse(optional.matches("{\"b\":2,\"a\":1}"));
    }

    @Test
    void theVocabularyAllowsTokensThatKeepAPrefix() {
        String[] texts = {"a", "ab", "abc", "b", "", null, "ba", "abd", "c"};
        Constraint.Vocabulary v = new Constraint.Vocabulary(texts);
        Constraint c = Constraint.regex("ab(c|d)?|ba");
        boolean[] ok = v.allowed(c.start());
        assertArrayEquals(new boolean[] {true, true, true, true, false, false, true, true, false}, ok);
        Constraint.Matcher afterA = v.advance(c.start(), 0);
        assertNotNull(afterA);
        assertArrayEquals(new boolean[] {false, false, false, true, false, false, false, false, false},
                v.allowed(afterA));
        // a brute force over many tokens agrees with the shared-prefix walk
        String[] many = new String[300];
        java.util.SplittableRandom r = new java.util.SplittableRandom(3);
        for (int i = 0; i < many.length; i++) {
            StringBuilder b = new StringBuilder();
            for (int k = r.nextInt(1, 5); k > 0; k--) {
                b.append("abc{}\"1".charAt(r.nextInt(7)));
            }
            many[i] = b.toString();
        }
        Constraint json = Constraint.regex("\\{\"(a|b)+\":1\\}");
        Constraint.Vocabulary big = new Constraint.Vocabulary(many);
        for (String prefix : new String[] {"", "{", "{\"", "{\"ab", "{\"ab\":"}) {
            Constraint.Matcher m = json.start().feed(prefix);
            boolean[] fast = big.allowed(m);
            for (int i = 0; i < many.length; i++) {
                assertEquals(m.feed(many[i]) != null, fast[i], prefix + " + " + many[i]);
            }
        }
    }

    @Test
    void generationFollowsTheConstraint() {
        String chars = "abcdefghijklmnopqrstuvwxyz0123456789{}[]\":, $";
        CharacterTokenizer tok = new CharacterTokenizer(chars);
        Llama model = Llama.random(LlamaConfig.small(chars.length(), 32, 2, 4, 2), 9, GeneratorTest.DEVICE,
                DType.FLOAT32);
        Generator gen = new Generator(model, tok);
        int stop = chars.indexOf('$');
        Constraint schema = Constraint.jsonSchema("""
                {"type": "object", "properties": {"id": {"type": "integer"}, "kind": {"enum": ["cat", "dog"]}},
                 "required": ["id", "kind"]}
                """);
        for (long seed = 0; seed < 4; seed++) {
            Generator.Options o = Generator.Options.sampled(60, Sampling.temperature(1.5f), seed)
                    .withStopTokens(Set.of(stop)).withConstraint(schema);
            Generator.Output out = gen.generate(List.of("data "), o, null).getFirst();
            assertTrue(schema.matches(out.text()) || out.finish() == Generator.Finish.LENGTH, out.text());
            if (out.finish() == Generator.Finish.STOP) {
                Map<String, Object> parsed = Json.object(Json.parse(out.text()));
                assertTrue(List.of("cat", "dog").contains(parsed.get("kind")), out.text());
            }
        }
        // no stop token: the sequence ends once nothing can follow
        Generator.Output pick = gen.generate(List.of("q"), Generator.Options.greedy(20).withStopTokens(Set.of())
                .withConstraint(Constraint.choice("yes", "no")), null).getFirst();
        assertTrue(pick.text().equals("yes") || pick.text().equals("no"), pick.text());
        assertEquals(Generator.Finish.STOP, pick.finish());
    }

    /** Balanced brackets up to a depth: not a regular language for unbounded depth. */
    private record Brackets(int depth, int length) implements Constraint.Matcher {
        @Override
        public Constraint.Matcher feed(char c) {
            if (c == '[' && depth < 6) {
                return new Brackets(depth + 1, length + 1);
            }
            if (c == ']' && depth > 0) {
                return new Brackets(depth - 1, length + 1);
            }
            return null;
        }

        @Override
        public boolean accepting() {
            return depth == 0 && length > 0;
        }

        @Override
        public boolean canContinue() {
            return true;
        }
    }

    @Test
    void customLanguagesPlugIn() {
        Constraint brackets = Constraint.of(() -> new Brackets(0, 0));
        assertNull(brackets.pattern());
        assertTrue(brackets.matches("[[]][]"));
        assertFalse(brackets.matches("[[]"));
        String chars = "abc[]$";
        CharacterTokenizer tok = new CharacterTokenizer(chars);
        Llama model = Llama.random(LlamaConfig.small(chars.length(), 32, 2, 4, 2), 13, GeneratorTest.DEVICE,
                DType.FLOAT32);
        Generator gen = new Generator(model, tok);
        for (long seed = 0; seed < 4; seed++) {
            Generator.Output out = gen.generate(List.of("a"), Generator.Options.sampled(30, Sampling.temperature(
                    1.5f), seed).withStopTokens(Set.of(chars.indexOf('$'))).withConstraint(brackets), null)
                    .getFirst();
            assertTrue(out.text().matches("[\\[\\]]*"), out.text());
            assertTrue(brackets.matches(out.text()) || out.finish() == Generator.Finish.LENGTH, out.text());
        }
    }

    @Test
    void anyJsonObjectIsAcceptedAndNothingElse() {
        Constraint json = Constraint.jsonObject();
        for (String ok : List.of("{}", "{\"a\":1}", "{\"a\": 1, \"b\": [true, null, -2.5e3, \"x\\\"y\\u00e9\"]}",
                "{\n  \"nested\": {\"deep\": [[], {}]},\n  \"s\": \"\"\n}")) {
            assertTrue(json.matches(ok), ok);
        }
        for (String bad : List.of("", "[1]", "\"text\"", "{\"a\":}", "{a:1}", "{\"a\":01}", "{\"a\":1,}", "{\"a\":1} and more")) {
            assertFalse(json.matches(bad), bad);
        }
        // an unfinished object can still be finished, text that left JSON cannot
        Constraint.Matcher m = json.start();
        for (char c : "{\"a\": [1, ".toCharArray()) {
            m = m.feed(c);
        }
        assertTrue(m.canContinue());
        assertFalse(m.accepting());
    }
}
