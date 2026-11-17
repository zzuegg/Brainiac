package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.llm.Constraint;
import io.github.zzuegg.brainiac.llm.Generator;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.LlamaConfig;
import io.github.zzuegg.brainiac.llm.Sampling;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class StructuredDecodingTest {

    @Test
    void regexMatchesLikeJava() {
        String[] patterns = {"a(b|c)*d", "[0-9]{2,3}x?", "\\d+(\\.\\d+)?", "[^\"]*", "(ab){2}|c+"};
        String[] texts = {"", "ad", "abcbd", "12", "123x", "1234", "3.14", "3.", "x\"y", "abab", "ccc", "ab"};
        for (String p : patterns) {
            RegexAutomaton a = RegexAutomaton.compile(p);
            for (String t : texts) {
                assertTrue(a.matches(t) == t.matches(p), p + " on '" + t + "'");
            }
        }
    }

    /** Characters plus a few multi-character tokens, and an end token. */
    private static final String[] PIECES = {"{", "}", "\"", ":", ",", "a", "b", "e", "l", "n", "m", "r", "s", "t",
        "u", "f", "g", "o", "k", "i", "0", "1", "2", "9", "-", "true", "false", "\"name\":", "ab", "1}", " ", null};

    @Test
    void untrainedModelWritesSchemaValidJson() {
        String regex = JsonSchemaLite.object().string("name", 1, 6).integer("age").bool("ok")
                .enumeration("tier", "ab", "ba").toRegex();
        int eos = PIECES.length - 1;
        GrammarConstraint g = new GrammarConstraint(RegexAutomaton.compile(regex), PIECES, eos);
        HookedDecoder m = HookedDecoder.create(HookedDecoder.Settings.tiny(PIECES.length), 5);
        for (int seed = 0; seed < 3; seed++) {
            int[] out = DecodingPipeline.of(m).sampler(TokenSampler.ancestral()).constraint(g).eos(eos)
                    .generate(new int[] {0}, 60, new SplittableRandom(seed));
            StringBuilder text = new StringBuilder();
            for (int t : out) {
                text.append(PIECES[t]);
            }
            assertTrue(text.toString().matches(regex), "not valid: " + text);
            assertTrue(text.toString().startsWith("{\"name\":\""));
        }
        assertFalse(RegexAutomaton.compile(regex).matches("{\"name\":\"a\",\"age\":01,\"ok\":true,\"tier\":\"ab\"}"));
        assertTrue(RegexAutomaton.compile(regex).matches("{\"name\":\"a b\",\"age\":-10,\"ok\":true,\"tier\":\"ab\"}"));
    }

    @Test
    void watermarkIsDetectedOnlyWhenApplied() {
        HookedDecoder m = HookedDecoder.create(HookedDecoder.Settings.tiny(64), 9);
        GreenListWatermark w = new GreenListWatermark(42, 0.25, 4f);
        int[] marked = DecodingPipeline.of(m).processor(w).sampler(TokenSampler.ancestral())
                .generate(new int[] {1}, 60, new SplittableRandom(1));
        int[] plain = DecodingPipeline.of(m).sampler(TokenSampler.ancestral())
                .generate(new int[] {1}, 60, new SplittableRandom(1));
        int[] withPrompt = new int[marked.length + 1];
        withPrompt[0] = 1;
        System.arraycopy(marked, 0, withPrompt, 1, marked.length);
        int[] plainPrompt = new int[plain.length + 1];
        plainPrompt[0] = 1;
        System.arraycopy(plain, 0, plainPrompt, 1, plain.length);
        double zMarked = w.zScore(withPrompt);
        double zPlain = w.zScore(plainPrompt);
        assertTrue(zMarked > 4, "watermarked z " + zMarked);
        assertTrue(zPlain < 3, "plain z " + zPlain);
        assertTrue(new GreenListWatermark(7, 0.25, 4f).zScore(withPrompt) < 3, "a wrong key sees no watermark");
    }

    @Test
    void theSameGrammarDrivesTheCoreGenerator() {
        JsonSchemaLite schema = JsonSchemaLite.object().integer("id").enumeration("kind", "cat", "dog");
        Constraint core = schema.toConstraint();
        String chars = "abcdefghijklmnopqrstuvwxyz0123456789{}[]\":,- $";
        CharacterTokenizer tok = new CharacterTokenizer(chars);
        int stop = chars.indexOf('$');
        // the token-level adapter and the core constraint accept the same texts
        GrammarConstraint g = GrammarConstraint.of(core, tok, stop);
        assertTrue(g.constraint() == core);
        assertTrue(g.automaton().matches("{\"id\":-12,\"kind\":\"dog\"}"));
        assertFalse(g.automaton().matches("{\"id\": 12,\"kind\":\"dog\"}"));
        Llama model = Llama.random(LlamaConfig.small(chars.length(), 32, 2, 4, 2), 3, Device.cpu(), DType.FLOAT32);
        Generator gen = new Generator(model, tok);
        for (long seed = 0; seed < 2; seed++) {
            Generator.Output out = gen.generate(List.of("x"), Generator.Options.sampled(40, Sampling.temperature(1.5f),
                    seed).withStopTokens(Set.of(stop)).withConstraint(g.constraint()), null).getFirst();
            assertTrue(out.finish() == Generator.Finish.LENGTH || out.text().matches(schema.toRegex()), out.text());
            assertTrue(out.text().startsWith("{\"id\":"), out.text());
        }
        // and a core JSON schema numbers into the same int-state automaton
        RegexAutomaton a = RegexAutomaton.of(Constraint.jsonSchema(
                "{\"type\": \"object\", \"properties\": {\"ok\": {\"type\": \"boolean\"}}, \"required\": [\"ok\"]}"));
        assertTrue(a.matches("{\"ok\": true}"));
        assertFalse(a.matches("{\"ok\": 1}"));
    }
}
