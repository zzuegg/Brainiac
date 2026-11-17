package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.tensor.DType;

import java.util.List;
import java.util.Set;

import org.junit.jupiter.api.Test;

class GrammarTest {

    static final String ARITHMETIC = """
            # nested arithmetic, which no regular expression describes
            root   ::= expr
            expr   ::= term ([+*-] term)*
            term   ::= num
                     | "(" expr ")"
            num    ::= [0-9]{1,3}
            """;

    @Test
    void acceptsTheLanguage() {
        Grammar g = Grammar.parse(ARITHMETIC);
        for (String ok : List.of("1", "12+3", "(1+2)*3", "((((7))))", "999-(1*(2+3))")) {
            assertTrue(g.matches(ok), ok);
        }
        for (String bad : List.of("", "1+", "(1", "1)", "1234", "a", "(()")) {
            assertFalse(g.matches(bad), bad);
        }
    }

    @Test
    void matcherAnswersWhatMayFollow() {
        Constraint.Matcher m = Grammar.parse(ARITHMETIC).start();
        assertFalse(m.accepting());
        assertNull(m.feed(')'));
        Constraint.Matcher open = m.feed("((1");
        assertFalse(open.accepting());
        assertTrue(open.canContinue());
        Constraint.Matcher done = open.feed("))");
        assertTrue(done.accepting());
        assertTrue(done.canContinue());   // "+ ..." may follow
        assertNull(done.feed(')'));
    }

    @Test
    void literalsClassesAndRepetitions() {
        Grammar g = Grammar.parse("""
                root ::= greeting " " name "!"? ("\\n" | "\\t")?
                greeting ::= "hi" | "hello"
                name ::= [A-Z] [a-z]* ([-] [A-Z] [a-z]+)?
                """);
        assertTrue(g.matches("hi Ann"));
        assertTrue(g.matches("hello Mary-Jane!\n"));
        assertFalse(g.matches("hello mary"));
        assertFalse(g.matches("hey Ann"));
        Grammar any = Grammar.parse("root ::= \"\\\"\" [^\"\\\\]* \"\\\"\" .{2}");
        assertTrue(any.matches("\"quoted\"ab"));
        assertFalse(any.matches("\"quo\"te\"ab"));
        Grammar counts = Grammar.parse("root ::= \"a\"{2,} \"b\"{0,2} [\\x41-\\u0043]{3}");
        assertTrue(counts.matches("aaBAC"));
        assertTrue(counts.matches("aaaaabbCCC"));
        assertFalse(counts.matches("aABC"));
        assertFalse(counts.matches("aabbbABC"));
    }

    @Test
    void refusesBadGrammars() {
        assertThrows(IllegalArgumentException.class, () -> Grammar.parse("expr ::= \"a\""));
        assertThrows(IllegalArgumentException.class, () -> Grammar.parse("root ::= missing"));
        assertThrows(IllegalArgumentException.class, () -> Grammar.parse("root ::= root \"a\" | \"a\""));
        assertThrows(IllegalArgumentException.class, () -> Grammar.parse("root ::= x? root \"a\" | \"b\"\nx ::= \"c\""));
        assertThrows(IllegalArgumentException.class, () -> Grammar.parse("root ::= \"a"));
        assertThrows(IllegalArgumentException.class, () -> Grammar.parse("root ::= (\"a\""));
        assertThrows(IllegalArgumentException.class, () -> Grammar.parse("root ::= \"a\"\nroot ::= \"b\""));
        // right recursion is fine
        assertTrue(Grammar.parse("root ::= \"a\" root | \"b\"").matches("aaab"));
    }

    @Test
    void longRepetitionsDoNotGrowTheStack() {
        Grammar g = Grammar.parse("root ::= [ab]*");
        assertTrue(g.matches("ab".repeat(20_000)));
    }

    @Test
    void constrainsGeneration() {
        String chars = "0123456789+*-() abcxyz$";
        CharacterTokenizer tok = new CharacterTokenizer(chars);
        Llama model = Llama.random(LlamaConfig.small(chars.length(), 32, 2, 4, 2), 21, GeneratorTest.DEVICE,
                DType.FLOAT32);
        Generator gen = new Generator(model, tok);
        Grammar g = Grammar.parse(ARITHMETIC);
        int stop = chars.indexOf('$');
        int finished = 0;
        for (long seed = 0; seed < 6; seed++) {
            Generator.Options o = Generator.Options.sampled(40, Sampling.temperature(1.5f), seed)
                    .withStopTokens(Set.of(stop)).withConstraint(Constraint.grammar(ARITHMETIC));
            Generator.Output out = gen.generate(List.of("x"), o, null).getFirst();
            assertNotNull(g.start().feed(out.text()), out.text());
            if (out.finish() == Generator.Finish.STOP) {
                assertTrue(g.matches(out.text()), out.text());
                finished++;
            }
        }
        assertTrue(finished > 0, "no sample finished");
    }
}
