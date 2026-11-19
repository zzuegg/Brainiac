package io.github.zzuegg.brainiac.studio.agent.js;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** The script interpreter: the language agents write, and calls into the host. */
class InterpreterTest {

    private final List<String> logs = new ArrayList<>();
    private final List<String> called = new ArrayList<>();

    private Object run(String code) {
        Map<String, Interpreter.HostFunction> tools = new LinkedHashMap<>();
        tools.put("add", args -> {
            called.add("add");
            Map<?, ?> a = (Map<?, ?>) args.getFirst();
            return ((Number) a.get("a")).doubleValue() + ((Number) a.get("b")).doubleValue();
        });
        tools.put("list", args -> List.of(Map.of("name", "a.txt", "size", 3), Map.of("name", "b.md", "size", 5)));
        tools.put("fail", args -> {
            throw new Interpreter.HostError("ToolError", "no such file");
        });
        Interpreter.Host host = new Interpreter.Host() {
            @Override
            public Map<String, Map<String, Interpreter.HostFunction>> namespaces() {
                return Map.of("tools", tools);
            }

            @Override
            public void log(String line) {
                logs.add(line);
            }
        };
        return new Interpreter(host, new Interpreter.Limits(200_000, 5_000, 100, 100_000)).run(code);
    }

    @Test
    void theLanguage() {
        assertEquals(7L, run("1 + 2 * 3"));
        assertEquals(512L, run("2 ** 3 ** 2"));
        assertEquals("a-b-c", run("['a','b','c'].join('-')"));
        assertEquals(List.of(2L, 4L), run("[1,2,3,4].filter(x => x % 2 === 0)"));
        assertEquals(10L, run("let s = 0; for (let i = 0; i < 5; i++) { s += i; } s"));
        assertEquals(6L, run("const {a, b: [c, d = 3]} = {a: 1, b: [2]}; a + c + d"));
        assertEquals("x=1, y=two", run("const o = {x: 1, y: 'two'}; Object.entries(o).map(([k, v]) => `${k}=${v}`).join(', ')"));
        assertEquals(120L, run("function f(n) { return n <= 1 ? 1 : n * f(n - 1); } return f(5);"));
        assertEquals("caught boom", run("try { throw new Error('boom'); } catch (e) { 'caught ' + e.message }"));
        assertEquals("{\"a\":[1,\"b\"]}", run("JSON.stringify({a: [1, 'b']})"));
        assertEquals(3L, run("JSON.parse('{\"n\": 3}').n"));
        assertEquals("HELLO wörld", run("'hello wörld'.replace(/hello/, s => s.toUpperCase())"));
        assertEquals(true, run("null ?? undefined ?? true"));
        assertEquals(1L, run("const m = new Map(); m.set('k', 1); m.get('k')"));
        assertEquals("number", run("typeof 1.5"));
        assertEquals(4L, run("let n = 0; for (const x of [1,2,3]) { if (x === 2) continue; n += x; } n"));
    }

    @Test
    void callsTheHost() {
        assertEquals(5L, run("const r = await tools.add({a: 2, b: 3}); r"));
        assertEquals(List.of("add"), called);
        assertEquals(8L, run("tools.list().reduce((s, f) => s + f.size, 0)"));
        assertEquals("ToolError: no such file", run("try { tools.fail({}) } catch (e) { `${e.name}: ${e.message}` }"));
        run("console.log('found', 2, {a: 1})");
        assertEquals(List.of("found 2 {\"a\":1}"), logs);
    }

    @Test
    void limits() {
        ScriptError loop = assertThrows(ScriptError.class, () -> run("while (true) {}"));
        assertTrue(loop.getMessage().contains("steps"), loop.getMessage());
        assertThrows(ScriptError.class, () -> run("function f() { return f(); } f()"));
        assertThrows(ScriptError.class, () -> run("let x = ;"));
        ScriptError uncaught = assertThrows(ScriptError.class, () -> run("throw new TypeError('bad')"));
        assertTrue(uncaught.getMessage().contains("TypeError: bad"), uncaught.getMessage());
    }
}
