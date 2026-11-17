package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class JsonTest {

    @Test
    void parsesEveryKindOfValue() {
        Map<String, Object> o = Json.object(Json.parse(
                " {\"a\": 1, \"b\": -2.5e1, \"c\": [true, false, null], \"d\": \"x\\\"\\n\\u00e9\\ud83d\\ude00\","
                        + " \"e\": {}, \"f\": [], \"g\": 12345678901234567890} "));
        assertEquals(1L, o.get("a"));
        assertEquals(-25.0, o.get("b"));
        assertEquals(Arrays.asList(true, false, null), o.get("c"));
        assertEquals("x\"\né😀", o.get("d"));
        assertEquals(Map.of(), o.get("e"));
        assertEquals(List.of(), o.get("f"));
        assertEquals(1.2345678901234567e19, o.get("g"));
        assertEquals(List.of("a", "b", "c", "d", "e", "f", "g"), List.copyOf(o.keySet()));
        assertEquals(1, Json.integer(o, "a"));
        assertEquals(7, Json.integer(o, "missing", 7));
        assertNull(Json.string(o, "missing", null));
    }

    @Test
    void rejectsBadJson() {
        assertThrows(IllegalArgumentException.class, () -> Json.parse("{\"a\" 1}"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("[1, 2"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("1 2"));
        assertThrows(IllegalArgumentException.class, () -> Json.parse("\"abc"));
    }

    @Test
    void writesWhatItReads() {
        Map<String, Object> o = new LinkedHashMap<>();
        o.put("s", "tab\there \"quoted\" \u0001");
        o.put("n", 3L);
        o.put("x", 0.5);
        o.put("l", Arrays.asList(1L, null, true));
        o.put("m", Map.of("k", "v"));
        String text = Json.write(o);
        assertEquals(o, Json.parse(text));
        assertEquals("[1,2]", Json.write(new int[] {1, 2}));
    }
}
