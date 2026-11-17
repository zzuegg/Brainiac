package io.github.zzuegg.brainiac.experimental.llm;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class VirtualContextPagerTest {

    static int words(String s) {
        return (int) Arrays.stream(s.trim().split("\\s+")).filter(w -> !w.isEmpty()).count();
    }

    /** A hashed bag of words. */
    static float[] embed(String s) {
        float[] v = new float[64];
        for (String w : s.toLowerCase(Locale.ROOT).split("\\W+")) {
            if (!w.isEmpty()) {
                v[Math.floorMod(w.hashCode(), 64)] += 1;
            }
        }
        return v;
    }

    /** Stands in for the model's summary: keeps the lines marked important. */
    static String keepImportant(String text) {
        return String.join("\n", text.lines().filter(l -> l.contains("important")).toList());
    }

    static VirtualContextPager pager() {
        return VirtualContextPager.of(50, VirtualContextPagerTest::words, VirtualContextPagerTest::embed, 64)
                .system("You are a helpful agent.").summarizer(VirtualContextPagerTest::keepImportant)
                .thresholds(0.7, 0.5);
    }

    @Test
    void evictsToRecallAndSummaryWithinTheBudget() {
        VirtualContextPager p = pager();
        List<VirtualContextPager.Event> events = new ArrayList<>();
        events.add(p.add("user", "my cat is named Tiberius important"));
        events.add(p.add("user", "the wifi password is hunter2"));
        for (int i = 0; i < 40; i++) {
            events.add(p.add(i % 2 == 0 ? "user" : "assistant", "some small talk number " + i));
            assertTrue(p.contextTokens() <= 50, p.toString());
        }
        int pressure = events.indexOf(VirtualContextPager.Event.PRESSURE);
        int evicted = events.indexOf(VirtualContextPager.Event.EVICTED);
        assertTrue(pressure >= 0 && pressure < evicted, "pressure is reported before anything is evicted");
        assertTrue(p.evictions() > 1);
        assertEquals(42, p.recallSize());
        // the summary keeps what the summarizer kept, through repeated evictions
        assertTrue(p.summary().contains("Tiberius"), p.summary());
        assertFalse(p.render().contains("hunter2"));
        // ... and recall search pages the rest back in
        String found = p.call("conversation_search", "wifi password");
        assertTrue(found.contains("hunter2"), found);
        assertTrue(p.render().contains("hunter2"), p.render());
        assertTrue(p.contextTokens() <= 50);
        assertEquals("no results", p.call("conversation_search", "zebra"));
    }

    @Test
    void archivalAndCoreMemoryAreFunctionCalls() {
        VirtualContextPager p = pager().coreLimit(12);
        p.call("archival_memory_insert", "The user's dog is a border collie called Pip.");
        p.call("archival_memory_insert", "The user works as a nurse on night shifts.");
        p.call("archival_memory_insert", "The user is learning Portuguese for a trip.");
        assertEquals("The user's dog is a border collie called Pip.", p.archivalSearch("what dog does the user have", 1)
                .getFirst());
        assertTrue(p.call("archival_memory_search", "night shifts work").startsWith("The user works as a nurse"));

        assertEquals("ok", p.call("core_memory_append", "Name: Ada"));
        assertEquals("ok", p.call("core_memory_append", "Likes: tea"));
        assertEquals("ok", p.call("core_memory_replace", "tea\ncoffee"));
        assertEquals("Name: Ada\nLikes: coffee", p.core());
        assertEquals("core memory full", p.coreAppend("a very long line that will not fit in the limit"));
        assertTrue(p.render().contains("[core memory]\nName: Ada\nLikes: coffee"));
        assertTrue(p.render().contains("archival: 3 entries"));
    }
}
