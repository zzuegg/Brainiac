package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SuffixDecodingTest {

    @Test
    void suffixTreeAcceptsLongDraftsOnRepetitiveTraces() {
        Map<String, SuffixDecoding.Outcome> r = SuffixDecoding.run(200, 16, 40, 1);
        StringBuilder line = new StringBuilder("SuffixDecoding, tokens/pass and verified/token:");
        r.forEach((k, v) -> line.append(String.format(" %s %.2f/%.2f;", k, v.tokensPerPass(), v.verifiedPerToken())));
        System.out.println(line);
        for (Map.Entry<String, SuffixDecoding.Outcome> e : r.entrySet()) {
            assertArrayEquals(r.get("greedy").tokens(), e.getValue().tokens(), e.getKey());
        }
        SuffixDecoding.Outcome suffix = r.get("suffix tree (adaptive)");
        assertTrue(suffix.tokensPerPass() > 1.5 * r.get("prompt lookup").tokensPerPass(), line.toString());
        assertTrue(suffix.verifiedPerToken() < r.get("prompt lookup").verifiedPerToken(), line.toString());
    }
}
