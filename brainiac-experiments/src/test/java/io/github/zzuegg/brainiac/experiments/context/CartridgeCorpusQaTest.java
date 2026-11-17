package io.github.zzuegg.brainiac.experiments.context;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.16: cartridges against full context, eviction and prefix tuning, by KV bytes. */
class CartridgeCorpusQaTest {

    @Test
    void aSmallCartridgeMatchesTheFullContext() {
        int[] budgets = {2, 4, 8};
        Map<String, CartridgeCorpusQa.Cell[]> r = CartridgeCorpusQa.run(budgets, 400, 400, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, cells) -> {
            sb.append(' ').append(k).append(" [");
            for (int i = 0; i < cells.length; i++) {
                sb.append(String.format("%s%dB:%.2f", i == 0 ? "" : " ", cells[i].kvBytes(), cells[i].accuracy()));
            }
            sb.append("];");
        });
        System.out.printf("corpus QA accuracy by KV bytes:%s%n", sb);
        double full = r.get("full context")[0].accuracy();
        double cart = r.get("cartridge (self-study distillation)")[1].accuracy();
        assertTrue(full > 0.8, "the reader answers from context: " + full);
        assertTrue(cart >= full - 0.1, "a 4-entry cartridge (4x smaller) nearly matches the full context: " + cart);
        assertTrue(cart > r.get("recency eviction")[1].accuracy() + 0.3, "and beats recency eviction");
    }
}
