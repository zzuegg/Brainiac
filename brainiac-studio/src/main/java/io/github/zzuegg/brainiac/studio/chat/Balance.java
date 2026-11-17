package io.github.zzuegg.brainiac.studio.chat;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * What is left on a key, as the provider reports it.
 *
 * @param amount the money left, or NaN when the provider sets no limit
 * @param currency the currency, such as {@code USD} or {@code CNY}
 * @param detail more words from the provider, such as how much was granted
 */
public record Balance(double amount, String currency, String detail) {

    Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        if (!Double.isNaN(amount)) {
            m.put("amount", amount);
        }
        m.put("currency", currency);
        m.put("detail", detail);
        return m;
    }
}
