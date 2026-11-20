package io.github.zzuegg.brainiac.studio.items;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One item as lists show it.
 *
 * @param kind its kind, such as {@code chat}
 * @param id its id within the kind
 * @param title what it is called
 * @param project the project it belongs to, or null when its kind keeps no project
 * @param updated when it last changed, in milliseconds since the epoch, or 0
 * @param extra anything else a list shows for it (state, counts), never null
 */
public record Item(String kind, String id, String title, String project, long updated, Map<String, Object> extra) {

    /** Copies the extra fields, so an item never changes. */
    public Item {
        extra = extra == null ? Map.of() : java.util.Collections.unmodifiableMap(new LinkedHashMap<>(extra));
    }

    /**
     * Returns this item in another project.
     *
     * @param p the project
     * @return the item
     */
    public Item in(String p) {
        return new Item(kind, id, title, p, updated, extra);
    }

    /**
     * Describes the item for a page.
     *
     * @return {@code {kind, id, title, project, updated, ...extra}}
     */
    public Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>(extra);
        m.put("kind", kind);
        m.put("id", id);
        m.put("title", title);
        m.put("project", project);
        m.put("updated", updated);
        return m;
    }
}
