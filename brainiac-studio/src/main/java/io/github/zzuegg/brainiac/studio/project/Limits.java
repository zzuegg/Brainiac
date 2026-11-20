package io.github.zzuegg.brainiac.studio.project;

import java.util.ArrayList;
import java.util.List;

/** Combines a project's limits with a parent's: what both allow. Null means "everything". */
final class Limits {

    private Limits() {
    }

    /**
     * Tools both allow.
     *
     * @param a one list, or null for all
     * @param b the other, or null for all
     * @return the tools in both
     */
    static List<String> tools(List<String> a, List<String> b) {
        if (a == null) {
            return b;
        }
        if (b == null) {
            return a;
        }
        return a.stream().filter(b::contains).toList();
    }

    /**
     * Models both allow; an entry is a {@code provider/id} key or a {@code provider/} prefix.
     *
     * @param a one list, or null for all
     * @param b the other, or null for all
     * @return the entries each allows that the other covers
     */
    static List<String> models(List<String> a, List<String> b) {
        if (a == null) {
            return b;
        }
        if (b == null) {
            return a;
        }
        List<String> out = new ArrayList<>();
        for (String x : a) {
            if (covers(b, x) && !out.contains(x)) {
                out.add(x);
            }
        }
        for (String x : b) {
            if (covers(a, x) && !out.contains(x)) {
                out.add(x);
            }
        }
        return out;
    }

    // whether a list allows everything an entry names
    private static boolean covers(List<String> list, String entry) {
        for (String e : list) {
            if (e.equals(entry) || (e.endsWith("/") && entry.startsWith(e))) {
                return true;
            }
        }
        return false;
    }
}
