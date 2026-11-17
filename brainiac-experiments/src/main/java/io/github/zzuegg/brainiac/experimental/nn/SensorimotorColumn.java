package io.github.zzuegg.brainiac.experimental.nn;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A cortical column in the sense of the Thousand Brains theory (Hawkins, Ahmad and Cui,
 * <a href="https://doi.org/10.3389/fncir.2017.00081">Front. Neural Circuits 2017</a>; the
 * Thousand Brains Project, <a href="https://arxiv.org/abs/2412.18354">2412.18354</a>): it
 * learns objects as features at locations in the object's own reference frame, and
 * recognizes them by sensing and moving. Its state is a set of hypotheses "this is object
 * {@code o} and my sensor is at location {@code l}":
 *
 * <ul>
 * <li>{@link #sense} keeps the hypotheses whose stored feature at {@code l} matches (the first
 *     sensation creates every matching hypothesis);</li>
 * <li>{@link #move} path-integrates: every hypothesis shifts by the known motor displacement,
 *     and those that leave their object are dropped;</li>
 * <li>{@link #vote} is the long-range lateral connection between columns: each column only
 *     keeps objects that every column still considers possible, so columns sensing different
 *     parts of the same object agree in fewer movements than one column alone.</li>
 * </ul>
 *
 * <p>Locations are 2-D integer grid points; features are integers.
 */
public final class SensorimotorColumn {
    private final Map<Integer, Map<Long, Integer>> objects = new HashMap<>();
    private Set<Long> hypotheses;   // packed (object, x, y)

    private SensorimotorColumn() {
    }

    /** @return an empty column */
    public static SensorimotorColumn create() {
        return new SensorimotorColumn();
    }

    private static long location(int x, int y) {
        return ((long) x << 32) ^ (y & 0xffffffffL);
    }

    private static long hypothesis(int object, int x, int y) {
        return ((long) object << 40) | ((long) (x & 0xfffff) << 20) | (y & 0xfffff);
    }

    private static int object(long h) {
        return (int) (h >>> 40);
    }

    private static int hx(long h) {
        return (int) ((h >> 20) & 0xfffff) << 12 >> 12;
    }

    private static int hy(long h) {
        return (int) (h & 0xfffff) << 12 >> 12;
    }

    /**
     * Learns one feature of an object.
     *
     * @param object the object id, 0 or more
     * @param x the location in the object's frame
     * @param y the location in the object's frame
     * @param feature the feature sensed there
     */
    public void learn(int object, int x, int y, int feature) {
        objects.computeIfAbsent(object, k -> new HashMap<>()).put(location(x, y), feature);
    }

    /** Forgets the current hypotheses, before a new object is presented. */
    public void reset() {
        hypotheses = null;
    }

    /**
     * Takes in one sensation.
     *
     * @param feature the sensed feature
     */
    public void sense(int feature) {
        if (hypotheses == null) {
            hypotheses = new HashSet<>();
            for (Map.Entry<Integer, Map<Long, Integer>> o : objects.entrySet()) {
                for (Map.Entry<Long, Integer> f : o.getValue().entrySet()) {
                    if (f.getValue() == feature) {
                        long loc = f.getKey();
                        hypotheses.add(hypothesis(o.getKey(), (int) (loc >> 32), (int) loc));
                    }
                }
            }
            return;
        }
        hypotheses.removeIf(h -> {
            Integer stored = objects.get(object(h)).get(location(hx(h), hy(h)));
            return stored == null || stored != feature;
        });
    }

    /**
     * Moves the sensor.
     *
     * @param dx the displacement
     * @param dy the displacement
     */
    public void move(int dx, int dy) {
        if (hypotheses == null) {
            return;
        }
        Set<Long> moved = new HashSet<>();
        for (long h : hypotheses) {
            int o = object(h);
            int x = hx(h) + dx;
            int y = hy(h) + dy;
            if (objects.get(o).containsKey(location(x, y))) {
                moved.add(hypothesis(o, x, y));
            }
        }
        hypotheses = moved;
    }

    /** @return the objects still possible, or all learned objects before the first sensation */
    public Set<Integer> candidates() {
        if (hypotheses == null) {
            return new HashSet<>(objects.keySet());
        }
        Set<Integer> c = new HashSet<>();
        for (long h : hypotheses) {
            c.add(object(h));
        }
        return c;
    }

    /** @return the number of (object, location) hypotheses */
    public int hypothesisCount() {
        return hypotheses == null ? 0 : hypotheses.size();
    }

    /**
     * Votes across columns: every column drops the objects that some column has ruled out.
     *
     * @param columns the columns sensing the same object
     * @return the objects all columns agree on
     */
    public static Set<Integer> vote(List<SensorimotorColumn> columns) {
        Set<Integer> consensus = null;
        for (SensorimotorColumn c : columns) {
            Set<Integer> own = c.candidates();
            if (consensus == null) {
                consensus = own;
            } else {
                consensus.retainAll(own);
            }
        }
        Set<Integer> agreed = consensus == null ? Set.of() : consensus;
        for (SensorimotorColumn c : columns) {
            if (c.hypotheses != null) {
                c.hypotheses.removeIf(h -> !agreed.contains(object(h)));
            }
        }
        return new HashSet<>(agreed);
    }

    /** @return the ids of the learned objects */
    public List<Integer> objectIds() {
        return new ArrayList<>(objects.keySet());
    }
}
