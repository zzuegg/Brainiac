package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * A synthetic family-tree knowledge graph with one relation per classic relation
 * pattern, for knowledge-graph completion:
 * <ul>
 *   <li>{@link #SIBLING}: symmetric ({@code sibling(a, b)} iff {@code sibling(b, a)});</li>
 *   <li>{@link #PARENT}: antisymmetric ({@code parent(a, b)} excludes {@code parent(b, a)});</li>
 *   <li>{@link #CHILD}: the inverse of parent;</li>
 *   <li>{@link #GRANDPARENT}: the composition parent then parent.</li>
 * </ul>
 * Each family is a three-generation tree (a root, 2-3 children, 2-3 grandchildren each).
 * The test split holds out triples whose answer follows from training triples through the
 * relation's pattern: sibling, parent and child triples whose mirror or inverse is in training,
 * and grandparent triples whose two parent links are in training.
 */
public final class FamilyTreeGraph {

    /** Relation id of the symmetric sibling relation. */
    public static final int SIBLING = 0;
    /** Relation id of the antisymmetric parent relation. */
    public static final int PARENT = 1;
    /** Relation id of child, the inverse of parent. */
    public static final int CHILD = 2;
    /** Relation id of grandparent, the composition of parent with itself. */
    public static final int GRANDPARENT = 3;
    /** The relation names, by id. */
    public static final List<String> RELATIONS = List.of("sibling", "parent", "child", "grandparent");

    private FamilyTreeGraph() {
    }

    /**
     * The knowledge graph.
     *
     * @param entities the person count
     * @param train training triples {@code {head, relation, tail}}
     * @param test held-out triples
     * @param all every true triple (train and test)
     */
    public record Data(int entities, List<int[]> train, List<int[]> test, List<int[]> all) {
    }

    /**
     * Generates families.
     *
     * @param families how many families
     * @param testFraction the fraction of eligible triples held out
     * @param seed the seed
     * @return the knowledge graph
     */
    public static Data generate(int families, double testFraction, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> all = new ArrayList<>();
        int next = 0;
        for (int f = 0; f < families; f++) {
            int root = next++;
            List<Integer> kids = new ArrayList<>();
            int k = 2 + r.nextInt(2);
            for (int i = 0; i < k; i++) {
                kids.add(next++);
            }
            List<List<Integer>> grandkids = new ArrayList<>();
            for (int kid : kids) {
                List<Integer> g = new ArrayList<>();
                int m = 2 + r.nextInt(2);
                for (int i = 0; i < m; i++) {
                    g.add(next++);
                }
                grandkids.add(g);
            }
            addFamily(all, root, kids);
            for (int i = 0; i < kids.size(); i++) {
                addFamily(all, kids.get(i), grandkids.get(i));
                for (int gk : grandkids.get(i)) {
                    all.add(new int[] {root, GRANDPARENT, gk});
                }
            }
        }
        Set<Long> held = new HashSet<>();
        List<int[]> train = new ArrayList<>();
        List<int[]> test = new ArrayList<>();
        List<int[]> shuffled = new ArrayList<>(all);
        for (int i = shuffled.size() - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int[] t = shuffled.get(i);
            shuffled.set(i, shuffled.get(j));
            shuffled.set(j, t);
        }
        for (int[] t : shuffled) {
            boolean eligible = r.nextDouble() < testFraction && switch (t[1]) {
                case SIBLING -> !held.contains(key(t[2], SIBLING, t[0]));
                case CHILD -> !held.contains(key(t[2], PARENT, t[0]));
                case PARENT -> !held.contains(key(t[2], CHILD, t[0]));
                default -> true;
            };
            if (eligible) {
                held.add(key(t[0], t[1], t[2]));
                test.add(t);
            } else {
                train.add(t);
            }
        }
        // a held-out grandparent needs both parent links in training
        List<int[]> keepTest = new ArrayList<>();
        for (int[] t : test) {
            if (t[1] == GRANDPARENT && !hasPath(train, t[0], t[2])) {
                train.add(t);
            } else {
                keepTest.add(t);
            }
        }
        return new Data(next, train, keepTest, all);
    }

    private static boolean hasPath(List<int[]> train, int a, int c) {
        for (int[] t : train) {
            if (t[1] == PARENT && t[0] == a) {
                for (int[] u : train) {
                    if (u[1] == PARENT && u[0] == t[2] && u[2] == c) {
                        return true;
                    }
                }
            }
        }
        return false;
    }

    private static void addFamily(List<int[]> all, int parent, List<Integer> kids) {
        for (int a : kids) {
            all.add(new int[] {parent, PARENT, a});
            all.add(new int[] {a, CHILD, parent});
            for (int b : kids) {
                if (a != b) {
                    all.add(new int[] {a, SIBLING, b});
                }
            }
        }
    }

    private static long key(int h, int r, int t) {
        return ((long) h * 8 + r) * 1_000_000L + t;
    }
}
