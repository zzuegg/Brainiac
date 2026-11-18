package io.github.zzuegg.brainiac.experimental.logic;

import java.util.ArrayList;
import java.util.List;

/**
 * A binary constraint satisfaction problem: variables with finite domains {@code 0..d-1} and
 * binary constraints given by their allowed value pairs (as a table or intensionally). N-queens, graph
 * colouring, Sudoku and Model RB random CSPs ({@code ConstraintInstances}) all fit this form.
 * {@link #toCnf} gives the direct SAT encoding (one Boolean per variable-value pair, at least and
 * at most one value per variable, one binary clause per forbidden pair).
 */
public final class Csp {

    /**
     * A binary constraint.
     *
     * @param x the first variable
     * @param y the second variable
     * @param relation whether {@code x = a, y = b} is allowed, as a function of
     *     {@code (x, y, a, b)}
     */
    public record Constraint(int x, int y, Relation relation) {

        /**
         * A constraint given by its table.
         *
         * @param x the first variable
         * @param y the second variable
         * @param allowed {@code allowed[a][b]} when {@code x = a, y = b} is allowed
         * @return the constraint
         */
        public static Constraint table(int x, int y, boolean[][] allowed) {
            return new Constraint(x, y, (u, v, a, b) -> allowed[a][b]);
        }

        /**
         * @param a the value of {@code x}
         * @param b the value of {@code y}
         * @return whether allowed
         */
        public boolean allowed(int a, int b) {
            return relation.test(x, y, a, b);
        }
    }

    private final String name;
    private final int[] domain;
    private final List<Constraint> constraints;
    private final List<List<Integer>> byVariable = new ArrayList<>();

    /**
     * Creates a CSP.
     *
     * @param name the instance name
     * @param domain the domain size of every variable
     * @param constraints the constraints
     */
    public Csp(String name, int[] domain, List<Constraint> constraints) {
        this.name = name;
        this.domain = domain.clone();
        this.constraints = List.copyOf(constraints);
        for (int i = 0; i < domain.length; i++) {
            byVariable.add(new ArrayList<>());
        }
        for (int c = 0; c < this.constraints.size(); c++) {
            byVariable.get(this.constraints.get(c).x()).add(c);
            byVariable.get(this.constraints.get(c).y()).add(c);
        }
    }

    /** @return the instance name */
    public String name() {
        return name;
    }

    /** @return the number of variables */
    public int variables() {
        return domain.length;
    }

    /**
     * @param v a variable
     * @return its domain size
     */
    public int domain(int v) {
        return domain[v];
    }

    /** @return the constraints */
    public List<Constraint> constraints() {
        return constraints;
    }

    /**
     * @param v a variable
     * @return indices of the constraints on it
     */
    public List<Integer> constraintsOf(int v) {
        return byVariable.get(v);
    }

    /**
     * Whether constraint {@code c} allows {@code v = a, other = b}.
     *
     * @param c the constraint index
     * @param v one of its variables
     * @param a the value of {@code v}
     * @param b the value of the other variable
     * @return whether allowed
     */
    public boolean allows(int c, int v, int a, int b) {
        Constraint k = constraints.get(c);
        return k.x() == v ? k.allowed(a, b) : k.allowed(b, a);
    }

    /**
     * @param c a constraint index
     * @param v one of its variables
     * @return the other variable
     */
    public int other(int c, int v) {
        Constraint k = constraints.get(c);
        return k.x() == v ? k.y() : k.x();
    }

    /**
     * @param assignment a value per variable (-1 unassigned)
     * @return the number of violated constraints among fully assigned ones
     */
    public int violations(int[] assignment) {
        int n = 0;
        for (Constraint k : constraints) {
            int a = assignment[k.x()];
            int b = assignment[k.y()];
            if (a >= 0 && b >= 0 && !k.allowed(a, b)) {
                n++;
            }
        }
        return n;
    }

    /**
     * Builds a CSP with one predicate for all constraints over the given variable pairs.
     *
     * @param name the name
     * @param domain the domain sizes
     * @param pairs the constrained pairs
     * @param allowed the relation, as a function of (x, y, a, b)
     * @return the CSP
     */
    public static Csp of(String name, int[] domain, List<int[]> pairs, Relation allowed) {
        List<Constraint> cs = new ArrayList<>();
        for (int[] p : pairs) {
            cs.add(new Constraint(p[0], p[1], allowed));
        }
        return new Csp(name, domain, cs);
    }

    /** A binary relation given intensionally. */
    @FunctionalInterface
    public interface Relation {

        /**
         * @param x the first variable
         * @param y the second variable
         * @param a its value
         * @param b the second's value
         * @return whether allowed
         */
        boolean test(int x, int y, int a, int b);
    }

    /**
     * The CNF variable of {@code x = a} in {@link #toCnf}.
     *
     * @param x a CSP variable
     * @param a a value
     * @return the DIMACS variable
     */
    public int cnfVariable(int x, int a) {
        int off = 0;
        for (int i = 0; i < x; i++) {
            off += domain[i];
        }
        return off + a + 1;
    }

    /** @return the number of CNF variables of {@link #toCnf} */
    public int cnfVariables() {
        int s = 0;
        for (int d : domain) {
            s += d;
        }
        return s;
    }

    /** @return the direct encoding as DIMACS clauses */
    public List<int[]> toCnf() {
        List<int[]> out = new ArrayList<>();
        for (int x = 0; x < domain.length; x++) {
            int[] alo = new int[domain[x]];
            for (int a = 0; a < domain[x]; a++) {
                alo[a] = cnfVariable(x, a);
                for (int b = a + 1; b < domain[x]; b++) {
                    out.add(new int[] {-cnfVariable(x, a), -cnfVariable(x, b)});
                }
            }
            out.add(alo);
        }
        for (Constraint k : constraints) {
            for (int a = 0; a < domain[k.x()]; a++) {
                for (int b = 0; b < domain[k.y()]; b++) {
                    if (!k.allowed(a, b)) {
                        out.add(new int[] {-cnfVariable(k.x(), a), -cnfVariable(k.y(), b)});
                    }
                }
            }
        }
        return out;
    }

    /**
     * Decodes a model of {@link #toCnf}.
     *
     * @param model the model (index 0 for CNF variable 1)
     * @return the assignment
     */
    public int[] decode(boolean[] model) {
        int[] out = new int[domain.length];
        for (int x = 0; x < domain.length; x++) {
            out[x] = -1;
            for (int a = 0; a < domain[x]; a++) {
                if (model[cnfVariable(x, a) - 1]) {
                    out[x] = a;
                    break;
                }
            }
        }
        return out;
    }
}
