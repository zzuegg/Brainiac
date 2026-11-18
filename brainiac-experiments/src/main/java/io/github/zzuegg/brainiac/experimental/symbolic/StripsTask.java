package io.github.zzuegg.brainiac.experimental.symbolic;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A grounded STRIPS planning task (Fikes &amp; Nilsson, AIJ 1971): facts are bit indices, a state is
 * the {@link BitSet} of the facts that hold, and every action has a precondition, an add and a
 * delete list and a non-negative cost. Applying an action removes its deletes and then adds its
 * adds.
 *
 * <p>Besides the propositional core, the task keeps the lifted structure the learned planners
 * need: every fact is a predicate applied to objects and every action an action schema applied
 * to objects, so object-centric graph encodings ({@link WlFeatures}) and argument-wise action
 * instantiation ({@link PartialSpaceSearch}) work across instance sizes. Optional mutex groups
 * (sets of facts of which at most one holds in any reachable state) give the multi-valued SAS+
 * view used by {@link PatternDatabases}.
 *
 * <p>Tasks are built with a {@link Builder}; facts and objects are interned by name.
 */
public final class StripsTask {

    /**
     * A ground action.
     *
     * @param id the index in {@link #actions()}
     * @param schema the action schema index in {@link #schemas()}
     * @param args the object arguments
     * @param pre precondition facts
     * @param add added facts
     * @param del deleted facts
     * @param cost the non-negative cost
     */
    public record Action(int id, int schema, int[] args, int[] pre, int[] add, int[] del, double cost) {

        /**
         * @param state a state
         * @return whether the action is applicable
         */
        public boolean applicable(BitSet state) {
            for (int p : pre) {
                if (!state.get(p)) {
                    return false;
                }
            }
            return true;
        }

        /**
         * @param state a state where the action is applicable
         * @return the successor
         */
        public BitSet apply(BitSet state) {
            BitSet next = (BitSet) state.clone();
            for (int d : del) {
                next.clear(d);
            }
            for (int a : add) {
                next.set(a);
            }
            return next;
        }
    }

    private final String name;
    private final List<String> objects;
    private final List<String> predicates;
    private final int[] predicateArity;
    private final List<String> schemas;
    private final int[] schemaArity;
    private final List<String> factNames;
    private final int[] factPredicate;
    private final int[][] factArgs;
    private final BitSet init;
    private final int[] goal;
    private final List<Action> actions;
    private final List<int[]> mutexGroups;
    private final int[][] achievers;
    private final int[][] consumers;

    private StripsTask(Builder b, BitSet init, int[] goal) {
        this.name = b.name;
        this.objects = List.copyOf(b.objects);
        this.predicates = List.copyOf(b.predicates);
        this.predicateArity = b.predicateArity.stream().mapToInt(Integer::intValue).toArray();
        this.schemas = List.copyOf(b.schemas);
        this.schemaArity = b.schemaArity.stream().mapToInt(Integer::intValue).toArray();
        this.factNames = List.copyOf(b.factNames);
        this.factPredicate = b.factPredicate.stream().mapToInt(Integer::intValue).toArray();
        this.factArgs = b.factArgs.toArray(new int[0][]);
        this.init = (BitSet) init.clone();
        this.goal = goal.clone();
        this.actions = List.copyOf(b.actions);
        this.mutexGroups = List.copyOf(b.mutexGroups);
        int f = factNames.size();
        List<List<Integer>> ach = new ArrayList<>();
        List<List<Integer>> con = new ArrayList<>();
        for (int i = 0; i < f; i++) {
            ach.add(new ArrayList<>());
            con.add(new ArrayList<>());
        }
        for (Action a : actions) {
            for (int x : a.add()) {
                ach.get(x).add(a.id());
            }
            for (int x : a.pre()) {
                con.get(x).add(a.id());
            }
        }
        this.achievers = new int[f][];
        this.consumers = new int[f][];
        for (int i = 0; i < f; i++) {
            achievers[i] = ach.get(i).stream().mapToInt(Integer::intValue).toArray();
            consumers[i] = con.get(i).stream().mapToInt(Integer::intValue).toArray();
        }
    }

    /** @return the instance name */
    public String name() {
        return name;
    }

    /** @return the number of facts */
    public int facts() {
        return factNames.size();
    }

    /**
     * @param fact a fact
     * @return its printable name
     */
    public String factName(int fact) {
        return factNames.get(fact);
    }

    /**
     * @param fact a fact
     * @return its predicate index
     */
    public int factPredicate(int fact) {
        return factPredicate[fact];
    }

    /**
     * @param fact a fact
     * @return its object arguments (do not modify)
     */
    public int[] factArgs(int fact) {
        return factArgs[fact];
    }

    /** @return the object names */
    public List<String> objects() {
        return objects;
    }

    /** @return the predicate names */
    public List<String> predicates() {
        return predicates;
    }

    /**
     * @param predicate a predicate index
     * @return its arity
     */
    public int predicateArity(int predicate) {
        return predicateArity[predicate];
    }

    /** @return the action schema names */
    public List<String> schemas() {
        return schemas;
    }

    /**
     * @param schema a schema index
     * @return its arity
     */
    public int schemaArity(int schema) {
        return schemaArity[schema];
    }

    /** @return the ground actions */
    public List<Action> actions() {
        return actions;
    }

    /** @return the initial state (a copy) */
    public BitSet initialState() {
        return (BitSet) init.clone();
    }

    /** @return the goal facts (a copy) */
    public int[] goal() {
        return goal.clone();
    }

    /** @return the declared mutex groups */
    public List<int[]> mutexGroups() {
        return mutexGroups;
    }

    /**
     * @param fact a fact
     * @return the actions adding it (do not modify)
     */
    public int[] achievers(int fact) {
        return achievers[fact];
    }

    /**
     * @param fact a fact
     * @return the actions requiring it (do not modify)
     */
    public int[] consumers(int fact) {
        return consumers[fact];
    }

    /**
     * @param state a state
     * @return whether every goal fact holds
     */
    public boolean isGoal(BitSet state) {
        for (int g : goal) {
            if (!state.get(g)) {
                return false;
            }
        }
        return true;
    }

    /**
     * @param state a state
     * @return the applicable actions
     */
    public List<Action> applicable(BitSet state) {
        List<Action> out = new ArrayList<>();
        for (Action a : actions) {
            if (a.applicable(state)) {
                out.add(a);
            }
        }
        return out;
    }

    /**
     * Checks a plan.
     *
     * @param plan the action sequence
     * @return its cost, or infinity if some step is inapplicable or the goal is not reached
     */
    public double validate(List<Action> plan) {
        BitSet s = initialState();
        double c = 0;
        for (Action a : plan) {
            if (!a.applicable(s)) {
                return Double.POSITIVE_INFINITY;
            }
            s = a.apply(s);
            c += a.cost();
        }
        return isGoal(s) ? c : Double.POSITIVE_INFINITY;
    }

    /**
     * The same task with another initial state (same facts and actions).
     *
     * @param state the new initial state
     * @return the task
     */
    public StripsTask withInitial(BitSet state) {
        StripsTask t = new StripsTask(source, state, goal);
        t.source = source;
        return t;
    }

    /**
     * The same task with every action cost set to 1.
     *
     * @return the unit-cost task
     */
    public StripsTask unitCost() {
        Builder b = source.copy();
        List<Action> unit = new ArrayList<>();
        for (Action a : b.actions) {
            unit.add(new Action(a.id(), a.schema(), a.args(), a.pre(), a.add(), a.del(), 1));
        }
        b.actions.clear();
        b.actions.addAll(unit);
        StripsTask t = new StripsTask(b, init, goal);
        t.source = b;
        return t;
    }

    private Builder source;

    @Override
    public String toString() {
        return name + " (" + facts() + " facts, " + actions.size() + " actions)";
    }

    /**
     * @return a new builder
     * @param name the instance name
     */
    public static Builder builder(String name) {
        return new Builder(name);
    }

    /** Builds a task: intern objects and facts, add actions, set the initial state and goal. */
    public static final class Builder {

        private final String name;
        private final List<String> objects = new ArrayList<>();
        private final Map<String, Integer> objectIndex = new HashMap<>();
        private final List<String> predicates = new ArrayList<>();
        private final List<Integer> predicateArity = new ArrayList<>();
        private final Map<String, Integer> predicateIndex = new HashMap<>();
        private final List<String> schemas = new ArrayList<>();
        private final List<Integer> schemaArity = new ArrayList<>();
        private final Map<String, Integer> schemaIndex = new HashMap<>();
        private final List<String> factNames = new ArrayList<>();
        private final List<Integer> factPredicate = new ArrayList<>();
        private final List<int[]> factArgs = new ArrayList<>();
        private final Map<String, Integer> factIndex = new HashMap<>();
        private final List<Action> actions = new ArrayList<>();
        private final List<int[]> mutexGroups = new ArrayList<>();

        private Builder(String name) {
            this.name = name;
        }

        private Builder copy() {
            Builder b = new Builder(name);
            b.objects.addAll(objects);
            b.objectIndex.putAll(objectIndex);
            b.predicates.addAll(predicates);
            b.predicateArity.addAll(predicateArity);
            b.predicateIndex.putAll(predicateIndex);
            b.schemas.addAll(schemas);
            b.schemaArity.addAll(schemaArity);
            b.schemaIndex.putAll(schemaIndex);
            b.factNames.addAll(factNames);
            b.factPredicate.addAll(factPredicate);
            b.factArgs.addAll(factArgs);
            b.factIndex.putAll(factIndex);
            b.actions.addAll(actions);
            b.mutexGroups.addAll(mutexGroups);
            return b;
        }

        /**
         * Interns an object.
         *
         * @param object the object name
         * @return its index
         */
        public int object(String object) {
            return objectIndex.computeIfAbsent(object, k -> {
                objects.add(k);
                return objects.size() - 1;
            });
        }

        /**
         * Interns a fact {@code predicate(args...)}.
         *
         * @param predicate the predicate
         * @param args the object names
         * @return the fact index
         */
        public int fact(String predicate, String... args) {
            String key = predicate + "(" + String.join(",", args) + ")";
            Integer id = factIndex.get(key);
            if (id != null) {
                return id;
            }
            int p = predicateIndex.computeIfAbsent(predicate, k -> {
                predicates.add(k);
                predicateArity.add(args.length);
                return predicates.size() - 1;
            });
            int[] a = new int[args.length];
            for (int i = 0; i < args.length; i++) {
                a[i] = object(args[i]);
            }
            factNames.add(key);
            factPredicate.add(p);
            factArgs.add(a);
            factIndex.put(key, factNames.size() - 1);
            return factNames.size() - 1;
        }

        /**
         * Adds a ground action.
         *
         * @param schema the schema name
         * @param args the object arguments
         * @param pre preconditions
         * @param add adds
         * @param del deletes
         * @param cost the cost
         * @return this builder
         */
        public Builder action(String schema, String[] args, int[] pre, int[] add, int[] del, double cost) {
            int s = schemaIndex.computeIfAbsent(schema, k -> {
                schemas.add(k);
                schemaArity.add(args.length);
                return schemas.size() - 1;
            });
            int[] a = new int[args.length];
            for (int i = 0; i < args.length; i++) {
                a[i] = object(args[i]);
            }
            actions.add(new Action(actions.size(), s, a, pre.clone(), add.clone(), del.clone(), cost));
            return this;
        }

        /**
         * Declares a mutex group: at most one of these facts holds in any reachable state.
         *
         * @param facts the facts
         * @return this builder
         */
        public Builder mutex(int... facts) {
            mutexGroups.add(facts.clone());
            return this;
        }

        /**
         * Finishes the task.
         *
         * @param init the initial facts
         * @param goal the goal facts
         * @return the task
         */
        public StripsTask build(int[] init, int[] goal) {
            BitSet s = new BitSet();
            for (int f : init) {
                s.set(f);
            }
            StripsTask t = new StripsTask(this, s, goal);
            t.source = this;
            return t;
        }
    }
}
