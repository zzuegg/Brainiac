package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.llm.MemoryStream;
import io.github.zzuegg.brainiac.experimental.llm.ReflexionLoop;
import io.github.zzuegg.brainiac.experimental.llm.SkillLibrary;
import io.github.zzuegg.brainiac.experiments.reasoning.ListPrograms;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Voyager's skill library on the {@link ListPrograms} DSL (item 5.2). An automatic
 * curriculum hands out synthesis tasks (one named concept, then pairs and triples of
 * concepts); a program writer proposes programs within a budget, and a verifier runs them on
 * the task's examples. Three agents share the curriculum, the writer and the budget:
 * <ul>
 *   <li><b>skill library</b>: verified programs are stored under the task's description,
 *       and the writer sees the three skills retrieved for the new task's description, which
 *       it may call as single units;</li>
 *   <li><b>no library</b>: the same writer over primitives only;</li>
 *   <li><b>Reflexion</b>: the writer over primitives, plus a {@link ReflexionLoop} memory of
 *       the last failed programs, which it does not repeat.</li>
 * </ul>
 * The writer stands in for the code-writing LLM: it glues one to three units (two most often) (retrieved
 * skills or primitives) at random and never repeats a proposal the critic rejected.
 */
public final class SkillCurriculum {

    /**
     * Tasks solved over the run.
     *
     * @param tasks the tasks in the curriculum's pool
     * @param library tasks solved with the skill library, after each curriculum step
     * @param noLibrary tasks solved without it, after each step
     * @param reflexion tasks solved by Reflexion, after each step
     * @param skills skills in the library at the end
     */
    public record Result(int tasks, int[] library, int[] noLibrary, int[] reflexion, int skills) {
        /** @return the final count with the library */
        public int libraryFinal() {
            return library[library.length - 1];
        }

        /** @return the final count without */
        public int noLibraryFinal() {
            return noLibrary[noLibrary.length - 1];
        }

        /** @return the final count of Reflexion */
        public int reflexionFinal() {
            return reflexion[reflexion.length - 1];
        }
    }

    private SkillCurriculum() {
    }

    /** The writer: one to three units (two most often), each a retrieved skill (when allowed) or a primitive. */
    static List<String> write(List<SkillLibrary.Skill<List<String>>> skills, Set<String> rejected,
            SplittableRandom random) {
        List<String> prims = ListPrograms.primitives();
        List<String> p = new ArrayList<>();
        for (int tries = 0; tries < 20; tries++) {
            p = new ArrayList<>();
            double u0 = random.nextDouble();
            int units = u0 < 0.2 ? 1 : u0 < 0.7 ? 2 : 3;
            for (int u = 0; u < units; u++) {
                if (!skills.isEmpty() && random.nextBoolean()) {
                    p.addAll(skills.get(random.nextInt(skills.size())).program());
                } else {
                    p.add(prims.get(random.nextInt(prims.size())));
                }
            }
            if (!rejected.contains(p.toString())) {
                break;
            }
        }
        return p;
    }

    /**
     * @param steps curriculum steps
     * @param budget proposals per step
     * @param seed the seed
     * @return the solve counts over time
     */
    public static Result run(int steps, int budget, long seed) {
        List<ListPrograms.Task> pool = ListPrograms.curriculum(12, 8, 4, seed);
        SkillLibrary<List<String>> library = SkillLibrary.of(MemoryStream.hashedBagOfWords(64), 64);
        int[] withLib = arm(pool, steps, (task, random) -> {
            Set<String> rejected = new HashSet<>();
            return library.solve(task.name(), (name, skills, critiques, r) -> write(skills, rejected, r), prog -> {
                if (task.solvedBy(prog)) {
                    return null;
                }
                rejected.add(prog.toString());
                return "fails " + (task.inputs().length - task.examplesRight(prog)) + " examples";
            }, 3, budget, random).success();
        }, seed);
        int[] without = arm(pool, steps, (task, random) -> {
            Set<String> rejected = new HashSet<>();
            for (int i = 0; i < budget; i++) {
                List<String> prog = write(List.of(), rejected, random);
                if (task.solvedBy(prog)) {
                    return true;
                }
                rejected.add(prog.toString());
            }
            return false;
        }, seed);
        int[] reflexion = arm(pool, steps, (task, random) -> ReflexionLoop.<List<String>>of(
                (lessons, r) -> write(List.of(), new HashSet<>(lessons), r),
                prog -> new ReflexionLoop.Feedback(task.solvedBy(prog), ""),
                (prog, f) -> prog.toString()).memorySize(3).run(budget, random).success(), seed);
        return new Result(pool.size(), withLib, without, reflexion, library.size());
    }

    private interface Solver {
        boolean solve(ListPrograms.Task task, SplittableRandom random);
    }

    private static int[] arm(List<ListPrograms.Task> pool, int steps, Solver solver, long seed) {
        SplittableRandom random = new SplittableRandom(seed + 7);
        SkillLibrary.Curriculum<ListPrograms.Task> curriculum = SkillLibrary.Curriculum.of(pool,
                ListPrograms.Task::level);
        int[] solved = new int[steps];
        for (int s = 0; s < steps; s++) {
            ListPrograms.Task task = curriculum.next();
            if (task != null) {
                curriculum.report(task, solver.solve(task, random));
            }
            solved[s] = curriculum.solved().size();
        }
        return solved;
    }
}
