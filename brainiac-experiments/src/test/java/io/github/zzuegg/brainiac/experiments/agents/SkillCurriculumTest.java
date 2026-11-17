package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.MemoryStream;
import io.github.zzuegg.brainiac.experimental.llm.SkillLibrary;
import java.util.List;
import org.junit.jupiter.api.Test;

class SkillCurriculumTest {

    @Test
    void retrievesTheSkillsNamedInTheTask() {
        SkillLibrary<String> lib = SkillLibrary.of(MemoryStream.hashedBagOfWords(64), 64);
        lib.add("descending", "sort rev");
        lib.add("evens", "even uniq");
        lib.add("shifted", "+3 *3");
        List<String> got = lib.retrieve("descending evens", 2).stream().map(SkillLibrary.Skill::program).toList();
        assertTrue(got.containsAll(List.of("sort rev", "even uniq")), got.toString());
        SkillLibrary.Curriculum<String> c = SkillLibrary.Curriculum.of(List.of("hardest", "easy"), s -> s.length());
        assertEquals("easy", c.next());
        c.report("easy", false);
        assertEquals("hardest", c.next());
    }

    @Test
    void skillLibrarySolvesMoreTasksOverTime() {
        SkillCurriculum.Result r = SkillCurriculum.run(90, 400, 1);
        System.out.printf("tasks solved of %d (step 30 / 60 / 90): skill library %d/%d/%d (%d skills), no library "
                + "%d/%d/%d, Reflexion %d/%d/%d%n", r.tasks(), r.library()[29], r.library()[59], r.libraryFinal(),
                r.skills(), r.noLibrary()[29], r.noLibrary()[59], r.noLibraryFinal(), r.reflexion()[29],
                r.reflexion()[59], r.reflexionFinal());
        assertTrue(r.libraryFinal() > r.noLibraryFinal() + 5, r.toString());
        assertTrue(r.libraryFinal() > r.reflexionFinal() + 5, r.toString());
    }
}
