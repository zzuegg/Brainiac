package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.Generator;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Sequence-level knowledge distillation (Kim and Rush 2016,
 * <a href="https://arxiv.org/abs/1606.07947">arXiv:1606.07947</a>): instead of matching the
 * teacher's soft distribution at every gold position, the student is fine-tuned with plain
 * cross-entropy on the teacher's own <em>decoded</em> outputs (greedy or beam search). The
 * targets are the teacher's mode, which is easier for a small student to fit than the full
 * distribution, and the teacher runs once per prompt (generation) rather than on every
 * training batch, so the data set can be built once and reused.
 *
 * <pre>{@code
 * SequenceDistillation.Data data = SequenceDistillation.greedy(teacher, prompts, maxTokens, eos);
 * student.supervisedStep(prompts, data.targets(), optimizer);
 * }</pre>
 */
public final class SequenceDistillation {

    /**
     * A distilled data set.
     *
     * @param targets the teacher's completion for each prompt
     * @param teacherTokens how many token positions the teacher scored to build it
     */
    public record Data(List<int[]> targets, long teacherTokens) {
    }

    private SequenceDistillation() {
    }

    /**
     * Decodes every prompt greedily with the teacher.
     *
     * @param teacher the teacher
     * @param prompts the prompts
     * @param maxNew the most tokens per completion
     * @param end the end token
     * @return the teacher's completions
     */
    public static Data greedy(WindowLm teacher, List<int[]> prompts, int maxNew, int end) {
        int[][] outs = teacher.sample(prompts.toArray(int[][]::new), maxNew, end, 0,
                new java.util.SplittableRandom(0));
        List<int[]> targets = new ArrayList<>();
        long tokens = 0;
        for (int[] o : outs) {
            targets.add(o);
            tokens += o.length;
        }
        return new Data(targets, tokens);
    }

    /**
     * Decodes every prompt with the core {@link Generator}'s beam search.
     *
     * @param teacher the teacher
     * @param prompts the prompts
     * @param maxNew the most tokens per completion
     * @param end the end token
     * @param beams the beam width
     * @return the teacher's best beams, with the end token appended when a beam stopped early
     */
    public static Data beam(WindowLm teacher, List<int[]> prompts, int maxNew, int end, int beams) {
        Generator generator = new Generator(teacher, null);
        List<int[]> targets = new ArrayList<>();
        long tokens = 0;
        for (int[] p : prompts) {
            int[] beamOut = generator.beamSearch(p, beams, maxNew, 0, Set.of(end));
            int[] out = beamOut.length < maxNew ? Arrays.copyOf(beamOut, beamOut.length + 1) : beamOut;
            if (out != beamOut) {
                out[out.length - 1] = end;
            }
            targets.add(out);
            tokens += (long) out.length * beams;
        }
        return new Data(targets, tokens);
    }
}
