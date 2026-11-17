package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.MultiturnRewardTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.ClarifyingQuestions.Training;
import org.junit.jupiter.api.Test;

/** Wave 5, 3.11: CollabLLM's multiturn-aware reward against single-turn GRPO on underspecified sums. */
class ClarifyingQuestionsTest {

    @Test
    void multiturnRewardTeachesTheModelToAsk() {
        ClarificationDialogue d = new ClarificationDialogue(4, ClarificationDialogue.Mode.COLLABORATIVE);
        WindowLm sft = ClarifyingQuestions.sft(d, 150, 0.2, 1);
        ClarifyingQuestions.Result[] rs = {
            ClarifyingQuestions.evaluate(d, sft, Training.SFT),
            ClarifyingQuestions.evaluate(d, ClarifyingQuestions.train(d, sft, MultiturnRewardTrainer.Mode.SINGLE_TURN,
                    300, 1), Training.SINGLE_TURN_GRPO),
            ClarifyingQuestions.evaluate(d, ClarifyingQuestions.train(d, sft, MultiturnRewardTrainer.Mode.MULTITURN,
                    300, 1), Training.COLLABLLM),
        };
        for (ClarifyingQuestions.Result r : rs) {
            System.out.printf("%-16s success %.2f (specified %.2f, underspecified %.2f), %.2f turns, asks when needed %.2f,"
                    + " needlessly %.2f%n", r.training(), r.success(), r.specifiedSuccess(), r.underspecifiedSuccess(),
                    r.turns(), r.askRate(), r.needlessAskRate());
        }
        assertTrue(rs[2].underspecifiedSuccess() > rs[1].underspecifiedSuccess() + 0.3, "CollabLLM asks and succeeds");
        assertTrue(rs[2].specifiedSuccess() > 0.8, "and still answers fully specified requests");
    }
}
