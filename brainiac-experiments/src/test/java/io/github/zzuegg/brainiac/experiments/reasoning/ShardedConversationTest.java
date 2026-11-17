package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.MultiturnRewardTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.ClarifyingQuestions.Training;
import org.junit.jupiter.api.Test;

/** Wave 5, 3.16: single-turn against sharded multi-turn accuracy for SFT, single-turn GRPO and CollabLLM. */
class ShardedConversationTest {

    @Test
    void shardingHurtsTheSingleTurnModelsMost() {
        ClarificationDialogue d = new ClarificationDialogue(4, ClarificationDialogue.Mode.COLLABORATIVE);
        WindowLm sft = ClarifyingQuestions.sft(d, 150, 0.2, 1);
        WindowLm[] models = {
            sft,
            ClarifyingQuestions.train(d, sft, MultiturnRewardTrainer.Mode.SINGLE_TURN, 300, 1),
            ClarifyingQuestions.train(d, sft, MultiturnRewardTrainer.Mode.MULTITURN, 300, 1),
        };
        ShardedConversation.Result[] rs = new ShardedConversation.Result[3];
        for (int m = 0; m < 3; m++) {
            rs[m] = ShardedConversation.evaluate(d, models[m], Training.values()[m], 10, 5);
            System.out.printf("%-16s single-turn %.2f -> sharded %.2f (drop %.2f); aptitude %.2f -> %.2f,"
                    + " unreliability %.2f -> %.2f%n", rs[m].training(), rs[m].singleTurn(), rs[m].sharded(),
                    rs[m].drop(), rs[m].singleAptitude(), rs[m].shardedAptitude(), rs[m].singleUnreliability(),
                    rs[m].shardedUnreliability());
        }
        assertTrue(rs[0].drop() > 0.3, "the supervised model gets lost in sharded conversations");
        assertTrue(rs[2].drop() < rs[0].drop() - 0.3, "CollabLLM, which learned to wait for the shard, loses far less");
    }
}
