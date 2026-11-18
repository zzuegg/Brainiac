package io.github.zzuegg.brainiac.experiments.security;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.robust.AdversarialTraining;

import java.util.List;

import org.junit.jupiter.api.Test;

class AttackShootoutTest {

    @Test
    void squareAttackApproachesWhiteBoxPgdFromScoresOnly() {
        for (boolean robust : new boolean[] {false, true}) {
            List<AttackShootout.Score> rows = AttackShootout.scoreBased(robust, 0.2f, 500, 50, 1);
            StringBuilder b = new StringBuilder();
            for (AttackShootout.Score s : rows) {
                b.append(String.format(" %s %.2f/%.0fq", s.attack(), s.success(), s.medianQueries()));
            }
            System.out.printf("Score-based eps 0.2, 500 queries, %s victim:%s%n", robust ? "PGD-AT" : "standard", b);
            double pgd = rows.get(0).success();
            double random = rows.get(1).success();
            double square = rows.get(5).success();
            assertTrue(square > random, "square " + square + " random " + random);
            if (!robust) {
                assertTrue(square > pgd - 0.25, "square " + square + " pgd " + pgd);
            }
        }
    }

    @Test
    void hopSkipJumpIsAheadOfTheBoundaryAttackAtSmallBudgets() {
        long[] budgets = {100, 500, 2000};
        List<AttackShootout.Curve> rows = AttackShootout.decisionBased(budgets, 20, 1);
        AttackShootout.Curve boundary = rows.get(0);
        AttackShootout.Curve hsj = rows.get(1);
        double white = rows.get(2).median()[0];
        System.out.printf("Decision-based median l2 at 100/500/2000 queries: boundary %.2f %.2f %.2f, HopSkipJump "
                + "%.2f %.2f %.2f; white-box PGD minimal l2 %.2f%n", boundary.median()[0], boundary.median()[1],
                boundary.median()[2], hsj.median()[0], hsj.median()[1], hsj.median()[2], white);
        assertTrue(hsj.median()[2] <= hsj.median()[0], "HSJ improves with queries");
        assertTrue(hsj.median()[1] < boundary.median()[1], "HSJ ahead at 500 queries");
        assertTrue(hsj.median()[2] < 2 * white, "HSJ within 2x of white-box at 2000 queries");
    }

    @Test
    void momentumAndEnsemblesRaiseTransferAndAUniversalPerturbationFools() {
        List<AttackShootout.Transfer> rows = AttackShootout.transfer(0.2f, 0.3f, 1);
        StringBuilder b = new StringBuilder();
        for (AttackShootout.Transfer t : rows) {
            b.append(String.format(" %s %.3f;", t.attack(), t.success()));
        }
        System.out.printf("Transfer eps 0.2 (UAP eps 0.3):%s%n", b);
        double random = rows.get(0).success();
        double pgd = rows.get(1).success();
        double miEns = rows.get(4).success();
        assertTrue(pgd > random, "transfer beats noise");
        assertTrue(miEns >= pgd - 0.05, "MI-FGSM ensemble " + miEns + " vs PGD " + pgd);
        assertTrue(rows.get(6).success() > random, "UAP fools more than noise");
    }

    @Test
    void autoAttackIsAtLeastAsStrongAsFixedStepPgd() {
        for (AdversarialTraining.Method m : new AdversarialTraining.Method[] {AdversarialTraining.Method.STANDARD,
                AdversarialTraining.Method.PGD_AT}) {
            List<AttackShootout.Robust> rows = AttackShootout.autoAttack(m, 0.1f, 150, 1);
            StringBuilder b = new StringBuilder();
            for (AttackShootout.Robust r : rows) {
                b.append(String.format(" %s %.3f", r.evaluation(), r.robust()));
            }
            System.out.printf("AutoAttack eps 0.1 %-8s:%s%n", m, b);
            assertTrue(rows.get(5).robust() <= rows.get(1).robust() + 1e-9, "AutoAttack <= PGD-10");
        }
    }
}
