package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experiments.control.ClassicalControl.PlannerScore;
import io.github.zzuegg.brainiac.experiments.control.ClassicalControl.Task;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ClassicalPlannersTest {

    private static String format(List<PlannerScore> rows) {
        StringBuilder sb = new StringBuilder();
        for (PlannerScore r : rows) {
            sb.append(String.format("%s %.2f (success %.2f, %.2f ms/step); ", r.method(), r.cost(), r.success(),
                    r.millisPerStep()));
        }
        return sb.toString();
    }

    private static PlannerScore get(List<PlannerScore> rows, String method) {
        return rows.stream().filter(r -> r.method().equals(method)).findFirst().orElseThrow();
    }

    @Test
    void plannersOnKnownModels() {
        List<PlannerScore> cart = ClassicalControl.planners(Task.CART_POLE_BALANCE, 100, 15, 32, 4, 3);
        List<PlannerScore> pend = ClassicalControl.planners(Task.PENDULUM_SWING_UP, 100, 30, 64, 4, 3);
        List<PlannerScore> mass = ClassicalControl.planners(Task.POINT_MASS, 60, 30, 64, 4, 3);
        double reach = ClassicalControl.massSpringIlqr(3, 30, 1);
        System.out.printf("cart-pole balance: %s| pendulum swing-up: %s| point mass: %s| mass-spring open-loop "
                + "iLQR final distance %.4f (BPTT controller 0.024)%n", format(cart), format(pend), format(mass), reach);
        assertTrue(get(cart, "LQR").success() == 1 && get(cart, "iLQR-MPC").success() == 1, "LQR and iLQR balance");
        assertTrue(get(pend, "iLQR-MPC").success() >= 0.66 && get(pend, "MBD").success() >= 0.66,
                "iLQR and MBD swing up");
        assertTrue(get(mass, "MBD").success() >= 0.66, "MBD reaches the goal around the obstacle");
        assertTrue(reach < 0.05, "iLQR reaches the mass-spring goal");
    }
}
