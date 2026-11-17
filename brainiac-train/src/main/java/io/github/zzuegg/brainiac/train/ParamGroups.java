package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Gives each parameter to the first group that takes it, the rest to a default optimizer. */
final class ParamGroups implements Optimizer {
    private final Optimizer rest;
    private final List<Group> groups;

    ParamGroups(Optimizer rest, List<Group> groups) {
        this.rest = rest;
        this.groups = List.copyOf(groups);
    }

    private List<List<Var>> split(List<Var> parameters) {
        List<List<Var>> parts = new ArrayList<>();
        for (int i = 0; i <= groups.size(); i++) {
            parts.add(new ArrayList<>());
        }
        for (Var p : parameters) {
            int g = 0;
            while (g < groups.size() && !groups.get(g).takes().test(p)) {
                g++;
            }
            parts.get(g).add(p);
        }
        return parts;
    }

    private Optimizer optimizer(int g) {
        return g < groups.size() ? groups.get(g).optimizer() : rest;
    }

    @Override
    public void step(List<Var> parameters) {
        List<List<Var>> parts = split(parameters);
        for (int g = 0; g < parts.size(); g++) {
            if (!parts.get(g).isEmpty()) {
                optimizer(g).step(parts.get(g));
            }
        }
    }

    @Override
    public float learningRate() {
        return rest.learningRate();
    }

    /** Scales every group's rate by the same factor, so their ratios stay. */
    @Override
    public void learningRate(float learningRate) {
        float factor = learningRate / rest.learningRate();
        for (Group g : groups) {
            g.optimizer().learningRate(g.optimizer().learningRate() * factor);
        }
        rest.learningRate(learningRate);
    }

    @Override
    public Map<String, Tensor> state(List<Var> parameters) {
        Map<String, Tensor> out = new LinkedHashMap<>();
        List<List<Var>> parts = split(parameters);
        for (int g = 0; g < parts.size(); g++) {
            String prefix = "group" + g + ".";
            optimizer(g).state(parts.get(g)).forEach((k, v) -> out.put(prefix + k, v));
        }
        return out;
    }

    @Override
    public void loadState(List<Var> parameters, Map<String, Tensor> state) {
        List<List<Var>> parts = split(parameters);
        for (int g = 0; g < parts.size(); g++) {
            String prefix = "group" + g + ".";
            Map<String, Tensor> mine = new LinkedHashMap<>();
            state.forEach((k, v) -> {
                if (k.startsWith(prefix)) {
                    mine.put(k.substring(prefix.length()), v);
                }
            });
            optimizer(g).loadState(parts.get(g), mine);
        }
    }

    @Override
    public String toString() {
        return "Groups(" + groups + ", rest=" + rest + ")";
    }
}
