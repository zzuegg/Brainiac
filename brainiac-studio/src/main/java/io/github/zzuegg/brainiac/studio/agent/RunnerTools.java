package io.github.zzuegg.brainiac.studio.agent;

import io.github.zzuegg.brainiac.llm.Tools;
import io.github.zzuegg.brainiac.studio.chat.ChatService;
import io.github.zzuegg.brainiac.studio.chat.GpuResidency;
import io.github.zzuegg.brainiac.studio.chat.VastAi;
import io.github.zzuegg.brainiac.studio.project.Project;
import io.github.zzuegg.brainiac.studio.project.Projects;
import io.github.zzuegg.brainiac.studio.train.VastRentals;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.function.Supplier;

/**
 * Tools for where work runs: rented GPU machines and the project's routes per kind of task.
 */
public final class RunnerTools {

    private final ChatService chat;
    private final Supplier<VastRentals> rentals;
    private final Projects projects;

    /**
     * Creates the tools.
     *
     * @param chat the models and Vast.ai
     * @param rentals the machines rented, or a supplier of null where renting does not work
     * @param projects the projects, or null
     */
    public RunnerTools(ChatService chat, Supplier<VastRentals> rentals, Projects projects) {
        this.chat = chat;
        this.rentals = rentals;
        this.projects = projects;
    }

    @Tools.Tool(name = "find_machines", value = "Lists GPU machines that can be rented from Vast.ai now, cheapest "
            + "first, with their price per hour.")
    @Tier(value = Risk.READ, resultChars = 10_000)
    @Returns("[{id, gpu, price per hour, ...}], at most 15")
    public List<Map<String, Object>> findMachines(@Tools.Param("a GPU name, such as 'RTX 4090'; any if left out") Optional<String> gpu,
            @Tools.Param("the least GPU memory in GB") Optional<Double> memory,
            @Tools.Param("how many GPUs, 1 if left out") Optional<Integer> gpus,
            @Tools.Param("the most dollars an hour") Optional<Double> maxPrice) {
        try {
            List<Map<String, Object>> all = chat.vast().offers(gpu.orElse(""), memory.orElse(0d), gpus.orElse(1),
                    maxPrice.orElse(0d));
            return all.size() > 15 ? all.subList(0, 15) : all;
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AgentRuns.Stopped();
        }
    }

    @Tools.Tool(name = "list_rentals", value = "Lists the machines the studio has rented, with what each has cost so far.")
    @Tier(value = Risk.READ)
    @Returns("[{instance, state, gpu, price per hour, ...}]")
    public List<Map<String, Object>> listRentals() {
        return need().list();
    }

    @Tools.Tool(name = "rent_machine", value = "Rents a GPU machine from Vast.ai as a worker for the studio, for a "
            + "number of hours, after which it is destroyed. Costs money: the user is asked first.")
    @Tier(value = Risk.SPEND, what = "rents a GPU machine, which costs money by the hour")
    @Returns("{instance, state, ...}")
    public Map<String, Object> rentMachine(@Tools.Param("the offer's id from find_machines") long offer,
            @Tools.Param("the price per hour the offer showed") double pricePerHour,
            @Tools.Param("how many hours it may run, at most the studio's limit") double hours) {
        VastRentals r = need();
        try {
            Map<String, Object> o = chat.vast().offer(offer).orElseThrow(() -> new IllegalArgumentException(
                    "that machine is no longer offered; find machines again"));
            return r.rent(VastAi.toOffer(o), pricePerHour, hours);
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AgentRuns.Stopped();
        }
    }

    @Tools.Tool(name = "stop_machine", value = "Destroys a rented machine, so it costs nothing more. Its files are lost.")
    @Tier(value = Risk.WRITE, what = "destroys a rented machine")
    @Returns("text: that the machine is going")
    public String stopMachine(@Tools.Param("the instance's id from list_rentals") long instance) {
        try {
            need().destroy(instance);
            return "destroying instance " + instance + "; list_rentals shows when it is gone";
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    @Tools.Tool(name = "set_route", value = "Sets which model the project uses for a kind of task (text, image, "
            + "speech, transcribe, video), and what happens when the GPU is busy.")
    @Tier(value = Risk.WRITE, what = "changes the project's routes")
    @Returns("{project, routes}")
    public Map<String, Object> setRoute(@Tools.Param("text, image, speech, transcribe or video") String medium,
            @Tools.Param("the model, or 'auto'") String model,
            @Tools.Param("when the GPU is busy: swap, queue, remote or ask; swap if left out") Optional<String> policy,
            @Tools.Param("a model to fall back to") Optional<String> fallback) {
        if (projects == null) {
            throw new IllegalArgumentException("there are no projects here");
        }
        Project p = FileTools.project();
        if (!Project.MEDIA.contains(medium)) {
            throw new IllegalArgumentException("the kind of task is one of " + Project.MEDIA);
        }
        GpuResidency.Policy pol;
        try {
            pol = GpuResidency.Policy.valueOf(policy.orElse("swap").strip().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException("the policy is swap, queue, remote or ask");
        }
        Map<String, ChatService.Route> routes = new LinkedHashMap<>(p.routes());
        routes.put(medium, new ChatService.Route(model, pol, fallback.orElse(null)));
        try {
            Project saved = projects.save(new Project(p.id(), p.name(), p.description(), p.folder(), p.tools(), p.models(),
                    routes, p.runners()), false);
            return Map.of("project", saved.id(), "routes", saved.toJson().get("routes"));
        } catch (IOException e) {
            throw new UncheckedIOException(e.getMessage(), e);
        }
    }

    private VastRentals need() {
        VastRentals r = rentals.get();
        if (r == null) {
            throw new IllegalArgumentException("renting machines does not work in this studio");
        }
        r.unavailable().ifPresent(u -> {
            throw new IllegalArgumentException(u);
        });
        return r;
    }
}
