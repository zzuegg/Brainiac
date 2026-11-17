package io.github.zzuegg.brainiac.backend.vulkan;

import static org.lwjgl.system.MemoryStack.stackPush;
import static org.lwjgl.vulkan.VK12.*;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.LongBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedQueue;

import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.vulkan.EXTDebugUtils;
import org.lwjgl.vulkan.VkApplicationInfo;
import org.lwjgl.vulkan.VkBufferCopy;
import org.lwjgl.vulkan.VkBufferCreateInfo;
import org.lwjgl.vulkan.VkBufferDeviceAddressInfo;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.lwjgl.vulkan.VkCommandBufferAllocateInfo;
import org.lwjgl.vulkan.VkCommandBufferBeginInfo;
import org.lwjgl.vulkan.VkCommandPoolCreateInfo;
import org.lwjgl.vulkan.VkComputePipelineCreateInfo;
import org.lwjgl.vulkan.VkDebugUtilsMessengerCallbackDataEXT;
import org.lwjgl.vulkan.VkDebugUtilsMessengerCreateInfoEXT;
import org.lwjgl.vulkan.VkDevice;
import org.lwjgl.vulkan.VkDeviceCreateInfo;
import org.lwjgl.vulkan.VkDeviceQueueCreateInfo;
import org.lwjgl.vulkan.VkFenceCreateInfo;
import org.lwjgl.vulkan.VkInstance;
import org.lwjgl.vulkan.VkInstanceCreateInfo;
import org.lwjgl.vulkan.VkMemoryAllocateFlagsInfo;
import org.lwjgl.vulkan.VkMemoryAllocateInfo;
import org.lwjgl.vulkan.VkMemoryBarrier;
import org.lwjgl.vulkan.VkMemoryRequirements;
import org.lwjgl.vulkan.VkPhysicalDevice;
import org.lwjgl.vulkan.VkPhysicalDeviceFeatures2;
import org.lwjgl.vulkan.VkPhysicalDeviceMemoryProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceProperties;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan11Features;
import org.lwjgl.vulkan.VkPhysicalDeviceVulkan12Features;
import org.lwjgl.vulkan.VkPipelineLayoutCreateInfo;
import org.lwjgl.vulkan.VkPipelineShaderStageCreateInfo;
import org.lwjgl.vulkan.VkPushConstantRange;
import org.lwjgl.vulkan.VkQueue;
import org.lwjgl.vulkan.VkQueueFamilyProperties;
import org.lwjgl.vulkan.VkShaderModuleCreateInfo;
import org.lwjgl.vulkan.VkSubmitInfo;

/**
 * The Vulkan objects behind {@link VulkanBackend}: one device and compute queue, memory
 * pools, compute pipelines and command recording.
 *
 * <p>Shaders reach buffers through device addresses passed as push constants
 * (buffer device address, core in Vulkan 1.2), so there are no descriptor sets. Commands
 * are recorded into a command buffer and submitted in batches; a memory barrier is inserted
 * only before a command that reads or writes memory an earlier command wrote (or writes
 * memory one read) since the last barrier, so independent commands may overlap.
 *
 * <p>Device memory comes in large blocks, each bound to one buffer; storages get slots of
 * those blocks, pooled by rounded size, which keeps the number of Vulkan allocations small.
 * Not thread-safe: the backend calls it under its own lock, except {@link #free}, which the
 * cleaner thread may call.
 */
final class VkContext {

    /** The size of the push constant block every pipeline declares. */
    static final int PUSH_BYTES = 128;
    /** Slots up to this size are carved from shared chunks; larger ones get their own block. */
    private static final long SMALL_SLOT = 4L << 20;
    private static final long CHUNK = 64L << 20;
    private static final long STAGING_CHUNK = 16L << 20;
    private static final int MAX_IN_FLIGHT = 3;
    private static final int SUBMIT_EVERY = Integer.getInteger("brainiac.vulkan.submitEvery", 64);
    /** {@code -Dbrainiac.vulkan.validate=true} turns on the Khronos validation layer. */
    private static final boolean VALIDATE = Boolean.getBoolean("brainiac.vulkan.validate");
    /** Below this many recorded commands, an idle queue still waits for more before a submit. */
    private static final int SUBMIT_IDLE = Integer.getInteger("brainiac.vulkan.submitIdle", 8);

    /** A block of memory bound to one buffer; host-visible blocks are mapped at {@link #mapped}. */
    static final class Block {
        long memory;
        long buffer;
        long address;
        long size;
        long used;
        long mapped;
    }

    /** A part of a block, owned by one storage at a time. */
    record Slot(Block block, long offset, long bytes, long address) {
    }

    final VkInstance instance;
    final VkPhysicalDevice physical;
    final VkDevice device;
    final VkQueue queue;
    final String deviceName;
    final String deviceList;
    final long deviceMemory;
    final long budget;
    final int maxGroups;
    /** Whether shaders can use doubles (float64 tensors). */
    final boolean fp64;
    /** Whether storage buffers can hold 16-bit values (float16 tensors). */
    final boolean storage16;
    /** Whether shaders have 16-bit integers (with {@link #storage16}: bfloat16 tensors). */
    final boolean int16;
    private final int family;
    private final long commandPool;
    private final long pipelineLayout;
    private final int deviceLocalType;
    private final int uploadType;
    private final int readbackType;

    private final Map<String, Long> pipelines = new HashMap<>();
    private final Map<Long, ArrayDeque<Slot>> free = new HashMap<>();
    private final ConcurrentLinkedQueue<Slot> released = new ConcurrentLinkedQueue<>();
    private final List<Block> dedicated = new ArrayList<>();
    private Block chunk;
    private long allocated;
    private long inUse;
    private long peakInUse;

    private final ArrayDeque<Block> stagingPool = new ArrayDeque<>();
    private Block staging;
    private Block readback;

    private final ArrayDeque<Batch> idleBatches = new ArrayDeque<>();
    private final ArrayDeque<Batch> inFlight = new ArrayDeque<>();
    private Batch current;
    private final Set<Long> written = new HashSet<>();
    private final Set<Long> read = new HashSet<>();
    private final ByteBuffer push = ByteBuffer.allocateDirect(PUSH_BYTES).order(ByteOrder.nativeOrder());
    private final ByteBuffer inline = MemoryUtil.memAlloc(65536);

    long dispatches;
    long barriers;
    long submits;
    long waits;
    long reclaims;
    long collections;
    private long createdSinceCollect;

    /** Commands recorded together, submitted with one fence. */
    private static final class Batch {
        VkCommandBuffer commands;
        long fence;
        int count;
        final List<Runnable> done = new ArrayList<>();
    }

    VkContext() {
        try (MemoryStack st = stackPush()) {
            VkApplicationInfo app = VkApplicationInfo.calloc(st).sType$Default()
                    .pApplicationName(st.UTF8("brainiac")).pEngineName(st.UTF8("brainiac"))
                    .apiVersion(VK_API_VERSION_1_2);
            VkInstanceCreateInfo ici = VkInstanceCreateInfo.calloc(st).sType$Default().pApplicationInfo(app);
            if (VALIDATE) {
                // needs the Khronos validation layer (Vulkan SDK); messages go to standard error
                ici.ppEnabledLayerNames(st.pointers(st.UTF8("VK_LAYER_KHRONOS_validation")))
                        .ppEnabledExtensionNames(st.pointers(st.UTF8(EXTDebugUtils.VK_EXT_DEBUG_UTILS_EXTENSION_NAME)));
            }
            PointerBuffer pp = st.mallocPointer(1);
            check(vkCreateInstance(ici, null, pp), "vkCreateInstance");
            instance = new VkInstance(pp.get(0), ici);
            if (VALIDATE) {
                VkDebugUtilsMessengerCreateInfoEXT mci = VkDebugUtilsMessengerCreateInfoEXT.calloc(st).sType$Default()
                        .messageSeverity(EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                | EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                        .messageType(EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                                | EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                                | EXTDebugUtils.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
                        .pfnUserCallback((severity, types, data, user) -> {
                            System.err.println("[vulkan] " + VkDebugUtilsMessengerCallbackDataEXT.create(data)
                                    .pMessageString());
                            return VK_FALSE;
                        });
                LongBuffer lp = st.mallocLong(1);
                check(EXTDebugUtils.vkCreateDebugUtilsMessengerEXT(instance, mci, null, lp),
                        "vkCreateDebugUtilsMessengerEXT");
            }
        }
        try {
            Candidate chosen = choose();
            physical = chosen.device;
            deviceName = chosen.name;
            deviceList = candidatesText;
            family = chosen.family;
            maxGroups = chosen.maxGroups;
            fp64 = chosen.fp64;
            storage16 = chosen.storage16;
            int16 = chosen.int16;
            try (MemoryStack st = stackPush()) {
                // chained by address: LWJGL's typed pNext setters insert a struct in front of
                // the existing chain and would drop the links made before
                VkPhysicalDeviceVulkan12Features f12 = VkPhysicalDeviceVulkan12Features.calloc(st).sType$Default()
                        .bufferDeviceAddress(true);
                VkPhysicalDeviceVulkan11Features f11 = VkPhysicalDeviceVulkan11Features.calloc(st).sType$Default()
                        .pNext(f12.address()).storageBuffer16BitAccess(storage16);
                VkPhysicalDeviceFeatures2 f2 = VkPhysicalDeviceFeatures2.calloc(st).sType$Default()
                        .pNext(f11.address());
                f2.features().shaderInt64(true).shaderFloat64(fp64).shaderInt16(int16);
                VkDeviceQueueCreateInfo.Buffer q = VkDeviceQueueCreateInfo.calloc(1, st).sType$Default()
                        .queueFamilyIndex(family).pQueuePriorities(st.floats(1f));
                // by address: LWJGL's typed pNext setters insert the struct in front of the
                // existing chain, which would drop f2's link to f12
                VkDeviceCreateInfo dci = VkDeviceCreateInfo.calloc(st).sType$Default().pNext(f2.address())
                        .pQueueCreateInfos(q);
                PointerBuffer pp = st.mallocPointer(1);
                check(vkCreateDevice(physical, dci, null, pp), "vkCreateDevice");
                device = new VkDevice(pp.get(0), physical, dci, VK_API_VERSION_1_2);
                vkGetDeviceQueue(device, family, 0, pp);
                queue = new VkQueue(pp.get(0), device);

                VkCommandPoolCreateInfo pci = VkCommandPoolCreateInfo.calloc(st).sType$Default()
                        .flags(VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT).queueFamilyIndex(family);
                LongBuffer lp = st.mallocLong(1);
                check(vkCreateCommandPool(device, pci, null, lp), "vkCreateCommandPool");
                commandPool = lp.get(0);

                VkPushConstantRange.Buffer range = VkPushConstantRange.calloc(1, st)
                        .stageFlags(VK_SHADER_STAGE_COMPUTE_BIT).offset(0).size(PUSH_BYTES);
                VkPipelineLayoutCreateInfo lci = VkPipelineLayoutCreateInfo.calloc(st).sType$Default()
                        .pPushConstantRanges(range);
                check(vkCreatePipelineLayout(device, lci, null, lp), "vkCreatePipelineLayout");
                pipelineLayout = lp.get(0);

                VkPhysicalDeviceMemoryProperties mp = VkPhysicalDeviceMemoryProperties.calloc(st);
                vkGetPhysicalDeviceMemoryProperties(physical, mp);
                deviceLocalType = memoryType(mp, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0);
                uploadType = memoryType(mp, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                int cached = memoryType(mp, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                        | VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
                readbackType = cached >= 0 ? cached : uploadType;
                if (deviceLocalType < 0 || uploadType < 0) {
                    throw new IllegalStateException("no suitable memory types");
                }
                deviceMemory = mp.memoryHeaps(mp.memoryTypes(deviceLocalType).heapIndex()).size();
                Integer mb = Integer.getInteger("brainiac.vulkan.memory");
                budget = mb != null ? (long) mb << 20 : deviceMemory / 10 * 8;
            }
        } catch (RuntimeException | Error e) {
            vkDestroyInstance(instance, null);
            throw e;
        }
    }

    // ------------------------------------------------------------------ device choice

    private record Candidate(VkPhysicalDevice device, String name, int type, int family, int maxGroups,
            boolean fp64, boolean storage16, boolean int16, String problem) {
    }

    private String candidatesText = "";

    private Candidate choose() {
        String types = System.getProperty("brainiac.vulkan.devices", "gpu").toLowerCase(Locale.ROOT);
        List<Candidate> all = new ArrayList<>();
        try (MemoryStack st = stackPush()) {
            IntBuffer ip = st.mallocInt(1);
            check(vkEnumeratePhysicalDevices(instance, ip, null), "vkEnumeratePhysicalDevices");
            PointerBuffer handles = st.mallocPointer(ip.get(0));
            check(vkEnumeratePhysicalDevices(instance, ip, handles), "vkEnumeratePhysicalDevices");
            for (int i = 0; i < handles.capacity(); i++) {
                all.add(describe(new VkPhysicalDevice(handles.get(i), instance)));
            }
        }
        StringBuilder text = new StringBuilder();
        List<Candidate> usable = new ArrayList<>();
        for (int i = 0; i < all.size(); i++) {
            Candidate c = all.get(i);
            String problem = c.problem;
            if (problem == null && c.type == VK_PHYSICAL_DEVICE_TYPE_CPU && types.equals("gpu")) {
                problem = "a CPU device; set -Dbrainiac.vulkan.devices=all to use it";
            }
            text.append(i == 0 ? "" : ", ").append(i).append(": ").append(c.name)
                    .append(problem == null ? "" : " (" + problem + ")");
            if (problem == null) {
                usable.add(c);
            }
        }
        candidatesText = text.toString();
        Integer index = Integer.getInteger("brainiac.vulkan.device");
        if (index != null) {
            if (index < 0 || index >= all.size() || all.get(index).problem != null) {
                throw new IllegalStateException("brainiac.vulkan.device=" + index + " is not usable; devices: "
                        + candidatesText);
            }
            return all.get(index);
        }
        if (usable.isEmpty()) {
            throw new IllegalStateException("no usable Vulkan device; devices: "
                    + (all.isEmpty() ? "none" : candidatesText));
        }
        usable.sort(Comparator.comparingInt(c -> switch (c.type) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU -> 0;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU -> 1;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU -> 2;
            case VK_PHYSICAL_DEVICE_TYPE_CPU -> 4;
            default -> 3;
        }));
        return usable.get(0);
    }

    private static Candidate describe(VkPhysicalDevice pd) {
        try (MemoryStack st = stackPush()) {
            VkPhysicalDeviceProperties props = VkPhysicalDeviceProperties.calloc(st);
            vkGetPhysicalDeviceProperties(pd, props);
            String name = props.deviceNameString();
            // unsigned: AMD reports 2^32 - 1 here, which as an int would be -1
            int maxGroups = (int) Math.min(Integer.toUnsignedLong(props.limits().maxComputeWorkGroupCount(0)), 65535);
            String problem = null;
            boolean fp64 = false;
            boolean storage16 = false;
            boolean int16 = false;
            if (VK_API_VERSION_MAJOR(props.apiVersion()) == 1 && VK_API_VERSION_MINOR(props.apiVersion()) < 2) {
                problem = "needs Vulkan 1.2";
            } else if (props.limits().maxPushConstantsSize() < PUSH_BYTES) {
                problem = "push constants too small";
            } else {
                VkPhysicalDeviceVulkan12Features f12 = VkPhysicalDeviceVulkan12Features.calloc(st).sType$Default();
                VkPhysicalDeviceVulkan11Features f11 = VkPhysicalDeviceVulkan11Features.calloc(st).sType$Default()
                        .pNext(f12.address());
                VkPhysicalDeviceFeatures2 f2 = VkPhysicalDeviceFeatures2.calloc(st).sType$Default()
                        .pNext(f11.address());
                vkGetPhysicalDeviceFeatures2(pd, f2);
                fp64 = f2.features().shaderFloat64();
                int16 = f2.features().shaderInt16();
                storage16 = f11.storageBuffer16BitAccess();
                if (!f12.bufferDeviceAddress()) {
                    problem = "no buffer device address";
                } else if (!f2.features().shaderInt64()) {
                    problem = "no 64-bit integers in shaders";
                }
            }
            int family = -1;
            IntBuffer ip = st.mallocInt(1);
            vkGetPhysicalDeviceQueueFamilyProperties(pd, ip, null);
            VkQueueFamilyProperties.Buffer families = VkQueueFamilyProperties.calloc(ip.get(0), st);
            vkGetPhysicalDeviceQueueFamilyProperties(pd, ip, families);
            for (int f = 0; f < families.capacity() && family < 0; f++) {
                if ((families.get(f).queueFlags() & VK_QUEUE_COMPUTE_BIT) != 0) {
                    family = f;
                }
            }
            if (family < 0 && problem == null) {
                problem = "no compute queue";
            }
            return new Candidate(pd, name, props.deviceType(), family, maxGroups, fp64, storage16, int16, problem);
        }
    }

    private static int memoryType(VkPhysicalDeviceMemoryProperties mp, int required, int preferred) {
        int fallback = -1;
        for (int i = 0; i < mp.memoryTypeCount(); i++) {
            int flags = mp.memoryTypes(i).propertyFlags();
            if ((flags & required) == required) {
                if ((flags & preferred) == preferred) {
                    return i;
                }
                if (fallback < 0) {
                    fallback = i;
                }
            }
        }
        return fallback;
    }

    // ------------------------------------------------------------------ memory

    static long bucket(long bytes) {
        long b = Math.max(256, bytes);
        long step = Long.highestOneBit(b) >> 3;
        return (b + step - 1) / step * step;
    }

    /** A device-local slot of at least {@code bytes}. Contents are undefined. */
    Slot allocate(long bytes) {
        long size = bucket(bytes);
        drainReleased();
        ArrayDeque<Slot> list = free.get(size);
        Slot s = list == null ? null : list.poll();
        if (s == null) {
            if (allocated + (size > SMALL_SLOT ? size : CHUNK) > budget && allocated > 0) {
                reclaim();
                list = free.get(size);
                s = list == null ? null : list.poll();
            }
            if (s == null) {
                s = create(size);
            }
        }
        inUse += s.bytes;
        peakInUse = Math.max(peakInUse, inUse);
        return s;
    }

    private Slot create(long size) {
        createdSinceCollect += size;
        if (createdSinceCollect > Math.min(128L << 20, budget / 16)) {
            // Slots return to the pool only once the JVM collects their tensors, which can
            // take long when little Java memory is allocated (inference queues many batches
            // and releases nothing). Ask for it rather than filling the device with new blocks.
            createdSinceCollect = 0;
            collections++;
            System.gc();
            drainReleased();
            ArrayDeque<Slot> list = free.get(size);
            Slot s = list == null ? null : list.poll();
            if (s != null) {
                return s;
            }
        }
        if (size > SMALL_SLOT) {
            Block b = block(size, deviceLocalType, false);
            dedicated.add(b);
            return new Slot(b, 0, size, b.address);
        }
        if (chunk == null || chunk.used + size > chunk.size) {
            chunk = block(CHUNK, deviceLocalType, false);
        }
        Slot s = new Slot(chunk, chunk.used, size, chunk.address + chunk.used);
        chunk.used += size;
        return s;
    }

    /** Called from any thread, such as the cleaner's. */
    void free(Slot slot) {
        released.add(slot);
    }

    private void drainReleased() {
        Slot s;
        while ((s = released.poll()) != null) {
            inUse -= s.bytes;
            free.computeIfAbsent(s.bytes, k -> new ArrayDeque<>()).add(s);
        }
    }

    /** Frees unused dedicated blocks after collecting garbage, to stay within the budget. */
    private void reclaim() {
        reclaims++;
        finish();
        System.gc();
        try {
            Thread.sleep(20);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        drainReleased();
        for (var it = dedicated.iterator(); it.hasNext(); ) {
            Block b = it.next();
            ArrayDeque<Slot> list = free.get(b.size);
            if (list != null && list.removeIf(s -> s.block == b)) {
                destroy(b);
                it.remove();
            }
        }
    }

    private Block block(long size, int type, boolean map) {
        try {
            return tryBlock(size, type, map);
        } catch (OutOfDeviceMemory e) {
            if (type == deviceLocalType) {
                reclaim();
                return tryBlock(size, type, map);
            }
            throw e;
        }
    }

    private static final class OutOfDeviceMemory extends IllegalStateException {
        private static final long serialVersionUID = 1L;

        OutOfDeviceMemory(String message) {
            super(message);
        }
    }

    private Block tryBlock(long size, int type, boolean map) {
        try (MemoryStack st = stackPush()) {
            Block b = new Block();
            b.size = size;
            VkBufferCreateInfo bci = VkBufferCreateInfo.calloc(st).sType$Default().size(size)
                    .usage(VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                            | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)
                    .sharingMode(VK_SHARING_MODE_EXCLUSIVE);
            LongBuffer lp = st.mallocLong(1);
            check(vkCreateBuffer(device, bci, null, lp), "vkCreateBuffer");
            b.buffer = lp.get(0);
            VkMemoryRequirements req = VkMemoryRequirements.calloc(st);
            vkGetBufferMemoryRequirements(device, b.buffer, req);
            if ((req.memoryTypeBits() & (1 << type)) == 0) {
                vkDestroyBuffer(device, b.buffer, null);
                throw new IllegalStateException("memory type " + type + " cannot hold storage buffers");
            }
            VkMemoryAllocateFlagsInfo flags = VkMemoryAllocateFlagsInfo.calloc(st).sType$Default()
                    .flags(VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT);
            VkMemoryAllocateInfo mai = VkMemoryAllocateInfo.calloc(st).sType$Default().pNext(flags)
                    .allocationSize(req.size()).memoryTypeIndex(type);
            int r = vkAllocateMemory(device, mai, null, lp);
            if (r != VK_SUCCESS) {
                vkDestroyBuffer(device, b.buffer, null);
                if (r == VK_ERROR_OUT_OF_DEVICE_MEMORY || r == VK_ERROR_OUT_OF_HOST_MEMORY) {
                    throw new OutOfDeviceMemory("out of Vulkan memory allocating " + (size >> 20) + " MB ("
                            + (allocated >> 20) + " MB allocated)");
                }
                check(r, "vkAllocateMemory");
            }
            b.memory = lp.get(0);
            check(vkBindBufferMemory(device, b.buffer, b.memory, 0), "vkBindBufferMemory");
            b.address = vkGetBufferDeviceAddress(device,
                    VkBufferDeviceAddressInfo.calloc(st).sType$Default().buffer(b.buffer));
            if (map) {
                PointerBuffer pp = st.mallocPointer(1);
                check(vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, pp), "vkMapMemory");
                b.mapped = pp.get(0);
            }
            if (type == deviceLocalType) {
                allocated += size;
            }
            return b;
        }
    }

    private void destroy(Block b) {
        if (b.mapped != 0) {
            vkUnmapMemory(device, b.memory);
        }
        vkDestroyBuffer(device, b.buffer, null);
        vkFreeMemory(device, b.memory, null);
        if (b.memory != 0 && b.mapped == 0) {
            allocated -= b.size;
        }
    }

    long allocated() {
        return allocated;
    }

    long peakInUse() {
        return peakInUse;
    }

    // ------------------------------------------------------------------ pipelines

    private long pipeline(String name) {
        Long p = pipelines.get(name);
        if (p != null) {
            return p;
        }
        byte[] code;
        try (InputStream in = VkContext.class.getResourceAsStream("shaders/" + name + ".spv")) {
            if (in == null) {
                throw new IllegalStateException("missing shader " + name);
            }
            code = in.readAllBytes();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        long p2 = createPipeline(code);
        pipelines.put(name, p2);
        return p2;
    }

    /** Whether a pipeline is known by this name, precompiled or {@link #register registered}. */
    boolean hasPipeline(String name) {
        return pipelines.containsKey(name);
    }

    /** Makes SPIR-V compiled at run time (a fused shader) available to {@link #dispatch} by name. */
    void register(String name, byte[] code) {
        pipelines.put(name, createPipeline(code));
    }

    private long createPipeline(byte[] code) {
        ByteBuffer spirv = MemoryUtil.memAlloc(code.length);
        try (MemoryStack st = stackPush()) {
            spirv.put(code).flip();
            LongBuffer lp = st.mallocLong(1);
            check(vkCreateShaderModule(device, VkShaderModuleCreateInfo.calloc(st).sType$Default().pCode(spirv),
                    null, lp), "vkCreateShaderModule");
            long module = lp.get(0);
            VkPipelineShaderStageCreateInfo stage = VkPipelineShaderStageCreateInfo.calloc(st).sType$Default()
                    .stage(VK_SHADER_STAGE_COMPUTE_BIT).module(module).pName(st.UTF8("main"));
            VkComputePipelineCreateInfo.Buffer ci = VkComputePipelineCreateInfo.calloc(1, st).sType$Default()
                    .stage(stage).layout(pipelineLayout);
            check(vkCreateComputePipelines(device, VK_NULL_HANDLE, ci, null, lp), "vkCreateComputePipelines");
            vkDestroyShaderModule(device, module, null);
            return lp.get(0);
        } finally {
            MemoryUtil.memFree(spirv);
        }
    }

    // ------------------------------------------------------------------ recording

    /** The push constant buffer to fill for the next {@link #dispatch}; cleared. */
    ByteBuffer push() {
        push.clear();
        return push;
    }

    private Batch recording() {
        if (current == null) {
            Batch b = idleBatches.poll();
            try (MemoryStack st = stackPush()) {
                if (b == null) {
                    b = new Batch();
                    VkCommandBufferAllocateInfo ai = VkCommandBufferAllocateInfo.calloc(st).sType$Default()
                            .commandPool(commandPool).level(VK_COMMAND_BUFFER_LEVEL_PRIMARY).commandBufferCount(1);
                    PointerBuffer pp = st.mallocPointer(1);
                    check(vkAllocateCommandBuffers(device, ai, pp), "vkAllocateCommandBuffers");
                    b.commands = new VkCommandBuffer(pp.get(0), device);
                    LongBuffer lp = st.mallocLong(1);
                    check(vkCreateFence(device, VkFenceCreateInfo.calloc(st).sType$Default(), null, lp),
                            "vkCreateFence");
                    b.fence = lp.get(0);
                }
                check(vkBeginCommandBuffer(b.commands, VkCommandBufferBeginInfo.calloc(st).sType$Default()
                        .flags(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)), "vkBeginCommandBuffer");
            }
            current = b;
        }
        return current;
    }

    /** Inserts a barrier first if this access conflicts with an earlier one since the last barrier. */
    private void access(long write, long... reads) {
        boolean hazard = write != 0 && (written.contains(write) || read.contains(write));
        for (int i = 0; i < reads.length && !hazard; i++) {
            hazard = written.contains(reads[i]);
        }
        if (hazard) {
            barrier();
        }
        if (write != 0) {
            written.add(write);
        }
        for (long r : reads) {
            read.add(r);
        }
    }

    private void barrier() {
        try (MemoryStack st = stackPush()) {
            VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, st).sType$Default()
                    .srcAccessMask(VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT)
                    .dstAccessMask(VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                            | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT);
            int stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
            vkCmdPipelineBarrier(recording().commands, stages, stages, 0, mb, null, null);
        }
        written.clear();
        read.clear();
        barriers++;
    }

    /**
     * Records a compute dispatch of {@code groups} work groups with the push constants in
     * {@link #push()}. {@code write} and {@code reads} are the slots touched (their base addresses).
     */
    void dispatch(String shader, long groups, Slot write, Slot... reads) {
        dispatch(shader, groups, 1, write, reads);
    }

    void dispatch(String shader, long groups, int groupsZ, Slot write, Slot... reads) {
        if (groups == 0 || groupsZ == 0) {
            return;
        }
        long x = Math.min(groups, maxGroups);
        long y = (groups + x - 1) / x;
        dispatch(shader, (int) x, (int) y, groupsZ, write, reads);
    }

    void dispatch(String shader, int x, int y, int z, Slot write, Slot... reads) {
        dispatch(shader, x, y, z, new Slot[] {write}, reads);
    }

    /** A dispatch that writes several slots. */
    void dispatch(String shader, long groups, Slot[] writes, Slot... reads) {
        long x = Math.min(groups, maxGroups);
        dispatch(shader, (int) x, (int) ((groups + x - 1) / x), 1, writes, reads);
    }

    private void dispatch(String shader, int x, int y, int z, Slot[] writes, Slot... reads) {
        long p = pipeline(shader);
        long[] w = addresses(writes);
        boolean hazard = false;
        for (long a : w) {
            hazard |= a != 0 && (written.contains(a) || read.contains(a));
        }
        if (hazard) {
            barrier();
        }
        for (int i = 0; i < w.length; i++) {
            access(w[i], i == 0 ? addresses(reads) : new long[0]);
        }
        Batch b = recording();
        vkCmdBindPipeline(b.commands, VK_PIPELINE_BIND_POINT_COMPUTE, p);
        push.flip();
        vkCmdPushConstants(b.commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push);
        vkCmdDispatch(b.commands, x, y, z);
        dispatches++;
        recorded(b);
    }

    private static long[] addresses(Slot[] slots) {
        long[] a = new long[slots.length];
        for (int i = 0; i < slots.length; i++) {
            a[i] = slots[i].address;
        }
        return a;
    }

    /** Sets {@code bytes} bytes at {@code offset} of a slot to a 32-bit pattern. */
    void fill(Slot s, long offset, long bytes, int pattern) {
        if (bytes == 0) {
            return;
        }
        access(s.address);
        Batch b = recording();
        vkCmdFillBuffer(b.commands, s.block.buffer, s.offset + offset, bytes, pattern);
        recorded(b);
    }

    /** Copies bytes between slots. */
    void copy(Slot from, long fromOffset, Slot to, long toOffset, long bytes) {
        if (bytes == 0) {
            return;
        }
        access(to.address, from.address);
        Batch b = recording();
        try (MemoryStack st = stackPush()) {
            vkCmdCopyBuffer(b.commands, from.block.buffer, to.block.buffer, VkBufferCopy.calloc(1, st)
                    .srcOffset(from.offset + fromOffset).dstOffset(to.offset + toOffset).size(bytes));
        }
        recorded(b);
    }

    /**
     * Writes host data into a slot: small amounts inline in the command buffer, larger ones
     * through a staging buffer. {@code writer} fills a direct buffer of {@code bytes} bytes.
     */
    void upload(Slot s, long bytes, java.util.function.Consumer<ByteBuffer> writer) {
        if (bytes == 0) {
            return;
        }
        access(s.address);
        Batch b = recording();
        if (bytes <= inline.capacity()) {
            // whole 32-bit words, as vkCmdUpdateBuffer requires; the padding stays inside the slot
            int words = (int) ((bytes + 3) & ~3L);
            inline.clear().limit(words);
            inline.putInt(words - 4, 0);
            writer.accept(inline);
            inline.position(0).limit(words);
            vkCmdUpdateBuffer(b.commands, s.block.buffer, s.offset, inline);
        } else {
            Block from;
            long at;
            if (bytes > STAGING_CHUNK / 2) {
                from = block(bytes, uploadType, true);
                at = 0;
                b.done.add(() -> destroy(from));
            } else {
                at = stage(b, bytes);
                from = staging;
            }
            writer.accept(MemoryUtil.memByteBuffer(from.mapped + at, (int) bytes).order(ByteOrder.nativeOrder()));
            try (MemoryStack st = stackPush()) {
                vkCmdCopyBuffer(b.commands, from.buffer, s.block.buffer,
                        VkBufferCopy.calloc(1, st).srcOffset(at).dstOffset(s.offset).size(bytes));
            }
        }
        recorded(b);
    }

    /** Room for {@code bytes} bytes in the staging chunk, for commands recorded into {@code b}. */
    private long stage(Batch b, long bytes) {
        if (staging == null || staging.used + bytes > staging.size) {
            // a staging chunk serves one batch (submit lets go of it) and returns when it is done
            Block fresh = stagingPool.poll();
            staging = fresh != null ? fresh : block(STAGING_CHUNK, uploadType, true);
            staging.used = 0;
            Block mine = staging;
            b.done.add(() -> stagingPool.add(mine));
        }
        long at = staging.used;
        staging.used += (bytes + 255) & ~255L;
        return at;
    }

    /**
     * Host data that commands recorded next read by device address, such as the pointers and
     * shapes of a fused shader: written into host-visible staging memory now (coherent, so
     * visible at submit) instead of copied by a command. {@code writer} fills {@code bytes} bytes.
     *
     * @return the device address of the data
     */
    long hostData(int bytes, java.util.function.Consumer<ByteBuffer> writer) {
        Batch b = recording();
        long at = stage(b, bytes);
        writer.accept(MemoryUtil.memByteBuffer(staging.mapped + at, bytes).order(ByteOrder.nativeOrder()));
        return staging.address + at;
    }

    /** Copies a slot's first {@code bytes} bytes to the host, waiting for all work before it. */
    ByteBuffer download(Slot s, long bytes) {
        if (readback == null || readback.size < bytes) {
            if (readback != null) {
                finish();
                destroy(readback);
            }
            readback = block(Math.max(bucket(bytes), 1L << 20), readbackType, true);
        }
        if (bytes > 0) {
            access(0, s.address);
            Batch b = recording();
            try (MemoryStack st = stackPush()) {
                vkCmdCopyBuffer(b.commands, s.block.buffer, readback.buffer,
                        VkBufferCopy.calloc(1, st).srcOffset(s.offset).dstOffset(0).size(bytes));
                VkMemoryBarrier.Buffer mb = VkMemoryBarrier.calloc(1, st).sType$Default()
                        .srcAccessMask(VK_ACCESS_TRANSFER_WRITE_BIT).dstAccessMask(VK_ACCESS_HOST_READ_BIT);
                vkCmdPipelineBarrier(b.commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, mb,
                        null, null);
            }
        }
        finish();
        return MemoryUtil.memByteBuffer(readback.mapped, (int) bytes).order(ByteOrder.nativeOrder());
    }

    private void recorded(Batch b) {
        b.count++;
        if (b.count >= SUBMIT_EVERY) {
            submit();
        } else if (b.count >= SUBMIT_IDLE) {
            retire();
            if (inFlight.isEmpty()) {
                submit();
            }
        }
    }

    /** Submits the recorded commands without waiting. */
    void submit() {
        Batch b = current;
        if (b == null) {
            return;
        }
        current = null;
        staging = null;   // its space belongs to this batch until it completes
        while (inFlight.size() >= MAX_IN_FLIGHT) {
            complete(inFlight.poll());
        }
        try (MemoryStack st = stackPush()) {
            check(vkEndCommandBuffer(b.commands), "vkEndCommandBuffer");
            VkSubmitInfo si = VkSubmitInfo.calloc(st).sType$Default().pCommandBuffers(st.pointers(b.commands));
            check(vkQueueSubmit(queue, si, b.fence), "vkQueueSubmit");
        }
        submits++;
        inFlight.add(b);
    }

    /** Recycles batches the device has finished. */
    private void retire() {
        while (!inFlight.isEmpty() && vkGetFenceStatus(device, inFlight.peek().fence) == VK_SUCCESS) {
            complete(inFlight.poll());
        }
    }

    private void complete(Batch b) {
        try (MemoryStack st = stackPush()) {
            LongBuffer fence = st.longs(b.fence);
            if (vkGetFenceStatus(device, b.fence) != VK_SUCCESS) {
                waits++;
                check(vkWaitForFences(device, fence, true, Long.MAX_VALUE), "vkWaitForFences");
            }
            check(vkResetFences(device, fence), "vkResetFences");
        }
        check(vkResetCommandBuffer(b.commands, 0), "vkResetCommandBuffer");
        for (Runnable r : b.done) {
            r.run();
        }
        b.done.clear();
        b.count = 0;
        idleBatches.add(b);
    }

    /** Submits everything recorded and waits until the device has run it. */
    void finish() {
        submit();
        while (!inFlight.isEmpty()) {
            complete(inFlight.poll());
        }
        written.clear();
        read.clear();
    }

    static void check(int result, String what) {
        if (result != VK_SUCCESS) {
            throw new IllegalStateException(what + " failed: VkResult " + result);
        }
    }
}
