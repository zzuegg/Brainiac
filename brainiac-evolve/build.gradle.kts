plugins {
    id("brainiac.java-library")
}

dependencies {
    api(project(":brainiac-train"))
    testRuntimeOnly(project(":brainiac-backend-cpu"))
}

dependencies {
    // the GPU tests run on PoCL in CI and are skipped where no OpenCL driver is installed
    testRuntimeOnly(project(":brainiac-backend-opencl"))
}

// Sub-buffers are pinned off: whether the backend carves buffers from slabs is otherwise
// decided by timing, and PoCL in CI aborted on a sub-buffer of a sub-buffer
// (pocl_kernel_collect_mem_objs: buf->parent == NULL) while these tests ran.
tasks.test {
    jvmArgs("--enable-native-access=ALL-UNNAMED", "-Dbrainiac.opencl.devices=all",
        "-Dbrainiac.opencl.subBuffers=false")
}
