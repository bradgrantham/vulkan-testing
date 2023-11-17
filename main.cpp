#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <string>

#include <cstring>
#include <cassert>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#if defined(_WIN32)
#define PLATFORM_WINDOWS
// Windows supported
#elif defined(__linux__)
#define PLATFORM_LINUX
// Linux supported
#elif defined(__APPLE__) && defined(__MACH__)
// MacOS not supported yet but want to get to runtime
#define PLATFORM_MACOS
#else
#error Platform not supported.
#endif

#define DEFAULT_FENCE_TIMEOUT 100000000000

// Styled somewhat after Sascha Willem's triangle

VkInstance instance;
VkPhysicalDevice physical_device;
const uint32_t NO_QUEUE_FAMILY = 0xffffffff;
uint32_t preferred_queue_family = NO_QUEUE_FAMILY;
VkDevice device;
VkPhysicalDeviceMemoryProperties memory_properties;
VkQueue queue;
VkCommandPool command_pool;
VkSurfaceKHR surface;
VkSwapchainKHR swapchain;
uint32_t swapchainIndex;
std::vector<VkCommandBuffer> commandBuffers;
uint32_t swapchainImageCount = 3;
std::vector<VkImage> swapchainImages;
std::vector<VkSemaphore> image_acquired_semaphores;
std::vector<VkSemaphore> draw_completed_semaphores;
std::vector<VkFence> draw_completed_fences;
VkRenderPass renderPass;
VkPipeline pipeline;
std::vector<VkFramebuffer> framebuffers;

int windowWidth, windowHeight;

#ifdef PLATFORM_MACOS
void *window;
#endif // PLATFORM_MACOS

bool be_noisy = true;
bool enable_validation = false;
bool dump_vulkan_calls = false;
bool do_the_wrong_thing = false;

struct vertex {
    float v[3];
    float c[3];
};

struct buffer {
    VkDeviceMemory mem;
    VkBuffer buf;
};

buffer vertex_buffer;
buffer index_buffer;

#if 0

VkDeviceMemory vs_uniform_block_memory;
VkBuffer vs_uniform_block_buffer;
VkDescriptorBufferInfo vs_uniform_block_descriptor;

struct {
    float projection[16];
    float model[16];
    float view[16];
};

VkPipelineLayout pipeline_layout;
VkPipeline pipeline;
VkDescriptorSetLayout descriptor_set_layout;
VkDescriptorSet descriptor_set;

VkSemaphore present_complete;
VkSemaphore render_complete;
std::vector<VkFence> wait_fences;

#endif

#define STR(f) #f

std::map<VkResult, std::string> vkresult_name_map =
{
    {VK_ERROR_OUT_OF_HOST_MEMORY, "OUT_OF_HOST_MEMORY"},
    {VK_ERROR_OUT_OF_DEVICE_MEMORY, "OUT_OF_DEVICE_MEMORY"},
    {VK_ERROR_INITIALIZATION_FAILED, "INITIALIZATION_FAILED"},
    {VK_ERROR_DEVICE_LOST, "DEVICE_LOST"},
    {VK_ERROR_MEMORY_MAP_FAILED, "MEMORY_MAP_FAILED"},
    {VK_ERROR_LAYER_NOT_PRESENT, "LAYER_NOT_PRESENT"},
    {VK_ERROR_EXTENSION_NOT_PRESENT, "EXTENSION_NOT_PRESENT"},
    {VK_ERROR_FEATURE_NOT_PRESENT, "FEATURE_NOT_PRESENT"},
};

#define VK_CHECK(f) \
{ \
    VkResult result = (f); \
    static const std::set<VkResult> okay{VK_SUCCESS, VK_SUBOPTIMAL_KHR, VK_THREAD_IDLE_KHR, VK_THREAD_DONE_KHR, VK_OPERATION_DEFERRED_KHR, VK_OPERATION_NOT_DEFERRED_KHR}; \
    if(!okay.contains(result)) { \
	if(vkresult_name_map.count(f) > 0) { \
	    std::cerr << "VkResult from " STR(f) " was " << vkresult_name_map[result] << " at line " << __LINE__ << "\n"; \
	} else { \
	    std::cerr << "VkResult from " STR(f) " was " << result << " at line " << __LINE__ << "\n"; \
        } \
	exit(EXIT_FAILURE); \
    } \
}


void print_implementation_information()
{
    uint32_t ext_count;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::unique_ptr<VkExtensionProperties[]> exts(new VkExtensionProperties[ext_count]);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.get());
    if(be_noisy) {
        printf("Vulkan instance extensions:\n");
        for(uint32_t i = 0; i < ext_count; i++) {
            printf("\t%s, %08X\n", exts[i].extensionName, exts[i].specVersion);
        }
    }
}

void create_instance(VkInstance* instance)
{
    std::set<std::string> extension_set;
    std::set<std::string> layer_set;

    uint32_t glfw_reqd_extension_count;
    const char** glfw_reqd_extensions = glfwGetRequiredInstanceExtensions(&glfw_reqd_extension_count);
    extension_set.insert(glfw_reqd_extensions, glfw_reqd_extensions + glfw_reqd_extension_count);

    extension_set.insert(VK_KHR_SURFACE_EXTENSION_NAME);
#if defined(PLATFORM_WINDOWS)
    extension_set.insert("VK_KHR_win32_surface");
#elif defined(PLATFORM_LINUX)
    extension_set.insert("VK_KHR_xcb_surface");
#elif defined(PLATFORM_MACOS)
    extension_set.insert("VK_MVK_macos_surface");
    extension_set.insert(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

    if(enable_validation) {
	layer_set.insert("VK_LAYER_KHRONOS_validation");
    }
    if(dump_vulkan_calls) {
	layer_set.insert("VK_LAYER_LUNARG_api_dump");
    }

    [&](const std::set<std::string> &extension_set, const std::set<std::string> &layer_set) {
        std::vector<const char*> extensions;
        std::vector<const char*> layers;

	for(auto& s: extension_set) {
	    extensions.push_back(s.c_str());
        }

	for(auto& s: layer_set) {
	    layers.push_back(s.c_str());
        }

	VkApplicationInfo app_info {};
	app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app_info.pApplicationName = "triangle";
	app_info.pEngineName = "triangle";
	app_info.apiVersion = VK_API_VERSION_1_2;

	VkInstanceCreateInfo create{};
	create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	create.pNext = nullptr;
#if defined(PLATFORM_MACOS)
        create.flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
#else
        create.flags = 0;
#endif
	create.pApplicationInfo = &app_info;
	create.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
	create.ppEnabledExtensionNames = extensions.data();
	create.enabledLayerCount = static_cast<uint32_t>(layers.size());
	create.ppEnabledLayerNames = layers.data();

	VK_CHECK(vkCreateInstance(&create, nullptr, instance));
    }(extension_set, layer_set);
}

void choose_physical_device(VkInstance instance, VkPhysicalDevice* physical_device)
{
    uint32_t gpu_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr));
    if(be_noisy) {
        std::cerr << gpu_count << " gpus enumerated\n";
    }
    VkPhysicalDevice physical_devices[32];
    gpu_count = std::min(32u, gpu_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, physical_devices));
    *physical_device = physical_devices[0];
}

const char* device_types[] = {
    "other",
    "integrated GPU",
    "discrete GPU",
    "virtual GPU",
    "CPU",
    "unknown",
};

std::map<uint32_t, std::string> memory_property_bit_name_map = {
    {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL"},
    {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE"},
    {VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT"},
    {VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED"},
    {VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, "LAZILY_ALLOCATED"},
};

void print_memory_property_bits(VkMemoryPropertyFlags flags)
{
    bool add_or = false;
    for(auto& bit : memory_property_bit_name_map) {
	if(flags & bit.first) {
	    printf("%s%s", add_or ? " | " : "", bit.second.c_str());
	    add_or = true;
	}
    }
}

void print_device_information(VkPhysicalDevice physical_device)
{
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device, &properties);

    printf("Physical Device Information\n");
    printf("    API     %d.%d.%d\n", properties.apiVersion >> 22, (properties.apiVersion >> 12) & 0x3ff, properties.apiVersion & 0xfff);
    printf("    driver  %X\n", properties.driverVersion);
    printf("    vendor  %X\n", properties.vendorID);
    printf("    device  %X\n", properties.deviceID);
    printf("    name    %s\n", properties.deviceName);
    printf("    type    %s\n", device_types[std::min(5, (int)properties.deviceType)]);

    uint32_t ext_count;

    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, nullptr);
    std::unique_ptr<VkExtensionProperties[]> exts(new VkExtensionProperties[ext_count]);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, exts.get());
    printf("    extensions:\n");
    for(uint32_t i = 0; i < ext_count; i++) {
	printf("        %s\n", exts[i].extensionName);
    }

    // VkPhysicalDeviceLimits              limits;
    // VkPhysicalDeviceSparseProperties    sparseProperties;
    //
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    std::unique_ptr<VkQueueFamilyProperties[]> queue_families(new VkQueueFamilyProperties[queue_family_count]);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.get());
    if(be_noisy) {
        for(uint32_t i = 0; i < queue_family_count; i++) {
            printf("queue %d:\n", i);
            printf("    flags:                       %04X\n", queue_families[i].queueFlags);
            printf("    queueCount:                  %d\n", queue_families[i].queueCount);
            printf("    timestampValidBits:          %d\n", queue_families[i].timestampValidBits);
            printf("    minImageTransferGranularity: (%d, %d, %d)\n",
                queue_families[i].minImageTransferGranularity.width,
                queue_families[i].minImageTransferGranularity.height,
                queue_families[i].minImageTransferGranularity.depth);
        }
    }

    for(uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        printf("memory type %d: flags ", i);
        print_memory_property_bits(memory_properties.memoryTypes[i].propertyFlags);
        printf("\n");
    }
}

void create_device(VkPhysicalDevice physical_device, VkDevice* device)
{
    std::vector<const char*> extensions;

    extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
#ifdef PLATFORM_MACOS
    extensions.push_back("VK_KHR_portability_subset" /* VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME */);
#endif
#if 0
    extensions.insert(extensions.end(), {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME
    });
#endif
    for(const auto &e: extensions) {
        printf("asked for %s\n", e);
    }

    VkDeviceQueueCreateInfo create_queues[1] = {};
    float queue_priorities[1] = {1.0f};
    create_queues[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    create_queues[0].pNext = nullptr;
    create_queues[0].flags = 0;
    create_queues[0].queueFamilyIndex = preferred_queue_family;
    create_queues[0].queueCount = 1;
    create_queues[0].pQueuePriorities = queue_priorities;

    VkDeviceCreateInfo create = {};

    create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create.pNext = nullptr;
    create.flags = 0;
    create.queueCreateInfoCount = 1;
    create.pQueueCreateInfos = create_queues;
    create.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    VK_CHECK(vkCreateDevice(physical_device, &create, nullptr, device));

    vkGetDeviceQueue(*device, preferred_queue_family, 0, &queue);

}

// Sascha Willem's 
uint32_t getMemoryTypeIndex(uint32_t type_bits, VkMemoryPropertyFlags properties)
{
    // Iterate over all memory types available for the device used in this example
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
	if (type_bits & (1 << i)) {
	    if ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
		return i;
            }
        }
    }

    throw "Could not find a suitable memory type!";
}

// Sascha Willem's 
VkCommandBuffer getCommandBuffer(bool begin)
{
    VkCommandBuffer cmdBuffer;

    VkCommandBufferAllocateInfo cmdBufAllocateInfo = {};
    cmdBufAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocateInfo.commandPool = command_pool;
    cmdBufAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocateInfo.commandBufferCount = 1;

    VK_CHECK(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &cmdBuffer));

    // If requested, also start the new command buffer
    if (begin) {
	VkCommandBufferBeginInfo cmdBufInfo = {};
	cmdBufInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	cmdBufInfo.flags = 0;
	VK_CHECK(vkBeginCommandBuffer(cmdBuffer, &cmdBufInfo));
    }

    return cmdBuffer;
}

// Sascha Willem's 
void flushCommandBuffer(VkCommandBuffer commandBuffer)
{
    assert(commandBuffer != VK_NULL_HANDLE);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo fenceCreateInfo = {};
    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceCreateInfo.flags = 0;
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

    // Submit to the queue
    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
    // Wait for the fence to signal that command buffer has finished executing
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, command_pool, 1, &commandBuffer);
}

// geometry data
static vertex vertices[3] = {
    {{0, 0, 0}, {1, 0, 0}},
    {{1, 0, 0}, {0, 1, 0}},
    {{0, 1, 0}, {0, 0, 1}},
};
static uint32_t indices[3] = {0, 1, 2}; 
int triangleCount = 1;

void create_vertex_buffers()
{
    // host-writable memory and buffers
    buffer vertex_staging;
    buffer index_staging;
    void *mapped; // when mapped, this points to the buffer

    // Tells us how much memory and which memory types (by bit) can hold this memory
    VkMemoryRequirements memory_req{};

    // Allocate memory
    VkMemoryAllocateInfo memory_alloc{};
    if(do_the_wrong_thing) {
        memory_alloc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    } else {
        memory_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    }
    memory_alloc.pNext = nullptr;

    // Create a buffer - buffers are used for things like vertex data
    // This one will be used as the source of a transfer to a GPU-addressable
    // buffer
    VkBufferCreateInfo create_staging_buffer{};
    create_staging_buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_staging_buffer.pNext = nullptr;
    create_staging_buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;


    // Create a buffer for vertices, allocate memory, map it, copy vertices to the memory, unmap, and then bind the vertex buffer to that memory.
    create_staging_buffer.size = sizeof(vertices);
    VK_CHECK(vkCreateBuffer(device, &create_staging_buffer, nullptr, &vertex_staging.buf));

    // Get the size and type requirements for memory containing this buffer
    vkGetBufferMemoryRequirements(device, vertex_staging.buf, &memory_req);

    memory_alloc.allocationSize = memory_req.size;
    // Find the type which this memory requires which is visible to the
    // CPU and also coherent, so when we unmap it it will be immediately
    // visible to the GPU
    memory_alloc.memoryTypeIndex = getMemoryTypeIndex(memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &vertex_staging.mem));

    // Map the memory, fill it, and unmap it
    VK_CHECK(vkMapMemory(device, vertex_staging.mem, 0, memory_alloc.allocationSize, 0, &mapped));
    memcpy(mapped, vertices, sizeof(vertices));
    vkUnmapMemory(device, vertex_staging.mem);

    // Tell Vulkan our buffer is in this memory at offset 0
    VK_CHECK(vkBindBufferMemory(device, vertex_staging.buf, vertex_staging.mem, 0));


    // Create a buffer for indices, allocate memory, map it, copy indices to the memory, unmap, and then bind the index buffer to that memory.
    create_staging_buffer.size = sizeof(indices);
    VK_CHECK(vkCreateBuffer(device, &create_staging_buffer, nullptr, &index_staging.buf));

    // Get the size and type requirements for memory containing this buffer
    vkGetBufferMemoryRequirements(device, index_staging.buf, &memory_req);

    memory_alloc.allocationSize = memory_req.size;
    // Find the type which this memory requires which is visible to the
    // CPU and also coherent, so when we unmap it it will be immediately
    // visible to the GPU
    memory_alloc.memoryTypeIndex = getMemoryTypeIndex(memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &index_staging.mem));

    // Map the memory, fill it, and unmap it
    VK_CHECK(vkMapMemory(device, index_staging.mem, 0, memory_alloc.allocationSize, 0, &mapped));
    memcpy(mapped, indices, sizeof(indices));
    vkUnmapMemory(device, index_staging.mem);

    // Tell Vulkan our buffer is in this memory at offset 0
    VK_CHECK(vkBindBufferMemory(device, index_staging.buf, index_staging.mem, 0));


    // This buffer will be used as the source of a transfer to a
    // GPU-addressable buffer
    VkBufferCreateInfo create_vertex_buffer = {};
    create_vertex_buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_vertex_buffer.pNext = nullptr;
    create_vertex_buffer.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // Create a buffer representing vertices on the GPU
    create_vertex_buffer.size = sizeof(vertices);
    VK_CHECK(vkCreateBuffer(device, &create_vertex_buffer, nullptr, &vertex_buffer.buf));

    // Get the size and type requirements for memory containing this buffer
    vkGetBufferMemoryRequirements(device, vertex_buffer.buf, &memory_req);

    // Create a new GPU accessible memory for vertices
    memory_alloc.allocationSize = memory_req.size;
    memory_alloc.memoryTypeIndex = getMemoryTypeIndex(memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &vertex_buffer.mem));
    VK_CHECK(vkBindBufferMemory(device, vertex_buffer.buf, vertex_buffer.mem, 0));

    // This buffer will be used as the source of a transfer to a
    // GPU-addressable buffer
    VkBufferCreateInfo create_index_buffer = {};
    create_index_buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_index_buffer.pNext = nullptr;
    create_index_buffer.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // Create a buffer representing indices on the GPU
    create_index_buffer.size = sizeof(indices);
    VK_CHECK(vkCreateBuffer(device, &create_index_buffer, nullptr, &index_buffer.buf));

    // Get the size and type requirements for memory containing this buffer
    vkGetBufferMemoryRequirements(device, index_buffer.buf, &memory_req);

    // Create a new GPU accessible memory for indices
    memory_alloc.allocationSize = memory_req.size;
    memory_alloc.memoryTypeIndex = getMemoryTypeIndex(memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &index_buffer.mem));
    VK_CHECK(vkBindBufferMemory(device, index_buffer.buf, index_buffer.mem, 0));

    // Copy from staging to the GPU-local buffers
    VkCommandBuffer commands = getCommandBuffer(true);
    VkBufferCopy copy = {};
    copy.size = sizeof(vertices);
    vkCmdCopyBuffer(commands, vertex_staging.buf, vertex_buffer.buf, 1, &copy);
    copy.size = sizeof(indices);
    vkCmdCopyBuffer(commands, index_staging.buf, index_buffer.buf, 1, &copy);
    flushCommandBuffer(commands);

    vkDestroyBuffer(device, vertex_staging.buf, nullptr);
    vkDestroyBuffer(device, index_staging.buf, nullptr);
    vkFreeMemory(device, vertex_staging.mem, nullptr);
    vkFreeMemory(device, index_staging.mem, nullptr);
}

void init_vulkan()
{
    print_implementation_information();
    create_instance(&instance);
    // get physical device surface support functions
    // get swapchain functions
    choose_physical_device(instance, &physical_device);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    std::unique_ptr<VkQueueFamilyProperties[]> queue_families(new VkQueueFamilyProperties[queue_family_count]);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.get());
    for(uint32_t i = 0; i < queue_family_count; i++) {
        if(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            preferred_queue_family = i;
        }
    }

    if(preferred_queue_family == NO_QUEUE_FAMILY) {
	std::cerr << "no desired queue family was found\n";
	exit(EXIT_FAILURE);
    }

    if(be_noisy) {
        print_device_information(physical_device);
    }
    create_device(physical_device, &device);
}

void cleanup_vulkan()
{
#if 0
    VkDestroyPipeline(device, pipeline, nullptr);
    VkDestroyPipelineLayout(device, pipeline_layout, nullptr);
    VkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);

    VkDestroyBuffer(device, vertex_buffer, nullptr);
    VkFreeMemory(device, vertex_memory, nullptr);

    VkDestroyBuffer(device, index_buffer, nullptr);
    VkFreeMemory(device, index_memory, nullptr);

    VkDestroyBuffer(device, vs_uniform_block_buffer, nullptr);
    VkFreeMemory(device, vs_uniform_block_memory, nullptr);

    VkDestroySemaphore(device, present_complete, nullptr);
    VkDestroySemaphore(device, render_complete, nullptr);

    for(auto& f : wait_fences)
	VkDestroyFence(device, f, nullptr);
#endif
}

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW: %s\n", description);
}

// From vkcube.cpp
static VkSurfaceFormatKHR pick_surface_format(const VkSurfaceFormatKHR *surfaceFormats, uint32_t count) {
    // Prefer non-SRGB formats...
    for (uint32_t i = 0; i < count; i++) {
        const VkFormat format = surfaceFormats[i].format;

        if (format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_B8G8R8A8_UNORM ||
            format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 ||
            format == VK_FORMAT_R16G16B16A16_SFLOAT) {
            return surfaceFormats[i];
        }
    }

    printf("Can't find our preferred formats... Falling back to first exposed format. Rendering may be incorrect.\n");

    assert(count >= 1);
    return surfaceFormats[0];
}

static std::vector<uint8_t> load_file(FILE *fp)
{
    long int start = ftell(fp);
    fseek(fp, 0, SEEK_END);
    long int end = ftell(fp);
    fseek(fp, start, SEEK_SET);

    std::vector<uint8_t> data(end - start);
    size_t result = fread(data.data(), 1, end - start, fp);
    assert(result == (end - start));

    return data;
}

std::vector<uint32_t> load_code(const std::string& filename) 
{
    std::vector<uint8_t> text = load_file(fopen(filename.c_str(), "rb"));
    std::vector<uint32_t> code((text.size() + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    memcpy(code.data(), text.data(), text.size());
    return code;
}

VkShaderModule create_shader_module(const std::vector<uint32_t>& code)
{
    VkShaderModule module;
    VkShaderModuleCreateInfo shader_create {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .codeSize = code.size() * sizeof(code[0]),
        .pCode = code.data(),
    };

    VK_CHECK(vkCreateShaderModule(device, &shader_create, NULL, &module));
    return module;
}


static void DrawFrame(GLFWwindow *window)
{
    VK_CHECK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, image_acquired_semaphores[swapchainIndex], VK_NULL_HANDLE, &swapchainIndex));

    auto cb = commandBuffers[swapchainIndex];

    VkCommandBufferBeginInfo begin {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0, // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    VK_CHECK(vkBeginCommandBuffer(cb, &begin));
    const VkClearValue clearValues [2] {
        {.color {.float32 {0.2f, 0.2f, 0.2f, 0.2f}}},
        {.depthStencil = {1.0f, 0}},
    };
    VkRenderPassBeginInfo beginRenderpass {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .pNext = nullptr,
        .renderPass = renderPass,
        .framebuffer = framebuffers[swapchainIndex],
        .renderArea = {{0, 0}, {static_cast<uint32_t>(windowWidth), static_cast<uint32_t>(windowHeight)}},
        .clearValueCount = static_cast<uint32_t>(std::size(clearValues)),
        .pClearValues = clearValues,
    };
    vkCmdBeginRenderPass(cb, &beginRenderpass, VK_SUBPASS_CONTENTS_INLINE);

    // 6. Bind the graphics pipeline state
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    // vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1,
    // descriptor_set[swapchainIndex], 0, NULL);

    // 8. Bind the texture resources - NA

    // 7. Bind the vertex and swapchainIndex buffers
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &vertex_buffer.buf, &offset);
    vkCmdBindIndexBuffer(cb, index_buffer.buf, 0, VK_INDEX_TYPE_UINT32);

    // 9. Set viewport and scissor parameters
    VkViewport viewport{};
    float viewport_dimension;
    if (windowWidth < windowHeight) {
        viewport_dimension = (float)windowWidth;
        viewport.y = (windowHeight - windowWidth) / 2.0f;
    } else {
        viewport_dimension = (float)windowHeight;
        viewport.x = (windowWidth - windowHeight) / 2.0f;
    }
    viewport.height = viewport_dimension;
    viewport.width = viewport_dimension;
    viewport.minDepth = (float)0.0f;
    viewport.maxDepth = (float)1.0f;
    vkCmdSetViewport(cb, 0, 1, &viewport);

    VkRect2D scissor {.offset{0, 0,}, .extent{static_cast<uint32_t>(windowWidth), static_cast<uint32_t>(windowHeight)}};
    vkCmdSetScissor(cb, 0, 1, &scissor);

    vkCmdDrawIndexed(cb, triangleCount * 3, 1, 0, 0, 0);
    vkCmdEndRenderPass(cb);
    VK_CHECK(vkEndCommandBuffer(cb));

    VkSubmitInfo submit {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &image_acquired_semaphores[swapchainIndex],
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
        .signalSemaphoreCount = 0, // 1,
        .pSignalSemaphores = nullptr, // &draw_completed_semaphores[swapchainIndex],
    };
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, draw_completed_fences[swapchainIndex]));
    VK_CHECK(vkWaitForFences(device, 1, &draw_completed_fences[swapchainIndex], VK_TRUE, DEFAULT_FENCE_TIMEOUT));
    VK_CHECK(vkResetFences(device, 1, &draw_completed_fences[swapchainIndex]));

    // 13. Present the rendered result
    uint32_t indices[] = {swapchainIndex};
    VkPresentInfoKHR present {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .waitSemaphoreCount = 0, // 1,
            .pWaitSemaphores = nullptr, // &draw_completed_semaphores[swapchainIndex],
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = indices,
            .pResults = nullptr,
    };
    VK_CHECK(vkQueuePresentKHR(queue, &present));
    swapchainIndex = (swapchainIndex + 1) % swapchainImageCount;

    printf("presented?!\n");
}

int main(int argc, char **argv)
{
    be_noisy = (getenv("BE_NOISY") != nullptr);
    enable_validation = (getenv("VALIDATE") != nullptr);
    do_the_wrong_thing = (getenv("BE_WRONG") != nullptr);

    glfwSetErrorCallback(error_callback);

    if(!glfwInit()) {
	std::cerr << "GLFW initialization failed.\n";
        exit(EXIT_FAILURE);
    }

    if (!glfwVulkanSupported()) {
	std::cerr << "GLFW reports Vulkan is not supported\n";
        exit(EXIT_FAILURE);
    }

#if 0

The pseudocode for initializing Vulkan and drawing an indexed, textured triangle mesh is as follows:

// Initialize Vulkan
1. Create a VkInstance using vkCreateInstance
2. Load all necessary instance-level extensions and validation layers
3. Enumerate physical devices and select one to use
4. Create a VkDevice for the chosen physical device
5. Load all necessary device-level extensions
6. Create a VkQueue for command submission
7. Create a VkCommandPool for allocating command buffers

#endif

    init_vulkan();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(512, 512, "vulkan test", nullptr, nullptr);

    VkResult err = glfwCreateWindowSurface(instance, window, nullptr, &surface);
    if (err) {
	std::cerr << "GLFW window creation failed " << err << "\n";
        exit(EXIT_FAILURE);
    }
	
    // PFN_vkCmdTraceRaysKHR cmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)vkGetInstanceProcAddr(instance, "vkCmdTraceRaysKHR");
    // assert(cmdTraceRaysKHR);

    printf("success creating device and window!\n");

    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    uint32_t formatCount;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, nullptr));
    std::unique_ptr<VkSurfaceFormatKHR[]> surfaceFormats(new VkSurfaceFormatKHR[formatCount]);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, surfaceFormats.get()));
    VkSurfaceFormatKHR surfaceFormat = pick_surface_format(surfaceFormats.get(), formatCount);
    VkFormat chosenFormat = surfaceFormat.format;
    VkColorSpaceKHR chosenColorSpace = surfaceFormat.colorSpace;

    VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    // TODO verify present mode with vkGetPhysicalDeviceSurfacePresentModesKHR

// 8. Create a VkSwapchain with desired parameters
    VkSwapchainCreateInfoKHR create {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .surface = surface,
        .minImageCount = swapchainImageCount,
        .imageFormat = chosenFormat,
        .imageColorSpace = chosenColorSpace,
        .imageExtent { static_cast<uint32_t>(windowWidth), static_cast<uint32_t>(windowHeight) },
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        .presentMode = swapchainPresentMode,
        .clipped = true,
        .oldSwapchain = VK_NULL_HANDLE, // oldSwapchain, // if we are recreating swapchain for when the window changes
    };
    VK_CHECK(vkCreateSwapchainKHR(device, &create, nullptr, &swapchain));

    printf("success creating swapchain!\n");

    VkCommandPoolCreateInfo create_command_pool{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = preferred_queue_family,
    };
    VK_CHECK(vkCreateCommandPool(device, &create_command_pool, nullptr, &command_pool));

    const VkCommandBufferAllocateInfo allocate{
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = nullptr,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    commandBuffers.resize(swapchainImageCount);
    for (uint32_t i = 0; i < swapchainImageCount; i++) {
        // Can't I just make commandBufferCount be swapchainImageCount here?
        VK_CHECK(vkAllocateCommandBuffers(device, &allocate, &commandBuffers[i]));
    }

#if 0
1. Create a VkRenderPass
2. Create a VkFramebuffer with the swapchain images
3. Create a VkPipelineLayout
4. Create a VkGraphicsPipeline
#endif

    uint32_t swapchainImageCountReturned;
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCountReturned, nullptr));
    assert(swapchainImageCountReturned == swapchainImageCount);
    swapchainImages.resize(swapchainImageCount);
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCount, swapchainImages.data()));

    VkFormat depthFormat = VK_FORMAT_D16_UNORM;
    VkImageCreateInfo createDepthInfo{
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = depthFormat,
        .extent{static_cast<uint32_t>(windowWidth), static_cast<uint32_t>(windowHeight), 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
    };
    VkImage depthImage;
    VK_CHECK(vkCreateImage(device, &createDepthInfo, nullptr, &depthImage));
    VkMemoryRequirements imageMemReqs;
    vkGetImageMemoryRequirements(device, depthImage, &imageMemReqs);
    uint32_t memoryTypeIndex = getMemoryTypeIndex(imageMemReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkDeviceMemory depthImageMemory;
    VkMemoryAllocateInfo depthAllocate{
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = imageMemReqs.size,
        .memoryTypeIndex = memoryTypeIndex,
    };
    VK_CHECK(vkAllocateMemory(device, &depthAllocate, nullptr, &depthImageMemory));
    VK_CHECK(vkBindImageMemory(device, depthImage, depthImageMemory, 0));
    // bind image to memory

    std::vector<VkImageView> colorImageViews(swapchainImageCount);
    for(uint32_t i = 0; i < swapchainImageCount; i++) {
        VkImageViewCreateInfo colorImageViewCreate{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .image = swapchainImages[i],
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = chosenFormat,
            .components{VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY},
            .subresourceRange{
                .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1
            },
        };
        VK_CHECK(vkCreateImageView(device, &colorImageViewCreate, nullptr, &colorImageViews[i]));
    }
    VkImageView depthImageView;
    VkImageViewCreateInfo depthImageViewCreate{
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .image = depthImage,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = depthFormat,
        .components{VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY},
        .subresourceRange{
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
    };
    VK_CHECK(vkCreateImageView(device, &depthImageViewCreate, nullptr, &depthImageView));

    VkAttachmentDescription attachments[2] = {
        {
            .flags = 0,
            .format = chosenFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        },
        {
            .flags = 0,
            .format = depthFormat,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        }
    };
    VkAttachmentReference colorReference{
        .attachment = 0,
        .layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    VkAttachmentReference depthReference{
        .attachment = 1,
        .layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    VkSubpassDescription subpass{
        .flags = 0,
        .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
        .inputAttachmentCount = 0,
        .pInputAttachments = nullptr,
        .colorAttachmentCount = 1,
        .pColorAttachments = &colorReference,
        .pResolveAttachments = nullptr,
        .pDepthStencilAttachment = &depthReference,
        .preserveAttachmentCount = 0,
        .pPreserveAttachments = nullptr,
    };
    VkSubpassDependency attachmentDependencies[2]{
        {
            // Image Layout Transition
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
            .dependencyFlags = 0,
        },
        {
            // Depth buffer is shared between swapchain images
            .srcSubpass = VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
            .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            .dependencyFlags = 0,
        },
    };
    VkRenderPassCreateInfo renderPassCreate{
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .attachmentCount = 2,
        .pAttachments = attachments,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 2,
        .pDependencies = attachmentDependencies,
    };
    VK_CHECK(vkCreateRenderPass(device, &renderPassCreate, nullptr, &renderPass));

    framebuffers.resize(swapchainImageCount);
    for(uint32_t i = 0; i < swapchainImageCount; i++) {
        VkImageView imageviews[2] = {colorImageViews[i], depthImageView};
        VkFramebufferCreateInfo framebufferCreate{
            .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .pNext = nullptr,
            .flags = 0,
            .renderPass = renderPass,
            .attachmentCount = 2,
            .pAttachments = imageviews,
            .width = static_cast<uint32_t>(windowWidth),
            .height = static_cast<uint32_t>(windowHeight),
            .layers = 1,
        };
        VK_CHECK(vkCreateFramebuffer(device, &framebufferCreate, nullptr, &framebuffers[i]));
    }

#if 0
9. Allocate memory for vertex and index buffers
10. Create a VkBuffer for each of them
#endif

    create_vertex_buffers();
    printf("success creating buffers!\n");

    // Create a graphics pipeline
    VkVertexInputBindingDescription vertex_input_binding {
        .binding = 0,
        .stride = sizeof(vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    VkVertexInputAttributeDescription vertex_position{
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(vertex, v),
    };
    VkVertexInputAttributeDescription vertex_color{
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(vertex, c),
    };
    std::vector<VkVertexInputAttributeDescription> vertex_input_attributes{vertex_position, vertex_color};

    VkPipelineVertexInputStateCreateInfo vertex_input_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_input_binding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_input_attributes.size()),
        .pVertexAttributeDescriptions = vertex_input_attributes.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .pNext = nullptr,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkPipelineRasterizationStateCreateInfo rasterization_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .pNext = nullptr,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };

    VkPipelineColorBlendAttachmentState att_state[1];
    memset(att_state, 0, sizeof(att_state));
    att_state[0].colorWriteMask = 0xf;
    att_state[0].blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo color_blend_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .attachmentCount = 1,
        .pAttachments = att_state,
    };

    VkStencilOpState keep_always{ .failOp = VK_STENCIL_OP_KEEP, .passOp = VK_STENCIL_OP_KEEP, .compareOp = VK_COMPARE_OP_ALWAYS };
    VkPipelineDepthStencilStateCreateInfo depth_stencil_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
        .pNext = nullptr,
        .depthTestEnable = VK_TRUE,
        .depthWriteEnable = VK_TRUE,
        .depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
        .depthBoundsTestEnable = VK_FALSE,
        .stencilTestEnable = VK_FALSE,
        .front = keep_always,
        .back = keep_always,
    };

    VkPipelineViewportStateCreateInfo viewport_state { 
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .scissorCount = 1,
    };

    VkPipelineMultisampleStateCreateInfo multisample_state{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .pNext = nullptr,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .pSampleMask = NULL,
    };

    VkDynamicState dynamicStateEnables[]{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    VkPipelineDynamicStateCreateInfo dynamicState{
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .pNext = nullptr,
        .dynamicStateCount = static_cast<uint32_t>(std::size(dynamicStateEnables)),
        .pDynamicStates = dynamicStateEnables,
    };

    std::vector<uint32_t> vertex_shader_code = load_code("testing.vert");
    printf("vert spir-v: %zd words (%x %x %x ... )\n", vertex_shader_code.size(), vertex_shader_code[0], vertex_shader_code[1], vertex_shader_code[2]);
    VkShaderModule vertex_shader_module = create_shader_module(vertex_shader_code);

    std::vector<uint32_t> fragment_shader_code = load_code("testing.frag");
    printf("frag spir-v: %zd words (%x %x %x ... )\n", fragment_shader_code.size(), fragment_shader_code[0], fragment_shader_code[1], fragment_shader_code[2]);
    VkShaderModule fragment_shader_module = create_shader_module(fragment_shader_code);

    printf("vert module = %p\n", vertex_shader_module);
    printf("frag module = %p\n", fragment_shader_module);

    VkPipelineShaderStageCreateInfo vertexShaderCreate {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_VERTEX_BIT,
        .module = vertex_shader_module,
        .pName = "main",
    };
    VkPipelineShaderStageCreateInfo fragmentShaderCreate {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = nullptr,
        .stage = VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = fragment_shader_module,
        .pName = "main",
    };
    std::vector<VkPipelineShaderStageCreateInfo> shaderStages{vertexShaderCreate, fragmentShaderCreate};

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VkPipelineLayoutCreateInfo create_layout {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
    };
    VK_CHECK(vkCreatePipelineLayout(device, &create_layout, nullptr, &pipeline_layout));

    VkGraphicsPipelineCreateInfo create_pipeline {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .pNext = nullptr,
        .stageCount = static_cast<uint32_t>(shaderStages.size()),
        .pStages = shaderStages.data(),
        .pVertexInputState = &vertex_input_state,
        .pInputAssemblyState = &input_assembly_state,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization_state,
        .pMultisampleState = &multisample_state,
        .pDepthStencilState = &depth_stencil_state,
        .pColorBlendState = &color_blend_state,
        .pDynamicState = &dynamicState,
        .layout = pipeline_layout,
        .renderPass = renderPass,
    };

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &create_pipeline, nullptr, &pipeline));

    // XXX create a timeline semaphore by chaining after a
    // VkSemaphoreTypeCreateInfo with VkSemaphoreTypeCreateInfo =
    // VK_SEMAPHORE_TYPE_TIMELINE
    VkSemaphoreCreateInfo sema_create = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
    };
    image_acquired_semaphores.resize(swapchainImageCount);
    for(uint32_t i = 0; i < swapchainImageCount; i++) {
        VK_CHECK(vkCreateSemaphore(device, &sema_create, NULL, &image_acquired_semaphores[i]));
    }
    draw_completed_semaphores.resize(swapchainImageCount);
    for(uint32_t i = 0; i < swapchainImageCount; i++) {
        VK_CHECK(vkCreateSemaphore(device, &sema_create, NULL, &draw_completed_semaphores[i]));
    }
    VkFenceCreateInfo fence_create = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = NULL,
        .flags = 0,
    };
    draw_completed_fences.resize(swapchainImageCount);
    for(uint32_t i = 0; i < swapchainImageCount; i++) {
        VK_CHECK(vkCreateFence(device, &fence_create, nullptr, &draw_completed_fences[i]));
    }

    // glfwSetKeyCallback(window, KeyCallback);
    // glfwSetMouseButtonCallback(window, ButtonCallback);
    // glfwSetCursorPosCallback(window, MotionCallback);
    // glfwSetScrollCallback(window, ScrollCallback);
    // glfwSetFramebufferSizeCallback(window, ResizeCallback);
    glfwSetWindowRefreshCallback(window, DrawFrame);

    while (!glfwWindowShouldClose(window)) {

        DrawFrame(window);

        // if(gStreamFrames)
            glfwPollEvents();
        // else
        // glfwWaitEvents();
    }

    cleanup_vulkan();

    glfwTerminate();
}

#if 0
THis is from vkcube, functions piped through uniq
vkCreateInstance                                create instance
vkEnumeratePhysicalDevices                      what physical devices exist
vkGetPhysicalDeviceProperties                   using this and the next 2 funcs, choose a physical device that meets our needs
vkGetPhysicalDeviceQueueFamilyProperties        
vkGetPhysicalDeviceFeatures
vkCreateWin32SurfaceKHR                         create a render target
vkGetPhysicalDeviceSurfaceSupportKHR            repeat until we have the one we want that also meets other needs?
vkCreateDevice                                  then create a device 
vkGetPhysicalDeviceImageFormatProperties2       get image format props
vkGetPhysicalDeviceQueueFamilyProperties        get device queue props, repeat until find one suitable
vkGetDeviceQueue                                then get a queue for the device?
vkGetPhysicalDeviceSurfaceFormatsKHR            get surface formats for the device to use to make swapchain
vkCreateFence                                   create fence
vkCreateSemaphore                               create sema
vkCreateFence                                   create fence
vkCreateSemaphore                               create sema
vkGetPhysicalDeviceMemoryProperties             get properties for memory types and heaps
vkGetPhysicalDeviceSurfaceCapabilitiesKHR       get what the surface is capable of
vkGetPhysicalDeviceSurfacePresentModesKHR       get surface presentation modes e.g. immediate vs FIFO
vkCreateSwapchainKHR                            create swapchain
vkGetSwapchainImagesKHR                         get images backing the swapchain
vkCreateImageView                               create ImageView that will be used to...?
vkCreateCommandPool                             create command pool - command buffers are allocated from here?
vkAllocateCommandBuffers                        allocate command buffers that are filled with commands
vkBeginCommandBuffer                            open a command buffer
vkCreateImage                                   create an image, no memory backs it yet
vkGetImageMemoryRequirements                    get size of memory required for image
vkAllocateMemory                                allocate it
vkBindImageMemory                               bind image and memory together
vkCreateImageView                               Create image view
vkGetPhysicalDeviceFormatProperties             get info about the format on the device, like optimal tiling
vkCreateImage                                   create image
vkGetImageMemoryRequirements                    get size required to hold image
vkAllocateMemory                                allocate some memory
vkBindImageMemory                               bind that memory to the image
vkGetImageSubresourceLayout                     get e.g. the offset, row pitch, etc for a MIP slice - not an object,
                                                this describes how the bytes for the subresource are laid out in memory
vkMapMemory                                     map VkMemory into memory
vkUnmapMemory                                   unmap (was presumably filled in interim)
vkCmdPipelineBarrier                            Insert a barrier to e.g. transition memory?
vkCreateSampler                                 create image sampler
vkCreateImageView                               create image view
vkCreateBuffer                                  create a buffer... but for?
vkGetBufferMemoryRequirements                   find out how big
vkAllocateMemory                                allocate
vkMapMemory                                     map (and fill)
vkBindBufferMemory                              bind buffer to memory
vkCreateBuffer                                  create buffer
vkGetBufferMemoryRequirements                   get size, alignment and memory type
vkAllocateMemory                                allocate memory using those params
vkMapMemory                                     map memory
vkBindBufferMemory                              bind buffer to that memory
vkCreateBuffer                                  create buffer
vkGetBufferMemoryRequirements                   get size, alignment and memory type
vkAllocateMemory                                allocate memory using those params
vkMapMemory                                     map memory
vkBindBufferMemory                              bind buffer to that memory, probably
vkCreateDescriptorSetLayout                     a descriptor describes a piece of data that can be used by a shader
vkCreatePipelineLayout                          a pipeline is based around the set of layouts to be used
vkCreateRenderPass                              describes a series of framebuffer attachments, subpasses, and dependencies between subpasses (a la ordering to take advantage of ARM framebuffer storage)
vkCreateShaderModule                            create a shader module from SPIR-V - compiled from Vulkan variant of GLSL - is vkcube's precompiled?
vkCreatePipelineCache                           create a cache object for pipelines - vkcube populates its cache with nullptr
vkCreateGraphicsPipelines                       create graphics pipelines - vkcube makes 1.  Contains all shaders for all stages; createinfos for the fixed-function parts of the pipeline like viewport, depth ops, multisample config; etc
vkDestroyShaderModule                           "A shader module can be destroyed while pipelines created using its shaders are still in use."
vkAllocateCommandBuffers                        allocate VkCommandBuffers from a command pool - vkcube makes 1 for each swapchain image
vkCreateDescriptorPool                          create backing store for descriptor sets?
vkAllocateDescriptorSets                        allocate descriptor sets within a pool - vkcube allocates 3; 1 for each swapchain
vkUpdateDescriptorSets                          fill descriptor sets with data
vkAllocateDescriptorSets                        #2
vkUpdateDescriptorSets                          #2
vkAllocateDescriptorSets                        #3
vkUpdateDescriptorSets                          #3
vkCreateFramebuffer                             create a framebuffer object which binds together ImageViews
vkBeginCommandBuffer                            begin a command buffer
vkCmdBeginRenderPass                            begin a "render pass"
vkCmdBindPipeline                               bind a pipeline (e.g. from vkCreateGraphicsPipelines)
vkCmdBindDescriptorSets                         bind descriptor sets (filled/updated with vkUpdateDescriptorSets)
vkCmdSetViewport                                set drawing viewport
vkCmdSetScissor                                 set drawing scissor
vkCmdDraw                                       draw some primitives
vkCmdEndRenderPass                              end a "render pass"
vkEndCommandBuffer                              end that command buffer
vkCreateFence                                   create a fence
vkQueueSubmit                                   submit all the stuff created in a command buffer including signaling the fence just created
vkWaitForFences                                 wait on those fences
vkFreeCommandBuffers                            free command buffers
vkDestroyFence                                  destroy fence we just created
vkWaitForFences                                 wait on frame fences but none has been issued at this point
vkResetFences                                   reset those fences
vkAcquireNextImageKHR                           acquire the index of the next image that is available
vkQueueSubmit                                   submit command buffers to queue
vkQueuePresentKHR                               queue a present of an image
vkWaitForFences                                 wait for frame fences
#endif
