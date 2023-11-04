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

	VkApplicationInfo app_info = {};
	app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app_info.pApplicationName = "triangle";
	app_info.pEngineName = "triangle";
	app_info.apiVersion = VK_API_VERSION_1_2;

	VkInstanceCreateInfo create = {};
	create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	create.pNext = NULL;
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


    VkDeviceQueueCreateInfo create_queues[1] = {};
    float queue_priorities[1] = {1.0f};
    create_queues[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    create_queues[0].pNext = NULL;
    create_queues[0].flags = 0;
    create_queues[0].queueFamilyIndex = preferred_queue_family;
    create_queues[0].queueCount = 1;
    create_queues[0].pQueuePriorities = queue_priorities;

    VkDeviceCreateInfo create = {};

    create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create.pNext = NULL;
    create.flags = 0;
    create.queueCreateInfoCount = 1;
    create.pQueueCreateInfos = create_queues;
    create.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    create.ppEnabledExtensionNames = extensions.data();
    VK_CHECK(vkCreateDevice(physical_device, &create, nullptr, device));

    vkGetDeviceQueue(*device, preferred_queue_family, 0, &queue);

    VkCommandPoolCreateInfo create_command_pool = {};
    create_command_pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    create_command_pool.flags = 0;
    create_command_pool.queueFamilyIndex = preferred_queue_family;
    VK_CHECK(vkCreateCommandPool(*device, &create_command_pool, nullptr, &command_pool));
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


void create_vertex_buffers()
{
    // geometry data
    static vertex vertices[3] = {
	{{0, 0, 0}, {1, 0, 0}},
	{{1, 0, 0}, {0, 1, 0}},
	{{0, 1, 0}, {0, 0, 1}},
    };
    static uint32_t indices[3] = {0, 1, 2}; 

    // host-writable memory and buffers
    buffer vertex_staging;
    buffer index_staging;
    void *mapped; // when mapped, this points to the buffer

    // Tells us how much memory and which memory types (by bit) can hold this memory
    VkMemoryRequirements memory_req = {};

    // Allocate memory
    VkMemoryAllocateInfo memory_alloc = {};
    if(do_the_wrong_thing) {
        memory_alloc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    } else {
        memory_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    }
    memory_alloc.pNext = nullptr;

    // Create a buffer - buffers are used for things like vertex data
    // This one will be used as the source of a transfer to a GPU-addressable
    // buffer
    VkBufferCreateInfo create_staging_buffer = {};
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

int main(int argc, char **argv)
{
    be_noisy = (getenv("BE_NOISY") != NULL);
    enable_validation = (getenv("VALIDATE") != NULL);
    do_the_wrong_thing = (getenv("BE_WRONG") != NULL);

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
    GLFWwindow* window = glfwCreateWindow(512, 512, "vulkan test", NULL, NULL);

    VkResult err = glfwCreateWindowSurface(instance, window, NULL, &surface);
    if (err) {
	std::cerr << "GLFW window creation failed " << err << "\n";
        exit(EXIT_FAILURE);
    }
	
    // PFN_vkCmdTraceRaysKHR cmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)vkGetInstanceProcAddr(instance, "vkCmdTraceRaysKHR");
    // assert(cmdTraceRaysKHR);

    printf("success creating device and window!\n");

    int window_width, window_height;
    glfwGetWindowSize(window, &window_width, &window_height);

    uint32_t formatCount;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, NULL));
    std::unique_ptr<VkSurfaceFormatKHR[]> surfaceFormats(new VkSurfaceFormatKHR[formatCount]);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, surfaceFormats.get()));
    VkSurfaceFormatKHR surfaceFormat = pick_surface_format(surfaceFormats.get(), formatCount);
    VkFormat chosenFormat = surfaceFormat.format;
    VkColorSpaceKHR chosenColorSpace = surfaceFormat.colorSpace;

    VkPresentModeKHR swapchainPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    // TODO verify present mode with vkGetPhysicalDeviceSurfacePresentModesKHR

// 8. Create a VkSwapchain with desired parameters
    uint32_t swapchainImageCount = 3;
    VkSwapchainCreateInfoKHR create = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .pNext = nullptr,
        .surface = surface,
        .minImageCount = swapchainImageCount,
        .imageFormat = chosenFormat,
        .imageColorSpace = chosenColorSpace,
        .imageExtent = { static_cast<uint32_t>(window_width), static_cast<uint32_t>(window_height) },
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
        .imageArrayLayers = 1,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = NULL,
        .presentMode = swapchainPresentMode,
        .oldSwapchain = VK_NULL_HANDLE, // oldSwapchain, // if we are recreating swapchain for when the window changes
        .clipped = true,
    };
    VkSwapchainKHR swapchain;
    VK_CHECK(vkCreateSwapchainKHR(device, &create, nullptr, &swapchain));

    printf("success creating swapchain!\n");

#if 0
11. Create a VkImage for the texture
12. Allocate device memory for it
13. Create a VkImageView for the texture image
14. Create a VkSampler for sampling the texture
15. Upload data to the allocated memory

// Draw an indexed, textured triangle mesh
1. Create a VkRenderPass
2. Create a VkFramebuffer with the swapchain images
3. Create a VkPipelineLayout
4. Create a VkGraphicsPipeline
#endif
#if 0
9. Allocate memory for vertex and index buffers
10. Create a VkBuffer for each of them
#endif

    create_vertex_buffers();
    printf("success creating buffers!\n");

    uint32_t index = 0;

    // while(1)
    {
#if 0

5. Begin a VkCommandBuffer
6. Bind the graphics pipeline state
7. Bind the vertex and index buffers
8. Bind the texture resources
9. Set viewport and scissor parameters
10. Issue draw commands
11. End the command buffer
12. Submit the command buffer

#endif 

        uint32_t indices[] = {index};
        index = (index + 1) % swapchainImageCount;
        VkPresentInfoKHR present {
            .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = nullptr,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = indices,
            .pResults = nullptr,
        };
// 13. Present the rendered result
        VK_CHECK(vkQueuePresentKHR(queue, &present));

    }

    cleanup_vulkan();
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
