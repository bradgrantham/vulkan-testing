#define _CRT_SECURE_NO_WARNINGS

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

#include "vectormath.h"

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

static constexpr uint64_t DEFAULT_FENCE_TIMEOUT = 100000000000;

#define STR(f) #f

std::map<VkResult, std::string> MapVkResultToName =
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
	if(MapVkResultToName.count(f) > 0) { \
	    std::cerr << "VkResult from " STR(f) " was " << MapVkResultToName[result] << " at line " << __LINE__ << "\n"; \
	} else { \
	    std::cerr << "VkResult from " STR(f) " was " << result << " at line " << __LINE__ << "\n"; \
        } \
	exit(EXIT_FAILURE); \
    } \
}

// From vkcube.cpp
static VkSurfaceFormatKHR PickSurfaceFormat(const VkSurfaceFormatKHR *surfaceFormats, uint32_t count)
{
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

// Sascha Willem's 
VkCommandBuffer getCommandBuffer(VkDevice device, VkCommandPool command_pool, bool begin)
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
void flushCommandBuffer(VkDevice device, VkQueue queue, VkCommandPool command_pool, VkCommandBuffer commandBuffer)
{
    assert(commandBuffer != VK_NULL_HANDLE);

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkSubmitInfo submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo fenceCreateInfo = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
    };
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

    // Submit to the queue
    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
    // Wait for the fence to signal that command buffer has finished executing
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));

    vkDestroyFence(device, fence, nullptr);
    vkFreeCommandBuffers(device, command_pool, 1, &commandBuffer);
}

// Sascha Willem's 
uint32_t getMemoryTypeIndex(VkPhysicalDeviceMemoryProperties memory_properties, uint32_t type_bits, VkMemoryPropertyFlags properties)
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

// Styled somewhat after Sascha Willem's triangle

static constexpr uint32_t NO_QUEUE_FAMILY = 0xffffffff;
static constexpr int MAX_IN_FLIGHT = 2;

struct Vertex
{
    float v[3];
    float n[3];
    float c[4];
    float t[2];
    Vertex(float v_[3], float n_[3], float c_[4], float t_[2])
    {
        std::copy(v_, v_ + 3, v);
        std::copy(n_, n_ + 3, n);
        std::copy(c_, c_ + 4, c);
        std::copy(t_, t_ + 2, t);
    }
    Vertex() {}
    // Vertex(const Vertex &v_)
    // {
        // std::copy(v_.v, v_.v + 3, v);
        // std::copy(v_.n, v_.n + 3, n);
        // std::copy(v_.c, v_.c + 4, c);
        // std::copy(v_.t, v_.t + 2, t);
    // }
};

struct Buffer
{
    VkDeviceMemory mem;
    VkBuffer buf;
    void* mapped;
};

mat4f modelview;
mat4f projection;

struct VertexUniforms
{
    float modelview[16];
    float modelview_normal[16];
    float projection[16];
};

void CreateInstance(VkInstance* instance, bool enableValidation)
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

    if(enableValidation) {
	layer_set.insert("VK_LAYER_KHRONOS_validation");
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

void ChoosePhysicalDevice(VkInstance instance, VkPhysicalDevice* physical_device, bool beVerbose)
{
    uint32_t gpu_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr));
    if(beVerbose) {
        std::cerr << gpu_count << " gpus enumerated\n";
    }
    VkPhysicalDevice physical_devices[32];
    gpu_count = std::min(32u, gpu_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, physical_devices));
    *physical_device = physical_devices[0];
}

const char* DeviceTypeDescriptions[] = {
    "other",
    "integrated GPU",
    "discrete GPU",
    "virtual GPU",
    "CPU",
    "unknown",
};

std::map<uint32_t, std::string> MemoryPropertyBitToNameMap = {
    {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL"},
    {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE"},
    {VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT"},
    {VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED"},
    {VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, "LAZILY_ALLOCATED"},
};

void PrintMemoryPropertyBits(VkMemoryPropertyFlags flags)
{
    bool add_or = false;
    for(auto& bit : MemoryPropertyBitToNameMap) {
	if(flags & bit.first) {
	    printf("%s%s", add_or ? " | " : "", bit.second.c_str());
	    add_or = true;
	}
    }
}

void PrintDeviceInformation(VkPhysicalDevice physical_device, VkPhysicalDeviceMemoryProperties &memory_properties)
{
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical_device, &properties);

    printf("Physical Device Information\n");
    printf("    API     %d.%d.%d\n", properties.apiVersion >> 22, (properties.apiVersion >> 12) & 0x3ff, properties.apiVersion & 0xfff);
    printf("    driver  %X\n", properties.driverVersion);
    printf("    vendor  %X\n", properties.vendorID);
    printf("    device  %X\n", properties.deviceID);
    printf("    name    %s\n", properties.deviceName);
    printf("    type    %s\n", DeviceTypeDescriptions[std::min(5, (int)properties.deviceType)]);

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

    for(uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        printf("memory type %d: flags ", i);
        PrintMemoryPropertyBits(memory_properties.memoryTypes[i].propertyFlags);
        printf("\n");
    }
}

static std::vector<uint8_t> GetFileContents(FILE *fp)
{
    long int start = ftell(fp);
    fseek(fp, 0, SEEK_END);
    long int end = ftell(fp);
    fseek(fp, start, SEEK_SET);

    std::vector<uint8_t> data(end - start);
    size_t result = fread(data.data(), 1, end - start, fp);
    assert(result == static_cast<size_t>(end - start));

    return data;
}

std::vector<uint32_t> GetFileAsCode(const std::string& filename) 
{
    std::vector<uint8_t> text = GetFileContents(fopen(filename.c_str(), "rb"));
    std::vector<uint32_t> code((text.size() + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    memcpy(code.data(), text.data(), text.size());
    return code;
}

VkShaderModule CreateShaderModule(VkDevice device, const std::vector<uint32_t>& code)
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

VkViewport CalculateViewport(uint32_t windowWidth, uint32_t windowHeight) 
{
    float viewport_dimension;
    float viewport_x = 0.0f;
    float viewport_y = 0.0f;
    if (windowWidth < windowHeight) {
        viewport_dimension = static_cast<float>(windowWidth);
        viewport_y = (windowHeight - windowWidth) / 2.0f;
    } else {
        viewport_dimension = static_cast<float>(windowHeight);
        viewport_x = (windowWidth - windowHeight) / 2.0f;
    }
    VkViewport viewport {
        .x = viewport_x,
        .y = viewport_y,
        .width = viewport_dimension,
        .height = viewport_dimension,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    return viewport;
}

void CreateDevice(VkPhysicalDevice physical_device, const std::vector<const char*>& extensions, uint32_t preferred_queue_family, VkDevice* device, VkQueue* queue)
{
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

    vkGetDeviceQueue(*device, preferred_queue_family, 0, queue);
}

void PrintImplementationInformation()
{
    uint32_t ext_count;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::unique_ptr<VkExtensionProperties[]> exts(new VkExtensionProperties[ext_count]);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.get());
    printf("Vulkan instance extensions:\n");
    for (uint32_t i = 0; i < ext_count; i++) {
        printf("\t%s, %08X\n", exts[i].extensionName, exts[i].specVersion);
    }
}

namespace VulkanApp
{

bool beVerbose = true;
bool enableValidation = false;
bool do_the_wrong_thing = false;

VkInstance instance;
VkPhysicalDevice physical_device;
VkDevice device;
VkPhysicalDeviceMemoryProperties memory_properties;
uint32_t preferred_queue_family = NO_QUEUE_FAMILY;
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
std::vector<VkDescriptorSet> descriptor_sets;
std::vector<VkFramebuffer> framebuffers;
std::vector<Buffer> uniform_buffers;
int draw_submission_index = 0;
VkPipelineLayout pipeline_layout;
VkDescriptorPool descriptor_pool;
VkRenderPass renderPass;
VkPipeline pipeline;
VkDescriptorSetLayout descriptor_set_layout;

Buffer vertex_buffer;
Buffer index_buffer;

template <typename T>
size_t ByteCount(const std::vector<T>& v) { return sizeof(T) * v.size(); }

// geometry data
std::vector<Vertex> vertices;
std::vector<uint32_t> indices;
int triangleCount;

void CreateGeometryBuffers()
{
    // host-writable memory and buffers
    Buffer vertex_staging;
    Buffer index_staging;
    void *mapped; // when mapped, this points to the buffer

    // Tells us how much memory and which memory types (by bit) can hold this memory
    VkMemoryRequirements memory_req{};


    // Create a buffer - buffers are used for things like vertex data
    // This one will be used as the source of a transfer to a GPU-addressable
    // buffer
    VkBufferCreateInfo create_staging_buffer{};
    create_staging_buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_staging_buffer.pNext = nullptr;
    create_staging_buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    // Create a buffer for vertices, allocate memory, map it, copy vertices to the memory, unmap, and then bind the vertex buffer to that memory.
    create_staging_buffer.size = ByteCount(vertices);
    VK_CHECK(vkCreateBuffer(device, &create_staging_buffer, nullptr, &vertex_staging.buf));

    // Get the size and type requirements for memory containing this buffer
    vkGetBufferMemoryRequirements(device, vertex_staging.buf, &memory_req);

    // Find the type which this memory requires which is visible to the
    // CPU and also coherent, so when we unmap it it will be immediately
    // visible to the GPU
    uint32_t memoryTypeIndex = getMemoryTypeIndex(memory_properties, memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // Allocate memory
    VkMemoryAllocateInfo memory_alloc {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = nullptr,
        .allocationSize = memory_req.size,
        .memoryTypeIndex = memoryTypeIndex,
    };
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &vertex_staging.mem));

    // Map the memory, fill it, and unmap it
    VK_CHECK(vkMapMemory(device, vertex_staging.mem, 0, memory_alloc.allocationSize, 0, &mapped));
    memcpy(mapped, vertices.data(), ByteCount(vertices));
    vkUnmapMemory(device, vertex_staging.mem);

    // Tell Vulkan our buffer is in this memory at offset 0
    VK_CHECK(vkBindBufferMemory(device, vertex_staging.buf, vertex_staging.mem, 0));


    // Create a buffer for indices, allocate memory, map it, copy indices to the memory, unmap, and then bind the index buffer to that memory.
    create_staging_buffer.size = ByteCount(indices);
    VK_CHECK(vkCreateBuffer(device, &create_staging_buffer, nullptr, &index_staging.buf));

    // Get the size and type requirements for memory containing this buffer
    vkGetBufferMemoryRequirements(device, index_staging.buf, &memory_req);

    memory_alloc.allocationSize = memory_req.size;
    // Find the type which this memory requires which is visible to the
    // CPU and also coherent, so when we unmap it it will be immediately
    // visible to the GPU
    memory_alloc.memoryTypeIndex = getMemoryTypeIndex(memory_properties, memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &index_staging.mem));

    // Map the memory, fill it, and unmap it
    VK_CHECK(vkMapMemory(device, index_staging.mem, 0, memory_alloc.allocationSize, 0, &mapped));
    memcpy(mapped, indices.data(), ByteCount(indices));
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
    create_vertex_buffer.size = ByteCount(vertices);
    VK_CHECK(vkCreateBuffer(device, &create_vertex_buffer, nullptr, &vertex_buffer.buf));

    // Get the size and type requirements for memory containing this buffer
    vkGetBufferMemoryRequirements(device, vertex_buffer.buf, &memory_req);

    // Create a new GPU accessible memory for vertices
    memory_alloc.allocationSize = memory_req.size;
    memory_alloc.memoryTypeIndex = getMemoryTypeIndex(memory_properties, memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &vertex_buffer.mem));
    VK_CHECK(vkBindBufferMemory(device, vertex_buffer.buf, vertex_buffer.mem, 0));

    // This buffer will be used as the source of a transfer to a
    // GPU-addressable buffer
    VkBufferCreateInfo create_index_buffer = {};
    create_index_buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_index_buffer.pNext = nullptr;
    create_index_buffer.usage = VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

    // Create a buffer representing indices on the GPU
    create_index_buffer.size = ByteCount(indices);
    VK_CHECK(vkCreateBuffer(device, &create_index_buffer, nullptr, &index_buffer.buf));

    // Get the size and type requirements for memory containing this buffer
    vkGetBufferMemoryRequirements(device, index_buffer.buf, &memory_req);

    // Create a new GPU accessible memory for indices
    memory_alloc.allocationSize = memory_req.size;
    memory_alloc.memoryTypeIndex = getMemoryTypeIndex(memory_properties, memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &index_buffer.mem));
    VK_CHECK(vkBindBufferMemory(device, index_buffer.buf, index_buffer.mem, 0));

    // Copy from staging to the GPU-local buffers
    VkCommandBuffer commands = getCommandBuffer(device, command_pool, true);
    VkBufferCopy copy = {};
    copy.size = ByteCount(vertices);
    vkCmdCopyBuffer(commands, vertex_staging.buf, vertex_buffer.buf, 1, &copy);
    copy.size = ByteCount(indices);
    vkCmdCopyBuffer(commands, index_staging.buf, index_buffer.buf, 1, &copy);
    flushCommandBuffer(device, queue, command_pool, commands);

    vkDestroyBuffer(device, vertex_staging.buf, nullptr);
    vkDestroyBuffer(device, index_staging.buf, nullptr);
    vkFreeMemory(device, vertex_staging.mem, nullptr);
    vkFreeMemory(device, index_staging.mem, nullptr);
}

void InitializeInstance()
{
    if (beVerbose) {
        PrintImplementationInformation();
    }
    CreateInstance(&instance, enableValidation);
}

void InitializeState(int windowWidth, int windowHeight)
{
    ChoosePhysicalDevice(instance, &physical_device, beVerbose);
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());
    for(uint32_t i = 0; i < queue_family_count; i++) {
        if(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            preferred_queue_family = i;
            break;
        }
    }

    if(preferred_queue_family == NO_QUEUE_FAMILY) {
	std::cerr << "no desired queue family was found\n";
	exit(EXIT_FAILURE);
    }

    if(beVerbose) {
        PrintDeviceInformation(physical_device, memory_properties);
    }

    std::vector<const char*> deviceExtensions;

    deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

#ifdef PLATFORM_MACOS
    deviceExtensions.push_back("VK_KHR_portability_subset" /* VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME */);
#endif

#if 0
    deviceExtensions.insert(extensions.end(), {
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME
        });
#endif

    if (beVerbose) {
        for (const auto& e : deviceExtensions) {
            printf("asked for %s\n", e);
        }
    }

    CreateDevice(physical_device, deviceExtensions, preferred_queue_family, &device, &queue);

    uint32_t formatCount;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, nullptr));
    std::unique_ptr<VkSurfaceFormatKHR[]> surfaceFormats(new VkSurfaceFormatKHR[formatCount]);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, surfaceFormats.get()));
    VkSurfaceFormatKHR surfaceFormat = PickSurfaceFormat(surfaceFormats.get(), formatCount);
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
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = swapchainPresentMode,
        .clipped = true,
        .oldSwapchain = VK_NULL_HANDLE, // oldSwapchain, // if we are recreating swapchain for when the window changes
    };
    VK_CHECK(vkCreateSwapchainKHR(device, &create, nullptr, &swapchain));

    uint32_t swapchainImageCountReturned;
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCountReturned, nullptr));
    assert(swapchainImageCountReturned == swapchainImageCount);
    swapchainImages.resize(swapchainImageCount);
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &swapchainImageCount, swapchainImages.data()));

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
        .commandBufferCount = swapchainImageCount,
    };
    commandBuffers.resize(swapchainImageCount);
    VK_CHECK(vkAllocateCommandBuffers(device, &allocate, commandBuffers.data()));

    VkFormat depthFormat = VK_FORMAT_D32_SFLOAT_S8_UINT;
    VkImageCreateInfo createDepthInfo {
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
    uint32_t memoryTypeIndex = getMemoryTypeIndex(memory_properties, imageMemReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
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
            .aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT,
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
            .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
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
    VkSubpassDescription subpass {
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
    VkSubpassDependency colorAttachmentDependency {
        // Image Layout Transition
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
        .dependencyFlags = 0,
    };
    VkSubpassDependency depthAttachmentDependency {
        // Depth buffer is shared between swapchain images
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = 0,
    };
    VkSubpassDependency attachmentDependencies[2] {
        depthAttachmentDependency,
        colorAttachmentDependency,
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

    CreateGeometryBuffers();

    // Create a graphics pipeline
    VkVertexInputBindingDescription vertex_input_binding {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    VkVertexInputAttributeDescription vertex_position{
        .location = 0,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, v),
    };
    VkVertexInputAttributeDescription vertex_normal{
        .location = 1,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32_SFLOAT,
        .offset = offsetof(Vertex, n),
    };
    VkVertexInputAttributeDescription vertex_color{
        .location = 2,
        .binding = 0,
        .format = VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = offsetof(Vertex, c),
    };
    VkVertexInputAttributeDescription vertex_texcoord{
        .location = 3,
        .binding = 0,
        .format = VK_FORMAT_R32G32_SFLOAT,
        .offset = offsetof(Vertex, t),
    };
    std::vector<VkVertexInputAttributeDescription> vertex_input_attributes{vertex_position, vertex_normal, vertex_color, vertex_texcoord};

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
        .cullMode = VK_CULL_MODE_NONE, // VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };

    VkPipelineColorBlendAttachmentState att_state[1] = {
        {
            .blendEnable = VK_FALSE,
            .colorWriteMask = 0xf,
        },
    };

    VkPipelineColorBlendStateCreateInfo color_blend_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .pNext = nullptr,
        .attachmentCount = 1,
        .pAttachments = att_state,
    };

    VkStencilOpState keep_always{ .failOp = VK_STENCIL_OP_KEEP, .passOp = VK_STENCIL_OP_KEEP, .compareOp = VK_COMPARE_OP_ALWAYS };
    VkPipelineDepthStencilStateCreateInfo depth_stencil_state {
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

    std::vector<uint32_t> vertex_shader_code = GetFileAsCode("testing.vert");
    VkShaderModule vertex_shader_module = CreateShaderModule(device, vertex_shader_code);

    std::vector<uint32_t> fragment_shader_code = GetFileAsCode("testing.frag");
    VkShaderModule fragment_shader_module = CreateShaderModule(device, fragment_shader_code);

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

    VkDescriptorSetLayoutBinding binding {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_VERTEX_BIT,
        .pImmutableSamplers = nullptr,
    };

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .bindingCount = 1,
        .pBindings = &binding,
    };
    VK_CHECK(vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create, nullptr, &descriptor_set_layout));

    VkDescriptorPoolSize pool_sizes = {
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = swapchainImageCount,
    };
    VkDescriptorPoolCreateInfo create_descriptor_pool {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = nullptr,
        .flags = 0,
        .maxSets = swapchainImageCount, // XXX could limit to MAX_IN_FLIGHT?
        .poolSizeCount = 1,
        .pPoolSizes = &pool_sizes,
    };
    VK_CHECK(vkCreateDescriptorPool(device, &create_descriptor_pool, nullptr, &descriptor_pool));

    descriptor_sets.resize(swapchainImageCount);
    for(uint32_t i = 0 ; i < swapchainImageCount; i++) {
        VkDescriptorSetAllocateInfo allocate_descriptor_set {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = nullptr,
            .descriptorPool = descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &descriptor_set_layout,
        };
        VK_CHECK(vkAllocateDescriptorSets(device, &allocate_descriptor_set, &descriptor_sets[i]));
    }

    uniform_buffers.resize(swapchainImageCount);
    for(uint32_t i = 0 ; i < swapchainImageCount; i++) {
        VkBufferCreateInfo create_buffer {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = nullptr,
            .size = sizeof(VertexUniforms),
            .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        };

        VK_CHECK(vkCreateBuffer(device, &create_buffer, nullptr, &uniform_buffers[i].buf));

        VkMemoryRequirements memory_req;
        vkGetBufferMemoryRequirements(device, uniform_buffers[i].buf, &memory_req);

        memoryTypeIndex = getMemoryTypeIndex(memory_properties, memory_req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkMemoryAllocateInfo memory_alloc {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = nullptr,
            .allocationSize = memory_req.size,
            .memoryTypeIndex = memoryTypeIndex,
        };
        VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &uniform_buffers[i].mem));
        VK_CHECK(vkBindBufferMemory(device, uniform_buffers[i].buf, uniform_buffers[i].mem, 0));
        VK_CHECK(vkMapMemory(device, uniform_buffers[i].mem, 0, sizeof(VertexUniforms), 0, &uniform_buffers[i].mapped));

    }

    for(uint32_t i = 0 ; i < swapchainImageCount; i++) {
        VkDescriptorBufferInfo buffer_info {
            .buffer = uniform_buffers[i].buf,
            .range = sizeof(VertexUniforms),
        };

        VkWriteDescriptorSet write_descriptor_set {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = descriptor_sets[i],
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .pImageInfo = nullptr,
            .pBufferInfo = &buffer_info,
            .pTexelBufferView = nullptr,
        };
        vkUpdateDescriptorSets(device, 1, &write_descriptor_set, 0, nullptr);
    }

    VkPipelineLayoutCreateInfo create_layout {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = nullptr,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
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
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    draw_completed_fences.resize(MAX_IN_FLIGHT);
    for(uint32_t i = 0; i < draw_completed_fences.size(); i++) {
        VK_CHECK(vkCreateFence(device, &fence_create, nullptr, &draw_completed_fences[i]));
    }
}

void Cleanup()
{
}

void rotation(float a, float x, float y, float z, float m[16])
{
    float c, s, t;

    c = (float)cos(a);
    s = (float)sin(a);
    t = 1.0f - c;

    m[0] = t * x * x + c;
    m[1] = t * x * y + s * z;
    m[2] = t * x * z - s * y;
    m[3] = 0;

    m[4] = t * x * y - s * z;
    m[5] = t * y * y + c;
    m[6] = t * y * z + s * x;
    m[7] = 0;

    m[8] = t * x * z + s * y;
    m[9] = t * y * z - s * x;
    m[10] = t * z * z + c;
    m[11] = 0;

    m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1;
}

static void DrawFrame([[maybe_unused]] GLFWwindow *window)
{
    VK_CHECK(vkWaitForFences(device, 1, &draw_completed_fences[draw_submission_index], VK_TRUE, DEFAULT_FENCE_TIMEOUT));
    VK_CHECK(vkResetFences(device, 1, &draw_completed_fences[draw_submission_index]));

    static float frame = 0.0;
    float time = frame / 100.0f;
    frame += 1;

    modelview = mat4f::rotation(time, 0, 1, 0);

    mat4f modelview_3x3 = modelview;
    modelview_3x3.m_v[12] = 0.0f;
    modelview_3x3.m_v[13] = 0.0f;
    modelview_3x3.m_v[14] = 0.0f;
    mat4f modelview_normal = inverse(transpose(modelview_3x3));

    uint8_t *ubo = static_cast<uint8_t*>(uniform_buffers[swapchainIndex].mapped);
    memcpy(ubo + sizeof(mat4f) * 0, modelview.m_v, sizeof(mat4f));
    memcpy(ubo + sizeof(mat4f) * 1, modelview_normal.m_v, sizeof(mat4f));
    memcpy(ubo + sizeof(mat4f) * 2, projection.m_v, sizeof(mat4f));

    VK_CHECK(vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, image_acquired_semaphores[swapchainIndex], VK_NULL_HANDLE, &swapchainIndex));

    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    auto cb = commandBuffers[swapchainIndex];

    VkCommandBufferBeginInfo begin {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = nullptr,
        .flags = 0, // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    VK_CHECK(vkResetCommandBuffer(cb, 0));
    VK_CHECK(vkBeginCommandBuffer(cb, &begin));
    const VkClearValue clearValues [2] {
        {.color {.float32 {0.1f, 0.1f, 0.2f, 1.0f}}},
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

    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_sets[swapchainIndex], 0, NULL);

    // 8. Bind the texture resources - NA

    // 7. Bind the vertex and swapchainIndex buffers
    VkDeviceSize offset = 0;
    vkCmdBindVertexBuffers(cb, 0, 1, &vertex_buffer.buf, &offset);
    vkCmdBindIndexBuffer(cb, index_buffer.buf, 0, VK_INDEX_TYPE_UINT32);

    // 9. Set viewport and scissor parameters
    VkViewport viewport = CalculateViewport(static_cast<uint32_t>(windowWidth), static_cast<uint32_t>(windowWidth));
    vkCmdSetViewport(cb, 0, 1, &viewport);

    VkRect2D scissor {
        .offset{0, 0},
        .extent{static_cast<uint32_t>(windowWidth), static_cast<uint32_t>(windowHeight)}};
    vkCmdSetScissor(cb, 0, 1, &scissor);

    vkCmdDrawIndexed(cb, triangleCount * 3, 1, 0, 0, 0);
    vkCmdEndRenderPass(cb);
    VK_CHECK(vkEndCommandBuffer(cb));

    VkPipelineStageFlags waitdststagemask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = nullptr,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &image_acquired_semaphores[swapchainIndex],
        .pWaitDstStageMask = &waitdststagemask,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &draw_completed_semaphores[swapchainIndex],
    };
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, draw_completed_fences[draw_submission_index]));
    draw_submission_index = (draw_submission_index + 1) % MAX_IN_FLIGHT;

    // 13. Present the rendered result
    uint32_t swapchainIndices[] = {swapchainIndex};
    VkPresentInfoKHR present {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .pNext = nullptr,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &draw_completed_semaphores[swapchainIndex],
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = swapchainIndices,
            .pResults = nullptr,
    };
    VK_CHECK(vkQueuePresentKHR(queue, &present));
    swapchainIndex = (swapchainIndex + 1) % swapchainImageCount;
}

};

static void ErrorCallback([[maybe_unused]] int error, const char* description)
{
    fprintf(stderr, "GLFW: %s\n", description);
}

static void KeyCallback(GLFWwindow *window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods)
{
    if(action == GLFW_PRESS) {
        switch(key) {
            case 'Q': case GLFW_KEY_ESCAPE: case '\033':
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
        }
    }
}

bool ParseTriSrc(FILE *fp, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices)
{
    char texture_name[512];
    char tag_name[512];
    float specular_color[4];
    float shininess;

    while(fscanf(fp,"\"%[^\"]\"", texture_name) == 1) {
        if(strcmp(texture_name, "*") == 0)
            texture_name[0] = '\0';

	if(fscanf(fp,"%s ", tag_name) != 1) {
	    fprintf(stderr, "couldn't read tag name\n");
	    return false;
	}

	if(fscanf(fp,"%g %g %g %g %g ", &specular_color[0], &specular_color[1],
	    &specular_color[2], &specular_color[3], &shininess) != 5) {
	    fprintf(stderr, "couldn't read specular properties\n");
	    return false;
	}

	if(shininess > 0 && shininess < 1) {
	    // shininess is not exponent - what is it?
	    shininess *= 10;
	}

        for(int i = 0; i < 3; i++) {
            float v[3];
            float n[3];
            float c[4];
            float t[2];

	    if(fscanf(fp,"%g %g %g %g %g %g %g %g %g %g %g %g ",
	        &v[0], &v[1], &v[2],
	        &n[0], &n[1], &n[2],
	        &c[0], &c[1], &c[2], &c[3],
	        &t[0], &t[1]) != 12) {

		fprintf(stderr, "couldn't read Vertex\n");
		return false;
	    }
            indices.push_back(vertices.size());
            vertices.push_back(Vertex(v, n, c, t));
        }

        // MATERIAL mtl(texture_name, specular_color, shininess);

        // sets.get_triangle_set(tag_name, mtl).add_triangle(verts[0], verts[1], verts[2]);
    }
    return true;
}

void LoadModel(const char *filename)
{
    using namespace VulkanApp;

    FILE* fp = fopen(filename, "r");
    if(fp == nullptr) {
        throw "couldn't open file";
    }

    ParseTriSrc(fp, vertices, indices);
    triangleCount = indices.size() / 3;

    fclose(fp);
}

void usage(const char *progName) 
{
    fprintf(stderr, "usage: %s modelFileName\n", progName);
}

int main(int argc, char **argv)
{
    using namespace VulkanApp;

    beVerbose = (getenv("BE_NOISY") != nullptr);
    enableValidation = (getenv("VALIDATE") != nullptr);
    do_the_wrong_thing = (getenv("BE_WRONG") != nullptr);

    const char *progName = argv[0];
    argv++;
    argc--;
    while(argc > 0 && argv[0][0] == '-') {
    }
    if(argc != 1) {
        fprintf(stderr, "expected a filename\n");
        usage(progName);
        exit(EXIT_FAILURE);
    }
    const char *input_filename = argv[0];

    LoadModel(input_filename);

    glfwSetErrorCallback(ErrorCallback);

    if(!glfwInit()) {
	std::cerr << "GLFW initialization failed.\n";
        exit(EXIT_FAILURE);
    }

    if (!glfwVulkanSupported()) {
	std::cerr << "GLFW reports Vulkan is not supported\n";
        exit(EXIT_FAILURE);
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(512, 512, "vulkan test", nullptr, nullptr);

    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);

    VulkanApp::InitializeInstance();

    VkResult err = glfwCreateWindowSurface(instance, window, nullptr, &surface);
    if (err) {
	std::cerr << "GLFW window creation failed " << err << "\n";
        exit(EXIT_FAILURE);
    }

    VulkanApp::InitializeState(windowWidth, windowHeight);

    glfwSetKeyCallback(window, KeyCallback);
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

    Cleanup();

    glfwTerminate();
}
