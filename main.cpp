#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <map>
#include <set>
#include <string>

#include <cstring>
#include <cassert>

#include <vulkan.h>
#include <GLFW/glfw3.h>

#if defined(_WIN32)
#define PLATFORM_WINDOWS
// Windows supported
#elif defined(__linux__)
#define PLATFORM_LINUX
// Linux supported
#elif defined(__APPLE__) && defined(__MACH__)
// MacOS not supported yet but want to get to runtime
#define PLATFORM_MOLTENVK
#else
#error Platform not supported.
#endif

#define DEFAULT_FENCE_TIMEOUT 100000000000

// Styled somewhat after Sascha Willem's triangle

using namespace std;

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

bool be_noisy = false;
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
std::Vector<VkFence> wait_fences;

#endif

#define STR(f) #f

map<VkResult, string> vkresult_name_map =
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
    if(result != VK_SUCCESS) { \
	if(vkresult_name_map.count(f) > 0) { \
	    cerr << "VkResult from " STR(f) " was " << vkresult_name_map[result] << " at line " << __LINE__ << "\n"; \
	} \
	else \
	    cerr << "VkResult from " STR(f) " was " << result << " at line " << __LINE__ << "\n"; \
	exit(EXIT_FAILURE); \
    } \
}


void print_implementation_information()
{
    uint32_t ext_count;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    unique_ptr<VkExtensionProperties[]> exts(new VkExtensionProperties[ext_count]);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.get());
    if(be_noisy) {
        printf("Vulkan instance extensions:\n");
        for(int i = 0; i < ext_count; i++)
            printf("    (%08X) %s\n", exts[i].specVersion, exts[i].extensionName);
    }
}

void create_instance(VkInstance* instance)
{
    set<string> extension_set;
    set<string> layer_set;

    uint32_t glfw_reqd_extension_count;
    const char** glfw_reqd_extensions = glfwGetRequiredInstanceExtensions(&glfw_reqd_extension_count);
    for(int i = 0; i < glfw_reqd_extension_count; i++) {
	extension_set.insert(glfw_reqd_extensions[i]);
    }

    extension_set.insert(VK_KHR_SURFACE_EXTENSION_NAME);
#if defined(PLATFORM_WINDOWS)
    extension_set.insert("VK_KHR_win32_surface");
#elif defined(PLATFORM_LINUX)
    extension_set.insert("VK_KHR_xcb_surface");
#elif defined(PLATFORM_MACOS)
    extension_set.insert("VK_MVK_macos_surface");
#endif

    if(enable_validation) {
	layer_set.insert("VK_LAYER_KHRONOS_validation");
	// layer_set.insert("VK_LAYER_LUNARG_standard_validation");
	// layer_set.insert("VK_LAYER_LUNARG_core_validation");
	// layer_set.insert("VK_LAYER_LUNARG_parameter_validation");
	// layer_set.insert("VK_LAYER_LUNARG_object_tracker");
    }
    if(dump_vulkan_calls) {
	layer_set.insert("VK_LAYER_LUNARG_api_dump");
    }

    {
	vector<const char*> extensions;
	vector<const char*> layers;
	// Careful - only valid for duration of sets where contents have not changed
	for(auto& s: extension_set)
	    extensions.push_back(s.c_str());

	for(auto& s: layer_set)
	    layers.push_back(s.c_str());

	VkApplicationInfo app_info = {};
	app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	app_info.pApplicationName = "triangle";
	app_info.pEngineName = "triangle";
	app_info.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo create = {};
	create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	create.pNext = NULL;
	create.pApplicationInfo = &app_info;
	create.enabledExtensionCount = extensions.size();
	create.ppEnabledExtensionNames = extensions.data();
	create.enabledLayerCount = layers.size();
	create.ppEnabledLayerNames = layers.data();

	VK_CHECK(vkCreateInstance(&create, nullptr, instance));
    }
}

void choose_physical_device(VkInstance instance, VkPhysicalDevice* physical_device)
{
    uint32_t gpu_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr));
    if(be_noisy) {
        cerr << gpu_count << " gpus enumerated\n";
    }
    VkPhysicalDevice physical_devices[32];
    gpu_count = min(32u, gpu_count);
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

map<uint32_t, string> memory_property_bit_name_map = {
    {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, "DEVICE_LOCAL"},
    {VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, "HOST_VISIBLE"},
    {VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, "HOST_COHERENT"},
    {VK_MEMORY_PROPERTY_HOST_CACHED_BIT, "HOST_CACHED"},
    {VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT, "LAZILY_ALLOCATED"},
};

void print_memory_property_bits(VkMemoryPropertyFlags flags)
{
    bool add_or = false;
    for(auto& bit : memory_property_bit_name_map)
	if(flags & bit.first) {
	    printf("%s%s", add_or ? " | " : "", bit.second.c_str());
	    add_or = true;
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
    printf("    type    %s\n", device_types[min(5, (int)properties.deviceType)]);

    uint32_t ext_count;

    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, nullptr);
    unique_ptr<VkExtensionProperties[]> exts(new VkExtensionProperties[ext_count]);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, exts.get());
    printf("    extensions:\n");
    for(int i = 0; i < ext_count; i++)
	printf("        %s\n", exts[i].extensionName);

    // VkPhysicalDeviceLimits              limits;
    // VkPhysicalDeviceSparseProperties    sparseProperties;
    //
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    unique_ptr<VkQueueFamilyProperties[]> queue_families(new VkQueueFamilyProperties[queue_family_count]);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.get());
    if(be_noisy) {
        for(int i = 0; i < queue_family_count; i++) {
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

    for(int i = 0; i < memory_properties.memoryTypeCount; i++) {
        printf("memory type %d: flags ", i);
        print_memory_property_bits(memory_properties.memoryTypes[i].propertyFlags);
        printf("\n");
    }
}

void create_device(VkPhysicalDevice physical_device, VkDevice* device)
{
    vector<const char*> extensions;

    extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

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
    create.enabledExtensionCount = extensions.size();
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
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
	if (type_bits & (1 << i))
	    if ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
		return i;

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
    unique_ptr<VkQueueFamilyProperties[]> queue_families(new VkQueueFamilyProperties[queue_family_count]);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.get());
    for(int i = 0; i < queue_family_count; i++) {
        if(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                preferred_queue_family = i;
        }
    }

    if(preferred_queue_family == NO_QUEUE_FAMILY) {
	cerr << "no desired queue family was found\n";
	exit(EXIT_FAILURE);
    }

    if(be_noisy) {
        print_device_information(physical_device);
    }
    create_device(physical_device, &device);
}

void prepare_vulkan()
{
    create_vertex_buffers();
    // create swapchain
    // create descriptor sets
    // load shader modules
    // create pipeline layout
    // create pipeline
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

int main(int argc, char **argv)
{
    be_noisy = (getenv("BE_NOISY") != NULL);
    enable_validation = (getenv("VALIDATE") != NULL);
    do_the_wrong_thing = (getenv("BE_WRONG") != NULL);

    glfwSetErrorCallback(error_callback);

    if(!glfwInit()) {
	cerr << "GLFW initialization failed.\n";
        exit(EXIT_FAILURE);
    }

    if (!glfwVulkanSupported()) {
	cerr << "GLFW reports Vulkan is not supported\n";
        exit(EXIT_FAILURE);
    }

    init_vulkan();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(512, 512, "vulkan test", NULL, NULL);

    VkResult err = glfwCreateWindowSurface(instance, window, NULL, &surface);
    if (err) {
	cerr << "GLFW window creation failed " << err << "\n";
        exit(EXIT_FAILURE);
    }

    prepare_vulkan();

    // while(1)
    {
        // start command buffer
        // enqueue draw command
        // end command buffer
        // enqueue command buffer for graphics
        // copy to present
    }

    cleanup_vulkan();
}

