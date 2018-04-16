#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <map>

#include "vulkan.h"

#ifdef _WIN32
// Platform supported
#else
#error Platform not supported.
#endif

// Styled somewhat after Sascha Willem's triangle

using namespace std;

VkInstance instance;
VkPhysicalDevice physical_device;
const uint32_t NO_QUEUE_FAMILY = 0xffffffff;
uint32_t preferred_queue_family = NO_QUEUE_FAMILY;
VkDevice device;
VkPhysicalDeviceMemoryProperties memory_properties;

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

void print_implementation_information()
{
    uint32_t ext_count;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    unique_ptr<VkExtensionProperties[]> exts(new VkExtensionProperties[ext_count]);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.get());
    printf("Vulkan implementation properties:\n");
    for(int i = 0; i < ext_count; i++)
	printf("%s\n", exts[i].extensionName);
}

void create_instance(VkInstance* instance)
{
    vector<const char*> extensions;

    extensions.push_back("VK_KHR_surface");
#ifdef WIN32
    extensions.push_back("VK_KHR_win32_surface");
#endif

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

    VkResult result = vkCreateInstance(&create, nullptr, instance);
    if(result != VK_SUCCESS) {
	cerr << "VkResult from VkCreateInstance was " << result;
	exit(EXIT_FAILURE);
    }
}

#define STR(f) #f

#define VK_CHECK(f) \
{ \
    VkResult result = (f); \
    if(result != VK_SUCCESS) { \
	cerr << "VkResult from " STR(f) " was " << result << " at line " << __LINE__ << "\n"; \
	exit(EXIT_FAILURE); \
    } \
}

void choose_physical_device(VkInstance instance, VkPhysicalDevice* physical_device)
{
    uint32_t gpu_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr));
    cerr << gpu_count << " gpus enumerated\n";
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
    for(int i = 0; i < queue_family_count; i++) {
	printf("queue %d:\n", i);
	printf("    flags:                       %04X\n", queue_families[i].queueFlags);
	printf("    queueCount:                  %d\n", queue_families[i].queueCount);
	printf("    timestampValidBits:          %d\n", queue_families[i].timestampValidBits);
	printf("    minImageTransferGranularity: (%d, %d, %d)\n",
	    queue_families[i].minImageTransferGranularity.width,
	    queue_families[i].minImageTransferGranularity.height,
	    queue_families[i].minImageTransferGranularity.depth);

	if(queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
	    preferred_queue_family = i;
    }

    if(preferred_queue_family == NO_QUEUE_FAMILY) {
	cerr << "no desired queue family was found\n";
	exit(EXIT_FAILURE);
    }
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    for(int i = 0; i < memory_properties.memoryTypeCount; i++) {
	printf("memory type %d: flags ", i);
	print_memory_property_bits(memory_properties.memoryTypes[i].propertyFlags);
	printf("\n");
    }
}

void create_device(VkPhysicalDevice physical_device, VkDevice* device)
{
    vector<const char*> extensions;

    extensions.push_back("VK_KHR_swapchain");

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
}

// (Sascha Willem's getMemoryTypeIndex)
uint32_t getMemoryTypeIndex(uint32_t type_bits, VkMemoryPropertyFlags properties)
{
    // Iterate over all memory types available for the device used in this example
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
	if (type_bits & (1 << i))
	    if ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
		return i;

    throw "Could not find a suitable memory type!";
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

    // Create a buffer - we can use buffers for things like vertex data
    VkBufferCreateInfo create_buffer = {};
    create_buffer.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    create_buffer.size = sizeof(vertices);
    create_buffer.pNext = nullptr;
    create_buffer.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VK_CHECK(vkCreateBuffer(device, &create_buffer, nullptr, &vertex_staging.buf));

    // Get the size and type requirements for memory containing this buffer
    VkMemoryRequirements memory_req = {}; // Tells us how much memory and which memory types (by bit) can hold this memory
    vkGetBufferMemoryRequirements(device, vertex_staging.buf, &memory_req);

    // Allocate memory
    VkMemoryAllocateInfo memory_alloc = {};
    memory_alloc.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memory_alloc.pNext = nullptr;
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

}

void vulkan_init()
{
    print_implementation_information();
    create_instance(&instance);
    choose_physical_device(instance, &physical_device);
    print_device_information(physical_device);
    create_device(physical_device, &device);
    create_vertex_buffers();
}

void vulkan_cleanup()
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

void main(int argc, char **argv)
{
    vulkan_init();

    // Render until quit

    vulkan_cleanup();
}

