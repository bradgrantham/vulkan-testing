#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>

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
VkDevice device;

#if 0

struct vertex {
    float v[3];
    float c[3];
};

VkDeviceMemory vertex_memory;
VkBuffer vertex_buffer;

VkDeviceMemory index_memory;
VkBuffer index_buffer;

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
    }
}

void create_device(VkPhysicalDevice physical_device, VkDevice* device)
{
#if 0
    VkDeviceCreateInfo create = {};

    create.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create.pNext = NULL;
    create.flags = 0;
    create.queueCreateInfoCount = 1;
    create.pQueueCreateInfos = queue_create;
    VK_CHECK(vkCreateDevice, &create, nullptr, device);
#endif
}

void vulkan_init()
{
    print_implementation_information();
    create_instance(&instance);
    choose_physical_device(instance, &physical_device);
    print_device_information(physical_device);
    create_device(physical_device, &device);
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

