#include <iostream>
#include "vulkan.h"

// Styled somewhat after Sascha Willem's triangle

using namespace std;

VkInstance instance;

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

void create_instance()
{
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "triangle";
    app_info.pEngineName = "triangle";
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pNext = NULL;
    instance_info.pApplicationInfo = &app_info;

    VkResult result = vkCreateInstance(&instance_info, nullptr, &instance);
    if(result != VK_SUCCESS) {
	cerr << "VkResult from VkCreateInstance was " << result;
	exit(EXIT_FAILURE);
    }
    cout << "Hey, I have an instance!\n";
}

void vulkan_init()
{
    create_instance();
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

