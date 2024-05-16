#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <algorithm>
#include <memory>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <string>
#include <array>
#include <filesystem>

#include <cstring>
#include <cassert>
#include <cinttypes>

#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include "vectormath.h"
#include "manipulator.h"

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

template <typename T>
T align(T value, T align) { return ((value + align - 1) / align) * align; }

template <typename T>
size_t ByteCount(const std::vector<T>& v) { return sizeof(T) * v.size(); }

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
    VkResult result_ = (f); \
    static const std::set<VkResult> okay{VK_SUCCESS, VK_SUBOPTIMAL_KHR, VK_THREAD_IDLE_KHR, VK_THREAD_DONE_KHR, VK_OPERATION_DEFERRED_KHR, VK_OPERATION_NOT_DEFERRED_KHR}; \
    if(!okay.contains(result_)) { \
	if(MapVkResultToName.count(f) > 0) { \
	    std::cerr << "VkResult from " STR(f) " was " << MapVkResultToName[result_] << " at line " << __LINE__ << "\n"; \
	} else { \
	    std::cerr << "VkResult from " STR(f) " was " << result_ << " at line " << __LINE__ << "\n"; \
        } \
	exit(EXIT_FAILURE); \
    } \
}

// Adapted from vkcube.cpp
VkSurfaceFormatKHR PickSurfaceFormat(std::vector<VkSurfaceFormatKHR>& surfaceFormats)
{
    // Prefer non-SRGB formats...
    int which = 0;
    for (const auto& surfaceFormat: surfaceFormats) {
        const VkFormat format = surfaceFormat.format;

        if (format == VK_FORMAT_R8G8B8A8_UNORM || format == VK_FORMAT_B8G8R8A8_UNORM) {
            return surfaceFormats[which];
        }
        which++;
    }

    printf("Can't find our preferred formats... Falling back to first exposed format. Rendering may be incorrect.\n");

    return surfaceFormats[0];
}

VkCommandPool CreateCommandPool(VkDevice device, uint32_t queue)
{
    VkCommandPoolCreateInfo create_command_pool{
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = queue,
    };
    VkCommandPool command_pool;
    VK_CHECK(vkCreateCommandPool(device, &create_command_pool, nullptr, &command_pool));
    return command_pool;
}

VkCommandBuffer AllocateCommandBuffer(VkDevice device, VkCommandPool command_pool)
{
    VkCommandBufferAllocateInfo cmdBufAllocateInfo {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer command_buffer;
    VK_CHECK(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &command_buffer));
    return command_buffer;
}

void BeginCommandBuffer(VkCommandBuffer command_buffer)
{
    VkCommandBufferBeginInfo info {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
    };
    VK_CHECK(vkBeginCommandBuffer(command_buffer, &info));
}

void FlushCommandBuffer(VkDevice device, VkQueue queue, VkCommandBuffer command_buffer)
{
    VkSubmitInfo submitInfo {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &command_buffer,
    };

    // Create fence to ensure that the command buffer has finished executing
    VkFenceCreateInfo fenceCreateInfo {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = 0,
    };
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

    // Submit to the queue
    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
    // Wait for the fence to signal that command buffer has finished executing
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));

    vkDestroyFence(device, fence, nullptr);
}

uint32_t GetMemoryTypeIndex(VkPhysicalDeviceMemoryProperties memory_properties, uint32_t type_bits, VkMemoryPropertyFlags properties)
{
    // Adapted from Sascha Willem's 
    // Iterate over all memory types available for the device used in this example
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
	if (type_bits & (1 << i)) {
	    if ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties) {
		return i;
            }
        }
    }

    throw std::runtime_error("Could not find a suitable memory type!");
}

static constexpr uint32_t NO_QUEUE_FAMILY = 0xffffffff;

struct Vertex
{
    vec3 v;
    vec3 n;
    vec4 c;
    vec3 t;

    Vertex(const vec3& v, const vec3& n, const vec4& c, const vec2& t) :
        v(v),
        n(n),
        c(c),
        t(t)
    { }
    Vertex() {}

    static std::vector<VkVertexInputAttributeDescription> GetVertexInputAttributeDescription()
    {
        std::vector<VkVertexInputAttributeDescription> vertex_input_attributes;
        vertex_input_attributes.push_back({0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, v)});
        vertex_input_attributes.push_back({1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, n)});
        vertex_input_attributes.push_back({2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, c)});
        vertex_input_attributes.push_back({3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, t)});
        return vertex_input_attributes;
    }
};

struct Buffer
{
    VkDevice device;
    VkDeviceMemory mem { VK_NULL_HANDLE };
    VkBuffer buf { VK_NULL_HANDLE };
    void* mapped { nullptr };

    uint64_t GetDeviceAddress(PFN_vkGetBufferDeviceAddressKHR GetBufferDeviceAddress)
    {
        assert(buf != VK_NULL_HANDLE);
        assert(mem != VK_NULL_HANDLE);
        VkBufferDeviceAddressInfoKHR address_info {
            .sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = buf,
        };
        return GetBufferDeviceAddress(device, &address_info);
    }

    void Create(VkPhysicalDevice physical_device, VkDevice device_, size_t size, VkBufferUsageFlags usage_flags, VkMemoryPropertyFlags properties)
    {
        Release();

        device = device_;

        VkBufferCreateInfo create_buffer {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage_flags,
        };

        VK_CHECK(vkCreateBuffer(device, &create_buffer, nullptr, &buf));

        VkMemoryRequirements memory_req;
        vkGetBufferMemoryRequirements(device, buf, &memory_req);

        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

        uint32_t memoryTypeIndex = GetMemoryTypeIndex(memory_properties, memory_req.memoryTypeBits, properties);
        VkMemoryAllocateFlagsInfo memory_flags {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
            .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
        };

        VkMemoryAllocateInfo memory_alloc {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &memory_flags,
            .allocationSize = memory_req.size,
            .memoryTypeIndex = memoryTypeIndex,
        };
        VK_CHECK(vkAllocateMemory(device, &memory_alloc, nullptr, &mem));
        VK_CHECK(vkBindBufferMemory(device, buf, mem, 0));
    }

    void Release()
    {
        if(mapped) {
            vkUnmapMemory(device, mem);
            mapped = nullptr;
        }
        if(mem != VK_NULL_HANDLE) {
            vkFreeMemory(device, mem, nullptr);
            mem = VK_NULL_HANDLE;
            vkDestroyBuffer(device, buf, nullptr);
            buf = VK_NULL_HANDLE;
        }
    }
};

struct Descriptor 
{
    uint32_t binding;
    VkShaderStageFlags stageFlags;
    VkDescriptorType type;
    size_t size;
    Descriptor(int binding, VkShaderStageFlags stageFlags, VkDescriptorType type) :
        binding(binding),
        stageFlags(stageFlags),
        type(type),
        size(0)
    {
    }
    Descriptor(int binding, VkShaderStageFlags stageFlags, VkDescriptorType type, size_t size) :
        binding(binding),
        stageFlags(stageFlags),
        type(type),
        size(size)
    {
    }
};

struct ShadingUniforms
{
    vec3 specular_color;
    float shininess;
};

struct RayTracingCamera
{
    mat4f modelviewInverse;
    mat4f projectionInverse;
};

struct VertexUniforms
{
    mat4f modelview;
    mat4f modelview_normal;
    mat4f projection;
};

struct FragmentUniforms
{
    vec3 light_position;
    float pad; // XXX grr - why can't I get this right with "pragma align"?
    vec3 light_color;
    float pad2;
};

VkInstance CreateInstance(bool enable_validation)
{
    VkInstance instance;
    std::set<std::string> extension_set;
    std::set<std::string> layer_set;

    uint32_t glfw_reqd_extension_count;
    const char** glfw_reqd_extensions = glfwGetRequiredInstanceExtensions(&glfw_reqd_extension_count);
    extension_set.insert(glfw_reqd_extensions, glfw_reqd_extensions + glfw_reqd_extension_count);

#if defined(PLATFORM_MACOS)
    extension_set.insert(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
#endif

    if(enable_validation) {
	layer_set.insert("VK_LAYER_KHRONOS_validation");
    }

    // Make this an immediately invoked lambda so I know the c_str() I called remains
    // valid through the scope of this lambda.
    [&](const std::set<std::string> &extension_set, const std::set<std::string> &layer_set) {

        std::vector<const char*> extensions;
        std::vector<const char*> layers;

	for(auto& s: extension_set) {
	    extensions.push_back(s.c_str());
        }

	for(auto& s: layer_set) {
	    layers.push_back(s.c_str());
        }

	VkApplicationInfo app_info {
            .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pApplicationName = "vulkan-test-grantham",
            .pEngineName = "vulkan-test-grantham",
            .apiVersion = VK_API_VERSION_1_3,
        };

	VkInstanceCreateInfo create {
            .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
#if defined(PLATFORM_MACOS)
            .flags = VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR,
#endif
            .pApplicationInfo = &app_info,
            .enabledLayerCount = static_cast<uint32_t>(layers.size()),
            .ppEnabledLayerNames = layers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
        };

	VK_CHECK(vkCreateInstance(&create, nullptr, &instance));

    }(extension_set, layer_set);

    return instance;
}

VkPhysicalDevice ChoosePhysicalDevice(VkInstance instance, uint32_t specified_gpu)
{
    uint32_t gpu_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, nullptr));

    std::vector<VkPhysicalDevice> physical_devices(gpu_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &gpu_count, physical_devices.data()));

    if(specified_gpu >= gpu_count) {
        fprintf(stderr, "requested device #%d but max device index is #%d.\n", specified_gpu, gpu_count);
        exit(EXIT_FAILURE);
    }

    return physical_devices[specified_gpu];
}


const std::vector<std::string> DeviceTypeDescriptions = {
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
    printf("    type    %s\n", DeviceTypeDescriptions[std::min(5, (int)properties.deviceType)].c_str());

    uint32_t ext_count;

    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateDeviceExtensionProperties(physical_device, nullptr, &ext_count, exts.data());
    printf("    extensions:\n");
    for(const auto& ext: exts) {
	printf("        %s\n", ext.extensionName);
    }

    // VkPhysicalDeviceLimits              limits;
    // VkPhysicalDeviceSparseProperties    sparseProperties;
    //
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());
    int queue_index = 0;
    for(const auto& queue_family: queue_families) {
        printf("queue %d:\n", queue_index++);
        printf("    flags:                       %04X\n", queue_family.queueFlags);
        printf("    queueCount:                  %d\n", queue_family.queueCount);
        printf("    timestampValidBits:          %d\n", queue_family.timestampValidBits);
        printf("    minImageTransferGranularity: (%d, %d, %d)\n",
            queue_family.minImageTransferGranularity.width,
            queue_family.minImageTransferGranularity.height,
            queue_family.minImageTransferGranularity.depth);
    }

    for(uint32_t i = 0; i < memory_properties.memoryTypeCount; i++) {
        printf("memory type %d: flags ", i);
        PrintMemoryPropertyBits(memory_properties.memoryTypes[i].propertyFlags);
        printf("\n");
    }
}

std::vector<uint8_t> GetFileContents(FILE *fp)
{
    long int start = ftell(fp);
    fseek(fp, 0, SEEK_END);
    long int end = ftell(fp);
    fseek(fp, start, SEEK_SET);

    std::vector<uint8_t> data(end - start);
    [[maybe_unused]] size_t result = fread(data.data(), 1, end - start, fp);
    if(result != static_cast<size_t>(end - start)) {
        fprintf(stderr, "short read\n");
        abort();
    }

    return data;
}

std::vector<uint32_t> GetFileAsCode(const std::string& filename) 
{
    FILE *fp = fopen(filename.c_str(), "rb");
    if(fp == nullptr) {
        fprintf(stderr, "couldn't open \"%s\" for reading\n", filename.c_str());
        abort();
    }
    std::vector<uint8_t> text = GetFileContents(fp);
    fclose(fp);

    std::vector<uint32_t> code((text.size() + sizeof(uint32_t) - 1) / sizeof(uint32_t));
    memcpy(code.data(), text.data(), text.size()); // XXX this is probably UB that just happens to work... also maybe endian

    return code;
}

VkShaderModule CreateShaderModule(VkDevice device, const std::vector<uint32_t>& code)
{
    VkShaderModuleCreateInfo shader_create {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .flags = 0,
        .codeSize = code.size() * sizeof(code[0]),
        .pCode = code.data(),
    };

    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(device, &shader_create, NULL, &module));
    return module;
}


VkDevice CreateDevice(VkPhysicalDevice physical_device, const std::vector<const char*>& extensions, uint32_t queue_family, void *device_create_pnext)
{
    float queue_priorities = 1.0f;

    VkDeviceQueueCreateInfo create_queues {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .flags = 0,
        .queueFamilyIndex = queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priorities,
    };

    VkDeviceCreateInfo create_device {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = device_create_pnext,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &create_queues,
        .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
        .ppEnabledExtensionNames = extensions.data(),
    };

    VkDevice device;
    VK_CHECK(vkCreateDevice(physical_device, &create_device, nullptr, &device));
    return device;
}

VkSampler CreateSampler(VkDevice device, VkSamplerMipmapMode mipMode, VkSamplerAddressMode wrapMode)
{
    VkSamplerCreateInfo create_sampler {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .flags = 0,
        .magFilter = VK_FILTER_LINEAR,
        .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = mipMode,
        .addressModeU = wrapMode,
        .addressModeV = wrapMode,
        .addressModeW = wrapMode,
        .mipLodBias = 0.0f,
        .anisotropyEnable = VK_FALSE,
        .maxAnisotropy = 0.0f,
        .compareEnable = VK_FALSE,
        .compareOp = VK_COMPARE_OP_ALWAYS,
        .minLod = 0,
        .maxLod = VK_LOD_CLAMP_NONE,
        // .borderColor
        .unnormalizedCoordinates = VK_FALSE,
    };
    VkSampler textureSampler;
    VK_CHECK(vkCreateSampler(device, &create_sampler, nullptr, &textureSampler));
    return textureSampler;
}

VkImageView CreateImageView(VkDevice device, VkFormat format, VkImage image, VkImageAspectFlags aspect)
{
    VkImageViewCreateInfo imageViewCreate {
        .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .flags = 0,
        .image = image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format = format,
        .components {VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY,VK_COMPONENT_SWIZZLE_IDENTITY},
        .subresourceRange{
            .aspectMask = aspect,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1
        },
    };
    VkImageView imageView;
    VK_CHECK(vkCreateImageView(device, &imageViewCreate, nullptr, &imageView));
    return imageView;
}

VkFramebuffer CreateFramebuffer(VkDevice device, const std::vector<VkImageView>& imageviews, VkRenderPass render_pass, uint32_t width, uint32_t height)
{
    VkFramebufferCreateInfo framebufferCreate {
        .sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
        .flags = 0,
        .renderPass = render_pass,
        .attachmentCount = static_cast<uint32_t>(imageviews.size()),
        .pAttachments = imageviews.data(),
        .width = width,
        .height = height,
        .layers = 1,
    };
    VkFramebuffer framebuffer;
    VK_CHECK(vkCreateFramebuffer(device, &framebufferCreate, nullptr, &framebuffer));
    return framebuffer;
}

void PrintImplementationInformation()
{
    uint32_t ext_count;
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> exts(ext_count);
    vkEnumerateInstanceExtensionProperties(nullptr, &ext_count, exts.data());
    printf("Vulkan instance extensions:\n");
    for (const auto& ext: exts) {
        printf("\t%s, %08X\n", ext.extensionName, ext.specVersion);
    }
}

uint32_t FindQueueFamily(VkPhysicalDevice physical_device, VkQueueFlags queue_flags)
{
    uint32_t queue_family_count;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.data());
    for(uint32_t i = 0; i < queue_family_count; i++) {
        if((queue_families[i].queueFlags & queue_flags) == queue_flags) {
            return i;
        }
    }
    return NO_QUEUE_FAMILY;
}

// XXX This need to be refactored
void CreateGeometryBuffers(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, const std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices, Buffer* vertex_buffer, Buffer* index_buffer)
{
    uint32_t transfer_queue = FindQueueFamily(physical_device, VK_QUEUE_TRANSFER_BIT);
    if(transfer_queue == NO_QUEUE_FAMILY) {
        throw std::runtime_error("couldn't find a transfer queue\n");
    }

    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    // host-writable memory and buffers
    Buffer vertex_staging;
    Buffer index_staging;

    vertex_staging.Create(physical_device, device, ByteCount(vertices), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VK_CHECK(vkMapMemory(device, vertex_staging.mem, 0, ByteCount(vertices), 0, &vertex_staging.mapped));
    memcpy(vertex_staging.mapped, vertices.data(), ByteCount(vertices));
    vkUnmapMemory(device, vertex_staging.mem);

    index_staging.Create(physical_device, device, ByteCount(indices), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    VK_CHECK(vkMapMemory(device, index_staging.mem, 0, ByteCount(indices), 0, &index_staging.mapped));
    memcpy(index_staging.mapped, indices.data(), ByteCount(indices));
    vkUnmapMemory(device, index_staging.mem);

    vertex_buffer->Create(physical_device, device, ByteCount(vertices), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); // VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    index_buffer->Create(physical_device, device, ByteCount(indices), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); // VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkCommandPool command_pool = CreateCommandPool(device, transfer_queue);
    VkCommandBuffer transfer_commands = AllocateCommandBuffer(device, command_pool);

    BeginCommandBuffer(transfer_commands);
    {
        VkBufferCopy copy {0, 0, ByteCount(vertices)};
        vkCmdCopyBuffer(transfer_commands, vertex_staging.buf, vertex_buffer->buf, 1, &copy);
    }
    {
        VkBufferCopy copy {0, 0, ByteCount(indices)};
        vkCmdCopyBuffer(transfer_commands, index_staging.buf, index_buffer->buf, 1, &copy);
    }
    VK_CHECK(vkEndCommandBuffer(transfer_commands));

    FlushCommandBuffer(device, queue, transfer_commands);

    vkFreeCommandBuffers(device, command_pool, 1, &transfer_commands);
    vkDestroyBuffer(device, vertex_staging.buf, nullptr);
    vkDestroyBuffer(device, index_staging.buf, nullptr);
    vkFreeMemory(device, vertex_staging.mem, nullptr);
    vkFreeMemory(device, index_staging.mem, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
}

void TransitionBaseImageLayout(VkCommandBuffer cb, VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout, VkImageAspectFlags aspect_bits)
{
    VkImageMemoryBarrier barrier {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = 0,
        .oldLayout = oldLayout,
        .newLayout = newLayout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = image,
        .subresourceRange = {aspect_bits, 0, 1, 0, 1}
    };
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}


template <class TEXTURE>
void CreateDeviceTextureImage(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue, TEXTURE texture, VkImage* textureImage, VkDeviceMemory* textureMemory, VkImageUsageFlags usage_flags, VkImageLayout final_layout)
{
    Buffer staging_buffer;
    staging_buffer.Create(physical_device, device, texture->GetSize(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkMapMemory(device, staging_buffer.mem, 0, texture->GetSize(), 0, &staging_buffer.mapped));
    memcpy(staging_buffer.mapped, texture->GetData(), texture->GetSize());
    vkUnmapMemory(device, staging_buffer.mem);

    auto [image, memory] = CreateBound2DImage(physical_device, device, texture->GetVulkanFormat(), static_cast<uint32_t>(texture->GetWidth()), static_cast<uint32_t>(texture->GetHeight()), usage_flags, VK_IMAGE_LAYOUT_PREINITIALIZED, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    *textureImage = image;
    *textureMemory = memory;

    uint32_t transfer_queue = FindQueueFamily(physical_device, VK_QUEUE_TRANSFER_BIT);
    if(transfer_queue == NO_QUEUE_FAMILY) {
        fprintf(stderr, "couldn't find a transfer queue\n");
        abort();
    }
    VkCommandPool command_pool = CreateCommandPool(device, transfer_queue);
    VkCommandBuffer transfer_commands = AllocateCommandBuffer(device, command_pool);

    BeginCommandBuffer(transfer_commands);

    VkImageMemoryBarrier transfer_dst_optimal {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_PREINITIALIZED,
        .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *textureImage,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    vkCmdPipelineBarrier(transfer_commands, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &transfer_dst_optimal);

    VkBufferImageCopy copy {
        .bufferOffset = 0,
        .bufferRowLength = static_cast<uint32_t>(texture->GetWidth()),
        .bufferImageHeight = static_cast<uint32_t>(texture->GetHeight()),
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {static_cast<uint32_t>(texture->GetWidth()), static_cast<uint32_t>(texture->GetHeight()), 1},
    };
    vkCmdCopyBufferToImage(transfer_commands, staging_buffer.buf, *textureImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

    VkImageMemoryBarrier shader_read_optimal {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
        .srcAccessMask = 0,
        .dstAccessMask = 0,
        .oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        .newLayout = final_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = *textureImage,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}
    };
    vkCmdPipelineBarrier(transfer_commands, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &shader_read_optimal);

    VK_CHECK(vkEndCommandBuffer(transfer_commands));

    FlushCommandBuffer(device, queue, transfer_commands);
    vkFreeCommandBuffers(device, command_pool, 1, &transfer_commands);
    vkDestroyBuffer(device, staging_buffer.buf, nullptr);
    vkFreeMemory(device, staging_buffer.mem, nullptr);
    vkDestroyCommandPool(device, command_pool, nullptr);
}

struct RGBA8UNormImage
{
private:
    int width;
    int height;
    std::vector<uint8_t> rgba8_unorm;

public:
    RGBA8UNormImage(int width, int height, std::vector<uint8_t>& rgba8_unorm) :
        width(width),
        height(height),
        rgba8_unorm(std::move(rgba8_unorm))
    {}

    VkFormat GetVulkanFormat() { return VK_FORMAT_R8G8B8A8_UNORM; }
    int GetWidth() { return width; }
    int GetHeight() { return height; }
    void* GetData() { return rgba8_unorm.data(); }
    size_t GetSize() { return rgba8_unorm.size(); }
};

// Can't be Drawable because that conflicts with a type name in X11
struct DrawableShape
{
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    int triangleCount;
    aabox bounds;

    float specular_color[4];
    float shininess;

    std::shared_ptr<RGBA8UNormImage> texture;

    VkImage textureImage { VK_NULL_HANDLE };
    VkDeviceMemory textureMemory { VK_NULL_HANDLE };
    VkImageView textureImageView { VK_NULL_HANDLE };
    VkSampler textureSampler { VK_NULL_HANDLE };

    constexpr static int VERTEX_BUFFER = 0;
    constexpr static int INDEX_BUFFER = 1;
    typedef std::array<Buffer, 2> DrawableShapeBuffersOnDevice;
    std::map<VkDevice, DrawableShapeBuffersOnDevice> buffers_by_device;

    DrawableShape(std::vector<Vertex>& vertices, std::vector<uint32_t>& indices,
        float specular_color[4], float shininess, std::shared_ptr<RGBA8UNormImage> texture) :
            vertices(vertices),
            indices(indices),
            shininess(shininess),
            texture(texture)
    {
        this->specular_color[0] = specular_color[0];
        this->specular_color[1] = specular_color[1];
        this->specular_color[2] = specular_color[2];
        this->specular_color[3] = specular_color[3];
        triangleCount = static_cast<int>(indices.size() / 3);
        for(uint32_t i = 0; i < vertices.size(); i++) {
            bounds += vertices[i].v;
        }
    }

    void CreateDeviceData(VkPhysicalDevice physical_device, VkDevice device, VkQueue queue)
    {
        if(texture) {
            CreateDeviceTextureImage(physical_device, device, queue, texture, &textureImage, &textureMemory, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            textureSampler = CreateSampler(device, VK_SAMPLER_MIPMAP_MODE_NEAREST, VK_SAMPLER_ADDRESS_MODE_REPEAT);
            textureImageView = CreateImageView(device, texture->GetVulkanFormat(), textureImage, VK_IMAGE_ASPECT_COLOR_BIT);
        }

        Buffer vertex_buffer, index_buffer;
        CreateGeometryBuffers(physical_device, device, queue, vertices, indices, &vertex_buffer, &index_buffer);
        buffers_by_device.insert({device, {vertex_buffer, index_buffer}});
    }

    void BindForDraw(VkDevice device, VkCommandBuffer cmdbuf)
    {
        VkDeviceSize offset = 0;
        auto buffers = buffers_by_device.at(device);
        auto vbuf = buffers[VERTEX_BUFFER].buf;
        vkCmdBindVertexBuffers(cmdbuf, 0, 1, &vbuf, &offset);
        vkCmdBindIndexBuffer(cmdbuf, buffers[INDEX_BUFFER].buf, 0, VK_INDEX_TYPE_UINT32);
    }

    void ReleaseDeviceData(VkDevice device)
    {
        for(auto& buffer : buffers_by_device.at(device)) {
            if(buffer.mapped) {
                vkUnmapMemory(device, buffer.mem);
            }
            vkDestroyBuffer(device, buffer.buf, nullptr);
            vkFreeMemory(device, buffer.mem, nullptr);
        }

        buffers_by_device.erase(device);
    }
};

std::tuple<VkImage, VkDeviceMemory> CreateBound2DImage(VkPhysicalDevice physical_device, VkDevice device, VkFormat format, uint32_t width, uint32_t height, VkImageUsageFlags usage_flags, VkImageLayout initial_layout, VkMemoryPropertyFlags properties)
{
    VkImageCreateInfo create_image {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .flags = 0,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = format,
        .extent{width, height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = usage_flags,
        .initialLayout = initial_layout,
    };
    VkImage image;
    VK_CHECK(vkCreateImage(device, &create_image, nullptr, &image));

    VkMemoryRequirements imageMemReqs;
    vkGetImageMemoryRequirements(device, image, &imageMemReqs);
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    uint32_t memoryTypeIndex = GetMemoryTypeIndex(memory_properties, imageMemReqs.memoryTypeBits, properties);
    VkMemoryAllocateInfo allocate {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = imageMemReqs.size,
        .memoryTypeIndex = memoryTypeIndex,
    };
    VkDeviceMemory image_memory;
    VK_CHECK(vkAllocateMemory(device, &allocate, nullptr, &image_memory));
    VK_CHECK(vkBindImageMemory(device, image, image_memory, 0));
    return {image, image_memory};
}

VkSwapchainKHR CreateSwapchain(VkDevice device, VkSurfaceKHR surface, int32_t min_image_count, VkFormat chosen_color_format, VkColorSpaceKHR chosen_color_space, VkPresentModeKHR swapchain_present_mode, uint32_t width, uint32_t height)
{
    // XXX verify present mode with vkGetPhysicalDeviceSurfacePresentModesKHR

    VkSwapchainCreateInfoKHR create {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = static_cast<uint32_t>(min_image_count),
        .imageFormat = chosen_color_format,
        .imageColorSpace = chosen_color_space,
        .imageExtent { width, height },
        .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE,
        .queueFamilyIndexCount = 0,
        .pQueueFamilyIndices = nullptr,
        .preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = swapchain_present_mode,
        .clipped = true,
        .oldSwapchain = VK_NULL_HANDLE,
    };
    VkSwapchainKHR swapchain;
    VK_CHECK(vkCreateSwapchainKHR(device, &create, nullptr, &swapchain));
    return swapchain;
}


std::vector<VkImage> GetSwapchainImages(VkDevice device, VkSwapchainKHR swapchain)
{
    uint32_t swapchain_image_count;
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, nullptr));
    std::vector<VkImage> swapchain_images(swapchain_image_count);
    VK_CHECK(vkGetSwapchainImagesKHR(device, swapchain, &swapchain_image_count, swapchain_images.data()));
    return swapchain_images;
}

struct StorageImage
{
    VkDevice device;
    VkImageLayout layout;
    VkImageView image_view;
    VkDeviceMemory memory;
    VkImage image;
    void Create(VkPhysicalDevice physical_device, VkDevice device_, VkQueue queue, uint32_t queue_family, uint32_t width, uint32_t height, VkFormat format)
    {
        device = device_;
        layout = VK_IMAGE_LAYOUT_UNDEFINED;
        std::tie(image, memory) = CreateBound2DImage(physical_device, device, format, width, height, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_STORAGE_BIT, layout, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        image_view = CreateImageView(device, format, image, VK_IMAGE_ASPECT_COLOR_BIT);

        VkCommandPool command_pool = CreateCommandPool(device, queue_family);
        VkCommandBuffer transition_commands = AllocateCommandBuffer(device, command_pool);

        BeginCommandBuffer(transition_commands);
        TransitionLayout(transition_commands, VK_IMAGE_LAYOUT_GENERAL);
        VK_CHECK(vkEndCommandBuffer(transition_commands));
        FlushCommandBuffer(device, queue, transition_commands);

        vkFreeCommandBuffers(device, command_pool, 1, &transition_commands);
        vkDestroyCommandPool(device, command_pool, nullptr);
    }
    void Destroy()
    {
        vkDestroyImageView(device, image_view, nullptr);
        image_view = VK_NULL_HANDLE;
        vkDestroyImage(device, image, nullptr);
        image = VK_NULL_HANDLE;
        vkFreeMemory(device, memory, nullptr);
        memory = VK_NULL_HANDLE;
    }
    void TransitionLayout(VkCommandBuffer cb, VkImageLayout newLayout)
    {
        TransitionBaseImageLayout(cb, image, layout, newLayout, VK_IMAGE_ASPECT_COLOR_BIT);
        layout = newLayout;
    }
};

namespace VulkanApp
{

bool be_verbose = true;
bool enable_validation = false;

// non-frame stuff - instance, queue, device, etc
VkInstance instance = VK_NULL_HANDLE;
VkPhysicalDevice physical_device = VK_NULL_HANDLE;
VkDevice device = VK_NULL_HANDLE;
uint32_t graphics_queue_family = NO_QUEUE_FAMILY;
VkSurfaceKHR surface = VK_NULL_HANDLE;
VkSwapchainKHR swapchain = VK_NULL_HANDLE;
VkCommandPool command_pool = VK_NULL_HANDLE;
VkQueue queue = VK_NULL_HANDLE;

// In flight rendering stuff
int submission_index = 0;
struct Submission {
    VkCommandBuffer command_buffer { VK_NULL_HANDLE };
    bool draw_completed_fence_submitted { false };
    VkFence draw_completed_fence { VK_NULL_HANDLE };
    VkSemaphore draw_completed_semaphore { VK_NULL_HANDLE };
    VkDescriptorSet rz_descriptor_set { VK_NULL_HANDLE };
    VkDescriptorSet rt_descriptor_set { VK_NULL_HANDLE };
    std::vector<Buffer> rz_uniform_buffers;
    std::vector<Buffer> rt_uniform_buffers;
    StorageImage rt_storage_image;
};
static constexpr int SUBMISSIONS_IN_FLIGHT = 2;
std::vector<Submission> submissions(SUBMISSIONS_IN_FLIGHT);

// per-frame stuff - swapchain image, current layout, indices, fences, semaphores
VkSurfaceFormatKHR chosen_surface_format;
VkPresentModeKHR swapchain_present_mode = VK_PRESENT_MODE_FIFO_KHR;
VkFormat chosen_color_format;
VkFormat chosen_depth_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
uint32_t swapchain_image_count = 3;
struct PerSwapchainImage
{
    VkImageLayout layout;
    VkImage image;
    VkImageView image_view;
    VkFramebuffer framebuffer;
    void TransitionLayout(VkCommandBuffer cb, VkImageLayout newLayout)
    {
        TransitionBaseImageLayout(cb, image, layout, newLayout, VK_IMAGE_ASPECT_COLOR_BIT);
        layout = newLayout;
    }
};
std::vector<PerSwapchainImage> per_swapchainimage;
uint32_t swapchainimage_semaphore_index = 0;
std::vector<VkSemaphore> swapchainimage_semaphores;
VkImage depth_image = VK_NULL_HANDLE;
VkDeviceMemory depth_image_memory = VK_NULL_HANDLE;
VkImageView depth_image_view = VK_NULL_HANDLE;
uint32_t swapchain_width, swapchain_height;

// rasterizer rendering stuff - pipelines, binding & drawing commands
std::vector<Descriptor> rz_descriptors;
VkPipelineLayout rz_pipeline_layout = VK_NULL_HANDLE;
VkDescriptorPool rz_descriptor_pool = VK_NULL_HANDLE;
VkRenderPass render_pass = VK_NULL_HANDLE;
VkPipeline rz_pipeline = VK_NULL_HANDLE;
VkDescriptorSetLayout rz_descriptor_set_layout = VK_NULL_HANDLE;

// ray-tracer rendering stuff - pipelines, binding & drawing commands
PFN_vkCmdTraceRaysKHR CmdTraceRaysKHR;
PFN_vkCreateRayTracingPipelinesKHR CreateRayTracingPipelinesKHR;
PFN_vkGetBufferDeviceAddressKHR GetBufferDeviceAddressKHR;
PFN_vkGetAccelerationStructureBuildSizesKHR GetAccelerationStructureBuildSizesKHR;
PFN_vkCreateAccelerationStructureKHR CreateAccelerationStructureKHR;
PFN_vkCmdBuildAccelerationStructuresKHR CmdBuildAccelerationStructuresKHR;
PFN_vkGetRayTracingShaderGroupHandlesKHR GetRayTracingShaderGroupHandlesKHR;
VkPhysicalDeviceRayTracingPipelinePropertiesKHR  rt_pipeline_properties;
std::vector<Descriptor> rt_descriptors;
VkDescriptorPool rt_descriptor_pool = VK_NULL_HANDLE;
VkDescriptorSetLayout rt_descriptor_set_layout = VK_NULL_HANDLE;
VkPipelineLayout rt_pipeline_layout = VK_NULL_HANDLE;
VkPipeline rt_pipeline = VK_NULL_HANDLE;
Buffer rt_blas_buffer;
VkAccelerationStructureKHR rt_blas;
Buffer rt_tlas_buffer;
VkAccelerationStructureKHR rt_tlas;
Buffer rt_raygen_sbt_buffer, rt_miss_sbt_buffer, rt_hit_sbt_buffer;
std::vector<std::tuple<std::string, VkShaderStageFlagBits, VkRayTracingShaderGroupTypeKHR>> rt_shader_binaries {
    {"testrt.rgen", VK_SHADER_STAGE_RAYGEN_BIT_KHR, VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR},
    {"testrt.rmiss", VK_SHADER_STAGE_MISS_BIT_KHR, VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR},
    {"testrt.rchit", VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR, VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR},
};
int rt_raygen_group_index = -1;
int rt_miss_group_index = -1;
int rt_hit_group_index = -1;



// interaction data


manipulator object_manip;
manipulator light_manip;
manipulator* current_manip;
int buttonPressed = -1;
bool motionReported = true;
double oldMouseX;
double oldMouseY;
float fov = 45;

// geometry data
typedef std::unique_ptr<DrawableShape> DrawableShapePtr;
DrawableShapePtr drawable;

void InitializeInstance()
{
    if (be_verbose) {
        PrintImplementationInformation();
    }
    instance = CreateInstance(enable_validation);
}

void WaitForAllDrawsCompleted()
{
    for(auto& submission: submissions) {
        if(submission.draw_completed_fence_submitted) {
            VK_CHECK(vkWaitForFences(device, 1, &submission.draw_completed_fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
            VK_CHECK(vkResetFences(device, 1, &submission.draw_completed_fence));
            submission.draw_completed_fence_submitted = false;
        }
    }
}

void DestroySwapchainData()
{
    WaitForAllDrawsCompleted();

    for(auto& sema: swapchainimage_semaphores) {
        vkDestroySemaphore(device, sema, nullptr);
        sema = VK_NULL_HANDLE;
    }
    swapchainimage_semaphores.clear();

    vkDestroyImageView(device, depth_image_view, nullptr);
    depth_image_view = VK_NULL_HANDLE;
    vkDestroyImage(device, depth_image, nullptr);
    depth_image = VK_NULL_HANDLE;
    vkFreeMemory(device, depth_image_memory, nullptr);
    depth_image_memory = VK_NULL_HANDLE;

    for(uint32_t i = 0; i < swapchain_image_count; i++) {
        auto& per_image = per_swapchainimage[i];
        vkDestroyImageView(device, per_image.image_view, nullptr);
        per_image.image_view = VK_NULL_HANDLE;
        vkDestroyFramebuffer(device, per_image.framebuffer, nullptr);
        per_image.framebuffer = VK_NULL_HANDLE;
    }
    per_swapchainimage.clear();

    vkDestroySwapchainKHR(device, swapchain, nullptr);
    swapchain = VK_NULL_HANDLE;
}

void CreateSwapchainData(/*VkPhysicalDevice physical_device, VkDevice device, VkSurfaceKHR surface */)
{
    VkSurfaceCapabilitiesKHR surfcaps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surfcaps));
    uint32_t width = surfcaps.currentExtent.width;
    uint32_t height = surfcaps.currentExtent.height;
    swapchain = CreateSwapchain(device, surface, swapchain_image_count, chosen_color_format, chosen_surface_format.colorSpace, swapchain_present_mode, width, height);

    std::tie(depth_image, depth_image_memory) = CreateBound2DImage(physical_device, device, chosen_depth_format, width, height, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    swapchain_width = width;
    swapchain_height = height;

    depth_image_view = CreateImageView(device, chosen_depth_format, depth_image, VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

    std::vector<VkImage> swapchain_images = GetSwapchainImages(device, swapchain);
    assert(swapchain_image_count == swapchain_images.size());
    swapchain_image_count = static_cast<uint32_t>(swapchain_images.size());

    per_swapchainimage.resize(swapchain_image_count);
    for(uint32_t i = 0; i < swapchain_image_count; i++) {
        auto& per_image = per_swapchainimage[i];
        per_image.image = swapchain_images[i];
        per_image.layout = VK_IMAGE_LAYOUT_UNDEFINED;
        per_image.image_view = CreateImageView(device, chosen_color_format, per_swapchainimage[i].image, VK_IMAGE_ASPECT_COLOR_BIT);
        per_image.framebuffer = CreateFramebuffer(device, {per_image.image_view, depth_image_view}, render_pass, width, height);
    }

    swapchainimage_semaphores.resize(swapchain_image_count);
    for(uint32_t i = 0; i < swapchain_image_count; i++) {
        // XXX create a timeline semaphore by chaining after a
        // VkSemaphoreTypeCreateInfo with VkSemaphoreTypeCreateInfo =
        // VK_SEMAPHORE_TYPE_TIMELINE
        VkSemaphoreCreateInfo sema_create {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .flags = 0,
        };
        VK_CHECK(vkCreateSemaphore(device, &sema_create, NULL, &swapchainimage_semaphores[i]));

    }
}

void CreatePipelineDescriptorInfo(const std::vector<Descriptor>& descriptors, VkDescriptorPool& descriptor_pool, VkDescriptorSetLayout& descriptor_set_layout, VkPipelineLayout& pipeline_layout)
{
    std::vector<VkDescriptorSetLayoutBinding> layout_bindings;
    std::vector<VkDescriptorPoolSize> pool_sizes;

    for(const auto& descriptor: descriptors) {
        VkDescriptorSetLayoutBinding binding {
            .binding = descriptor.binding,
            .descriptorType = descriptor.type,
            .descriptorCount = 1,
            .stageFlags = descriptor.stageFlags,
            .pImmutableSamplers = nullptr,
        };
        layout_bindings.push_back(binding);

        pool_sizes.push_back({ descriptor.type, SUBMISSIONS_IN_FLIGHT });
    }

    VkDescriptorPoolCreateInfo create_descriptor_pool {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = 0,
        .maxSets = SUBMISSIONS_IN_FLIGHT,
        .poolSizeCount = static_cast<uint32_t>(pool_sizes.size()),
        .pPoolSizes = pool_sizes.data(),
    };
    VK_CHECK(vkCreateDescriptorPool(device, &create_descriptor_pool, nullptr, &descriptor_pool));

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = 0,
        .bindingCount = static_cast<uint32_t>(layout_bindings.size()),
        .pBindings = layout_bindings.data(),
    };
    VK_CHECK(vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create, nullptr, &descriptor_set_layout));

    VkPipelineLayoutCreateInfo create_layout {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };
    VK_CHECK(vkCreatePipelineLayout(device, &create_layout, nullptr, &pipeline_layout));
}

void CreateRasterizationPipeline()
{
    // XXX Should probe these from shader code somehow
    // XXX at the moment this order is assumed for "struct submission"
    // rz_descriptors Buffer structs, see *_uniforms setting code in DrawFrame
    rz_descriptors.push_back({0, VK_SHADER_STAGE_VERTEX_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sizeof(VertexUniforms)});
    rz_descriptors.push_back({1, VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sizeof(FragmentUniforms)});
    rz_descriptors.push_back({2, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sizeof(ShadingUniforms)});
    rz_descriptors.push_back({3, VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER});

    CreatePipelineDescriptorInfo(rz_descriptors, rz_descriptor_pool, rz_descriptor_set_layout, rz_pipeline_layout);

    // ---------- Graphics pipeline

    VkAttachmentDescription color_attachment_description {
        .flags = 0,
        .format = chosen_color_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };
    VkAttachmentDescription depth_attachment_description {
        .flags = 0,
        .format = chosen_depth_format,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    VkAttachmentDescription attachments[] = { color_attachment_description, depth_attachment_description };

    VkAttachmentReference colorReference { 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL };
    VkAttachmentReference depthReference { 1, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL };

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

    VkSubpassDependency color_attachment_dependency {
        // Image Layout Transition
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT,
        .dependencyFlags = 0,
    };
    VkSubpassDependency depth_attachment_dependency {
        // Depth buffer is shared between swapchain images
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
        .srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        .dependencyFlags = 0,
    };
    VkSubpassDependency attachment_dependencies[] = {
        depth_attachment_dependency,
        color_attachment_dependency,
    };

    VkRenderPassCreateInfo render_pass_create {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .flags = 0,
        .attachmentCount = static_cast<uint32_t>(std::size(attachments)),
        .pAttachments = attachments,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = static_cast<uint32_t>(std::size(attachment_dependencies)),
        .pDependencies = attachment_dependencies,
    };
    VK_CHECK(vkCreateRenderPass(device, &render_pass_create, nullptr, &render_pass));

    // Swapchain and per-swapchainimage stuff
    // Creating the framebuffer requires the renderPass
    CreateSwapchainData(/*physical_device, device, surface*/);

    VkVertexInputBindingDescription vertex_input_binding {
        .binding = 0,
        .stride = sizeof(Vertex),
        .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
    };

    std::vector<VkVertexInputAttributeDescription> vertex_input_attributes = Vertex::GetVertexInputAttributeDescription();

    VkPipelineVertexInputStateCreateInfo vertex_input_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_input_binding,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_input_attributes.size()),
        .pVertexAttributeDescriptions = vertex_input_attributes.data(),
    };

    VkPipelineInputAssemblyStateCreateInfo input_assembly_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
    };

    VkPipelineRasterizationStateCreateInfo rasterization_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = VK_FALSE,
        .rasterizerDiscardEnable = VK_FALSE,
        .polygonMode = VK_POLYGON_MODE_FILL,
        .cullMode = VK_CULL_MODE_NONE, // VK_CULL_MODE_BACK_BIT,
        .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = VK_FALSE,
        .lineWidth = 1.0f,
    };

    VkPipelineColorBlendAttachmentState att_state[] {
        {
            .blendEnable = VK_FALSE,
            .colorWriteMask = 0xf,
        },
    };

    VkPipelineColorBlendStateCreateInfo color_blend_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .attachmentCount = static_cast<uint32_t>(std::size(att_state)),
        .pAttachments = att_state,
    };

    VkStencilOpState keep_always{ .failOp = VK_STENCIL_OP_KEEP, .passOp = VK_STENCIL_OP_KEEP, .compareOp = VK_COMPARE_OP_ALWAYS };
    VkPipelineDepthStencilStateCreateInfo depth_stencil_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
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

    VkPipelineMultisampleStateCreateInfo multisample_state {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
        .pSampleMask = NULL,
    };

    VkDynamicState dynamicStateEnables[]{ VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

    VkPipelineDynamicStateCreateInfo dynamicState {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
        .dynamicStateCount = static_cast<uint32_t>(std::size(dynamicStateEnables)),
        .pDynamicStates = dynamicStateEnables,
    };


    std::vector<std::pair<std::string, VkShaderStageFlagBits>> shader_binaries {
        {"testing.vert", VK_SHADER_STAGE_VERTEX_BIT},
        {"testing.frag", VK_SHADER_STAGE_FRAGMENT_BIT}
    };
    
    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;

    for(const auto& [name, stage]: shader_binaries) {
        std::vector<uint32_t> shader_code = GetFileAsCode(name);
        VkShaderModule shader_module = CreateShaderModule(device, shader_code);
        VkPipelineShaderStageCreateInfo shader_create {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = stage,
            .module = shader_module,
            .pName = "main",
        };
        shader_stages.push_back(shader_create);
    }

    VkGraphicsPipelineCreateInfo create_pipeline {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
        .stageCount = static_cast<uint32_t>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .pVertexInputState = &vertex_input_state,
        .pInputAssemblyState = &input_assembly_state,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterization_state,
        .pMultisampleState = &multisample_state,
        .pDepthStencilState = &depth_stencil_state,
        .pColorBlendState = &color_blend_state,
        .pDynamicState = &dynamicState,
        .layout = rz_pipeline_layout,
        .renderPass = render_pass,
    };

    VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &create_pipeline, nullptr, &rz_pipeline));
}

void CreateRayTracingPipeline()
{
    // XXX Should probe these from shader code somehow

    rt_descriptors.push_back({0, VK_SHADER_STAGE_RAYGEN_BIT_KHR, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR});
    rt_descriptors.push_back({1, VK_SHADER_STAGE_RAYGEN_BIT_KHR, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE});
    // rt_descriptors.push_back({2, VK_SHADER_STAGE_RAYGEN_BIT_KHR, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER});

    // XXX at the moment this order is assumed for "struct submission"
    // rt_uniforms Buffer structs, see *_uniforms setting code in DrawFrame
    rt_descriptors.push_back({2, VK_SHADER_STAGE_RAYGEN_BIT_KHR, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sizeof(RayTracingCamera)});
    // rt_uniforms.push_back({1, VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sizeof(FragmentUniforms)});
    // rt_uniforms.push_back({2, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, sizeof(ShadingUniforms)});

    CreatePipelineDescriptorInfo(rt_descriptors, rt_descriptor_pool, rt_descriptor_set_layout, rt_pipeline_layout);

    std::vector<VkPipelineShaderStageCreateInfo> shader_stages;
    std::vector<VkRayTracingShaderGroupCreateInfoKHR> shader_groups;
    
    int index = 0;
    for(const auto& [name, stage, group_type]: rt_shader_binaries) {
        std::vector<uint32_t> shader_code = GetFileAsCode(name);
        VkShaderModule shader_module = CreateShaderModule(device, shader_code);

        VkPipelineShaderStageCreateInfo shader_create {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = stage,
            .module = shader_module,
            .pName = "main",
        };
        shader_stages.push_back(shader_create);

        uint32_t stage_index = static_cast<uint32_t>(shader_stages.size()) - 1;
        VkRayTracingShaderGroupCreateInfoKHR shader_group {
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            .type = group_type,
            .generalShader = (group_type == VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR) ? stage_index : VK_SHADER_UNUSED_KHR,
            .closestHitShader = (group_type == VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR) ? stage_index : VK_SHADER_UNUSED_KHR,
            .anyHitShader = VK_SHADER_UNUSED_KHR,
            .intersectionShader = VK_SHADER_UNUSED_KHR,
        };
        shader_groups.push_back(shader_group);

        // XXX !!!
        switch(stage) {
            case VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR:
                rt_hit_group_index = index;
                break;
            case VK_SHADER_STAGE_MISS_BIT_KHR:
                rt_miss_group_index = index;
                break;
            case VK_SHADER_STAGE_RAYGEN_BIT_KHR:
                rt_raygen_group_index = index;
                break;
            default:
                abort();
        }
        index++;
    }
    assert(rt_hit_group_index != -1);
    assert(rt_raygen_group_index != -1);
    assert(rt_miss_group_index != -1);

    VkRayTracingPipelineCreateInfoKHR create_pipeline {
        .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
        .pNext = nullptr,
        .flags = 0,
        .stageCount = static_cast<uint32_t>(shader_stages.size()),
        .pStages = shader_stages.data(),
        .groupCount = static_cast<uint32_t>(shader_groups.size()),
        .pGroups = shader_groups.data(),
        .maxPipelineRayRecursionDepth = 1,
        .pLibraryInfo = nullptr,
        .pLibraryInterface = nullptr,
        .pDynamicState = nullptr,
        .layout = rt_pipeline_layout,
        .basePipelineHandle = VK_NULL_HANDLE,
        .basePipelineIndex = 0,
    };

    VK_CHECK(CreateRayTracingPipelinesKHR(device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &create_pipeline, nullptr, &rt_pipeline));

    uint32_t handle_size_aligned = align(rt_pipeline_properties.shaderGroupHandleSize, rt_pipeline_properties.shaderGroupHandleAlignment);
    uint32_t group_count = static_cast<uint32_t>(shader_groups.size());
    uint32_t sbt_size = group_count * handle_size_aligned;

    std::vector<uint8_t> shader_handle_storage(sbt_size);
    VK_CHECK(GetRayTracingShaderGroupHandlesKHR(device, rt_pipeline, 0, group_count, sbt_size, shader_handle_storage.data()));

    std::vector<std::tuple<Buffer&,int>> sbt_buffers = {
        {rt_raygen_sbt_buffer, rt_raygen_group_index},
        {rt_hit_sbt_buffer, rt_hit_group_index},
        {rt_miss_sbt_buffer, rt_miss_group_index},
    };

    for(const auto& [buffer, group_index]: sbt_buffers) {
        buffer.Create(physical_device, device, handle_size_aligned, VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT); // | VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VK_CHECK(vkMapMemory(device, buffer.mem, 0, handle_size_aligned, 0, &buffer.mapped));
        memcpy(buffer.mapped, shader_handle_storage.data() + group_index * handle_size_aligned, handle_size_aligned);
        vkUnmapMemory(device, buffer.mem);
    }
}

void CreatePerSubmissionDescriptors(VkDescriptorPool descriptor_pool, VkDescriptorSetLayout descriptor_set_layout, VkDescriptorSet& descriptor_set, const std::vector<Descriptor>& descriptors, std::vector<Buffer>& uniform_buffers, const StorageImage& storage_image, VkAccelerationStructureKHR acceleration_structure)
{
    VkDescriptorSetAllocateInfo allocate_descriptor_set {
	.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
	    .descriptorPool = descriptor_pool,
	    .descriptorSetCount = 1,
	    .pSetLayouts = &descriptor_set_layout,
    };
    VK_CHECK(vkAllocateDescriptorSets(device, &allocate_descriptor_set, &descriptor_set));

    for(const auto& descriptor: descriptors) {
        VkWriteDescriptorSet write {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = nullptr,
            .dstSet = descriptor_set,
            .dstBinding = descriptor.binding,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = descriptor.type,
        };

        if(descriptor.type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
            Buffer uniform_buffer;

            uniform_buffer.Create(physical_device, device, descriptor.size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            VK_CHECK(vkMapMemory(device, uniform_buffer.mem, 0, descriptor.size, 0, &uniform_buffer.mapped));
            uniform_buffers.push_back(uniform_buffer);

            VkDescriptorBufferInfo buffer_info { .buffer = uniform_buffer.buf, .offset = 0, .range = VK_WHOLE_SIZE};

            write.pBufferInfo = &buffer_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

        } else if(descriptor.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {

            // XXX!!! assumes that there is only one texture, the drawble->textureSampler!
            // rewrite this so that there's a generalized "Descriptor" thingie also containing values

            VkDescriptorImageInfo image_info {
                .sampler = drawable->textureSampler, // XXX AHHHH
                .imageView = drawable->textureImageView, // XXX AHHHH
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };

            write.pImageInfo = &image_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

        } else if(descriptor.type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {

            // XXX!!! assumes that there is only one storageImage
            // rewrite this so that there's a generalized "Descriptor" thingie also containing values
            VkDescriptorImageInfo image_info {
                .imageView = storage_image.image_view, // XXX AHHHH
                .imageLayout = storage_image.layout,
            };

            write.pImageInfo = &image_info;
            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);

        } else if(descriptor.type == VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR) {

            // XXX!!! assumes that there is only one AS
            // rewrite this so that there's a generalized "Descriptor" thingie also containing values
            VkWriteDescriptorSetAccelerationStructureKHR write_as {
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                .pNext = nullptr,
                .accelerationStructureCount = 1,
                .pAccelerationStructures = &acceleration_structure, // XXX AHHH
            };
            write.pNext = &write_as;

            vkUpdateDescriptorSets(device, 1, &write, 0, nullptr);
        }
    }
}

void CreatePerSubmissionData()
{
    for(uint32_t i = 0; i < SUBMISSIONS_IN_FLIGHT; i++) {

        auto& submission = submissions[i];

        const VkCommandBufferAllocateInfo allocate {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        VK_CHECK(vkAllocateCommandBuffers(device, &allocate, &submission.command_buffer));

        VkFenceCreateInfo fence_create {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = 0,
        };
        VK_CHECK(vkCreateFence(device, &fence_create, nullptr, &submission.draw_completed_fence));
        submission.draw_completed_fence_submitted = false;

        VkSemaphoreCreateInfo sema_create {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .flags = 0,
        };
        VK_CHECK(vkCreateSemaphore(device, &sema_create, NULL, &submission.draw_completed_semaphore));

        CreatePerSubmissionDescriptors(rz_descriptor_pool, rz_descriptor_set_layout, submission.rz_descriptor_set, rz_descriptors, submission.rz_uniform_buffers, {}, VK_NULL_HANDLE);

        submission.rt_storage_image.Create(physical_device, device, queue, graphics_queue_family, 512, 512, chosen_color_format);

        CreatePerSubmissionDescriptors(rt_descriptor_pool, rt_descriptor_set_layout, submission.rt_descriptor_set, rt_descriptors, submission.rt_uniform_buffers, submission.rt_storage_image, rt_tlas);
    }
}


void InitializeState(uint32_t specified_gpu)
{
    // non-frame stuff
    physical_device = ChoosePhysicalDevice(instance, specified_gpu);

    uint32_t formatCount;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, nullptr));
    std::vector<VkSurfaceFormatKHR> surfaceFormats(formatCount);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device, surface, &formatCount, surfaceFormats.data()));

    VkSurfaceCapabilitiesKHR surfcaps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &surfcaps));

    chosen_surface_format = PickSurfaceFormat(surfaceFormats);
    chosen_color_format = chosen_surface_format.format;

    graphics_queue_family = FindQueueFamily(physical_device, VK_QUEUE_GRAPHICS_BIT);
    if(graphics_queue_family == NO_QUEUE_FAMILY) {
        fprintf(stderr, "couldn't find a graphics queue\n");
        abort();
    }

    if(be_verbose) {
        VkPhysicalDeviceMemoryProperties memory_properties;
        vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
        PrintDeviceInformation(physical_device, memory_properties);
    }

    std::vector<const char*> device_extensions;

    device_extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

#ifdef PLATFORM_MACOS
    device_extensions.push_back("VK_KHR_portability_subset" /* VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME */);
#endif

    device_extensions.insert(device_extensions.end(), {
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME,
        VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
        VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME,
        VK_KHR_RAY_QUERY_EXTENSION_NAME,
        VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
        });

    if (be_verbose) {
        for (const auto& e : device_extensions) {
            printf("asked for %s\n", e);
        }
    }

    // Enable features required for ray tracing using feature chaining via pNext		
    VkPhysicalDeviceBufferDeviceAddressFeatures enable_buffer_device_address{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
        .bufferDeviceAddress = VK_TRUE,
    };

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR enable_ray_tracing_pipeline{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
        .pNext = &enable_buffer_device_address,
        .rayTracingPipeline = VK_TRUE,
    };

    VkPhysicalDeviceAccelerationStructureFeaturesKHR enable_acceleration_structure{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        .pNext = &enable_ray_tracing_pipeline,
        .accelerationStructure = VK_TRUE,
    };

    VkPhysicalDeviceDescriptorIndexingFeaturesEXT enable_descriptor_indexing{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES_EXT,
        .pNext = &enable_acceleration_structure,
        .shaderSampledImageArrayNonUniformIndexing = VK_TRUE,
    };

    device = CreateDevice(physical_device, device_extensions, graphics_queue_family, &enable_descriptor_indexing);

    // Ray-tracing extensions and properties
    CmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(device, "vkCmdTraceRaysKHR");
    CreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR");
    CreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(device, "vkCreateRayTracingPipelinesKHR");
    GetBufferDeviceAddressKHR = (PFN_vkGetBufferDeviceAddressKHR)vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddressKHR");
    GetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(device, "vkGetAccelerationStructureBuildSizesKHR");
    CreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(device, "vkCreateAccelerationStructureKHR");
    CmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(device, "vkCmdBuildAccelerationStructuresKHR");
    GetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(device, "vkGetRayTracingShaderGroupHandlesKHR");

    rt_pipeline_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 device_properties2{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &rt_pipeline_properties,
    };
    vkGetPhysicalDeviceProperties2(physical_device, &device_properties2);

    // Queues, CommmandPools, etc
    vkGetDeviceQueue(device, graphics_queue_family, 0, &queue);

    VkCommandPoolCreateInfo create_command_pool {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphics_queue_family,
    };
    VK_CHECK(vkCreateCommandPool(device, &create_command_pool, nullptr, &command_pool));

    drawable->CreateDeviceData(physical_device, device, queue);

    CreateRasterizationPipeline();

    VkCommandBuffer build_commands = AllocateCommandBuffer(device, command_pool);

    // bottom-level acceleration structure (BLAS)

    VkTransformMatrixKHR blas_transform {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };
    Buffer blas_transform_buffer;
    blas_transform_buffer.Create(physical_device, device, sizeof(blas_transform), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkMapMemory(device, blas_transform_buffer.mem, 0, sizeof(blas_transform), 0, &blas_transform_buffer.mapped));
    memcpy(blas_transform_buffer.mapped, &blas_transform, sizeof(blas_transform));
    vkUnmapMemory(device, blas_transform_buffer.mem);
    blas_transform_buffer.mapped = nullptr;

    VkAccelerationStructureGeometryTrianglesDataKHR triangles_data {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
        .vertexFormat = VK_FORMAT_R32G32B32_SFLOAT,
        .vertexData = drawable->buffers_by_device[device][DrawableShape::VERTEX_BUFFER].GetDeviceAddress(VulkanApp::GetBufferDeviceAddressKHR),
        .vertexStride = sizeof(Vertex),
        .maxVertex = static_cast<uint32_t>(drawable->vertices.size() - 1),
        .indexType = VK_INDEX_TYPE_UINT32,
        .indexData = drawable->buffers_by_device[device][DrawableShape::INDEX_BUFFER].GetDeviceAddress(VulkanApp::GetBufferDeviceAddressKHR),
        .transformData = blas_transform_buffer.GetDeviceAddress(VulkanApp::GetBufferDeviceAddressKHR),
    };

    VkAccelerationStructureGeometryKHR blas_geometry {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
        .geometry = triangles_data,
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };

    VkAccelerationStructureBuildGeometryInfoKHR blas_geometry_info {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .srcAccelerationStructure = VK_NULL_HANDLE,
        .dstAccelerationStructure = VK_NULL_HANDLE,
        .geometryCount = 1,
        .pGeometries = &blas_geometry,
        .ppGeometries = nullptr,
    };

    uint32_t triangle_count = static_cast<uint32_t>(drawable->indices.size() / 3);

    uint32_t blas_primitive_counts = triangle_count;
    VkAccelerationStructureBuildSizesInfoKHR blas_sizes {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    VulkanApp::GetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blas_geometry_info, &blas_primitive_counts, &blas_sizes);

    Buffer blas_scratch;
    blas_scratch.Create(physical_device, device, blas_sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    rt_blas_buffer.Create(physical_device, device, blas_sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkAccelerationStructureCreateInfoKHR create_blas {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .pNext = nullptr,
        // .createFlags,
        .buffer = rt_blas_buffer.buf,
        .offset = 0,
        .size = blas_sizes.accelerationStructureSize,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
    };
    VK_CHECK(VulkanApp::CreateAccelerationStructureKHR(device, &create_blas, nullptr, &rt_blas));

    VkAccelerationStructureBuildRangeInfoKHR blas_range {
        .primitiveCount = triangle_count,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0,
    };
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> blas_ranges = { &blas_range };

    BeginCommandBuffer(build_commands);
    blas_geometry_info.dstAccelerationStructure = rt_blas;
    blas_geometry_info.scratchData.deviceAddress = blas_scratch.GetDeviceAddress(VulkanApp::GetBufferDeviceAddressKHR);
    VulkanApp::CmdBuildAccelerationStructuresKHR(build_commands, 1, &blas_geometry_info, blas_ranges.data());

    // memory barrier on BLAS buffer
    VkBufferMemoryBarrier blas_buffer_barrier {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .pNext = nullptr,
        .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = rt_blas_buffer.buf,
        .offset = 0,
        .size = blas_sizes.accelerationStructureSize,
    };

    vkCmdPipelineBarrier(build_commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 1, &blas_buffer_barrier, 0, nullptr);

    // top-level acceleration structure (TLAS)

    VkTransformMatrixKHR instance_transform {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f
    };

    VkAccelerationStructureInstanceKHR tlas_instance {
        .transform = instance_transform,
        .instanceCustomIndex = 0,
        .mask = 0xFF,
        .instanceShaderBindingTableRecordOffset = 0,
        .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
        .accelerationStructureReference = rt_blas_buffer.GetDeviceAddress(VulkanApp::GetBufferDeviceAddressKHR),
    };

    Buffer instances_buffer;
    instances_buffer.Create(physical_device, device, sizeof(tlas_instance), VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VK_CHECK(vkMapMemory(device, instances_buffer.mem, 0, sizeof(tlas_instance), 0, &instances_buffer.mapped));
    memcpy(instances_buffer.mapped, &tlas_instance, sizeof(tlas_instance));
    vkUnmapMemory(device, instances_buffer.mem);
    instances_buffer.mapped = nullptr;

    VkAccelerationStructureGeometryInstancesDataKHR instances_data {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
        .arrayOfPointers = VK_FALSE,
        .data = instances_buffer.GetDeviceAddress(VulkanApp::GetBufferDeviceAddressKHR),
    };

    VkAccelerationStructureGeometryKHR tlas_geometry {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
        .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
        .flags = VK_GEOMETRY_OPAQUE_BIT_KHR,
    };
    tlas_geometry.geometry.instances = instances_data;

    VkAccelerationStructureBuildGeometryInfoKHR tlas_geometry_info {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
        .mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
        .srcAccelerationStructure = VK_NULL_HANDLE,
        .dstAccelerationStructure = VK_NULL_HANDLE,
        .geometryCount = 1,
        .pGeometries = &tlas_geometry,
        .ppGeometries = nullptr,
    };

    uint32_t instance_count = 1;

    uint32_t tlas_primitive_counts = instance_count;
    VkAccelerationStructureBuildSizesInfoKHR tlas_sizes {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
    };
    VulkanApp::GetAccelerationStructureBuildSizesKHR(device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlas_geometry_info, &tlas_primitive_counts, &tlas_sizes);

    Buffer tlas_scratch;
    tlas_scratch.Create(physical_device, device, tlas_sizes.buildScratchSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    rt_tlas_buffer.Create(physical_device, device, tlas_sizes.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkAccelerationStructureCreateInfoKHR create_tlas {
        .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
        .pNext = nullptr,
        // .createFlags,
        .buffer = rt_tlas_buffer.buf,
        .offset = 0,
        .size = tlas_sizes.accelerationStructureSize,
        .type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
    };
    VK_CHECK(VulkanApp::CreateAccelerationStructureKHR(device, &create_tlas, nullptr, &rt_tlas));

    VkAccelerationStructureBuildRangeInfoKHR tlas_range {
        .primitiveCount = instance_count,
        .primitiveOffset = 0,
        .firstVertex = 0,
        .transformOffset = 0,
    };
    std::vector<VkAccelerationStructureBuildRangeInfoKHR*> tlas_ranges = { &tlas_range };

    tlas_geometry_info.dstAccelerationStructure = rt_tlas;
    tlas_geometry_info.scratchData.deviceAddress = tlas_scratch.GetDeviceAddress(VulkanApp::GetBufferDeviceAddressKHR);
    VulkanApp::CmdBuildAccelerationStructuresKHR(build_commands, 1, &tlas_geometry_info, tlas_ranges.data());

    VK_CHECK(vkEndCommandBuffer(build_commands));

    FlushCommandBuffer(device, queue, build_commands);

    vkFreeCommandBuffers(device, command_pool, 1, &build_commands);

    CreateRayTracingPipeline();

    CreatePerSubmissionData();
}

void Cleanup()
{
    WaitForAllDrawsCompleted();
    drawable->ReleaseDeviceData(device);
}

void DrawFrameRT([[maybe_unused]] GLFWwindow *window)
{
    auto& submission = submissions[submission_index];

    if(submission.draw_completed_fence_submitted) {
        // printf("waiting on fence %p\n", (void*)submission.draw_completed_fence);
        VK_CHECK(vkWaitForFences(device, 1, &submission.draw_completed_fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
        // printf("got fence %p\n", (void*)submission.draw_completed_fence);
        VK_CHECK(vkResetFences(device, 1, &submission.draw_completed_fence));
        // printf("reset fence %p\n", (void*)submission.draw_completed_fence);
        submission.draw_completed_fence_submitted = false;
    }

    mat4f modelview = inverse(object_manip.m_matrix);

    float nearClip = .1f; // XXX - gSceneManip->m_translation[2] - gSceneManip->m_reference_size;
    float farClip = 1000.0; // XXX - gSceneManip->m_translation[2] + gSceneManip->m_reference_size;
    float frustumTop = tan(fov / 180.0f * 3.14159f / 2) * nearClip;
    float frustumBottom = -frustumTop;
    float frustumRight = frustumTop * swapchain_width / swapchain_height;
    float frustumLeft = -frustumRight;
    mat4f projection = inverse(mat4f::frustum(frustumLeft, frustumRight, frustumTop, frustumBottom, nearClip, farClip));

    RayTracingCamera* rtcamera_uniforms = static_cast<RayTracingCamera*>(submission.rt_uniform_buffers[0].mapped);
    rtcamera_uniforms->modelviewInverse = modelview;
    rtcamera_uniforms->projectionInverse = projection.m_v;

    VkResult result;
    uint32_t swapchain_index;
    while((result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, swapchainimage_semaphores[swapchainimage_semaphore_index], VK_NULL_HANDLE, &swapchain_index)) != VK_SUCCESS) {
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            DestroySwapchainData();
            CreateSwapchainData(/*physical_device, device, surface */);
            abort(); //  XXX recreate storage image
        } else {
	    std::cerr << "VkResult from vkAcquireNextImageKHR was " << result << " at line " << __LINE__ << "\n";
            exit(EXIT_FAILURE);
        }
    }
    auto& per_image = per_swapchainimage[swapchain_index];

    auto cb = submission.command_buffer;

    VK_CHECK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo begin {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0,
        .pInheritanceInfo = nullptr,
    };
    VK_CHECK(vkBeginCommandBuffer(cb, &begin));

    submission.rt_storage_image.TransitionLayout(cb, VK_IMAGE_LAYOUT_GENERAL);

    uint32_t handle_size_aligned = align(rt_pipeline_properties.shaderGroupHandleSize, rt_pipeline_properties.shaderGroupHandleAlignment);
 
    VkStridedDeviceAddressRegionKHR raygen_sbt {
        .deviceAddress = rt_raygen_sbt_buffer.GetDeviceAddress(VulkanApp::GetBufferDeviceAddressKHR),
        .stride = handle_size_aligned,
        .size = handle_size_aligned,
    };
    VkStridedDeviceAddressRegionKHR miss_sbt {
        .deviceAddress = rt_miss_sbt_buffer.GetDeviceAddress(VulkanApp::GetBufferDeviceAddressKHR),
        .stride = handle_size_aligned,
        .size = handle_size_aligned,
    };
    VkStridedDeviceAddressRegionKHR hit_sbt {
        .deviceAddress = rt_hit_sbt_buffer.GetDeviceAddress(VulkanApp::GetBufferDeviceAddressKHR),
        .stride = handle_size_aligned,
        .size = handle_size_aligned,
    };
    VkStridedDeviceAddressRegionKHR callable_sbt {
        .deviceAddress = 0,
        .stride = 0,
        .size = 0,
    };

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline);

    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, rt_pipeline_layout, 0, 1, &submission.rt_descriptor_set, 0, NULL);

    VulkanApp::CmdTraceRaysKHR(
        cb,
        &raygen_sbt,
        &miss_sbt,
        &hit_sbt,
        &callable_sbt,
        swapchain_width,
        swapchain_height,
        1);

    VkImageSubresourceRange subresource_range = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    submission.rt_storage_image.TransitionLayout(cb, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    per_image.TransitionLayout(cb, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

    VkImageCopy region {
        .srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .srcOffset = { 0, 0, 0 },
        .dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 },
        .dstOffset = { 0, 0, 0 },
        .extent = { swapchain_width, swapchain_height, 1 },
    };

    vkCmdCopyImage(cb, submission.rt_storage_image.image, submission.rt_storage_image.layout, per_image.image, per_image.layout, 1, &region);
    per_image.TransitionLayout(cb, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

    VK_CHECK(vkEndCommandBuffer(cb));

    VkPipelineStageFlags waitdststagemask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &swapchainimage_semaphores[swapchainimage_semaphore_index],
        .pWaitDstStageMask = &waitdststagemask,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &submission.draw_completed_semaphore,
    };
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, submission.draw_completed_fence));
    //printf("submitted with completion fence %p\n", (void*)submission.draw_completed_fence);
    submission.draw_completed_fence_submitted = true;

    // XXX debug
    if(0) {
        printf("waiting on fence %p\n", (void*)submission.draw_completed_fence);
        VK_CHECK(vkWaitForFences(device, 1, &submission.draw_completed_fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
        printf("got fence %p\n", (void*)submission.draw_completed_fence);
        VK_CHECK(vkResetFences(device, 1, &submission.draw_completed_fence));
        printf("reset fence %p\n", (void*)submission.draw_completed_fence);
        submission.draw_completed_fence_submitted = false;
    }

    // Enqueue a Present
    VkPresentInfoKHR present {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &submission.draw_completed_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &swapchain_index,
        .pResults = nullptr,
    };
    result = vkQueuePresentKHR(queue, &present);
    if(result != VK_SUCCESS) {
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            DestroySwapchainData();
            CreateSwapchainData(/*physical_device, device, surface */);
            abort(); //  XXX recreate storage image
        } else {
	    std::cerr << "VkResult from vkQueuePresentKHR was " << result << " at line " << __LINE__ << "\n";
            exit(EXIT_FAILURE);
        }
    }

    submission_index = (submission_index + 1) % submissions.size();
    swapchainimage_semaphore_index = (swapchainimage_semaphore_index + 1) % swapchainimage_semaphores.size();
}

void DrawFrame([[maybe_unused]] GLFWwindow *window)
{
    auto& submission = submissions[submission_index];

    if(submission.draw_completed_fence_submitted) {
        VK_CHECK(vkWaitForFences(device, 1, &submission.draw_completed_fence, VK_TRUE, DEFAULT_FENCE_TIMEOUT));
        VK_CHECK(vkResetFences(device, 1, &submission.draw_completed_fence));
        submission.draw_completed_fence_submitted = false;
    }

    mat4f modelview = object_manip.m_matrix;
    mat4f modelview_3x3 = modelview;
    modelview_3x3.m_v[12] = 0.0f; modelview_3x3.m_v[13] = 0.0f; modelview_3x3.m_v[14] = 0.0f;
    mat4f modelview_normal = inverse(transpose(modelview_3x3));

    float nearClip = .1f; // XXX - gSceneManip->m_translation[2] - gSceneManip->m_reference_size;
    float farClip = 1000.0; // XXX - gSceneManip->m_translation[2] + gSceneManip->m_reference_size;
    float frustumTop = tan(fov / 180.0f * 3.14159f / 2) * nearClip;
    float frustumBottom = -frustumTop;
    float frustumRight = frustumTop * swapchain_width / swapchain_height;
    float frustumLeft = -frustumRight;
    mat4f projection = mat4f::frustum(frustumLeft, frustumRight, frustumTop, frustumBottom, nearClip, farClip);

    VertexUniforms* vertex_uniforms = static_cast<VertexUniforms*>(submission.rz_uniform_buffers[0].mapped);
    vertex_uniforms->modelview = modelview;
    vertex_uniforms->modelview_normal = modelview_normal;
    vertex_uniforms->projection = projection.m_v;

    vec4 light_position{1000, 1000, 1000, 0};
    vec3 light_color{1, 1, 1};

    light_position = light_position * light_manip.m_matrix;

    FragmentUniforms* fragment_uniforms = static_cast<FragmentUniforms*>(submission.rz_uniform_buffers[1].mapped);
    fragment_uniforms->light_position[0] = light_position[0];
    fragment_uniforms->light_position[1] = light_position[1];
    fragment_uniforms->light_position[2] = light_position[2];
    fragment_uniforms->light_color = light_color;

    ShadingUniforms* shading_uniforms = static_cast<ShadingUniforms*>(submission.rz_uniform_buffers[2].mapped);
    shading_uniforms->specular_color.set(drawable->specular_color); // XXX drops specular_color[3]
    shading_uniforms->shininess = drawable->shininess;

    VkResult result;
    uint32_t swapchain_index;
    while((result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, swapchainimage_semaphores[swapchainimage_semaphore_index], VK_NULL_HANDLE, &swapchain_index)) != VK_SUCCESS) {
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            DestroySwapchainData();
            CreateSwapchainData(/*physical_device, device, surface */);
        } else {
	    std::cerr << "VkResult from vkAcquireNextImageKHR was " << result << " at line " << __LINE__ << "\n";
            exit(EXIT_FAILURE);
        }
    }
    auto& per_image = per_swapchainimage[swapchain_index];

    auto cb = submission.command_buffer;

    VK_CHECK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo begin {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = 0, // VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = nullptr,
    };
    VK_CHECK(vkBeginCommandBuffer(cb, &begin));
    const VkClearValue clearValues [2] {
        {.color {.float32 {0.1f, 0.1f, 0.2f, 1.0f}}},
        {.depthStencil = {1.0f, 0}},
    };
    VkRenderPassBeginInfo beginRenderpass {
        .sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
        .renderPass = render_pass,
        .framebuffer = per_image.framebuffer,
        .renderArea = {{0, 0}, {swapchain_width, swapchain_height}},
        .clearValueCount = static_cast<uint32_t>(std::size(clearValues)),
        .pClearValues = clearValues,
    };
    vkCmdBeginRenderPass(cb, &beginRenderpass, VK_SUBPASS_CONTENTS_INLINE);

    // 6. Bind the graphics pipeline state
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rz_pipeline);

    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rz_pipeline_layout, 0, 1, &submission.rz_descriptor_set, 0, NULL);

    // 9. Set viewport and scissor parameters
    VkViewport viewport {
        .x = 0,
        .y = 0,
        .width = (float)swapchain_width,
        .height = (float)swapchain_height,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };
    vkCmdSetViewport(cb, 0, 1, &viewport);

    VkRect2D scissor {
        .offset{0, 0},
        .extent{swapchain_width, swapchain_height}};
    vkCmdSetScissor(cb, 0, 1, &scissor);

    drawable->BindForDraw(device, cb);
    vkCmdDrawIndexed(cb, drawable->triangleCount * 3, 1, 0, 0, 0);

    vkCmdEndRenderPass(cb);
    VK_CHECK(vkEndCommandBuffer(cb));

    VkPipelineStageFlags waitdststagemask = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo submit {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &swapchainimage_semaphores[swapchainimage_semaphore_index],
        .pWaitDstStageMask = &waitdststagemask,
        .commandBufferCount = 1,
        .pCommandBuffers = &cb,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &submission.draw_completed_semaphore,
    };
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, submission.draw_completed_fence));
    submission.draw_completed_fence_submitted = true;

    // 13. Present the rendered result
    VkPresentInfoKHR present {
        .sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &submission.draw_completed_semaphore,
        .swapchainCount = 1,
        .pSwapchains = &swapchain,
        .pImageIndices = &swapchain_index,
        .pResults = nullptr,
    };
    result = vkQueuePresentKHR(queue, &present);
    if(result != VK_SUCCESS) {
        if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            DestroySwapchainData();
            CreateSwapchainData(/*physical_device, device, surface */);
        } else {
	    std::cerr << "VkResult from vkQueuePresentKHR was " << result << " at line " << __LINE__ << "\n";
            exit(EXIT_FAILURE);
        }
    }

    submission_index = (submission_index + 1) % submissions.size();
    swapchainimage_semaphore_index = (swapchainimage_semaphore_index + 1) % swapchainimage_semaphores.size();
}

}

static void ErrorCallback([[maybe_unused]] int error, const char* description)
{
    fprintf(stderr, "GLFW: %s\n", description);
}

static void KeyCallback(GLFWwindow *window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods)
{
    using namespace VulkanApp;

    if(action == GLFW_PRESS) {
        switch(key) {
            case 'W':
                break;

            case 'R':
                current_manip = &object_manip;
                object_manip.m_mode = manipulator::ROTATE;
                break;

            case 'O':
                current_manip = &object_manip;
                object_manip.m_mode = manipulator::ROLL;
                break;

            case 'X':
                current_manip = &object_manip;
                object_manip.m_mode = manipulator::SCROLL;
                break;

            case 'Z':
                current_manip = &object_manip;
                object_manip.m_mode = manipulator::DOLLY;
                break;

            case 'L':
                current_manip = &light_manip;
                light_manip.m_mode = manipulator::ROTATE;
                break;

            case 'Q': case GLFW_KEY_ESCAPE: case '\033':
                glfwSetWindowShouldClose(window, GL_TRUE);
                break;
        }
    }
}

static void ButtonCallback(GLFWwindow *window, int b, int action, [[maybe_unused]] int mods)
{
    using namespace VulkanApp;

    double x, y;
    glfwGetCursorPos(window, &x, &y);

    if(b == GLFW_MOUSE_BUTTON_1 && action == GLFW_PRESS) {
        buttonPressed = 1;
	oldMouseX = x;
	oldMouseY = y;
    } else {
        buttonPressed = -1;
    }
}

static void MotionCallback(GLFWwindow *window, double x, double y)
{
    using namespace VulkanApp;

    // glfw/glfw#103
    // If no motion has been reported yet, we catch the first motion
    // reported and store the current location
    if(!motionReported) {
        motionReported = true;
        oldMouseX = x;
        oldMouseY = y;
    }

    double dx, dy;

    dx = x - oldMouseX;
    dy = y - oldMouseY;

    oldMouseX = x;
    oldMouseY = y;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    if(buttonPressed == 1) {
        current_manip->move(static_cast<float>(dx / width), static_cast<float>(dy / height));
    }
}

static void ScrollCallback(GLFWwindow *window, double dx, double dy)
{
    using namespace VulkanApp;

    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    current_manip->move(static_cast<float>(dx / width), static_cast<float>(dy / height));
}

bool ParseTriSrc(FILE *fp, std::vector<Vertex>& vertices, std::vector<uint32_t>& indices, std::string& texture_name, float specular_color[4], float& shininess)
{
    char texture_name_c_str[512];
    char tag_name_c_str[512];

    // XXX each triangle can reference a texture, but I ignore all but the last in this hacked parser
    while(fscanf(fp,"\"%[^\"]\"", texture_name_c_str) == 1) {
	if(fscanf(fp,"%s ", tag_name_c_str) != 1) {
            // XXX each triangle can reference a tag name, but I ignore all but the last in this hacked parser
            // It really can be ignored in a sense because it's a hint about part assemblies, spatial locality, etc
	    fprintf(stderr, "couldn't read tag name\n");
	    return false;
	}

        // XXX each triangle can reference object specular and shininess, but I ignore all but the last in this hacked parser
	if(fscanf(fp,"%f %f %f %f %f ", &specular_color[0], &specular_color[1],
	    &specular_color[2], &specular_color[3], &shininess) != 5) {
	    fprintf(stderr, "couldn't read specular properties\n");
	    return false;
	}

	if(shininess > 0 && shininess < 1) {
	    // XXX I forgot what shininess units are!
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
            indices.push_back(static_cast<uint32_t>(vertices.size()));
            vertices.push_back(Vertex(v, n, c, t));
        }

        // old code from another project for reference
        // MATERIAL mtl(texture_name, specular_color, shininess);
        // sets.get_triangle_set(tag_name, mtl).add_triangle(verts[0], verts[1], verts[2]);
    }

    texture_name = texture_name_c_str;

    return true;
}


// very old code
static void skipComments(FILE *fp)
{
    int c;

    while((c = fgetc(fp)) == '#')
        while((c = fgetc(fp)) != '\n');
    ungetc(c, fp);
}


int pnmRead(FILE *file, int *w, int *h, float **pixels)
{
    unsigned char	dummyByte;
    int			i;
    float		max = 1;
    char		token;
    int			width, height;
    float		*rgbPixels;

    fscanf(file, " ");

    skipComments(file);

    if(fscanf(file, "P%c ", &token) != 1) {
         fprintf(stderr, "pnmRead: Had trouble reading PNM tag\n");
	 return 0;
    }

    skipComments(file);

    if(fscanf(file, "%d ", &width) != 1) {
         fprintf(stderr, "pnmRead: Had trouble reading PNM width\n");
	 return 0;
    }

    skipComments(file);

    if(fscanf(file, "%d ", &height) != 1) {
         fprintf(stderr, "pnmRead: Had trouble reading PNM height\n");
	 return 0;
    }

    skipComments(file);

    if(token != '1' && token != '4') {
        if(fscanf(file, "%f", &max) != 1) {
             fprintf(stderr, "pnmRead: Had trouble reading PNM max value\n");
	     return 0;
        }
    }

    rgbPixels = static_cast<float*>(malloc(width * height * 4 * sizeof(float)));
    if(rgbPixels == NULL) {
         fprintf(stderr, "pnmRead: Couldn't allocate %zd bytes\n", width * height * 4 * sizeof(float));
         fprintf(stderr, "pnmRead: (For a %d by %d image)\n", width, height);
	 return 0;
    }

    if(token != '4') {
	skipComments(file);
    }

    if(token != '4') { 
        // ??
        fread(&dummyByte, 1, 1, file);	/* chuck white space */
    }

    if(token == '1')
    {
	for(i = 0; i < width * height; i++)
	{
	    int pixel;
	    fscanf(file, "%d", &pixel);
	    pixel = 1 - pixel;
	    rgbPixels[i * 4 + 0] = (float)pixel;
	    rgbPixels[i * 4 + 1] = (float)pixel;
	    rgbPixels[i * 4 + 2] = (float)pixel;
	    rgbPixels[i * 4 + 3] = 1.0;
	}
    }
    else if(token == '2')
    {
	for(i = 0; i < width * height; i++)
	{
	    int pixel;
	    fscanf(file, "%d", &pixel);
	    rgbPixels[i * 4 + 0] = pixel / max;
	    rgbPixels[i * 4 + 1] = pixel / max;
	    rgbPixels[i * 4 + 2] = pixel / max;
	    rgbPixels[i * 4 + 3] = 1.0;
	}
    }
    else if(token == '3')
    {
	for(i = 0; i < width * height; i++)
	{
	    int r, g, b;
	    fscanf(file, "%d %d %d", &r, &g, &b);
	    rgbPixels[i * 4 + 0] = r / max;
	    rgbPixels[i * 4 + 1] = g / max;
	    rgbPixels[i * 4 + 2] = b / max;
	    rgbPixels[i * 4 + 3] = 1.0;
	}
    }
    else if(token == '4')
    {
        int bitnum = 0;

	for(i = 0; i < width * height; i++)
	{
	    unsigned char pixel;
	    unsigned char value = 0;

	    if(bitnum == 0) {
	        fread(&value, 1, 1, file);
            }

	    pixel = (1 - ((value >> (7 - bitnum)) & 1));
	    rgbPixels[i * 4 + 0] = pixel;
	    rgbPixels[i * 4 + 1] = pixel;
	    rgbPixels[i * 4 + 2] = pixel;
	    rgbPixels[i * 4 + 3] = 1.0;

	    if(++bitnum == 8 || ((i + 1) % width) == 0) {
	        bitnum = 0;
            }
	}
    }
    else if(token == '5')
    {
	for(i = 0; i < width * height; i++)
	{
	    unsigned char pixel;
	    fread(&pixel, 1, 1, file);
	    rgbPixels[i * 4 + 0] = pixel / max;
	    rgbPixels[i * 4 + 1] = pixel / max;
	    rgbPixels[i * 4 + 2] = pixel / max;
	    rgbPixels[i * 4 + 3] = 1.0;
	}
    }
    else if(token == '6')
    {
	for(i = 0; i < width * height; i++)
	{
	    unsigned char rgb[3];
	    fread(rgb, 3, 1, file);
	    rgbPixels[i * 4 + 0] = rgb[0] / max;
	    rgbPixels[i * 4 + 1] = rgb[1] / max;
	    rgbPixels[i * 4 + 2] = rgb[2] / max;
	    rgbPixels[i * 4 + 3] = 1.0;
	}
    }
    *w = width;
    *h = height;
    *pixels = rgbPixels;
    return 1;
}

void LoadModel(const char *filename)
{
    using namespace VulkanApp;

    FILE* fp = fopen(filename, "r");
    if(fp == nullptr) {
        fprintf(stderr, "couldn't open file %s for reading\n", filename);
        exit(EXIT_FAILURE);
    }

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    std::string texture_name;
    float specular_color[4];
    float shininess;
    ParseTriSrc(fp, vertices, indices, texture_name, specular_color, shininess);

    std::shared_ptr<RGBA8UNormImage> texture;

    if(texture_name == "*") {

        // XXX Need a way to have no texture at some point
        int width = 1, height = 1;
        std::vector<uint8_t> rgba8_unorm = {255, 255, 255, 255};
        texture = std::make_shared<RGBA8UNormImage>(width, height, rgba8_unorm);

    } else {

        std::filesystem::path path {filename};
        std::filesystem::path texture_path = path.parent_path() / texture_name;
        FILE *texture_file = fopen(texture_path.string().c_str(), "rb");
        if(texture_file == nullptr) {
            fprintf(stderr, "couldn't open texture file %s for reading\n", texture_name.c_str());
            exit(EXIT_FAILURE);
        }

        int width = 0, height = 0;
        float *float_pixels = nullptr;
        std::vector<uint8_t> rgba8_unorm;
        int result = pnmRead(texture_file, &width, &height, &float_pixels);
        if(!result) {
            fprintf(stderr, "couldn't read PPM image from %s\n", texture_name.c_str());
            exit(EXIT_FAILURE);
        }

        rgba8_unorm.resize(4 * width * height);
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                for(int c = 0; c < 4; c++) {
                    rgba8_unorm[c + (x + y * width) * 4] = static_cast<uint8_t>(std::clamp(float_pixels[c + (x + (height - y - 1) * width) * 4] * 255.999f, 0.0f, 255.0f));
                    // rgba8_unorm[c + (x + y * width) * 4] = static_cast<uint8_t>(std::clamp(float_pixels[c + (x + y * width) * 4] * 255.999f, 0.0f, 255.0f));
                }
            }
        }
        texture = std::make_shared<RGBA8UNormImage>(width, height, rgba8_unorm);

    }

    drawable = std::make_unique<DrawableShape>(vertices, indices, specular_color, shininess, texture);

    object_manip = manipulator(drawable->bounds, fov / 180.0f * 3.14159f / 2);
    light_manip = manipulator(aabox(), fov / 180.0f * 3.14159f / 2);
    current_manip = &object_manip;

    fclose(fp);
}

void usage(const char *progName) 
{
    fprintf(stderr, "usage: %s modelFileName\n", progName);
}

extern "C" { void DebugBreak(void); }

int main(int argc, char **argv)
{
    uint32_t specified_gpu = 0;
    bool rz_only = false;

    using namespace VulkanApp;

    be_verbose = (getenv("BE_NOISY") != nullptr);
    enable_validation = (getenv("VALIDATE") != nullptr);

    [[maybe_unused]] const char *progName = argv[0];
    argv++;
    argc--;
    while(argc > 0 && argv[0][0] == '-') {
        if(strcmp(argv[0], "--gpu") == 0) {
            if(argc < 2) {
                usage(progName);
                printf("--gpu requires a GPU index (e.g. \"--gpu 1\")\n");
                exit(EXIT_FAILURE);
            }
            specified_gpu = atoi(argv[1]);
            argv += 2;
            argc -= 2;
        } else if(strcmp(argv[0], "--raster") == 0) {
            rz_only = true;
            argv += 1;
            argc -= 1;
        } else if(strcmp(argv[0], "--validate") == 0) {
            enable_validation = true;
            argv += 1;
            argc -= 1;
        } else {
            usage(progName);
            printf("unknown option \"%s\"\n", argv[0]);
            exit(EXIT_FAILURE);
        }
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

    if (false) {
        //time_t now = time(0);
        // while(time(0) - now < 5);
    }
    if(false) {
        DebugBreak();
    }
    VulkanApp::InitializeInstance();

    VkResult err = glfwCreateWindowSurface(instance, window, nullptr, &surface);
    if (err) {
	std::cerr << "GLFW window creation failed " << err << "\n";
        exit(EXIT_FAILURE);
    }

    VulkanApp::InitializeState(specified_gpu);

    glfwSetKeyCallback(window, KeyCallback);
    glfwSetMouseButtonCallback(window, ButtonCallback);
    glfwSetCursorPosCallback(window, MotionCallback);
    glfwSetScrollCallback(window, ScrollCallback);
    // glfwSetFramebufferSizeCallback(window, ResizeCallback);
    glfwSetWindowRefreshCallback(window, rz_only ? DrawFrame : DrawFrameRT);

    while (!glfwWindowShouldClose(window)) {

        // if(gStreamFrames)
            glfwPollEvents();
        // else
        // glfwWaitEvents();

        (rz_only ? DrawFrame : DrawFrameRT)(window);
    }

    Cleanup();

    glfwTerminate();
}
