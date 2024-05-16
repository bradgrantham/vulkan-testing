#version 460
#extension GL_EXT_ray_tracing : enable
#extension GL_EXT_nonuniform_qualifier : enable

// layout (binding = 2) uniform sampler2D color_texture;

layout(location = 0) rayPayloadInEXT vec3 hitValue;
hitAttributeEXT vec3 attribs;

struct Vertex
{
    vec4 v; // component 3 ignored
    vec4 n; // component 3 ignored
    vec4 c;
    vec4 t; // component 3 ignored
};

layout(binding = 3, set = 0) buffer Vertices { Vertex vertices[]; } vertex_buffer;
layout(binding = 4, set = 0) buffer Indices { uint indices[]; } index_buffer;

void main()
{
    uint i0 = index_buffer.indices[gl_PrimitiveID * 3 + 0];
    uint i1 = index_buffer.indices[gl_PrimitiveID * 3 + 1];
    uint i2 = index_buffer.indices[gl_PrimitiveID * 3 + 2];
    Vertex v0 = vertex_buffer.vertices[i0];
    Vertex v1 = vertex_buffer.vertices[i1];
    Vertex v2 = vertex_buffer.vertices[i2];

    vec3 specular_color = vec3(1, 1, 1); // from material
    float shininess = 10; // from material
    vec3 light_position = vec3(1000, 1000, 1000); // get global light position
    vec3 light_color = vec3(1, 1, 1); // get global light color

    vec3 bary = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);

    vec4 object_color = v0.c * bary.x + v1.c * bary.y + v2.c * bary.z;
    vec3 object_normal = normalize((v0.n * bary.x + v1.n * bary.y + v2.n * bary.z).xyz);
    vec3 object_position = (v0.v * bary.x + v1.v * bary.y + v2.v * bary.z).xyz;
    vec3 object_texcoord = (v0.t * bary.x + v1.t * bary.y + v2.t * bary.z).xyz;

    // XXX instance and object transformation are currently identity
    // so object is already in world space.  Rays are transformed to world
    // space by the camera transformation.
    vec3 world_normal = object_normal; // XXX
    vec3 world_position = object_position; // XXX

    vec3 eye_position = gl_WorldRayOriginEXT;
    vec3 edir = normalize(eye_position);

    if(dot(world_normal, edir) < 0) {
        world_normal *= -1;
    }

    vec3 ldir = normalize(light_position - world_position.xyz);
    vec3 refl = reflect(-ldir, world_normal);
    // vec3 diffuse = max(0, dot(world_normal, ldir)) * light_color * texture(color_texture, object_texcoord).xyz;
    vec3 diffuse = max(0, dot(world_normal, ldir)) * light_color;
    vec3 ambient = vec3(1, 1, 1); // ubo2.light_color;
    vec3 specular = pow(max(0, dot(refl, edir)), shininess) * specular_color * light_color;
    vec3 material_diffuse = vec3(.8, .8, .8);
    vec3 material_ambient = vec3(.1, .1, .1);
    vec3 material_specular = vec3(.8, .8, .8);

    vec3 result_color = object_color.xyz * (diffuse * material_diffuse + ambient * material_ambient + specular * material_specular);

    hitValue = result_color;
}

