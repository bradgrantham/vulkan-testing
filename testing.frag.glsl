#version 450

layout (binding = 1) uniform UBO2
{
    vec3 light_position;
    vec3 light_color;
} ubo2;

layout (binding = 2) uniform Shading
{
    vec3 specular_color;
    float shininess;
} shading;

layout (binding = 3) uniform sampler2D color_texture;

layout (location = 0) in vec4 vertex_position;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec4 vertex_color;
layout (location = 3) in vec2 vertex_texcoord;

layout (location = 0) out vec4 outFragColor;

void main()
{
    vec3 normal = normalize(vertex_normal);
    vec3 eye_position = -vertex_position.xyz; 
    vec3 edir = normalize(eye_position);
    if(dot(normal, edir) < 0) {
        normal *= -1;
    }
    vec3 ldir = normalize(ubo2.light_position - vertex_position.xyz);
    vec3 refl = reflect(-ldir, normal);
    vec3 diffuse = max(0, dot(normal, ldir)) * ubo2.light_color; // * texture(material_diffuse_texture, vertex_texcoord);
    vec3 ambient = ubo2.light_color;
    vec3 specular = pow(max(0, dot(refl, edir)), shading.shininess) * shading.specular_color * ubo2.light_color;
    vec3 material_diffuse = vec3(.8, .8, .8);
    vec3 texture_color = texture(color_texture, vertex_texcoord).xyz;
    vec3 material_ambient = vec3(.1, .1, .1);
    vec3 material_specular = vec3(.8, .8, .8);
    outFragColor = vec4(texture_color * vertex_color.xyz * (diffuse * material_diffuse + ambient * material_ambient + specular * material_specular ), 1.0);
}
