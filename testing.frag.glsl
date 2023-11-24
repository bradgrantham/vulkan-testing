#version 450

layout (location = 0) in vec4 vertex_position;
layout (location = 1) in vec3 vertex_normal;
layout (location = 2) in vec4 vertex_color;

layout (location = 0) out vec4 outFragColor;

void main()
{
    vec3 normal = normalize(vertex_normal);
    vec3 eye_position = -vertex_position.xyz; 
    vec3 edir = normalize(eye_position);
    if(dot(normal, edir) < 0) {
        normal *= -1;
    }
    vec3 light_pos = vec3(100, 100, 100); // light_position uniform
    vec3 light_color = vec3(1, 1, 1); // light_color uniform
    vec3 ldir = normalize(light_pos - vertex_position.xyz);
    vec3 refl = reflect(-ldir, normal);
    vec3 diffuse = max(0, dot(normal, ldir)) * light_color; // * texture(material_diffuse_texture, vertex_texcoord);
    vec3 ambient = light_color;
    // let specular = vec3<f32>(0, 0, 0); 
    vec3 specular = pow(max(0, dot(refl, edir)), 50 /* material_shininess */) * light_color;
    vec3 material_diffuse = vec3(.8, .8, .8);
    vec3 material_ambient = vec3(.1, .1, .1);
    vec3 material_specular = vec3(.8, .8, .8);
    outFragColor = vec4(vertex_color.xyz * (diffuse * material_diffuse + ambient * material_ambient + specular * material_specular ), 1.0);
}
