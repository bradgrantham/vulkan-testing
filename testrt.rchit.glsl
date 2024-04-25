#version 460

layout (binding = 0) uniform sampler2D color_texture;

// ?? layout (location = 0) out vec4 outFragColor;

void main()
{
    vec3 specular_color;
    float shininess;
    vec3 light_position = vec3(1000, 1000, 1000); // get global light position
    vec3 light_color = vec3(1, 1, 1); // get global light color
    vec3 vertex_normal = vec3(0, 0, 1); // interpolate normal from per-vertex normals;
    vec3 vertex_color = vec3(1, 1, 1); // interpolate color from per-color normals; 
    vec3 normal = normalize(vertex_normal);
    vec3 vertex_position = vec3(0, 0, 0); // interpolate position from vertex data; 
    vec2 vertex_texcoord = vec2(0, 0); // interpolate texcoord from per-vertex texcoords
    vec3 eye_position = -vertex_position.xyz; 
    vec3 edir = normalize(eye_position);

    if(dot(normal, edir) < 0) {
        normal *= -1;
    }

    vec3 ldir = normalize(light_position - vertex_position.xyz);
    vec3 refl = reflect(-ldir, normal);
    vec3 diffuse = max(0, dot(normal, ldir)) * light_color * texture(color_texture, vertex_texcoord).xyz;
    vec3 ambient = vec3(1, 1, 1); // ubo2.light_color;
    vec3 specular = pow(max(0, dot(refl, edir)), shininess) * specular_color * light_color;
    vec3 material_diffuse = vec3(.8, .8, .8);
    vec3 material_ambient = vec3(.1, .1, .1);
    vec3 material_specular = vec3(.8, .8, .8);

    vec3 result_color = vertex_color.xyz * (diffuse * material_diffuse + ambient * material_ambient + specular * material_specular );
    // store result_color
}

