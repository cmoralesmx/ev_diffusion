# version 330 core
layout (lines) in;
layout (triangle_strip, max_vertices=4) out;

in vData {
    vec2 unit_normal;
} vertices[];

uniform mat4 mvp;

void main(){
    float thickness = 0.1f;
    vec4 p0 = gl_in[0].gl_Position;
    vec4 p1 = gl_in[1].gl_Position;
    vec2 normal = vertices[0].unit_normal;

    // bottom left
    gl_Position = mvp * vec4(p1.xy - normal * thickness, 0, 1.0);
    EmitVertex();

    // bottom right
    gl_Position = mvp * vec4(p1.xy, 0, 1.0);
    EmitVertex();

    // top left
    gl_Position = mvp * vec4(p0.xy - normal * thickness, 0, 1.0);
    EmitVertex();

    // top right
    gl_Position = mvp * vec4(p0.xy, 0, 1.0);
    EmitVertex();

    EndPrimitive();
}