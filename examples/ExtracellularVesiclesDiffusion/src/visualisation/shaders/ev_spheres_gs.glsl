#version 330 core
// modified from https://learnopengl.com/Advanced-OpenGL/Geometry-Shader
//https://stackoverflow.com/questions/14909796/simple-pass-through-geometry-shader-with-normal-and-color
layout (points) in;
layout (triangle_strip, max_vertices=4) out;

in vData {
    float radius;
	vec3 sphereColor;
    vec4 transformedPosition;
	vec3 normalizedViewCoordinate;
} vertices[];

out fData {
    vec2 position;
    float radius;
	vec3 sphereColor;
    vec4 transformedPosition;
	vec3 normalizedViewCoordinate;
} frag;
//out vec3 sphereColor;

void fake_sphere(vec4 position){
    
    frag.sphereColor = vertices[0].sphereColor;
    frag.radius = vertices[0].radius;
    frag.transformedPosition = vertices[0].transformedPosition;
	frag.normalizedViewCoordinate = vertices[0].normalizedViewCoordinate;

    frag.position = vec2(-frag.radius, -frag.radius);
    gl_Position = position + vec4(frag.position, 0, 0);
    EmitVertex();
    frag.position = vec2( frag.radius, -frag.radius);
    gl_Position = position + vec4(frag.position, 0, 0);
    EmitVertex();
    frag.position = vec2(-frag.radius, frag.radius);
    gl_Position = position + vec4(frag.position, 0, 0);
    EmitVertex();
    frag.position = vec2( frag.radius, frag.radius);
    gl_Position = position + vec4(frag.position, 0, 0);
    EmitVertex();
    EndPrimitive();
}

void main(){
    fake_sphere(gl_in[0].gl_Position);
}