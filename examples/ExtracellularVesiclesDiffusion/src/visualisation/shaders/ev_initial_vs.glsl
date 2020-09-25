#version 330 core
// modified from https://stackoverflow.com/a/10506172/
layout (location=0) in vec2 position;
layout (location=1) in float radius;

out vData {
	float radius;
	vec3 sphereColor;
	vec4 transformedPosition;
	vec3 normalizedViewCoordinate;
} vertex;

uniform mat4 mvp;

void main(){
	vertex.radius = radius * 10.0f;
	vec4 transformedPosition = mvp * vec4(position, 0., 1.);
	vertex.sphereColor = vec3(1., .0, .0);
	vertex.normalizedViewCoordinate = (transformedPosition.xyz + 1.0) / 2.0;
	vertex.transformedPosition = transformedPosition;

	gl_Position = transformedPosition;
}

