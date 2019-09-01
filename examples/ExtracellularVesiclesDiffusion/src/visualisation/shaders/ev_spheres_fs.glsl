#version 330 core
// modified from https://stackoverflow.com/a/10506172/

in fData
{
	vec2 position;
    float radius;
	vec3 sphereColor;
	vec4 transformedPosition;
	vec3 normalizedViewCoordinate;
} frag;
//in vec3 sphereColor;

vec3 lightPosition = normalize(vec3(0.7, 0.7, 0.7));
out vec4 FragColor;

void main(){
	float distanceFromCenter = length(frag.position);
	if(distanceFromCenter > frag.radius) discard;

	float normalizedDepth = sqrt(frag.radius*1.5 - distanceFromCenter * distanceFromCenter);

	// current depth
	float fragmentDepth = frag.radius * 0.5 * normalizedDepth;
	float currentDepth = (frag.normalizedViewCoordinate.z - fragmentDepth - 0.0025);
	
	vec3 normal = vec3(frag.position, normalizedDepth);
	
	// ambient
	float lightingIntensity = 0.3 + 0.7 * clamp(dot(lightPosition, normal), 0., 1.);
	vec3 finalColor = frag.sphereColor * lightingIntensity;

	// per fragment specular lighting
	float diffuse = clamp(dot(lightPosition, normal), 0., 1.);
	float specular = pow(diffuse, 40.0);
	finalColor += vec3(0.4) * diffuse * specular * lightingIntensity;

	FragColor = vec4(finalColor, 1.0);
}
