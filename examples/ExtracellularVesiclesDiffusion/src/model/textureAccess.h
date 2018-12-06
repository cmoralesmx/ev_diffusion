#pragma once
#include <cuda_runtime.h>
#include "myVisualisation.h"

// we declare these pointers to as references to acces the textures in the model
__device__ float * d_boundaries_x;
__device__ float * d_boundaries_y;
__device__ float * d_boundaries_angles;

// corresponding textures are created to store the actual
texture<float, 1, cudaReadModeElementType> d_boundaries_x_t;
texture<float, 1, cudaReadModeElementType> d_boundaries_y_t;
texture<float, 1, cudaReadModeElementType> d_boundaries_angles_t;

size_t total_vertices_size;
size_t total_angles_size;

__device__ float4 get_boundary_coords(int boundaryIdx) {
	float4 b;
	b.x = tex1Dfetch(d_boundaries_x_t, boundaryIdx);
	b.y = tex1Dfetch(d_boundaries_y_t, boundaryIdx);
	b.z = tex1Dfetch(d_boundaries_x_t, boundaryIdx + 1);
	b.w = tex1Dfetch(d_boundaries_y_t, boundaryIdx + 1);
	return b;
}
__device__ float get_boundary_angle(int boundaryIdx) {
	return tex1Dfetch(d_boundaries_angles_t, boundaryIdx);
}

void loadDataToTextures() {
	printf("total memory size required in the GPU: %zi\n", 2 * total_vertices_size + total_angles_size);
	checkCudaErrors(cudaMalloc((void**)&d_boundaries_x, total_vertices_size));
	checkCudaErrors(cudaMemcpy((void*)d_boundaries_x, (void*)h_xs, total_vertices_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, d_boundaries_x_t, d_boundaries_x, d_boundaries_x_t.channelDesc,total_vertices_size));

	checkCudaErrors(cudaMalloc((void**)&d_boundaries_y, total_vertices_size));
	checkCudaErrors(cudaMemcpy((void*)d_boundaries_y, (void*)h_ys, total_vertices_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, d_boundaries_y_t, d_boundaries_y, d_boundaries_y_t.channelDesc, total_vertices_size));


	checkCudaErrors(cudaMalloc((void**)&d_boundaries_angles, total_angles_size));
	checkCudaErrors(cudaMemcpy((void*)d_boundaries_angles, (void*)h_angles, total_angles_size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaBindTexture(0, d_boundaries_angles_t, d_boundaries_angles, d_boundaries_angles_t.channelDesc, total_angles_size));
	printf("Completed GPU loading\n");
}

void loadData(char *  filename) {
	FILE *ifile;
	ifile = fopen(filename, "rb");
	if (ifile == NULL) {
		printf("Error opening Boundaries files");
	}

	//char buf[100];
	int number_of_visual_sections;
	fread(&number_of_visual_sections, sizeof(int), 1, ifile);
	//printf("Number of sections: %i\nVertices per section:\n", number_of_visual_sections);
	//get_n_sections();
	set_n_visual_sections(&number_of_visual_sections);

	int * vertices_per_visual_section;
	vertices_per_visual_section = (int*)malloc(sizeof(int) * number_of_visual_sections);
	fread(vertices_per_visual_section, sizeof(int) * number_of_visual_sections, 1, ifile);
	set_vertices_per_visual_section(vertices_per_visual_section);

	int * offset_visual_sections;
	int total_vertices = 0, total_angles = 0;
	int * offset_angles;
	offset_visual_sections = (int*)malloc(sizeof(int) * number_of_visual_sections);
	offset_angles = (int*)malloc(sizeof(int) * number_of_visual_sections);

	for (int i = 0; i < number_of_visual_sections; i++) {
		offset_visual_sections[i] = total_vertices;
		printf("offset section %i = %d, last idx=%i\n", i, total_vertices, total_vertices + vertices_per_visual_section[i] - 1);
		total_vertices += vertices_per_visual_section[i];

		offset_angles[i] = total_angles;
		printf("offset angles %i = %d, last idx=%i\n", i, total_angles, total_angles + vertices_per_visual_section[i] - 2);
		total_angles += vertices_per_visual_section[i] - 1;
		//printf("S%i) %i\n", i, vertices_per_section[i]);
	}
	printf("total vertices %i, total angles %i\n", total_vertices, total_angles);

	set_offset_visual_sections(offset_visual_sections);
	set_offset_angles(offset_angles);

	total_vertices_size = sizeof(float) * total_vertices;
	total_angles_size = sizeof(float) * total_angles;
	

	h_xs = (float*)malloc(total_vertices_size);
	h_ys = (float*)malloc(total_vertices_size);
	h_angles = (float*)malloc(total_angles_size);

	fread(h_xs, total_vertices_size, 1, ifile);
	fread(h_ys, total_vertices_size, 1, ifile);
	fread(h_angles, total_angles_size, 1, ifile);

	fprintf(stdout, "h_xs[0]=%f\n", h_xs[0]);
	printf("h_xs[0]=%f\n", h_xs[0]);
	fprintf(stdout, "h_xs[14968]=%f (hard coded index)\n", h_xs[14968]);
	fprintf(stdout, "h_ys[0]=%f\n", h_ys[0]);
	fprintf(stdout, "h_ys[%i]=%f\n", total_vertices - 1, h_ys[total_vertices - 1]);
	fprintf(stdout, "h_angles[0]=%f\n", h_angles[0]);
	fprintf(stdout, "h_angles[%i]=%f\n", total_angles - 1, h_angles[total_angles - 1]);
	fflush(stdout);

	fclose(ifile);
}

void loadStartingPoints(char * filename) {
	printf("\n\nloading starting points data... \n");
	FILE *ifile;
	ifile = fopen(filename, "rb");
	if (ifile == NULL) {
		printf("Error opening StartingPoints file");
	}

	//char buf[100];
	fread(&number_of_starting_points, sizeof(int), 1, ifile);
	printf("File contains %d points... \n",number_of_starting_points);

	size_t size_sps = sizeof(float4) * number_of_starting_points;
	// allocate enough memory to contain all the elements
	starting_points = (float4*)malloc(size_sps);
	fread(starting_points, sizeof(float4), number_of_starting_points, ifile);
	printf("values %f, %f\n", starting_points[0].x, starting_points[0].y);
	printf("last values %f, %f\n\n", starting_points[number_of_starting_points-1].x, 
		starting_points[number_of_starting_points-1].y);

	fclose(ifile);
}