
/*
 * Copyright 2011 University of Sheffield.
 * Author: Dr Paul Richmond 
 * Contact: p.richmond@sheffield.ac.uk (http://www.paulrichmond.staff.shef.ac.uk)
 *
 * University of Sheffield retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * University of Sheffield is strictly prohibited.
 *
 * For terms of licence agreement please attached licence or view licence 
 * on www.flamegpu.com website.s
 * 
 */

#ifndef _FLAMEGPU_FUNCTIONS
#define _FLAMEGPU_FUNCTIONS

#ifndef VISUALISATION
//#define INSTRUMENT_ITERATIONS 1
//#define INSTRUMENT_AGENT_FUNCTIONS 1
//#define INSTRUMENT_INIT_FUNCTIONS 1
//#define INSTRUMENT_STEP_FUNCTIONS 1
//#define INSTRUMENT_EXIT_FUNCTIONS 1
//#define OUTPUT_POPULATION_PER_ITERATION 1
#endif

#include <header.h>
#define _USE_MATH_DEFINES
#include <math_constants.h>
#include <math.h>
#include <float.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "myVisualisation.h"
//#include "textureAccess.h"

//#define radius 2.0

// Declare global scope variables for host-based agent creation, so allocation of host data is only performed once.
xmachine_memory_EV ** h_EV_AoS;
unsigned int h_EV_AoS_MAX;

__device__ __host__ float degrees(float radians){
	return radians * (180 / M_PI);
}

__device__ __host__ float radians(float degrees){
	return degrees * (M_PI / 180.0);
}

/**
* Compute the distance between two agents using the coordinates of their centers
*/
__device__ float euclidean_distance(xmachine_memory_EV* agent, float message_x, float message_y)
{
	return sqrt((agent->x - message_x) * (agent->x - message_x) +
		(agent->y - message_y) * (agent->y - message_y));
}

__FLAME_GPU_STEP_FUNC__ void exitFunction(){
	//float x = reduce_EV_default_x_variable();
	//float y = reduce_EV_default_y_variable();
	//float z = reduce_EV_default_z_variable();
	//printf("FlameGPU exit function, avg position is (%f, %f, %f)\n", x, y);
}

__FLAME_GPU_STEP_FUNC__ void print_iteration_no() {
	//float x = reduce_EV_default_x_variable();
	//float y = reduce_EV_default_y_variable();
	//float z = reduce_EV_default_z_variable();
	//printf("FlameGPU exit function, avg position is (%f, %f, %f)\n", x, y);
	//printf("iter: %d\n", ++iteration_number);
}

__FLAME_GPU_INIT_FUNC__ void initialize_new_ev_structures(){
	srand(time(NULL));

	if (*get_const_ev_collisions() > 0) {
		printf("Ev-Ev collisions are enabled\n");
	}
	else {
		printf("Ev-Ev collisions are not enabled\n");
	}
	iteration_number = 0;
	if(*get_introduce_n_new_evs() > 0){
		printf("Attempting introduction of new EVs every %f seconds with a probability threshold of %f\n",
		 *get_seconds_before_introducing_new_evs(), *get_new_evs_threshold());

		//h_EV_AoS_MAX = *get_introduce_n_new_evs();
		//h_EV_AoS = h_allocate_agent_EV_array(*get_introduce_n_new_evs());
	} else {
		printf("No new EVs will be introduced during simulation\n");
	}
}

__FLAME_GPU_INIT_FUNC__ void precompute_values() {
	float val = 6 * M_PI * *get_const_water_viscosity();
	set_const_6_pi_viscosity(&val);
	printf("set_const_6_pi_viscosity: %f\n", val);

	float val2 = *get_const_Boltzmann() * *get_const_Temperature_K();
	printf("boltzmann: %.35f\n", *get_const_Boltzmann());
	set_const_boltzmann_x_temp(&val2);
	printf("set_const_boltzmann_x_temp: %.30f\n", val2);
}

/**
 * output_data FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_EV. This represents a single agent instance and can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
 */
__FLAME_GPU_FUNC__ int output_data(xmachine_memory_EV* agent, xmachine_message_location_list* location_messages){
	add_location_message(location_messages, agent->id, agent->x, agent->y,
		agent->z, agent->radius_um, agent->mass_kg, agent->vx, agent->vy);
    return 0;
}

/**
* output_data FLAMEGPU Agent Function
* Automatically generated using functions.xslt
* @param agent Pointer to an agent structure of type xmachine_memory_EV. This represents a single agent instance and can be modified directly.
* @param location_messages Pointer to output message list of type xmachine_message_location_list. Must be passed as an argument to the add_location_message function ??.
*/
__FLAME_GPU_FUNC__ int output_ciliary_cell_location(xmachine_memory_CiliaryCell* agent, xmachine_message_ciliary_cell_location_list* location_messages) {

	add_ciliary_cell_location_message(location_messages, agent->id, agent->x, agent->y, 0,
		agent->p1_x, agent->p1_y, agent->p2_x, agent->p2_y, agent->direction_x, agent->direction_y,
		agent->direction_x_unit, agent->direction_y_unit, agent->direction_length, 
		agent->normal_x, agent->normal_y, agent->unit_normal_x, agent->unit_normal_y, agent->normal_length
	);
	return 0;
}

__FLAME_GPU_FUNC__ int output_secretory_cell_location(xmachine_memory_SecretoryCell* agent, xmachine_message_secretory_cell_location_list* location_messages) {

	add_secretory_cell_location_message(location_messages, agent->id, agent->x, agent->y, 0,
		agent->p1_x, agent->p1_y, agent->p2_x, agent->p2_y, agent->direction_x, agent->direction_y,
		agent->direction_x_unit, agent->direction_y_unit, agent->direction_length, 
		agent->normal_x, agent->normal_y, agent->unit_normal_x, agent->unit_normal_y, agent->normal_length
	);
	return 0;
}


//__device__ float perpendicular(float vec_x, float vec_y, float length_vec, int component=0) {
//	float x, y;
//	if (length_vec > 0) {
//		x = vec_y * (1 / length_vec);
//		y = -vec_x * (1 / length_vec);
//	}
//	else {
//		x = 0;// = make_float2(0, 0);
//		y = 0;
//	}
//	return component == 0 ? x : y;
//}

__device__ float dotprod(float x1, float y1, float x2, float y2) {
	return x1*x2 + y1*y2;
}

__device__ float vlength(float x, float y) {
	return sqrt(x*x + y*y);
}

__device__ float projection(float a_x, float a_y, float b_x, float b_y) {
	float length = vlength(a_x, a_y);
	float lengthVec = vlength(b_x, b_y);
	if (length == 0 || lengthVec == 0)
		return 0;
	return (a_x * b_x + a_y * b_y) / lengthVec;
}

__device__ float2 parallel(float vec_x, float vec_y, float u) {
	float factor = (u / vlength(vec_x, vec_y));
	return make_float2(vec_x * factor, vec_y * factor);
}

__device__ float4 vectors_from_ev_to_wall_endpoints(xmachine_memory_EV* agent, const float & p1_x, const float & p1_y,
	const float & p2_x, const float & p2_y) {
	return make_float4(p1_x - agent->x, p1_y - agent->y, p2_x - agent->x, p2_y - agent->y);
}

__device__ float2 project_vectors_onto_wall(float4 ev_wall, float cell_direction_x, float cell_direction_y) {
	return make_float2(
		projection(ev_wall.x, ev_wall.y, cell_direction_x, cell_direction_y),
		projection(ev_wall.z, ev_wall.w, cell_direction_x, cell_direction_y));
}

__device__ float2 perpendicular_distance_from_ev_to_wall(float4 ev_wall, float proj1, float cell_direction_x_unit, float cell_direction_y_unit) {
	return make_float2(
		ev_wall.x + cell_direction_x_unit * -proj1,
		ev_wall.y + cell_direction_y_unit * -proj1
	);
}


// solves direct collisions between EV and segments
__device__  int solve_segment_collision(xmachine_memory_EV* agent, float cell_dir_x, 
	float cell_dir_y, float cell_direction_length, float perp_dist_x, float perp_dist_y,
	float cell_unit_normal_x, float cell_unit_normal_y) {
	// 1. angle between velocity and wall 
	float angle = acosf(dotprod(agent->vx, agent->vy, cell_dir_x, cell_dir_y) / (vlength(agent->vx, agent->vy) * cell_direction_length));
	// 2. reposition object
	// var normal = wall.normal;
	// 2a compute the perpendicular to the cell
	float normal_x = cell_unit_normal_x;
	float normal_y = cell_unit_normal_y;
	//printf("wall normal x:%f y:%f\n", normal_x, normal_y);
	// 2b scale the normal by -1 if needed
	float d = dotprod(normal_x, normal_y, agent->vx, agent->vy);
	if (d > 0) { normal_x *= -1.0; normal_y *= -1.0; }

	// 3. compute deltaS
	// var deltaS = (obj.radius + dist.dotProduct(normal)) / Math.sin(angle);
	float deltaS = (agent->radius_um + dotprod(perp_dist_x, perp_dist_y, normal_x, normal_y)) / sin(angle);

	// 4. estimate what was the displacement = velocity parallel to delta
	// var displ = obj.velo2D.para(deltaS);
	float2 displ = parallel(agent->vx, agent->vy, deltaS);
	//float displ_x = parallel(agent->vx, agent->vy, deltaS, 0);
	//float displ_y = parallel(agent->vx, agent->vy, deltaS, 1);

	// 5. update position by subtracting the displacement
	// obj.pos2D = obj.pos2D.subtract(displ);
	agent->x -= displ.x;
	agent->y -= displ.y;

	// velocity correction factor
	//var vcor = 1-acc.dotProduct(displ)/obj.velo2D.lengthSquared();
	//float numerator = dotprod(acceleration_x, acceleration_y, displ_x, displ_y);
	//float sqr_vel = (agent->vx * agent->vx + agent->vy*agent->vy);
	//float vfrac = numerator / sqr_vel;
	//float vcor = 1 - vfrac;

	//printf("displx:%f disply:%f numerator:%f sqr_vel:%f vfrac:%f vcor:%f\n", displ_x, displ_y, numerator, sqr_vel, vfrac, vcor);
	// corrected velocity vector just before impact 
	// var Velo = obj.velo2D.multiply(vcor)
	//float new_velocity_x = agent->vx;// *vcor;
	//float new_velocity_y = agent->vy;// *vcor;

	// 6. decompose the velocity
	// velocity vector component perpendicular to wall just before impact
	// var normalVelo = dist.para(Velo.projection(dist));
	float velocityProjection = projection(agent->vx, agent->vy, perp_dist_x, perp_dist_y);
	float2 normalVelocity = parallel(perp_dist_x, perp_dist_y, velocityProjection);
	//float normalVelocity_x = parallel(perp_dist_x, perp_dist_y, velocityProjection, 0);
	//float normalVelocity_y = parallel(perp_dist_x, perp_dist_y, velocityProjection, 1);

	// velocity vector component parallel to wall; unchanged by impact
	// var tangentVelo = Velo.subtract(normalVelo);
	float tangentVelocity_x = agent->vx - normalVelocity.x;
	float tangentVelocity_y = agent->vy - normalVelocity.y;

	// velocity vector component perpendicular to wall just after impact
	// obj.velo2D = tangentVelo.addScaled(normalVelo, -vfac);
	agent->vx = tangentVelocity_x + (-normalVelocity.x);
	agent->vy = tangentVelocity_y + (-normalVelocity.y);

	return 0;
}

__device__  int solve_segment_end_point_collision(xmachine_memory_EV* agent, 
	float ev_wall_pt_x, float ev_wall_pt_y, float cell_direction_length,
	float dist_x, float dist_y) {
	// bounce off endpoint wall_p1
	float distp_x = agent->x - ev_wall_pt_x;
	float distp_y = agent->y - ev_wall_pt_y;
	//var distp = obj.pos2D.subtract(pEndpoint);
	//// move particle so that it just touches the endpoint			
	//var L = obj.radius - distp.length();
	//var vrel = obj.velo2D.length();
	//obj.pos2D = obj.pos2D.addScaled(obj.velo2D, -L / vrel);
	float L = agent->radius_um - vlength(distp_x, distp_y);
	float vrel = vlength(agent->vx, agent->vy);
	agent->x += (-L / vrel) * agent->vx;
	agent->y += (-L / vrel) * agent->vy;

	//// normal velocity vector just before the impact
	//var normalVelo = obj.velo2D.project(distp);
	float normalVelo = projection(agent->vx, agent->vy, distp_x, distp_y);
	//// tangential velocity vector
	//var tangentVelo = obj.velo2D.subtract(normalVelo);
	float tangentVelo_x = agent->vx - normalVelo;
	float tangentVelo_y = agent->vy - normalVelo;
	//// normal velocity vector after collision
	//normalVelo.scaleBy(-vfac);

	//// final velocity vector after collision
	//obj.velo2D = normalVelo.add(tangentVelo);
	agent->vx = normalVelo + tangentVelo_x;
	agent->vy = normalVelo + tangentVelo_y;
	agent->velocity_ms = 666.f;
	return 0;
}

/**
Collision resolution algorithm modified from Physics for Javascript Games, Animation, and Simulation Ch, 11
*/
__FLAME_GPU_FUNC__ int secretory_cell_collision_resolution(xmachine_memory_EV* agent, 
	xmachine_message_secretory_cell_collision_list* secretory_cell_collision_messages){

	xmachine_message_secretory_cell_collision* message = get_first_secretory_cell_collision_message(secretory_cell_collision_messages);
	
	while (message){
		// we verify the ev_id in the message matches this agent's id
		if (message->ev_id == agent->id) {
			float4 ev_wall_vectors = vectors_from_ev_to_wall_endpoints(agent, message->p1_x, message->p1_y, message->p2_x, message->p2_y);
			float2 projections = project_vectors_onto_wall(ev_wall_vectors, message->cell_direction_x, message->cell_direction_y);
			float2 perp_dist = perpendicular_distance_from_ev_to_wall(ev_wall_vectors, projections.x, message->cell_direction_x_unit, message->cell_direction_y_unit);

			bool test_needed = (abs(projections.x) < message->cell_direction_length) && (abs(projections.y) < message->cell_direction_length);

			if ((vlength(perp_dist.x, perp_dist.y) < agent->radius_um) && test_needed) {

				solve_segment_collision(agent, message->cell_direction_x, message->cell_direction_y, message->cell_direction_length, 
					perp_dist.x, perp_dist.y, message->unit_normal_x, message->unit_normal_y);
			}
			else if (abs(vlength(ev_wall_vectors.x, ev_wall_vectors.y)) < agent->radius_um) {
				// collision with 1st end point
				solve_segment_end_point_collision(agent, ev_wall_vectors.x, ev_wall_vectors.y, message->cell_direction_length, perp_dist.x, perp_dist.y);
			}
			else if (abs(vlength(ev_wall_vectors.z, ev_wall_vectors.w)) < agent->radius_um) {
				// collision with 2nd end point
				solve_segment_end_point_collision(agent, ev_wall_vectors.z, ev_wall_vectors.w, message->cell_direction_length, perp_dist.x, perp_dist.y);
			}
		}
		message = get_next_secretory_cell_collision_message(message, secretory_cell_collision_messages);
	}
	return 0;
}

/**
This function resolves a single collision between an EV and a ciliary cell
After this, the EV has new values for x, y, vx, and vy, compensated for the collision
*/
__FLAME_GPU_FUNC__ int ciliary_cell_collision_resolution(xmachine_memory_EV* agent, 
	xmachine_message_ciliary_cell_collision_list* ciliary_cell_collision_messages) {

	xmachine_message_ciliary_cell_collision* message = get_first_ciliary_cell_collision_message(ciliary_cell_collision_messages);
	//float acceleration_x = 0, acceleration_y = 0;
	while (message) {
		// we verify the ev_id in the message matches this agent's id
		if (message->ev_id == agent->id) {
			
			float4 ev_wall_vectors = vectors_from_ev_to_wall_endpoints(agent, message->p1_x, message->p1_y, message->p2_x, message->p2_y);
			float2 projections = project_vectors_onto_wall(ev_wall_vectors, message->cell_direction_x, message->cell_direction_y);
			float2 perp_dist = perpendicular_distance_from_ev_to_wall(ev_wall_vectors, projections.x, message->cell_direction_x_unit, message->cell_direction_y_unit);

			bool test_needed = (abs(projections.x) < message->cell_direction_length) && (abs(projections.y) < message->cell_direction_length);

			if ((vlength(perp_dist.x, perp_dist.y) < agent->radius_um) && test_needed) {

				solve_segment_collision(agent, message->cell_direction_x, message->cell_direction_y, message->cell_direction_length,
					perp_dist.x, perp_dist.y, message->unit_normal_x, message->unit_normal_y);
			}
			else if (abs(vlength(ev_wall_vectors.x, ev_wall_vectors.y)) < agent->radius_um) {
				// collision with 1st end point
				solve_segment_end_point_collision(agent, ev_wall_vectors.x, ev_wall_vectors.y, message->cell_direction_length, perp_dist.x, perp_dist.y);
			}
			else if (abs(vlength(ev_wall_vectors.z, ev_wall_vectors.w)) < agent->radius_um) {
				// collision with 2nd end point
				solve_segment_end_point_collision(agent, ev_wall_vectors.z, ev_wall_vectors.w, message->cell_direction_length, perp_dist.x, perp_dist.y);
			}
		}
		message = get_next_ciliary_cell_collision_message(message, ciliary_cell_collision_messages);
	}
	return 0;
}

/**
Computes the distance between a point (p0) and the segment of a line with end points (p1) and (p2).
We get the vectors A = p0 - p1 and B = p2 - p1. 
-param- is the scalar value that when multiplied by B gives you the point on the line closest to p0.
A collision is only possible if the point is projecting on the segment. Otherwise, the agent may be 
in collision with another segment first.
Modified from https://stackoverflow.com/a/6853926/3830240

Returns
float4 where:
	x - 1 if the point is projecting on the segment, 0 otherwise
	y - distance from the closest point on the segment to the EV position
	z - x coordinate of the closest point from the segment to the EV
	w - y coordinate of the closest point from the segment to the EV
*/
__device__ float4 point_to_line_segment_distance(float x, float y, float x1, float y1, float x2, float y2)
{
	float4 result;

	// vector components from point to one of the end points of the line segment
	float A = x - x1;
	float B = y - y1;
	// vector components of the line segment
	float C = x2 - x1;
	float D = y2 - y1;

	float dot = A * C + B * D; // vector projection
	float len_sq = C * C + D * D; // sqrd magnitude of the line segment
	float param = -1;
	float xx = 0.f, yy = 0.f, dx, dy;
	float projecting_on_segment = 0.0f;

	if (len_sq != 0){ // in case of 0 length line
		param = dot / len_sq; // ratio of the projected vectors over the length of the line segment
	}
	
	if (param < 0){ // the closest point is p1, point not projecting on segment
		xx = x1;
		yy = y1;
	} 
	else if (param > 1){ // the closest point is p2, point not projecting on segment
		xx = x2;
		yy = y2;
	}
	else {   // the point is projecting on the segment
		// interpolate to find the closes point on the line segment
		projecting_on_segment = 1.0f;
		xx = x1 + param * C;
		yy = y1 + param * D;
	}

	// dx / dy is the vector from p0 to the closest point on the segment
	dx = x - xx;
	dy = y - yy;

	result.x = projecting_on_segment;
	result.y = sqrt(dx * dx + dy * dy); // distance between p
	result.z = xx;
	result.w = yy;
	
	return result;
}

/**
Checks any possible collision with the secretory cells on the boundaries. 
If a collision occurred, a collision_message is generated and the agent will have the values:
	agent->closest_secretory_cell_id = closest_cell_id;
	agent->closest_secretory_cell_distance = closest_cell_distance;
Otherwise, no message is produced and the ev agent will have the default values unchanged:
	agent->closest_secretory_cell_id = -1;
	agent->closest_secretory_cell_distance = -1.0f;
*/
__FLAME_GPU_FUNC__ int test_secretory_cell_collision(xmachine_memory_EV* agent, xmachine_message_secretory_cell_location_list* location_messages,
	xmachine_message_secretory_cell_location_PBM* partition_matrix, xmachine_message_secretory_cell_collision_list* secretory_cell_collision_messages)
{
	int closest_cell = -1;
	float closest_cell_distance = 100.f;
	float4 res;
	float4 wall_direction;
	float4 wall_normal;
	float4 wall_pts;
	float wall_direction_length, wall_normal_length;

	xmachine_message_secretory_cell_location* message = get_first_secretory_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message) {
		// we only check for collisions if the EV is displacing in the same direction as the wall
		float dotProduct = dotprod(agent->vx, agent->vy, message->unit_normal_x, message->unit_normal_y);

		if (dotProduct < 0) {
			// get the distance between an agent's new position and a cell
			res = point_to_line_segment_distance(agent->x, agent->y, message->p1_x, message->p1_y, message->p2_x, message->p2_y);

			// if the ev new position is projecting over the segment and the distance to the segment < radius
			if (res.x == 1.0f && res.y < agent->radius_um)
			{
				if (res.y < closest_cell_distance)
				{
					closest_cell = message->id;
					closest_cell_distance = res.y;
					
					// wall direction values
					wall_direction.x = message->direction_x;
					wall_direction.y = message->direction_y;
					// wall direction unit values
					wall_direction.z = message->direction_x_unit;
					wall_direction.w = message->direction_y_unit;
					wall_direction_length = message->direction_length;
					// wall normal values
					wall_normal.x = message->normal_x;
					wall_normal.y = message->normal_y;
					// wall unit normal values
					wall_normal.z = message->unit_normal_x;
					wall_normal.w = message->unit_normal_y;
					wall_normal_length = message->normal_length;

					wall_pts.x = message->p1_x;
					wall_pts.y = message->p1_y;
					wall_pts.z = message->p2_x;
					wall_pts.w = message->p2_y;
				}
			}
		}

		// get the next message
		message = get_next_secretory_cell_location_message(message, location_messages, partition_matrix);
	}

	if (closest_cell > -1) { // a collision was detected
		// we store the reference in the agent for future comparisons
		agent->closest_secretory_cell_id = closest_cell;
		agent->closest_secretory_cell_distance = closest_cell_distance;
		// write the corresponding collision_message
		add_secretory_cell_collision_message(secretory_cell_collision_messages, agent->id,
			agent->closest_secretory_cell_id, agent->closest_secretory_cell_distance,
			wall_pts.x, wall_pts.y, wall_pts.z, wall_pts.w, wall_direction.x, wall_direction.y,
			wall_direction.z, wall_direction.w, wall_direction_length,
			wall_normal.x, wall_normal.y, wall_normal.z, wall_normal.w, wall_normal_length);
	}
	return 0;
}

/*
//Given three colinear points p, q, r, the function checks if
//point q lies on line segment 'pr'
//if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) && q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
//return true;
//return false;
__device__ int on_segment(float p_x, float p_y, float q_x, float q_y, float r_x, float r_y) {
	if (q_x <= max(p_x, r_x) && q_x >= min(p_x, r_x) && q_y <= max(p_y, r_y) && q_y >= min(p_y, r_y))
		return 1;
	return 0;
}


// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
__device__ int orientation(float p_x, float p_y, float q_x, float q_y, float r_x, float r_y)
{
	// See https://www.geeksforgeeks.org/orientation-3-ordered-points/ 
	// for details of below formula. 
	int val = (q_y - p_y) * (r_x - q_x) - (q_x - p_x) * (r_y - q_y);

	if (val == 0) return 0;  // colinear 

	return (val > 0) ? 1 : 2; // clock or counterclock wise 
}

// The main function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
// Adapted from:
// https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
__device__ bool segments_intersect(float p1_x, float p1_y, float q1_x, float q1_y, float p2_x, float p2_y, float q2_x, float q2_y)
{
	// Find the four orientations needed for general and 
	// special cases 
	int o1 = orientation(p1_x, p1_y, q1_x, q1_y, p2_x, p2_y);
	int o2 = orientation(p1_x, p1_y, q1_x, q1_y, q2_x, q2_y);
	int o3 = orientation(p2_x, p2_y, q2_x, q2_y, p1_x, p1_y);
	int o4 = orientation(p2_x, p2_y, q2_x, q2_y, q1_x, q1_y);

	// General case 
	if (o1 != o2 && o3 != o4)
		return true;

	// Special Cases 
	// p1, q1 and p2 are colinear and p2 lies on segment p1q1 
	if (o1 == 0 && onSegment(p1, p2, q1)) return true;

	// p1, q1 and q2 are colinear and q2 lies on segment p1q1 
	if (o2 == 0 && onSegment(p1, q2, q1)) return true;

	// p2, q2 and p1 are colinear and p1 lies on segment p2q2 
	if (o3 == 0 && onSegment(p2, p1, q2)) return true;

	// p2, q2 and q1 are colinear and q1 lies on segment p2q2 
	if (o4 == 0 && onSegment(p2, q1, q2)) return true;

	return false; // Doesn't fall in any of the above cases 
}

// this check works if we check for collisions in the past. 
// i.e.: we first displace the ev, then check for collisions and solve them, then we persist the state of the system
__FLAME_GPU_FUNC__ int test_secretory_cell_collision_v2(xmachine_memory_EV* agent, xmachine_message_secretory_cell_location_list* location_messages,
	xmachine_message_secretory_cell_location_PBM* partition_matrix, xmachine_message_secretory_cell_collision_list* secretory_cell_collision_messages)
{
	int closest_cell = -1;
	float closest_cell_distance = 10000.f;
	float3 closest;
	float4 res;
	float4 direction;
	float4 cell_pts;
	float direction_length;

	xmachine_message_secretory_cell_location* message = get_first_secretory_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message) {
		// 1. check if the displacement vector intersects with the wall/cell vector
		bool intersection = segments_intersect(agent->x_1, agent->y_1, agent->x, agent->y, message->p1_x, message->p1_y, message->p2_x, message->p2_y);
		// 2. if they intersect, we need to identify at what time they intersect.
		// Collisions must be resolved in order, on a first occurred - first solved basis.
		if (intersection) {

		}
		else {
			point_to_line_segment_distance(agent->x, agent->y, message->p1_x, message->p1_y, message->p2_x, message->p2_y);

		}


			//printf("secretory res.x: %f res.y:%f radius_um:%f\n", res.x, res.y, agent->radius_um);
			if (closest_cell == -1 || (closest_cell > -1 && res.y < closest_cell_distance)) // save the reference if this is the first candidate collision
			{
				closest_cell = message->id;
				closest_cell_distance = res.y;
				closest.x = res.z;
				closest.y = res.w;
				//closest.z = message->angle; //get_boundary_angle(offset_angles[agent->sec] + i);
				direction.x = message->direction_x;
				direction.y = message->direction_y;
				// direction unit values
				direction.z = message->direction_x_unit;
				direction.w = message->direction_y_unit;
				direction_length = message->direction_length;

				cell_pts.x = message->p1_x;
				cell_pts.y = message->p1_y;
				cell_pts.z = message->p2_x;
				cell_pts.w = message->p2_y;
			}

		// get the next message
		message = get_next_secretory_cell_location_message(message, location_messages, partition_matrix);
	}

	if (closest_cell > -1) { // a collision was detected
							 // we store the reference in the agent for future comparisons
		agent->closest_secretory_cell_id = closest_cell;
		agent->closest_secretory_cell_distance = closest_cell_distance;
		// write the corresponding collision_message
		add_secretory_cell_collision_message(secretory_cell_collision_messages, agent->id,
			agent->closest_secretory_cell_id, agent->closest_secretory_cell_distance,
			cell_pts.x, cell_pts.y, cell_pts.z, cell_pts.w, direction.x, direction.y,
			direction.z, direction.w, direction_length);
	}
	return 0;
}
*/

/**
Checks any possible collision with the secretory cells on the boundaries.
If a collision occurred, a collision_message is generated and the agent will have the values:
	agent->closest_ciliary_cell_id = closest_cell_id;
	agent->closest_ciliary_cell_distance = closest_cell_distance;
Otherwise, no message is produced and the ev agent will have the default values unchanged:
	agent->closest_ciliary_cell_id = -1;
	agent->closest_ciliary_cell_distance = -1.0f;
*/
__FLAME_GPU_FUNC__ int test_ciliary_cell_collision(xmachine_memory_EV* agent, xmachine_message_ciliary_cell_location_list* location_messages,
	xmachine_message_ciliary_cell_location_PBM* partition_matrix, xmachine_message_ciliary_cell_collision_list* ciliary_cell_collision_messages)
{
	int closest_cell = -1;
	float closest_cell_distance = 100.f;
	float4 res;
	float4 wall_direction;
	float4 wall_normal;
	float4 wall_pts;
	float wall_direction_length, wall_normal_length;

	xmachine_message_ciliary_cell_location* message = get_first_ciliary_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message) {
		// we only check for collisions if the EV is displacing in the same direction as the wall
		float dotProduct = dotprod(agent->vx, agent->vy, message->unit_normal_x, message->unit_normal_y);

		// the dot product will only be negative for vectors in opposite directions
		if (dotProduct < 0) {
			// get the distance between agent and cell
			res = point_to_line_segment_distance(agent->x, agent->y, message->p1_x, message->p1_y, message->p2_x, message->p2_y);

			// if the ev is in collision route and the distance to segment < radius
			if (res.x == 1.0f && res.y < agent->radius_um)
			{
				//printf("ciliary res.x: %f res.y:%f radius_um:%f\n", res.x, res.y, agent->radius_um);
				// save the reference if this is the first candidate collision
				// or if the agent is closer to the wall than the previous collision
				if (res.y < closest_cell_distance)
				{
					closest_cell = message->id;
					closest_cell_distance = res.y;
					// wall direction values
					wall_direction.x = message->direction_x;
					wall_direction.y = message->direction_y;
					// wall direction unit values
					wall_direction.z = message->direction_x_unit;
					wall_direction.w = message->direction_y_unit;
					wall_direction_length = message->direction_length;
					// wall normal values
					wall_normal.x = message->normal_x;
					wall_normal.y = message->normal_y;
					// wall unit normal values
					wall_normal.z = message->unit_normal_x;
					wall_normal.w = message->unit_normal_y;
					wall_normal_length = message->normal_length;

					wall_pts.x = message->p1_x;
					wall_pts.y = message->p1_y;
					wall_pts.z = message->p2_x;
					wall_pts.w = message->p2_y;
				}
			}
		}
		message = get_next_ciliary_cell_location_message(message, location_messages, partition_matrix);
	}
	
	if (closest_cell > -1) { // a collision was detected
		// agent is closest to the detected ciliary cell
		agent->closest_ciliary_cell_id = closest_cell;
		agent->closest_ciliary_cell_distance = closest_cell_distance;
		// write the corresponding collision_message
		add_ciliary_cell_collision_message(ciliary_cell_collision_messages, agent->id, 
			agent->closest_ciliary_cell_id, agent->closest_ciliary_cell_distance,
			wall_pts.x, wall_pts.y, wall_pts.z, wall_pts.w, wall_direction.x, wall_direction.y,
			wall_direction.z, wall_direction.w, wall_direction_length,
			wall_normal.x, wall_normal.y, wall_normal.z, wall_normal.w, wall_normal_length);
	}
	return 0;
}


/**
This function has two cases.
Current Ev is the EV1 in collision message.
Current Ev in the EV2 in collision message.
As a single message is generated per collisison, we must use it to compensate both EVs involved
*/
__FLAME_GPU_FUNC__ int ev_collision_resolution(xmachine_memory_EV* agent, xmachine_message_ev_collision_list* ev_collision_messages){
	float dmin, cs, sc, vp1, vp2, ddt, dx, dy, ax, ay, distance, va1, va2, vb, vaP;

	// fetch the message where this EV is involved in the collision
	xmachine_message_ev_collision* message = get_first_ev_collision_message(ev_collision_messages);
	while (message){
		if ( (message->ev1_id == agent->id && message->ev2_id == agent->closest_ev_id) 
			|| (message->ev2_id == agent->id && message->ev1_id == agent->closest_ev_id) ){
			// agent is ev1
			dmin = message->ev1_r_um + message->ev2_r_um;

			// compute the direction resulting of the collision
			cs = (message->ev2_x - message->ev1_x) / (message->distance == 0 ? FLT_MIN : message->distance);
			sc = (message->ev2_y - message->ev1_y) / (message->distance == 0 ? FLT_MIN : message->distance);

			// calculate the component of velocity in direction cs, sc for each particle
			vp1 = message->ev1_vx * cs + message->ev1_vy * sc; // ev1
			vp2 = message->ev2_vx * cs + message->ev2_vy * sc; // ev2

			// back to collision time
			float vp1vp2 = vp1 - vp2;
			ddt = (dmin - message->distance) / (vp1vp2 == 0 ? FLT_MIN : vp1vp2);

			// time of collision occuring after the time step ? don't affect current velocities
			// otherwise, get the next position back to the moment of collision
			if (ddt > dt) {
				ddt = 0;
			}
			else {
				if (ddt < 0)
					ddt = dt;
			}
			if (message->ev1_id == agent->id) {
				agent->x -= message->ev1_vx * ddt; // ev1 - agent->x
				agent->y -= message->ev1_vy * ddt;
			}
			else {
				agent->x -= message->ev2_vx * ddt; // ev2 - agent
				agent->y -= message->ev2_vx * ddt;
			}

			// calculate components of velocity
			dx = (message->ev2_x - message->ev1_x);
			dy = (message->ev2_y - message->ev1_y);

			// where x1, y1 are center of ball1, and x2, y2 are center of ball2
			distance = sqrt(dx*dx + dy*dy);
			// Unit vector in the direction of the collision
			ax = dx / (distance == 0 ? FLT_MIN : distance);
			ay = dy / (distance == 0 ? FLT_MIN : distance);
			// Projection of the velocities in these axes
			// vector sum for velocity p1, p2
			va1 = message->ev1_vx * ax + message->ev1_vy * ay;
			if (message->ev1_id == agent->id) {
				vb = -(message->ev1_vx) * ay + message->ev1_vy * ax;
			}
			else {
				vb = -message->ev2_vx * ay + message->ev2_vy * ax;
			}
			va2 = message->ev2_vx * ax + message->ev2_vy * ay;
			// New velocities in these axes(after collision)

			float vaP;

			if (message->ev1_id == agent->id) {
				float one_plus_mass1_div_mass2 = 1 + message->ev1_mass / message->ev2_mass;
				vaP = va1 + 2 * (va2 - va1) / (one_plus_mass1_div_mass2 == 0 ? FLT_MIN : one_plus_mass1_div_mass2);
			}
			else {
				float one_plus_mass2_div_mass1 = 1 + message->ev2_mass / message->ev1_mass;
				vaP = va2 + 2 * (va1 - va2) / (one_plus_mass2_div_mass1 == 0 ? FLT_MIN : one_plus_mass2_div_mass1);
			}
			// undo the projections, get back to the original coordinates
			agent->vx = vaP * ax - vb * ay;
			agent->vy = vaP * ay + vb * ax; // new vx, vy for ball 1 after collision;
			agent->velocity_ums = sqrtf(agent->vx * agent->vx + agent->vy * agent->vy);
			// Because we have moved back the time by dt, we need to move the time forward by dt.
			agent->x += agent->vx * (dt - ddt);
			agent->y += agent->vy * (dt - ddt);
		}
		message = get_next_ev_collision_message(message, ev_collision_messages);
	}
	return 0;
}

/**
Checks any possible collision with another EV
If a collision occurred, a collision_message is generated and the agent will have the values:
	agent->closest_ev_id = closest_ev_id;
	agent->closest_ev_distance = closest_ev_distance;
Otherwise, no message is produced and the ev agent will have the default values unchanged:
	agent->closest_ev_id = -1;
	agent->closest_ev_distance = -1.0f;
*/	
__FLAME_GPU_FUNC__ int test_ev_collision(xmachine_memory_EV* agent, xmachine_message_location_list* location_messages, 
	xmachine_message_location_PBM* partition_matrix, xmachine_message_ev_collision_list* ev_collision_messages){
	int closest_ev_id = -1;
	float closest_ev_distance = 10000.f;
	float distance;
	float radii = 0.0;
	float ev2_mass, ev2_r_um, ev2_x, ev2_y, ev2_vx, ev2_vy;

	xmachine_message_location* message = get_first_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	// Identify the closest EV colliding with this EV
	while (message)
	{
		if (message->id != agent->id)
		{
			// check for collision
			distance = euclidean_distance(agent, message->x, message->y);
			radii = agent->radius_um + message->radius_um;

			if (distance <= radii)
			{
				if(closest_ev_id == -1){
					closest_ev_id = message->id;
					closest_ev_distance = distance;

					ev2_mass = message->mass_kg;
					ev2_r_um = message->radius_um;
					ev2_x = message->x;
					ev2_y = message->y;
					ev2_vx = message->vx;
					ev2_vy = message->vy;
				}
				else if (distance < closest_ev_distance){
					closest_ev_id = message->id;
					closest_ev_distance = distance;

					ev2_mass = message->mass_kg;
					ev2_r_um = message->radius_um;
					ev2_x = message->x;
					ev2_y = message->y;
					ev2_vx = message->vx;
					ev2_vy = message->vy;
				}
			}
		}
		message = get_next_location_message(message, location_messages, partition_matrix);
	}
	// record the relevant data
	
	if (closest_ev_id != -1){
		agent->closest_ev_id = closest_ev_id;
		agent->closest_ev_distance = closest_ev_distance;
		// write the corresponding collision_message
		add_ev_collision_message(ev_collision_messages, agent->id, closest_ev_id, closest_ev_distance, 
			agent->mass_kg, agent->radius_um, agent->x, agent->y, agent->vx, agent->vy, 
			ev2_mass, ev2_r_um, ev2_x, ev2_y, ev2_vx, ev2_vy);
	}
	return 0;
}

__FLAME_GPU_FUNC__ int reset_state(xmachine_memory_EV* agent) {
	agent->closest_ciliary_cell_id = -1;
	agent->closest_ciliary_cell_distance = 100;
	agent->closest_secretory_cell_id = -1;
	agent->closest_secretory_cell_distance = 100;
	agent->closest_ev_id = -1;
	agent->closest_ev_distance = 100;

	return 0;
}

/*
Uses the Box-Mueller transform to generate a pair of normally distributed random numbers
by sampling from two uniformly distributed RNG.
In the original algorithm, the sampled values are transformed into cartesian coordinates,
here they become the new velocity for the next step
Due to limitations in FlameGPU, we sample both continuos numbers from the same RNG.
*/
__FLAME_GPU_FUNC__ int brownian_movement_1d(xmachine_memory_EV* agent, RNG_rand48* rand48) {
	float u1, u2, r, theta;
	if (agent->age > dt * 600) {
		u1 = rnd<CONTINUOUS>(rand48);
		u2 = rnd<CONTINUOUS>(rand48);
		r = sqrt(-2.0 * log(u1));
		theta = 2 * M_PI * u2;
		// the product of r * (cos|sin)(theta) becomes the displacement factor to use in this iteration
		agent->vx = agent->velocity_ums * r * cos(theta);
	}
	//agent->vx = agent->vx;
	//agent->vy = agent->vy;
	return 0;
}

/*
Uses the Box-Mueller transform to generate a pair of normally distributed random numbers
by sampling from two uniformly distributed RNG.
In the original algorithm, the sampled values are transformed into cartesian coordinates, 
here they become the new velocity for the next step.
After secretion, the EV displaces balistically for < 600 iterations, no brownian motion is computed in this case
Due to limitations in FlameGPU, we sample both continuos numbers from the same RNG.
*/
__FLAME_GPU_FUNC__ int brownian_movement_2d(xmachine_memory_EV* agent, RNG_rand48* rand48) {
	float u1, u2, r, theta; 
	agent->bm_vx = 0;
	agent->bm_vy = 0;
	if (agent->age > dt * const_iterations_in_ballistic_displacement) {
		u1 = rnd<CONTINUOUS>(rand48);
		u2 = rnd<CONTINUOUS>(rand48);
		// 'velocity_ums' comes form the SD value of the MSD (Mean squared displacement)
		r = sqrt(-2.0 * log(u1)) * agent->velocity_ums; // computes the radius of a circumference.
		theta = 2 * M_PI * u2; // computes the angle of rotation
		agent->vx = r * cos(theta);
		agent->vy = r * sin(theta);
		// we also save the new values due to brownian motion for debugging/validationn purposes
		agent->bm_vx = agent->vx;
		agent->bm_vy = agent->vy;
	}
	return 0;
}

/**
 * move FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_EV. This represents a single agent instance and can be modified directly.
 
 */
__FLAME_GPU_FUNC__ int move(xmachine_memory_EV* agent){
	
	agent->x_1 = agent->x;
	agent->y_1 = agent->y;

	agent->x += agent->vx * dt;
	agent->y += agent->vy * dt;
	agent->age += dt;

	//agent->velocity_ums = sqrt(agent->diff_rate_um_x_twice_dof * agent->age) / agent->age;
    return 0;
}

// This function can add up tu N agents between steps;
__FLAME_GPU_FUNC__ int secrete_ev(xmachine_memory_SecretoryCell* agent, xmachine_memory_EV_list* EVs, RNG_rand48* rand48){
	if (introduce_n_new_evs > 0)
	{
		// introduce new agents if the last introduction was longer ago than 
		// get_seconds_before_introducing_new_evs() and 
		// a RNG goes above get_new_evs_threshold() and
		//  there is space in the environment
		// (xmachine_memory_EV_MAX - get_agent_EV_default_count() > 0
		if(agent->time_since_last_secreted > seconds_before_introducing_new_evs){
			
			float rn = rnd<CONTINUOUS>(rand48);
			
			if( rn > new_evs_threshold)
			{
				int rand_i = (int)(rnd<CONTINUOUS>(rand48) * 120);
				// setup the new agent
				// our simulations use 30-150 nm EVs
				float diameter_nm = (rand_i % 120) + 30;
				float diameter_um = diameter_nm * 0.001;// / 1000;

				float radius_nm = diameter_nm * .5;
				float radius_um = diameter_um * .5; // ((30-150 nm) * 0.001/2) // (15 to 75) 
				float radius_m = radius_um * 1E-6; // / 1e+6;

				// compute the volume
				float volume = const_pi_4div3 * radius_nm * radius_nm * radius_nm;

				// to convert:
				// N square metres to square micrometre: multiply N * 1e+12
				// N metres to micrometres: multiply N * 1e+6
				float diffusion_rate_m = const_Boltzmann_x_Temp_K / (const_6_pi_viscosity * radius_m); // square metre
				float diffusion_rate_um = diffusion_rate_m * 1e+12; // square micrometre

				float velocity_um = sqrt(4 * diffusion_rate_um);
				float velocity_m = sqrt(4 * diffusion_rate_m);

				// decompose velocity
				float vx = 0;
				float vy = 0; 
				if (new_evs_random_direction > 0) {
					float r = 2 * M_PI * rnd<CONTINUOUS>(rand48);
					vx = velocity_um * cos(r);
					vy = velocity_um * sin(r);
				}
				else {
					vx = velocity_um * agent->direction_x;
					vy = velocity_um * agent->direction_y;
				}

				// displace the ev by a sigle step
				float x = agent->x + vx * dt;
				float y = agent->y + vy * dt;
				unsigned int id = generate_EV_id();
				
				add_EV_agent(EVs, id, x, y, 0,
					agent->x, agent->y, vx, vy, volume * const_mass_per_volume_unit, 0, // <- colour
					radius_um, radius_m, diffusion_rate_m, diffusion_rate_um, 
					diffusion_rate_um * 2 * dof,
					// closest elements and distances
					-1, 100, -1, 100, -1, 100, 0, 
					// place holders for brownian motion values
					0, 0,
					velocity_um, velocity_m);

				agent->time_since_last_secreted = 0;
			}
			else
				agent->time_since_last_secreted += dt;
		}
		else
			agent->time_since_last_secreted += dt;
	}
	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
