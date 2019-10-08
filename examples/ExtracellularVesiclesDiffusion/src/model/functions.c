
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

__device__ __host__ float degrees(float radians){
	return radians * (180 / M_PI);
}

__device__ __host__ float radians(float degrees){
	return degrees * (M_PI / 180.0);
}

/**
* Compute the distance between two agents using the coordinates of their centers
*/
__device__ float euclidean_distance(xmachine_memory_EV* agent, 
	float message_x, float message_y)
{
	return sqrt((agent->x - message_x) * (agent->x - message_x) +
		(agent->y - message_y) * (agent->y - message_y));
}

__FLAME_GPU_INIT_FUNC__ void precompute_values() {
	srand(time(NULL));

	if (*get_ev_collisions() > 0) {
		printf("Ev-Ev collisions are enabled\n");
	}
	else {
		printf("Ev-Ev collisions are not enabled\n");
	}
	iteration_number = 0;
	if (*get_introduce_new_evs() > 0) {
		printf("Attempting introduction of new EVs every %f seconds with a probability threshold of %f\n",
			*get_seconds_before_introducing_new_evs(), *get_new_evs_threshold());
	}
	else {
		printf("No new EVs will be introduced during simulation\n");
	}

	float val0 = 6 * M_PI * *get_const_water_dynamic_viscosity_pas();
	set_const_6_pi_dynamic_viscosity(&val0);
	printf("set_const_6_pi_dynamic_viscosity: %f\n", val0);

	float val1 = 6 * M_PI * *get_const_water_kinematic_viscosity_kg_ums();
	set_const_6_pi_kinematic_viscosity(&val1);
	printf("set_const_6_pi_kinematic_viscosity: %f\n", val1);

	float val2 = *get_const_Boltzmann() * *get_const_Temperature_K();
	printf("boltzmann: %.35f\n", *get_const_Boltzmann());
	set_const_boltzmann_x_temp(&val2);
	printf("set_const_boltzmann_x_temp: %.30f\n", val2);
}

/**
 * output_data FLAMEGPU Agent Function
 * Automatically generated using functions.xslt
 * @param agent Pointer to an agent structure of type xmachine_memory_EV. Represents an agent instance, can be modified directly.
 * @param location_messages Pointer to output message list of type xmachine_message_location_list.
 */
__FLAME_GPU_FUNC__ int output_location_ev_default(xmachine_memory_EV* agent, xmachine_message_location_ev_default_list* location_messages){
	add_location_ev_default_message(location_messages, agent->id, agent->x, agent->y,
		agent->z, agent->radius_um, agent->mass_kg, agent->mass_ag, agent->vx, agent->vy);
    return 0;
}
__FLAME_GPU_FUNC__ int output_location_ev_initial(xmachine_memory_EV* agent, xmachine_message_location_ev_initial_list* location_messages) {
	add_location_ev_initial_message(location_messages, agent->id, agent->x, agent->y,
		agent->z, agent->radius_um, agent->mass_kg, agent->mass_ag, agent->vx, agent->vy);
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
	agent->probability_of_secretion = 0; // used for debugging purposes only
	agent->time_to_next_secretion_attempt -= dt;
	add_secretory_cell_location_message(location_messages, agent->id, agent->x, agent->y, 0,
		agent->p1_x, agent->p1_y, agent->p2_x, agent->p2_y, agent->direction_x, agent->direction_y,
		agent->direction_x_unit, agent->direction_y_unit, agent->direction_length, 
		agent->normal_x, agent->normal_y, agent->unit_normal_x, agent->unit_normal_y, agent->normal_length
	);
	return 0;
}


__device__ float dotprod(float x1, float y1, float x2, float y2) {
	return x1*x2 + y1*y2;
}

__device__ float vlength(float x, float y) {
	return sqrt(x*x + y*y);
}

// https://github.com/takagi/cl-cuda/blob/master/include/float4.h
__device__ float2 float2_add(float2 a, float2 b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}
__device__ float4 float4_add(float4 a, float4 b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

__device__ float2 float2_sub(float2 a, float2 b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}
__device__ float4 float4_sub(float4 a, float4 b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

__device__ float2 float2_scale(float2 a, float k)
{
	return make_float2(a.x * k, a.y * k);
}
__device__ float4 float4_scale(float4 a, float k)
{
	return make_float4(a.x * k, a.y * k, a.z * k, a.w * k);
}
__device__ float float2_dot(float2 a, float2 b)
{
	return a.x * b.x + a.y * b.y;
}
__device__ float float4_dot(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

/*
Scales b by scale and adds it to a
*/
__device__ float2 add_scaled(float2 a, float2 b, float scale) {
	return float2_add(a, float2_scale(b, scale));
}

__device__ float2 parallel(float vec_x, float vec_y, float u) {
	float factor = (u / vlength(vec_x, vec_y));
	return make_float2(vec_x * factor, vec_y * factor);
}
/*
Returns the length of the projection of vector A in the direction of vector B
*/
__device__ float projection(float a_x, float a_y, float b_x, float b_y) {
	float length = vlength(a_x, a_y);
	float lengthVec = vlength(b_x, b_y);
	if (length == 0 || lengthVec == 0)
		return 0;
	return (a_x * b_x + a_y * b_y) / lengthVec;
}

__device__ float2 float2_project(float2 v1, float2 v2) {
	float mag = projection(v1.x, v1.y, v2.x, v2.y);
	return parallel(v2.x, v2.y, mag);
}
__device__ float2 project(float v1x, float v1y, float v2x, float v2y) {
	return float2_project(make_float2(v1x, v1y), make_float2(v2x, v2y));
}

__device__ float2 force_drag(xmachine_memory_EV* agent) {
	float vel_mag_ums = vlength(agent->vx, agent->vy); // in (um/s)
	float2 vel_unit = make_float2(agent->vx / vel_mag_ums, agent->vy / vel_mag_ums);
	// kinematic viscosity in kg/um.s
	float factor = const_6_pi_kinematic_viscosity * agent->radius_um * vel_mag_ums;
	return float2_scale(vel_unit, factor);
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
	// add_scaled(ballp1, wdir.unit(), proj1*-1)
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
	float angle = acosf(dotprod(agent->vx, agent->vy, cell_dir_x, cell_dir_y) 
		/ (vlength(agent->vx, agent->vy) * cell_direction_length));
	// 2. reposition object
	float normal_x = cell_unit_normal_x;
	float normal_y = cell_unit_normal_y;

	// 3. compute deltaS
	float dist_dp_normal = dotprod(perp_dist_x, perp_dist_y, normal_x, normal_y);
	float deltaS = (agent->radius_um + dist_dp_normal) / sin(angle);

	// 4. estimate what was the displacement = velocity parallel to delta
	float2 displ = parallel(agent->vx, agent->vy, deltaS);

	// 5. update position by subtracting the displacement
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
	float velocityProjection = projection(agent->vx, agent->vy, perp_dist_x, perp_dist_y);
	float2 normalVelocity = parallel(perp_dist_x, perp_dist_y, velocityProjection);

	// velocity vector component parallel to wall; unchanged by impact
	float tangentVelocity_x = agent->vx - normalVelocity.x;
	float tangentVelocity_y = agent->vy - normalVelocity.y;

	// velocity vector component perpendicular to wall just after impact
	agent->vx = tangentVelocity_x + (-normalVelocity.x);
	agent->vy = tangentVelocity_y + (-normalVelocity.y);

	return 0;
}

__device__  int solve_segment_end_point_collision(xmachine_memory_EV* agent, 
	float wall_pt_x, float wall_pt_y, float caller
	//, float cell_direction_length, float dist_x, float dist_y
) {
	// bounce off endpoint wall_p
	float distp_x = agent->x - wall_pt_x;
	float distp_y = agent->y - wall_pt_y;

	// move particle so that it just touches the endpoint			
	float L = agent->radius_um - vlength(distp_x, distp_y);
	float vrel = vlength(agent->vx, agent->vy);
	agent->x += agent->vx * (-L / vrel);
	agent->y += agent->vy * (-L / vrel);

	// normal velocity vector just before the impact
	float2 normalVelo = project(agent->vx, agent->vy, distp_x, distp_y);
	// tangential velocity vector
	float tangentVelo_x = agent->vx - normalVelo.x;
	float tangentVelo_y = agent->vy - normalVelo.y;
	// normal velocity vector after collision

	//// final velocity vector after collision
	agent->vx = -1.0 * normalVelo.x + tangentVelo_x;
	agent->vy = -1.0 * normalVelo.y + tangentVelo_y;
	agent->velocity_ms = caller;
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
				solve_segment_end_point_collision(agent, message->p1_x, message->p1_y, 601.0);
			}
			else if (abs(vlength(ev_wall_vectors.z, ev_wall_vectors.w)) < agent->radius_um) {
				// collision with 2nd end point
				solve_segment_end_point_collision(agent, message->p2_x, message->p2_y, 602.0);
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
			bool tunneling = false;

			/*
			this check can be done by comparing the triangles described by the points
			involved...
			if (wall.side*dist.dotProduct(wall.normal) < 0){
				testTunneling = true;
			} else{
				testTunneling = false;
			}
			*/

			if ((vlength(perp_dist.x, perp_dist.y) < agent->radius_um || tunneling) && test_needed) {

				solve_segment_collision(agent, message->cell_direction_x, message->cell_direction_y, message->cell_direction_length,
					perp_dist.x, perp_dist.y, message->unit_normal_x, message->unit_normal_y);
			}
			else if (abs(vlength(ev_wall_vectors.x, ev_wall_vectors.y)) < agent->radius_um) {
				// collision with 1st end point
				solve_segment_end_point_collision(agent, message->p1_x, message->p1_y, 603.0);
			}
			else if (abs(vlength(ev_wall_vectors.z, ev_wall_vectors.w)) < agent->radius_um) {
				// collision with 2nd end point
				solve_segment_end_point_collision(agent, message->p2_x, message->p2_y, 604.0);
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
A collision is possible if the point:
	- is projecting on the segment, 
	- is not yet proyecting on the segment but displacing towards the segment and within a shorter distance from the endpoints than its radius
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
	//float projecting_on_segment = 0.0f;

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
		//projecting_on_segment = 1.0f;
		xx = x1 + param * C;
		yy = y1 + param * D;
	}

	// dx / dy is the vector from p0 to the closest point on the segment
	dx = x - xx;
	dy = y - yy;

	result.x = param;
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
	float3 wall_normal;
	float2 wall_unit_normal;
	float4 wall_pts;
	float wall_direction_length;

	xmachine_message_secretory_cell_location* message = get_first_secretory_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message) {
		// we only check for collisions if the EV: -is displacing in opposite or perpendicular direction to the wall
		float dotProduct = dotprod(agent->vx, agent->vy, message->unit_normal_x, message->unit_normal_y);

		// the dot product will be negative for vectors in opposite directions and zero for those perpendicular to each other
		if (dotProduct <= 0) {
			// get the distance between an agent's new position and a cell
			res = point_to_line_segment_distance(agent->x, agent->y,
				message->p1_x, message->p1_y, message->p2_x, message->p2_y);

			// if the ev new position is projecting over the segment and the distance to the segment < radius
			if (res.y < agent->radius_um)
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
					wall_normal.z = message->normal_length;
					// wall unit normal values
					wall_unit_normal.x = message->unit_normal_x;
					wall_unit_normal.y = message->unit_normal_y;
					

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
			wall_normal.x, wall_normal.y, wall_unit_normal.x, wall_unit_normal.y, wall_normal.z);
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
	float3 wall_normal;
	float2 wall_unit_normal;
	float4 wall_pts;
	float wall_direction_length;

	xmachine_message_ciliary_cell_location* message = get_first_ciliary_cell_location_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	while (message) {
		// we only check for collisions if the EV: -is displacing in opposite or perpendicular direction to the wall
		float dotProduct = dotprod(agent->vx, agent->vy, message->unit_normal_x, message->unit_normal_y);

		// the dot product will be negative for vectors in opposite directions and zero for those perpendicular to each other
		if (dotProduct < 0) {
			// get the distance between agent and cell
			res = point_to_line_segment_distance(agent->x, agent->y,
				message->p1_x, message->p1_y, message->p2_x, message->p2_y);

			// if the distance to segment < radius
			if (res.y < agent->radius_um)
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
					wall_normal.z = message->normal_length;
					// wall unit normal values
					wall_unit_normal.x = message->unit_normal_x;
					wall_unit_normal.y = message->unit_normal_y;

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
			wall_normal.x, wall_normal.y, wall_unit_normal.x, wall_unit_normal.y, wall_normal.z);
	}
	return 0;
}

struct PostCollisionData {
	float2 correctedLocation;
	float2 correctedVelocity;
	float correctionFactor;
	float overlap;
	float vrel;
};
/*
Computes the new position and velocity after a collision for an agent.
It can compute the values for two agents involved, however, we can only update one
active agent at a time
*/
__device__ struct PostCollisionData solve_collision_ev_default_ev_default(float2 ev1_loc, float2 ev1_velo, float ev1_mass_ag,
	float2 ev2_loc, float2 ev2_velo, float ev2_mass_ag, float min_distance, float2 dist, float dist_length) {
	struct PostCollisionData pcd;
	// normal velocity vectors just before the impact
	float2 normal_velocity1 = float2_project(ev1_velo, dist);
	float2 normal_velocity2 = float2_project(ev2_velo, dist);
	// tangential velocity vectors
	float2 tangent_velocity1 = float2_sub(ev1_velo, normal_velocity1);

	// move particles so that they just touch
	pcd.overlap = min_distance - dist_length;
	float2 normal_velo_subtracted = float2_sub(normal_velocity1, normal_velocity2);
	pcd.vrel = vlength(normal_velo_subtracted.x, normal_velo_subtracted.y);
	pcd.correctionFactor = pcd.overlap / pcd.vrel;
	pcd.correctedLocation = add_scaled(ev1_loc, normal_velocity1, -pcd.correctionFactor);
	
	// normal velocity components after the impact
	float u1 = projection(normal_velocity1.x, normal_velocity1.y, dist.x, dist.y);
	float u2 = projection(normal_velocity2.x, normal_velocity2.y, dist.x, dist.y);
	float v1 = ((ev1_mass_ag - ev2_mass_ag)*u1 + 2 * ev2_mass_ag*u2) / (ev1_mass_ag + ev2_mass_ag);

	normal_velocity1 = parallel(dist.x, dist.y, v1);

	pcd.correctedVelocity = float2_add(normal_velocity1, tangent_velocity1);
	return pcd;
}

struct PostCollisionDataInitial {
	float2 correctedPosition;
	float2 correctedVelocity;
	float correctionFactor;
	float overlap;
	float vrel;
	float vnorm;
	float2 auxiliaryFactors;
};
__device__ struct PostCollisionDataInitial solve_collision_ev_default_ev_initial(float2 ev1_loc, float2 ev1_velo, float ev1_mass_ag,
	float2 ev2_loc, float2 ev2_velo, float ev2_mass_ag, float min_distance, float2 dist, float dist_length) {
	struct PostCollisionDataInitial pcdi;

	// normal velocity vectors just before the impact
	float2 normal_velocity1 = float2_project(ev1_velo, dist);
	float2 normal_velocity2 = float2_project(ev2_velo, dist);
	// tangential velocity vectors
	float2 tangent_velocity1 = float2_sub(ev1_velo, normal_velocity1);

	// move particles so that they just touch
	pcdi.overlap = min_distance - dist_length;
	float2 normal_velo_subtracted = float2_sub(normal_velocity1, normal_velocity2);
	pcdi.vrel = vlength(normal_velo_subtracted.x, normal_velo_subtracted.y);
    pcdi.vnorm = vlength(ev1_velo.x, ev1_velo.y);
    float dp = ev1_velo.x*ev2_velo.x + ev1_velo.y*ev2_velo.y;
	pcdi.auxiliaryFactors.x = pcdi.overlap / pcdi.vrel;
	pcdi.auxiliaryFactors.y = pcdi.auxiliaryFactors.x > 100 ? pcdi.auxiliaryFactors.x /10 : pcdi.auxiliaryFactors.x;
	pcdi.correctionFactor = (pcdi.overlap * 4) / pcdi.vnorm;
	if(dp < 0){
		pcdi.auxiliaryFactors.y = -pcdi.auxiliaryFactors.y;
		pcdi.correctionFactor = -pcdi.correctionFactor;
	}
	//float2 new_ev1_loc = add_scaled(ev1_loc, normal_velocity1, pcdi.correctionFactors.y);
    pcdi.correctedPosition = add_scaled(ev1_loc, normal_velocity1, pcdi.correctionFactor);

	// normal velocity components after the impact
	float u1 = projection(normal_velocity1.x, normal_velocity1.y, dist.x, dist.y);
	float u2 = projection(normal_velocity2.x, normal_velocity2.y, dist.x, dist.y);
	float v1 = ((ev1_mass_ag - ev2_mass_ag)*u1 + 2 * ev2_mass_ag*u2) / (ev1_mass_ag + ev2_mass_ag);
	normal_velocity1 = parallel(dist.x, dist.y, v1);
	pcdi.correctedVelocity = float2_add(normal_velocity1, tangent_velocity1);

	return pcdi;
}

/*
 *  This function is WIP.
 *  Collision scenarios:
 *  - default, same direction - cannot occur
 *  - default, oposite direction - ev_initial unchanged
 *  - default, orthogonal directions - ev_initial unchanged
 *  - initial, same direction - cannot occur
 *  - initial, oposite direction - They must deviate to a side
 *  - initial, orthogonal directions - Std collision resolution?
 */
__device__ float4 solve_collision_ev_initial_ev_any(float2 ev1_loc, float2 ev1_velo, float ev1_mass_ag, 
	float2 ev2_loc, float2 ev2_velo, float ev2_mass_ag, float min_distance, float2 dist, float dist_length) {
	// normal velocity vectors just before the impact
	float2 normal_velocity1 = float2_project(ev1_velo, dist);
	float2 normal_velocity2 = float2_project(ev2_velo, dist);
	// tangential velocity vectors
	float2 tangent_velocity1 = float2_sub(ev1_velo, normal_velocity1);
	//float2 tangent_velocity2 = ev2_velo - normal_velocity2;

	// move particles so that they just touch
	float L = min_distance - dist_length;
	float2 normal_velo_subtracted = float2_sub(normal_velocity1, normal_velocity2);
	float vrel = vlength(normal_velo_subtracted.x, normal_velo_subtracted.y);
	//float2 new_ev1_loc = add_scaled(make_float2(0, 0), normal_velocity1, -L / vrel);
	float dp = float2_dot(normal_velocity1, normal_velocity2);
	float2 new_ev1_loc = add_scaled(ev1_loc, normal_velocity1, -L / vrel);
	return make_float4(new_ev1_loc.x, new_ev1_loc.y, ev1_velo.x, ev1_velo.y);
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
__FLAME_GPU_FUNC__ int test_collision_ev_default_ev_default(xmachine_memory_EV* agent, xmachine_message_location_ev_default_list* location_messages, 
	xmachine_message_location_ev_default_PBM* partition_matrix){
	
	float distance, closest_ev_distance = 100.f;
	float radii = 0.0, radius_to_radius;
	float max_overlap = 0, overlap;

	int ev2_id = -1;
	float ev2_x, ev2_y, ev2_vx, ev2_vy, ev2_mass_ag;
	float2 distance_vector;
	//float4 new_values;
	struct PostCollisionData pcd;

	xmachine_message_location_ev_default* message = get_first_location_ev_default_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	// This check could be improved by identifying and solving only the first collision involving this agent.
	while (message)	{
		if (message->id != agent->id){
			// check for collision
			distance = euclidean_distance(agent, message->x, message->y);
			radii = agent->radius_um + message->radius_um;

			if (distance <= radii){
				overlap = radii - distance;
				if (ev2_id == -1 || overlap > max_overlap){
					ev2_id = message->id;
					closest_ev_distance = distance;
					radius_to_radius = radii;
					max_overlap = overlap;
					distance_vector = make_float2(agent->x - message->x, agent->y - message->y);
					ev2_x = message->x;
					ev2_y = message->y;
					ev2_vx = message->vx;
					ev2_vy = message->vy;
					ev2_mass_ag = message->mass_ag;
					//ev2_radius_um = message->radius_um;
				}
			}
		}
		message = get_next_location_ev_default_message(message, location_messages, partition_matrix);
	}
	if (ev2_id != -1) {
		pcd = solve_collision_ev_default_ev_default(
			make_float2(agent->x, agent->y), make_float2(agent->vx, agent->vy), agent->mass_ag,
			make_float2(ev2_x, ev2_y), make_float2(ev2_vx, ev2_vy), ev2_mass_ag,
			radius_to_radius, distance_vector, vlength(distance_vector.x, distance_vector.y));
		// we store the current position as the previous and update the values accordingly
		agent->x_1 = agent->x;
		agent->y_1 = agent->y;

		agent->x = pcd.correctedLocation.x;
		agent->y = pcd.correctedLocation.y;
		agent->vx = pcd.correctedVelocity.x;
		agent->vy = pcd.correctedVelocity.y;
		agent->bm_impulse_t_left = pcd.correctionFactor; // fac1
		agent->mass_kg = pcd.overlap;
		agent->colour = pcd.vrel;  // L
        /*
		agent->radius_m = ;
        agent->diffusion_rate_m;
        agent->velocity_ms;
		*/
		agent->closest_ev_id = ev2_id;
		agent->closest_ev_distance = closest_ev_distance;
	}

	return 0;
}

__FLAME_GPU_FUNC__ int test_collision_ev_default_ev_initial(xmachine_memory_EV* agent, xmachine_message_location_ev_initial_list* location_messages, 
	xmachine_message_location_ev_initial_PBM* partition_matrix){
	
	float distance, closest_ev_distance = 100.f;
	float radii = 0.0, radius_to_radius;
	float max_overlap = 0, overlap;

	int ev2_id = -1;
	float ev2_x, ev2_y, ev2_vx, ev2_vy, ev2_mass_ag;
	float2 distance_vector;
	struct PostCollisionDataInitial pcdi;

	xmachine_message_location_ev_initial* message = get_first_location_ev_initial_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	// This check could be improved by identifying and solving only the first collision involving this agent.
	while (message)	{
		if (message->id != agent->id){
			// check for collision
			distance = euclidean_distance(agent, message->x, message->y);
			radii = agent->radius_um + message->radius_um;

			if (distance <= radii){
				overlap = radii - distance;
				if (ev2_id == -1 || overlap > max_overlap){
					ev2_id = message->id;
					closest_ev_distance = distance;
					radius_to_radius = radii;
					max_overlap = overlap;
					distance_vector = make_float2(agent->x - message->x, agent->y - message->y);
					ev2_x = message->x;
					ev2_y = message->y;
					ev2_vx = message->vx;
					ev2_vy = message->vy;
					ev2_mass_ag = message->mass_ag;
					//ev2_radius_um = message->radius_um;
				}
			}
		}
		message = get_next_location_ev_initial_message(message, location_messages, partition_matrix);
	}
	if (ev2_id != -1) {
		pcdi = solve_collision_ev_default_ev_initial(
			make_float2(agent->x, agent->y), make_float2(agent->vx, agent->vy), agent->mass_ag,
			make_float2(ev2_x, ev2_y), make_float2(ev2_vx, ev2_vy), ev2_mass_ag,
			radius_to_radius, distance_vector, vlength(distance_vector.x, distance_vector.y));
		// we store the current position as the previous and update the values accordingly
		agent->x_1 = agent->x;
		agent->y_1 = agent->y;

		agent->x = pcdi.correctedPosition.x;
		agent->y = pcdi.correctedPosition.y;
		agent->vx = pcdi.correctedVelocity.x;
		agent->vy = pcdi.correctedVelocity.y;
		agent->bm_impulse_t_left = pcdi.correctionFactor;
		agent->mass_kg = pcdi.overlap;
		agent->colour = pcdi.vnorm;
		agent->radius_m = pcdi.vrel; // auxiliary
		agent->diffusion_rate_m = pcdi.auxiliaryFactors.x; // factor 1
        agent->velocity_ms = pcdi.auxiliaryFactors.y; // factor 2

		agent->closest_ev_id = ev2_id;
		agent->closest_ev_distance = closest_ev_distance;
	}

	return 0;
}

// the agent must be repositioned but velocity unchanged
__FLAME_GPU_FUNC__ int test_collision_ev_initial_ev_default(xmachine_memory_EV* agent, xmachine_message_location_ev_default_list* location_messages, 
	xmachine_message_location_ev_default_PBM* partition_matrix){
	
	float distance, closest_ev_distance = 100.f;
	float radii = 0.0, radius_to_radius;
	float max_overlap = 0, overlap;

	int ev2_id = -1;
	float ev2_x, ev2_y, ev2_vx, ev2_vy, ev2_mass_ag;
	float2 distance_vector;
	float4 new_values;

	xmachine_message_location_ev_default* message = get_first_location_ev_default_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	// This check could be improved by identifying and solving only the first collision involving this agent.
	while (message)	{
		if (message->id != agent->id){
			// check for collision
			distance = euclidean_distance(agent, message->x, message->y);
			radii = agent->radius_um + message->radius_um;

			if (distance <= radii){
				overlap = radii - distance;
				if (ev2_id == -1 || overlap > max_overlap){
					ev2_id = message->id;
					closest_ev_distance = distance;
					radius_to_radius = radii;
					max_overlap = overlap;
					distance_vector = make_float2(agent->x - message->x, agent->y - message->y);
					ev2_x = message->x;
					ev2_y = message->y;
					ev2_vx = message->vx;
					ev2_vy = message->vy;
					ev2_mass_ag = message->mass_ag;
					//ev2_radius_um = message->radius_um;
				}
			}
		}
		message = get_next_location_ev_default_message(message, location_messages, partition_matrix);
	}
	if (ev2_id != -1) {
		new_values = solve_collision_ev_initial_ev_any(
			make_float2(agent->x, agent->y), make_float2(agent->vx, agent->vy), agent->mass_ag,
			make_float2(ev2_x, ev2_y), make_float2(ev2_vx, ev2_vy), ev2_mass_ag,
			radius_to_radius, distance_vector, vlength(distance_vector.x, distance_vector.y));
		// we store the current position as the previous and update the values accordingly
		agent->x_1 = agent->x;
		agent->y_1 = agent->y;
		agent->x = new_values.x;
		agent->y = new_values.y;
		agent->vx = new_values.z;
		agent->vy = new_values.w;

		agent->closest_ev_id = ev2_id;
		agent->closest_ev_distance = closest_ev_distance;
	}

	return 0;
}
// the agent must be repositioned but velocity unchanged
__FLAME_GPU_FUNC__ int test_collision_ev_initial_ev_initial(xmachine_memory_EV* agent, xmachine_message_location_ev_initial_list* location_messages, 
	xmachine_message_location_ev_initial_PBM* partition_matrix){
	
	float distance, closest_ev_distance = 100.f;
	float radii = 0.0, radius_to_radius;
	float max_overlap = 0, overlap;

	int ev2_id = -1;
	float ev2_x, ev2_y, ev2_vx, ev2_vy, ev2_mass_ag;
	float2 distance_vector;
	float4 new_values;

	xmachine_message_location_ev_initial* message = get_first_location_ev_initial_message(location_messages, partition_matrix, agent->x, agent->y, agent->z);

	// This check could be improved by identifying and solving only the first collision involving this agent.
	while (message)	{
		if (message->id != agent->id){
			// check for collision
			distance = euclidean_distance(agent, message->x, message->y);
			radii = agent->radius_um + message->radius_um;

			if (distance <= radii){
				overlap = radii - distance;
				if (ev2_id == -1 || overlap > max_overlap){
					ev2_id = message->id;
					closest_ev_distance = distance;
					radius_to_radius = radii;
					max_overlap = overlap;
					distance_vector = make_float2(agent->x - message->x, agent->y - message->y);
					ev2_x = message->x;
					ev2_y = message->y;
					ev2_vx = message->vx;
					ev2_vy = message->vy;
					ev2_mass_ag = message->mass_ag;
					//ev2_radius_um = message->radius_um;
				}
			}
		}
		message = get_next_location_ev_initial_message(message, location_messages, partition_matrix);
	}
	if (ev2_id != -1) {
		new_values = solve_collision_ev_initial_ev_any(
			make_float2(agent->x, agent->y), make_float2(agent->vx, agent->vy), agent->mass_ag,
			make_float2(ev2_x, ev2_y), make_float2(ev2_vx, ev2_vy), ev2_mass_ag,
			radius_to_radius, distance_vector, vlength(distance_vector.x, distance_vector.y));
		// we store the current position as the previous and update the values accordingly
		agent->x_1 = agent->x;
		agent->y_1 = agent->y;
		agent->x = new_values.x;
		agent->y = new_values.y;
		agent->vx = new_values.z;
		agent->vy = new_values.w;

		agent->closest_ev_id = ev2_id;
		agent->closest_ev_distance = closest_ev_distance;
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

__FLAME_GPU_FUNC__ int initial_to_default(xmachine_memory_EV* agent) {
	//agent->colour = agent->age;
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
	
	u1 = rnd<CONTINUOUS>(rand48);
	u2 = rnd<CONTINUOUS>(rand48);
	r = sqrt(-2.0 * log(u1));
	theta = 2 * M_PI * u2;
	// the product of r * (cos|sin)(theta) becomes the displacement factor to use in this iteration
	agent->vx = agent->velocity_ums * r * cos(theta) * dt;
	
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
__FLAME_GPU_FUNC__ int brownian_movement_2d_v1(xmachine_memory_EV* agent, RNG_rand48* rand48) {
	float u1, u2, r, theta; 
	
	if (agent->bm_impulse_t_left <= 0) {
		u1 = rnd<CONTINUOUS>(rand48);
		u2 = rnd<CONTINUOUS>(rand48);
		agent->bm_impulse_t_left = 0;// rnd<CONTINUOUS>(rand48);
		// 'velocity_ums' comes form the SD value of the MSD (Mean squared displacement)
		r = sqrt(-2.0 * log(u1)) * agent->velocity_ums * dt; // computes the radius of a circumference smaller than velocity_ums.
		theta = 2 * M_PI * u2; // computes the angle of rotation
		agent->bm_vx = r * cos(theta);
		agent->bm_vy = r * sin(theta);
		// integrate the effect of brownian motion
		agent->vx += agent->bm_vx;
		agent->vy += agent->bm_vy;
	}
	else {
		agent->bm_impulse_t_left -= dt;
	}
	return 0;
}

__FLAME_GPU_FUNC__ int brownian_movement_2d_v2(xmachine_memory_EV* agent, RNG_rand48* rand48) {
	float u1, u2, r, theta;
	
	if (agent->bm_impulse_t_left <= 0) {
		u1 = rnd<CONTINUOUS>(rand48);
		u2 = rnd<CONTINUOUS>(rand48);
		agent->bm_impulse_t_left = 0;
										// 'velocity_ums' comes form the SD value of the MSD (Mean squared displacement)
		r = sqrt(-2.0 * log(u1)) * agent->velocity_ums * dt; // computes the radius of a circumference smaller than velocity_ums.
		theta = 2 * M_PI * u2; // computes the angle of rotation
		agent->bm_vx = r * cos(theta);
		agent->bm_vy = r * sin(theta);
		// integrate the effect of brownian motion
		agent->vx += agent->bm_vx;
		agent->vy += agent->bm_vy;
	}
	else {
		agent->bm_impulse_t_left -= dt;
	}
	return 0;
}

__FLAME_GPU_FUNC__ int brownian_movement_2d(xmachine_memory_EV* agent, RNG_rand48* rand48) {
	float u1, u2, r, theta;
	
	//u1 = rnd<CONTINUOUS>(rand48);
	u2 = rnd<CONTINUOUS>(rand48);
	//agent->bm_impulse_t_left = 0; // rnd<CONTINUOUS>(rand48);
	// 'velocity_ums' comes form the SD value of the MSD (Mean squared displacement)
	// compute the radius of a circumference
	r = sqrt(agent->vx * agent->vx + agent->vy * agent->vy); // we factor this radius by 0.1 to keep the values within [-1,1]
	theta = 2 * M_PI * u2; // computes the angle of rotation
	agent->bm_vx = (r * cos(theta));
	agent->bm_vy = (r * sin(theta));

	return 0;
}

__FLAME_GPU_FUNC__ int kill_ev(xmachine_memory_EV* agent, RNG_rand48* rand48){
	float rn = rnd<CONTINUOUS>(rand48);
	if(rn > 0.98){
		return 1;
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
    	return 0;
}
__FLAME_GPU_FUNC__ int moveInitial(xmachine_memory_EV* agent) {

	agent->x_1 = agent->x;
	agent->y_1 = agent->y;

	agent->x += agent->vx * dt;
	agent->y += agent->vy * dt;

	agent->age += dt;

	agent->closest_ev_id = -1;
	agent->closest_ev_distance = 100;

	return 0;
}

// This function can add up tu N agents between steps;
__FLAME_GPU_FUNC__ int secrete_ev(xmachine_memory_SecretoryCell* agent, xmachine_memory_EV_list* EVs, RNG_rand48* rand48){
	if(introduce_new_evs > 0)
	{
		// compute the next interval of secretion
		if(agent->time_to_next_secretion_attempt < 0)
		{
			//float min_next_secretion = ((0.3 / (0.16 * dt)) * dt) * 2;
			agent->time_to_next_secretion_attempt = seconds_before_introducing_new_evs/2 + (rnd<CONTINUOUS>(rand48) * seconds_before_introducing_new_evs);
			//agent->time_to_next_secretion = tt_next_secretion < min_next_secretion ? min_next_secretion : tt_next_secretion;

			agent->probability_of_secretion = rnd<CONTINUOUS>(rand48);
					
			if( agent->probability_of_secretion > new_evs_threshold)
			{
				int rand_i = (int)(rnd<CONTINUOUS>(rand48) * max_ev_radius);
				// our simulations work with EVs with diameters in the range 80-320 nm
				// therefore, the radius must be in the range 40-160 nm
				float radius_nm = (rand_i % max_ev_radius) + min_ev_radius;
				float radius_um = (radius_nm) / 1000; // faster than doing /1000
				float radius_m = radius_um / 1E6;

				// compute the volume
				//float radius_nm_3 = radius_nm * radius_nm * radius_nm;
				//float volume = const_pi_4div3 * radius_nm_3;
				//float mass_g = const_mass_per_volume_unit * volume;
				float mass_g = const_mass_p_vol_u_x_4div3_pi * radius_nm * radius_nm * radius_nm;
				float mass_kg = mass_g * 1e3;
				float mass_ag = mass_g / 1e-18;

				// to convert:
				// N square metres to square micrometre: multiply N * 1e+12
				// N metres to micrometres: multiply N * 1e+6
				float diffusion_rate_ms = const_Boltzmann_x_Temp_K / (const_6_pi_dynamic_viscosity * radius_m); // square metre
				float diffusion_rate_ums = diffusion_rate_ms * 1e+12; // square micrometre

				// compute the diffusion-rate dependant Mean Square Displacement
				float velocity_ums = sqrt(2 * dof * diffusion_rate_ums * dt);
				float velocity_ms = sqrt(2 * dof * diffusion_rate_ms * dt);

				// decompose velocity
				float vx = velocity_ums * agent->unit_normal_x;
				float vy = velocity_ums * agent->unit_normal_y;

				// choose a random starting point
				int rand_i2 = (int)(rnd<CONTINUOUS>(rand48) * agent->source_points) - 1;
				while(rand_i2 < 0 || rand_i2 == agent->last_source_point_secreting) {
					rand_i2 = (int)(rnd<CONTINUOUS>(rand48) * agent->source_points) - 1;
				}
				int ssp = rand_i2 * xmachine_memory_SecretoryCell_MAX;
				float x = agent->source_points_xs[ssp];
				float y = agent->source_points_ys[ssp];
				unsigned int id = generate_EV_id();
				// displace the startgint coordinates backwards by the diameter of the EV
				x -= agent->unit_normal_x * radius_um;
				y -= agent->unit_normal_y * radius_um;
				
				float time_in_initial = ((radius_um * 2.) / (velocity_ums * dt)) * dt;
				// EV_agent_list, id, x, y z, x_1, y_1, vx, vy, bm_vx, bm_vy, bm_impulse_t_left
				add_EV_agent(EVs, id, x, y, 0, x - vx * dt, y - vy * dt, vx, vy,
					// (float)rand_i,(float)rand_i2,volume,
					0, 0, 0,
					// mass_kg, mass_ag, colour, radius_um, radius_m, diffusion_rate_m,
					mass_kg, mass_ag, 0, radius_um, radius_m, diffusion_rate_ms,
					// diffusion_rate_um, diff_rate_um_x_twice_dof
					diffusion_rate_ums, diffusion_rate_ums * 2 * dof,
					// closest: ev_id, distance, secretory, distance, ciliary, distance, age, velocity_um, velocity_m
					-1, 100, -1, 100, -1, 100, 0, time_in_initial, velocity_ums, velocity_ms);
			}
		}
	}
	return 0;
}

#endif //_FLAMEGPU_FUNCTIONS
