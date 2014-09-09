#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <math.h>
#include <assert.h>
 
// gcc gmm_linear.c -Wall -O3 -g -march=native -mtune=native -mavx -o gmm_linear -lm

#define LOGSUM_LOOKUP_SIZE 10000
#define LOGSUM_LOOKUP_SCALE 1000.0f
#define LOGSUM_LIMIT 10.0f

void check_alloc(int alloc_result);
void initialize(int amount);
void initialize_gmm(int index, int gauss_count, float *mixture_values);
void initialize_gaussian(int gmm_index, int gauss_index, int dimension, float *means, float *presicions, float c);
void print_gmm(int index);
void print_floats(float *floats, int amount);
void print_gaussian(int gmm_index, int gauss_index);
void initialize_logsum();
float logsum(float loga, float logb);

 
/*	Represents a diagonal gaussian. However it only contains enough information to 
	calculate log likelihoods. presicions contain -0.5/variance values. */
	
typedef struct gaussian{
	float *means;
	float *presicions;
	float c;
	int dimension;
} gaussian;
 
/* Represents a gauss mixture model. mixture weights are actually log values. */
 
typedef struct gmm{
	float *mixture_weights;
	gaussian *gaussians;
	int count;
} gmm;
 
gmm *gmms;
int gmm_count;

float *logsum_lookup;

void initialize_logsum() {
	logsum_lookup=malloc(sizeof(float) * LOGSUM_LOOKUP_SIZE);
	int i;
	for(i = 0; i < LOGSUM_LOOKUP_SIZE; ++i) {
		logsum_lookup[i] = logf(1 + expf(((float)-i) / LOGSUM_LOOKUP_SCALE));
	}
}

float logsum(float loga, float logb) {
	if (loga > logb) {
		const float dif = loga - logb; // logA-logB because during lookup calculation dif is multiplied with -1
		return dif>=LOGSUM_LIMIT ? loga : loga + logsum_lookup[(int)(dif * LOGSUM_LOOKUP_SCALE)];
	} else {		
		const float dif = logb - loga;
		return dif>=LOGSUM_LIMIT ? logb : logb + logsum_lookup[(int)(dif * LOGSUM_LOOKUP_SCALE)];		
	}
}

/* Allocates for `amount` of gmms. */
 
void initialize(int amount) {  
	gmm_count = amount;	
	gmms=malloc(sizeof(gmm) * amount);	
}

/* Allocates 'gauss_count' amount of gaussian for gmm[index] with mixture values. */
void initialize_gmm(int index, int gauss_count, float *mixture_values) {
	gmm *g = &gmms[index];
	g->count = gauss_count;    
	g->mixture_weights = mixture_values;
	g->gaussians=malloc(sizeof(gaussian) * gauss_count);	
}
 
void initialize_gaussian(int gmm_index, int gauss_index, int dimension, float *means, float *presicions, float c) {  
 	gmm *g = &gmms[gmm_index];
	gaussian *gauss = &g->gaussians[gauss_index];
	gauss->dimension = dimension;
	gauss->means=means;
	gauss->presicions = presicions;
	gauss->c = c;
}
 

static inline float log_likelihood(gaussian *gauss, float *input) {	
	int i;
	float result = 0.0;
	for(i = 0; i<gauss->dimension; i++) {
		const float dif = input[i]-gauss->means[i];
		result += (dif*dif*gauss->presicions[i]);
	}
	return result + gauss->c;
}
 
float score_gmm (int gmm_index, float *input) { 
	gmm *g = &gmms[gmm_index];	
	const float *mixtures = g->mixture_weights;
	float score = 0.0f;
	int k=0;
	for(k = 0; k < g->count; k++) {
		float likelihood = log_likelihood( &g->gaussians[k], input) + mixtures[k];
		score = logsum(score, likelihood);			
	}
	return score;	
}
 
void print_gmm(int index) {
	printf("gmm index = %d\n", index);
	gmm g = gmms[index];
	printf("gauss count = %d\n", g.count);
	printf("mixture weights = ");
	print_floats(g.mixture_weights, g.count);
	printf("\n");
}
 
void print_gaussian(int gmm_index, int gauss_index) {
	gmm *g = &gmms[gmm_index];
	gaussian *gauss = &g->gaussians[gauss_index];
	printf("gmm, gauss_index = %d,%d\n", gmm_index, gauss_index);
	printf("means = ");	
	print_floats(gauss->means, gauss->dimension);
	printf("\n");
	printf("presicions = ");	
	print_floats(gauss->presicions, gauss->dimension);
	printf("\n");
}
 
void print_floats(float *floats, int amount) {
	int i;
	printf("[");
	for(i = 0; i < amount ; ++i) {
		printf("%.3f", floats[i]);
		if(i < amount-1) {
			printf(" ");
		}
	}
	printf("]");	
}
 
void check_alloc(int i) {
	if(i!=0) { 
		printf("Allocation failure %d\n", i);
		exit(i);
	}
}
 
void test_linear() {
	int gmm_count = 1000;
	int dimension = 40;
	int gauss_count = 32;
	int input_amount = 1000;
	int batch_size = 4;	
	
	initialize_logsum();

	initialize(gmm_count);
    printf("gmms allocated\n");
    
    // prepare gmms
    
    float mi = -0.00033f;
	float mi_start = -0.001f;
	
	int i,k,z;
 
    for(i = 0; i < gmm_count; ++i) {
    	float *mixtures = malloc(sizeof(float) * gauss_count);    	  	
    	for(k = 0; k < gauss_count; ++k) {
    		mixtures[k] = mi_start + (float)k * mi;
	   	}
	   	mi_start+=mi;	   	
	   	initialize_gmm(i, gauss_count, mixtures);
	}	
 
	// prepare gaussians
	
    float ma = -0.00011f;
	float ma_start = -0.003f;
	float pa = -0.00025f;
	float pa_start = -0.007f;
	
    for(i = 0; i < gmm_count; ++i) {
    	for(k = 0; k < gauss_count; ++k) {
	  		float *means = malloc(sizeof(float) * dimension);
	   		float *presicions = malloc(sizeof(float) * dimension);
	   		for(z = 0; z < dimension ; ++z) {
	   			means[z] = ma_start + (float)z * ma;
	   			presicions[z] = pa_start + (float)z * pa;
	   		}
	   		ma_start+=ma;
	   		pa_start+=pa;
	   		initialize_gaussian(i, k, dimension, means, presicions, -0.3f);
	   	}
	}
	printf("Gausses initialized.");
	
	// Prepare input
	float **input = malloc(sizeof(float*) * input_amount);
	
	for(i=0; i<input_amount; ++i) {
	  input[i] = malloc(sizeof(float) * dimension);
	}
	
    float ia = 0.0011f;
	float ia_start = -0.75f;
	
	for(i = 0; i < input_amount; ++i) {			
		for(k = 0; k < dimension; ++k) {
		  input[i][k] = ia_start + (float)k * ia;
		}
		ia_start+=ia;
	}
	
	// run test
    clock_t start = clock(), diff;	
	float result = 0.0f;
	for(i = 0; i < input_amount; i+=batch_size) {			
		for(k = 0; k < gmm_count; ++k) {
			int b;
			for(b=0; b<batch_size; ++b) {
				if(b+i>=input_amount) {				
					break;
				}
				result+= score_gmm(k, &input[b+i][0]);
			}
		}
	}
	diff = clock() - start;	
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("result = %f", result);
	printf("Time taken %d seconds %d milliseconds\n", msec/1000, msec%1000);		
}
 
int main( int argc, char *argv[] )
{   
	test_linear();		
    return 0;
}

