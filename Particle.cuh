#pragma once
#include <curand_kernel.h>

class Particle
{
	double* m_currentVelocity{};
	double* m_currentPosition{};

	double m_currentMinimum = 999999;

	int m_dimSize{};

	double (*m_function)(double*, int);

	curandState_t m_randState{};

public:
	__device__ Particle(int dimSize, double (*function)(double*, int), const curandState_t& randomState);

	__device__ Particle();
	__device__ ~Particle();
	__device__ void Run(double aInd, double aGrp, double w, double* currentPersonalBest, const double* currentGlobalBest, const double* boundsVector, const double* boundsSize);
	__device__ double GetMinimum() const;
};

