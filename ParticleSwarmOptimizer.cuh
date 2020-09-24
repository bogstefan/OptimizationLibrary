#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <vector>
#include "Particle.cuh"

namespace INTERNAL
{
	namespace PSO
	{
		struct InitFunctor
		{
			int m_dimSize;
			double (*m_function)(double*, int);

			InitFunctor(int dimSize, double (*function)(double*, int));

			__device__
			void operator()(const thrust::tuple<Particle&, int>& particleIndexTuple) const;

		};

		struct RunFunctor
		{
			double m_aInd;
			double m_aGrp;
			double m_w;
			double* m_currentGlobalBest;
			double* m_boundsVector;
			double* m_boundsSize;

			RunFunctor(double aInd, double aGrp, double w, double* currentGlobalBest, double* boundsVector, double* boundsSize);
			__device__
			void operator()(const thrust::tuple<Particle&, double*&>& particlePersonalBestTuple);
		};
	}
}

class ParticleSwarmOptimizer
{
	thrust::device_vector<double> mdv_currentGlobalBestPosition;
	thrust::device_vector<Particle> mdv_particles;
	std::vector<thrust::device_vector<double>> mdv_currentParticleBestPositions;
	thrust::device_vector<double*> currentParticleBestPositionsPointers;
	thrust::device_vector<double> mdv_boundsVector;
	thrust::device_vector<double> mdv_boundsSize;

public:
	ParticleSwarmOptimizer(int numParticles, double (*function)(double*, int), int numDimensions);
	void Run(int numIterations, double aInd, double aGrp, double w, const thrust::host_vector<double>& boundsVector, const thrust::host_vector<double>& boundsSize);
};

