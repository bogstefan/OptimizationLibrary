#include "ParticleSwarmOptimizer.cuh"
#include <thrust/iterator/counting_iterator.h>
#include <thrust/extrema.h>
#include <thrust/for_each.h>
#include <curand_kernel.h>

namespace INTERNAL
{
	namespace PSO
	{
		InitFunctor::InitFunctor(int dimSize, double (*function)(double*, int)) :
			m_dimSize(dimSize),
			m_function(function)
		{}

		__device__
			void InitFunctor::operator()(const thrust::tuple<Particle&, int>& particleIndexTuple) const
		{
			Particle& particle = particleIndexTuple.get<0>();
			int index = particleIndexTuple.get<1>();

			curandState_t randomState;
			curand_init(index, 0, 0, &randomState);

			particle = Particle(m_dimSize, m_function, randomState);
		}

		RunFunctor::RunFunctor(double aInd, double aGrp, double w, double* currentGlobalBest, double* boundsVector, double* boundsSize) :
			m_aInd(aInd),
			m_aGrp(aGrp),
			m_w(w),
			m_currentGlobalBest(currentGlobalBest),
			m_boundsVector(boundsVector),
			m_boundsSize(boundsSize)
		{}

		__device__
			void RunFunctor::operator()(const thrust::tuple<Particle&, double*&>& particlePersonalBestTuple)
		{
			Particle& particle = particlePersonalBestTuple.get<0>();
			double* personalBest = particlePersonalBestTuple.get<1>();
			particle.Run(m_aInd, m_aGrp, m_w, personalBest, m_currentGlobalBest, m_boundsVector, m_boundsSize);
		}
	}
}

ParticleSwarmOptimizer::ParticleSwarmOptimizer(int numParticles, double (*function)(double*, int), int numDimensions)
{
	thrust::counting_iterator<int> counter = thrust::make_counting_iterator(0);
	auto beginIterator = thrust::make_zip_iterator(thrust::make_tuple(mdv_particles.begin(), counter));
	auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(mdv_particles.end(), counter + numDimensions));
	INTERNAL::PSO::InitFunctor initFunctor(numDimensions, function);
	thrust::for_each(beginIterator, endIterator, initFunctor);
}

void ParticleSwarmOptimizer::Run(int numIterations, double aInd, double aGrp, double w, const thrust::host_vector<double>& boundsVector, const thrust::host_vector<double>& boundsSize)
{
	mdv_boundsVector = boundsVector;
	mdv_boundsSize = boundsSize;

	auto beginIterator = thrust::make_zip_iterator(thrust::make_tuple(mdv_particles.begin(), currentParticleBestPositionsPointers.begin()));
	auto endIterator = thrust::make_zip_iterator(thrust::make_tuple(mdv_particles.end(), currentParticleBestPositionsPointers.end()));
	INTERNAL::PSO::RunFunctor runFunctor(aInd, aGrp, w, mdv_currentGlobalBestPosition.data().get(), mdv_boundsVector.data().get(), mdv_boundsSize.data().get());

	for (int i = 0; i < numIterations; ++i)
	{
		thrust::for_each(beginIterator, endIterator, runFunctor);
		auto minElementIter = thrust::min_element(mdv_particles.begin(), mdv_particles.end(), [] __device__(const Particle & a, const Particle & b) { return a.GetMinimum() < b.GetMinimum(); });
		auto minElementIndex = minElementIter - mdv_particles.begin();


		mdv_currentGlobalBestPosition = mdv_currentParticleBestPositions[minElementIndex];
	}
}
