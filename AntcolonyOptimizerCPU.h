#pragma once
#include <vector>
#include "AntCPU.h"

class AntcolonyOptimizerCPU
{
public:
	AntcolonyOptimizerCPU(size_t numAnts, int initialLocation, std::vector<double>& distanceMatrix, std::vector<double>& pheromoneMatrix, size_t matrixDim);
	void Run(int numIterations, double alpha, double beta, double evaporationRate);
	void Report();
	double m_bestRouteLength;
private:

	std::vector<AntCPU> m_ants;

	
	std::vector<int> m_bestRoute;

	std::vector<double> m_distanceMatrix;
	std::vector<double> m_pheromoneMatrix;
	std::vector<double> m_pheromoneMatrixChanges;

	size_t m_matrixDim;

};

