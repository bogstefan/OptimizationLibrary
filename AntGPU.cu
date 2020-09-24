#include "AntGPU.cuh"
#include <stdio.h>

__device__
AntGPU::AntGPU(int initialLocation, int matrixDim, curandState_t randState) :
	position(initialLocation),
	possibleLocations(new int[matrixDim]),
	goodnessNumerators(new double[matrixDim]),
	possibleLocationsLastIndex(matrixDim - 1),
	m_randomState(randState)
{
}

__device__
void AntGPU::Venture(int* route, const double* distanceMatrix, const double* pheromoneMatrix, int matrixDim, double alpha, double beta)
{
	while (possibleLocationsLastIndex >= 0)
	{
		int nextHop = SelectNextHop(distanceMatrix, pheromoneMatrix, matrixDim, alpha, beta);
		GoTo(nextHop, route, distanceMatrix, matrixDim);
	}
	//printf("Distance Traveled: %f\n", distance + distanceMatrix[position]);
	route[matrixDim] = route[0];
	distance += distanceMatrix[route[matrixDim - 1] * matrixDim + route[0]];
}

__device__
int AntGPU::SelectNextHop(const double* distance_matrix, const double* pheromoneMatrix, int matrixDim, double alpha, double beta)
{
	double denominator = 0;
	for (int i = 0; i <= possibleLocationsLastIndex; ++i)
	{
		int possiblePosition = possibleLocations[i];
		double goodnessNumerator = pow(pheromoneMatrix[position * matrixDim + possiblePosition], alpha) * pow(1.0 / distance_matrix[position * matrixDim + possiblePosition], beta);

		goodnessNumerators[possiblePosition] = goodnessNumerator;
		denominator += goodnessNumerator;
	}
	
	double sum = 0;
	double random = curand_uniform_double(&m_randomState);

	for (int i = 0; i <= possibleLocationsLastIndex; ++i)
	{
		int possiblePosition = possibleLocations[i];
		double numerator = goodnessNumerators[possiblePosition];
		double probability = numerator / denominator;
		if (random <= sum + probability)
		{
			possibleLocationsNextIndex = i;
			return possiblePosition;
		}
		sum += probability;
	}
	return -1;
	
}

__device__
void AntGPU::GoTo(int next, int* route, const double* distanceMatrix, int matrixDim)
{
	route[routeIndex++] = next;
	possibleLocations[possibleLocationsNextIndex] = possibleLocations[possibleLocationsLastIndex--];
	distance += distanceMatrix[position * matrixDim + next];
	position = next;
}

__device__
void AntGPU::Reset(int* route, int initialLocation, int matrixDim)
{
	routeIndex = 0;
	distance = 0;
	position = initialLocation;
	possibleLocationsLastIndex = matrixDim - 1;
	for (int i = 0; i < matrixDim; ++i) { possibleLocations[i] = i; }
	route[routeIndex++] = initialLocation;
	possibleLocations[initialLocation] = possibleLocations[possibleLocationsLastIndex--];
}


