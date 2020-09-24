#include "AntSimple.cuh"
#include <stdio.h>

namespace SIMPLE
{
	__device__
		Ant::Ant(int initialLocation, int matrixDim, curandState_t randState) :
		visitedIndex(0),
		isVisited(new bool[matrixDim]),
		position(initialLocation),
		goodnessNumerators(new double[matrixDim]),
		m_randomState(randState)
	{
	}

	__device__
		void Ant::Venture(int* route, const double* distanceMatrix, const double* pheromoneMatrix, int matrixDim, double alpha, double beta)
	{
		while (visitedIndex < matrixDim)
		{
			int nextHop = SelectNextHop(distanceMatrix, pheromoneMatrix, matrixDim, alpha, beta);
			GoTo(nextHop, route, distanceMatrix, matrixDim);
		}
		route[matrixDim] = route[0];
		distance += distanceMatrix[route[matrixDim - 1] * matrixDim + route[0]];
		//printf("Distance Traveled: %f\n", distance);
	}

	__device__
		int Ant::SelectNextHop(const double* distance_matrix, const double* pheromoneMatrix, int matrixDim, double alpha, double beta)
	{
		double denominator = 0;
		for (int i = 0; i < matrixDim; ++i)
		{
			if (isVisited[i]) { continue; }
			int possiblePosition = i;

			double goodnessNumerator = pow(pheromoneMatrix[position * matrixDim + possiblePosition], alpha) * pow(1.0 / distance_matrix[position * matrixDim + possiblePosition], beta);

			goodnessNumerators[possiblePosition] = goodnessNumerator;
			denominator += goodnessNumerator;
		}

		//New
		/*
		for (int i = 0; i < matrixDim; ++i)
		{
			if (isVisited[i]) { continue; }
			goodnessNumerators[i] /= denominator;
		}

		double random = curand_uniform_double(&m_randomState);

		for (int i = 0; i < matrixDim; ++i)
		{
			if (isVisited[i]) { continue; }
			random -= goodnessNumerators[i];
			if (random <= 0) { return i; }
			
		}
		return -1;
		*/
		

		double sum = 0;
		double random = curand_uniform_double(&m_randomState);
		//printf("Random is %f\n", random);
		
		for (int i = 0; i < matrixDim; ++i)
		{
			if (isVisited[i]) { continue; }

			int possiblePosition = i;
			double numerator = goodnessNumerators[possiblePosition];
			double probability = numerator / denominator;
			if (random <= sum + probability)
			{
				return possiblePosition;
			}
			sum += probability;
		}
		return -1;
		
	}

	__device__
		void Ant::GoTo(int next, int* route, const double* distanceMatrix, int matrixDim)
	{
		route[visitedIndex++] = next;
		isVisited[next] = true;
		distance += distanceMatrix[position * matrixDim + next];
		position = next;
	}

	__device__
		void Ant::Reset(int* route, int initialLocation, int matrixDim)
	{
		visitedIndex = 0;
		distance = 0;
		position = initialLocation;
		for (int i = 0; i < matrixDim; ++i) { isVisited[i] = false; }
		isVisited[position] = true;
		route[visitedIndex++] = initialLocation;
	}


}


