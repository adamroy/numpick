package numpick;

import java.util.Arrays;
import java.util.Random;

public class ParameterizedGeneticAlgorithm
{
	private static Random rand = new Random(3289579843L);
	
	private static class Population
	{
		int size;
		Parameter[][] population;
		Parameter[][] tempPopulation;
		int best;
		double bestFitness;
		double [] fitnesses;
		double [] cumulativeFitnesses;
		
		public void initialize(int size, Parameter... prototypes)
		{
			this.size = size;
			population = new Parameter[size][prototypes.length];
			tempPopulation = new Parameter[size][prototypes.length];
			fitnesses = new double[size];
			cumulativeFitnesses = new double[size+1];
			
			for(int i=0; i<size; i++)
			{
				Parameter[] individual = new Parameter[prototypes.length];
				for(int j=0; j<prototypes.length; j++)
				{
					individual[j] = prototypes[j].clone();
					individual[j].mutate();
					individual[j].mutate();
					individual[j].mutate();
				}
				
				population[i] = individual;
			}
			
			best = 0;
		}
		
		public void select()
		{
			cumulativeFitnesses[0] = 0;
			for(int i=0; i<size; i++)
			{
				cumulativeFitnesses[i+1] = cumulativeFitnesses[i] + fitnesses[i];
			}
			
			System.arraycopy(population, 0, tempPopulation, 0, size);
			
			population[0] = tempPopulation[best].clone();
			best = 0;
			
			for(int i=1; i<size; i++)
			{
				double val = rand.nextDouble() * cumulativeFitnesses[size];
				int index = Arrays.binarySearch(cumulativeFitnesses, val);
				if(index < 0)
					index = -index - 1;
				
				population[i] = tempPopulation[index].clone();
			}
		}
		
		public void crossover()
		{
			double crossoverRate = 0.75;
			
			for(int i=0; i<size; i+=2)
			{
				if(rand.nextDouble() < crossoverRate)
				{
					Parameter[] one = population[i];
					Parameter[] two = population[i+1];
					
					for(int j=0; j<one.length; j++)
					{
						if(rand.nextDouble() < 0.5)
						{
							Parameter temp = one[j];
							one[j] = two[j];
							two[j] = temp;
						}
					}
				}
			}
		}
		
		public void mutate()
		{
			double mutationRate = 1.0 / population[0].length;
			
			for(int i=0; i<size; i++)
			{
				for(int j=0; j<population[i].length; j++)
				{
					if(rand.nextDouble() < mutationRate)
					{
						population[i][j].mutate();
					}
				}
			}
		}
		
		public void evaluate(Evaluator evaluator)
		{
			for(int i=0; i<size; i++)
			{
				fitnesses[i] = evaluator.evaluate(population[i]);
				if(fitnesses[i] > bestFitness)
				{
					best = i;
					bestFitness = fitnesses[i];
				}
			}
		}
		
		public double getFitness()
		{
			return bestFitness;
		}
		
		public Parameter[] getBest()
		{
			return population[best];
		}
	}
	
	public static interface Evaluator
	{
		public double evaluate(Parameter[] parameters);
	}
	
	public static class Parameter
	{
		String name;
		double value;
		double changeRate;
		
		public Parameter(String name, double initialValue, double changeRate)
		{
			this.name = name;
			this.value = initialValue;
			this.changeRate = changeRate;
		}
		
		public void mutate()
		{
			int r = rand.nextInt(10) - 5;
			value += changeRate * r;
		}
		
		public void copy(Parameter other)
		{
			this.value = other.value;
		}
		
		public Parameter clone()
		{
			return new Parameter(name, value, changeRate);
		}
		
		public double getValue()
		{
			return value;
		}
		
		@Override
		public String toString()
		{
			return name + ": " + value;
		}
	}
	
	public static Parameter[] run(double targetFitness, Evaluator evaluator, Parameter... params)
	{
		Population population = new Population();
		population.initialize(100, params);
		population.evaluate(evaluator);
		
		while(population.getFitness() < targetFitness)
		{
			population.select();
			population.crossover();
			population.mutate();
			population.evaluate(evaluator);
		}
		
		return population.getBest();
	}
	
}
