package numpick;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.concurrent.locks.Lock;

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
				for(int j=0; j<prototypes.length; j++)
				{
					population[i][j] = prototypes[j].clone();
					tempPopulation[i][j] = prototypes[j].clone();
					population[i][j].initialize();
				}
			}
			
			best = 0;
			bestFitness = 0;
		}
		
		public void select()
		{
			cumulativeFitnesses[0] = 0;
			for(int i=0; i<size; i++)
			{
				cumulativeFitnesses[i+1] = cumulativeFitnesses[i] + fitnesses[i];
			}
			
			for(int i=0; i<population[0].length; i++)
			{
				tempPopulation[0][i].copy(population[best][i].clone());
			}
			best = 0;
			
			for(int i=1; i<size; i++)
			{
				double val = rand.nextDouble() * cumulativeFitnesses[size];
				int index = Arrays.binarySearch(cumulativeFitnesses, val);
				if(index < 0)
					index = (-index - 1) - 1;
				if(index < 0)
					index = 0;
				if(index >= size)
					index = size-1;
				
				for(int j=0; j<population[0].length; j++)
				{
					tempPopulation[i][j].copy(population[index][j]);
				}
			}
			
			Parameter[][] holder = population;
			population = tempPopulation;
			tempPopulation = holder;
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
			// Thread the evaluation to speed it up
			InidividualEvaluationThread[] thread = new InidividualEvaluationThread[size];
			
			for(int i=0; i<size; i++)
			{
				thread[i] = new InidividualEvaluationThread(population[i], evaluator);
				thread[i].start();
			}
			
			for(int i=0; i<size; i++)
			{
				try
				{
					thread[i].join();
				} 
				catch (InterruptedException e)
				{
					e.printStackTrace();
				}
			}
			
			for(int i=0; i<size; i++)
			{
				fitnesses[i] = thread[i].getFitness();
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
		
		private class InidividualEvaluationThread extends Thread
		{
			private Parameter[] individual;
			private Evaluator eval;
			private double fitness;
			
			public InidividualEvaluationThread(Parameter[] individual, Evaluator eval)
			{
				this.individual = individual;
				this.eval = eval;
			}

			@Override
			public void run()
			{
				fitness = eval.evaluate(individual);
			}
			
			public double getFitness()
			{
				return fitness;
			}
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
		double start, end;
		
		public Parameter(String name, double start, double end, double changeRate)
		{
			this.name = name;
			this.changeRate = changeRate;
			this.start = start;
			this.end = end;
		}
		
		public void initialize()
		{
			this.value = start + (end - start) * rand.nextDouble();
		}
		
		public void mutate()
		{
			double r = rand.nextGaussian();
			value += changeRate * r;
			if(value < start)
				value = start;
			if(value > end)
				value = end;
		}
		
		public void copy(Parameter other)
		{
			this.value = other.value;
		}
		
		public Parameter clone()
		{
			Parameter p  = new Parameter(name, start, end, changeRate);
			p.value = this.value;
			return p; 
		}
		
		public double getValue()
		{
			return value;
		}
		
		@Override
		public String toString()
		{
			return String.format("%s: %.4f",  name, value);
		}
	}
	
	public static Parameter[] run(int populationSize, double targetFitness, Evaluator evaluator, Parameter... params)
	{
		Population population = new Population();
		population.initialize(populationSize, params);
		population.evaluate(evaluator);
		
		int gen = 0;
		while(population.getFitness() < targetFitness)
		{
			System.out.printf("Generation %d\n", gen++);
			population.select();
			population.crossover();
			population.mutate();
			population.evaluate(evaluator);
		}
		
		return population.getBest();
	}
	
}
