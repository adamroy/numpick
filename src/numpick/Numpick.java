package numpick;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import numpick.ParameterizedGeneticAlgorithm.Evaluator;
import numpick.ParameterizedGeneticAlgorithm.Parameter;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Numpick
{
	public static void main(String[] args)
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// Build the data for the GA
		final List<Mat> images = new ArrayList<Mat>();
		final List<Integer> counts = new ArrayList<Integer>();
		try
		{
			BufferedReader reader = new BufferedReader(new FileReader("unityPictures/toothpickCounts.txt"));
			String line;
			while((line = reader.readLine()) != null)
			{
				String filename = "unityPictures/" + line.substring(0, 40);
				int actualToothpicks = Integer.parseInt(line.substring(41));
				Mat image = ImagePreprocessor.processImage(filename); 
				
				if(image != null)
				{
					images.add(image);
					counts.add(actualToothpicks);
				}
			}
			reader.close();
			
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}

		System.out.println("Image processing done.");
		
		Parameter houghThreshold = new Parameter("houghThreshold", 50, 150, 2);
		Parameter splitThreshold = new Parameter("splitThreshold", 0.005, 0.03, 0.001);
		
		Parameter[] params = ParameterizedGeneticAlgorithm.run(20, 0.9, new Evaluator()
		{
			public double evaluate(Parameter[] parameters)
			{
				int houghThreshold = 0;
				double splitThreshold = 0;
				
				if(parameters[0].name.equals("houghThreshold"))
					houghThreshold = (int)parameters[0].value;
				if(parameters[1].name.equals("splitThreshold"))
					splitThreshold = parameters[1].value;
				
				for(int i=0; i<parameters.length; i++)
				{
					System.out.print(parameters[i] + " ");
				}

				int n = images.size();	
				double fitness = 0;
				for(int i=0; i<10; i++)
				{
					int estimatedToothpicks = countToothpicks(images.get(i), houghThreshold, splitThreshold);
					int difference = Math.abs(estimatedToothpicks - counts.get(i));
					fitness += 1 - (float)difference/counts.get(i);
				}
				
				fitness /= n; 
				System.out.printf("Fitness: %.2f\n", fitness);
				return fitness;
			}
		}, houghThreshold, splitThreshold);
		
		for(int i=0; i<params.length; i++)
		{
			System.out.println(params[i]);
		}
	}
	
	
	public static int countToothpicks(Mat preProcessedImage, int houghThreshold, double splitThreshold)
	{
		int count = 0;
		boolean customHough = false;
		
		if(customHough)
		{	
			Point[] lineArray = HoughParallelLines.run(preProcessedImage, 1, Math.PI/180, 1, 20, 150, 10);
			
			double maxRho = Math.max(preProcessedImage.width(), preProcessedImage.height()) * Math.sqrt(2);
			count = HieracrchicalClustering.countClusters(lineArray, splitThreshold, maxRho, Math.PI);
		}
		else
		{
			Mat lines = new Mat();
			Imgproc.HoughLines(preProcessedImage, lines, 1, Math.PI/180, houghThreshold);
			
			double maxRho = Math.max(preProcessedImage.width(), preProcessedImage.height()) * Math.sqrt(2);
			count = HieracrchicalClustering.countClusters(makePoints(lines), 0.015, maxRho, Math.PI);
		}
		
		return count;
	}
	
	static Point[] makePoints(Mat lines)
	{
		Point[] points = new Point[lines.rows()];
		for(int i=0; i<lines.rows(); i++)
		{
			double[] vec = lines.get(i, 0);
			double rho = vec[0], theta = vec[1];
			points[i] = new Point(rho, theta);
		}
		return points;
	}

}
