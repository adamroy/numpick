package numpick;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import numpick.ParameterizedGeneticAlgorithm.Evaluator;
import numpick.ParameterizedGeneticAlgorithm.Parameter;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class Numpick
{
	public static void main(String[] args)
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		// pictureTest();
		// trainWithGA();
		test();
	}
	
	// Harness for testing one picture
	public static void pictureTest()
	{
		ImagePreprocessor.pictureOutput = true;
		
		Mat raw = new Mat();
		AtomicReference<Double> dataPercent = new AtomicReference<Double>();
		//Mat img = ImagePreprocessor.processImage("pictures/miscount.png", raw, dataPercent);
		List<Point> normalLines = new ArrayList<Point>();
		Mat img = ImagePreprocessor.processImage("unityPictures/2f9778a4-9b76-495f-8d55-1e714615b958.png", raw, dataPercent);
		
		Mat cw = new Mat();
		img.copyTo(cw);;
		Core.transpose(cw, cw);
		Core.flip(cw, cw, 1);
		
		int count = countToothpicks(img, cw, 80, 0.0236, dataPercent.get(), raw, normalLines);
		System.out.println("Count: " + count);
	}
	
	static void transformLineCW(Point p, double cwImageWidth)
	{
		double rho = p.x;
		double theta = p.y;
		double h = rho / Math.cos(Math.PI - theta);
		double rhoPrime = (h + cwImageWidth) * rho / h;
		double thetaPrime = theta > Math.PI/2 ? theta - Math.PI/2 : theta + Math.PI/2;
		double x = h / Math.cos(theta);
		if (thetaPrime > Math.PI/2 && rho > x)
			rhoPrime *= -1;
		
		p.x = rhoPrime;
		p.y = thetaPrime;
	}
	
	static boolean approx(double d, double b, double delta)
	{
		return Math.abs(d - b) < delta;
	}
	
	public static void trainWithGA()
	{
		ImagePreprocessor.pictureOutput = false;
		
		// Build the data for the GA
		final List<Mat> images = new ArrayList<Mat>();
		final List<Integer> counts = new ArrayList<Integer>();
		final List<Double> dataPercents = new ArrayList<Double>();
		AtomicReference<Double> dataPercent = new AtomicReference<Double>();
		try
		{
			BufferedReader reader = new BufferedReader(new FileReader("unityPictures/toothpickCounts.txt"));
			String line;
			while((line = reader.readLine()) != null)
			{
				String filename = "unityPictures/" + line.substring(0, 40);
				int actualToothpicks = Integer.parseInt(line.substring(41));
				Mat image = ImagePreprocessor.processImage(filename, null, dataPercent); 
				
				if(image != null)
				{
					images.add(image);
					counts.add(actualToothpicks);
					dataPercents.add(dataPercent.get());
					System.out.println(dataPercent.get());
				}
				
				if(images.size() >= 500)
					break;
			}
			reader.close();
			
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}

		System.out.println("Image processing done.");
		
		Parameter houghThreshold = new Parameter("houghThreshold", 80, 150, 2);
		Parameter splitThreshold = new Parameter("splitThreshold", 0.005, 0.03, 0.001);
		//Parameter splitThreshold = new Parameter("splitThreshold", 1, 2, 0.05);
		
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
				for(int i=0; i<n; i++)
				{
					int estimatedToothpicks = countToothpicks(images.get(i), houghThreshold, dataPercents.get(i), splitThreshold);
					int difference = Math.abs(estimatedToothpicks - counts.get(i));
					fitness += 1 / ((double)difference/counts.get(i) + 1);
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
	
	public static void test()
	{
		ImagePreprocessor.pictureOutput = false;
		
		// Build the data
		final List<Mat> images = new ArrayList<Mat>();
		final List<Mat> imagesCW = new ArrayList<Mat>();
		final List<Integer> counts = new ArrayList<Integer>();
		final List<Double> dataPercents = new ArrayList<Double>();
		AtomicReference<Double> dataPercent = new AtomicReference<Double>();
		try
		{
			BufferedReader reader = new BufferedReader(new FileReader("unityPictures/toothpickCounts.txt"));
			String line;
			reader.skip(20000);
			reader.readLine();
			while((line = reader.readLine()) != null)
			{
				String filename = "unityPictures/" + line.substring(0, 40);
				int actualToothpicks = Integer.parseInt(line.substring(41));
				Mat image = ImagePreprocessor.processImage(filename, null, dataPercent); 
				
				if(image != null)
				{
					Mat cw = new Mat();
					image.copyTo(cw);;
					Core.transpose(cw, cw);
					Core.flip(cw, cw, 1);
					
					images.add(image);
					imagesCW.add(cw);
					counts.add(actualToothpicks);
					dataPercents.add(dataPercent.get());
				}
				
				if(images.size() > 100)
					break;
			}
			reader.close();
			
		}
		catch (Exception e)
		{
			e.printStackTrace();
		}
		
		int houghThreshold = 80;
		double splitThreshold = 0.0236;
		
		int n = images.size();	
		
		double totalFitness = 0;
		double[] fitnesses = new double[n];
		double minFitness = Double.MAX_VALUE;
		double maxFitness = Double.MIN_VALUE;
		
		for(int i=0; i<n; i++)
		{
			int estimatedToothpicks = countToothpicks(images.get(i), imagesCW.get(i), houghThreshold, dataPercents.get(i), splitThreshold, null, null);
			int difference = Math.abs(estimatedToothpicks - counts.get(i));
			fitnesses[i] = (double)difference/counts.get(i); 
			totalFitness += fitnesses[i];
			minFitness = Math.min(minFitness, fitnesses[i]);
			maxFitness = Math.max(maxFitness, fitnesses[i]);
		}
		double averageFitness = totalFitness/n;
		
		double variance = 0;
		for(int i=0; i<n; i++)
		{
			variance += (fitnesses[i] - averageFitness) * (fitnesses[i] - averageFitness);		
		}
		variance /= n;
		double standardDeviation = Math.sqrt(variance); 
		
		System.out.printf("Fitness stats\nAverage: %.2f\nStandard deviation: %.2f\nMax: %.2f\nMin: %.2f\n", averageFitness, standardDeviation, maxFitness, minFitness);
	}
	
	
	public static int countToothpicks(Mat preProcessedImage, int houghThreshold, double percentData, double splitThreshold)
	{
		return countToothpicks(preProcessedImage, null, houghThreshold, splitThreshold, percentData, null, null);
	}
	
	public static int countToothpicks(Mat preProcessedImage, Mat cw, int houghThreshold, double splitThreshold, double percentData, Mat raw, List<Point> lines)
	{
		int count = 0;
		boolean customHough = false;
		Point[] lineArray;
		Point[] lineArrayCW = null;
		
		if(customHough)
		{	
			lineArray = HoughParallelLines.run(preProcessedImage, 1, Math.PI/180, 0.25, 15, 80, 10);
		}
		else
		{
			Mat temp = new Mat();
			Imgproc.HoughLines(preProcessedImage, temp, 1, Math.PI/180, houghThreshold);
			lineArray = makePoints(temp);
			
			Mat temp2 = new Mat();
			Imgproc.HoughLines(cw, temp2, 1, Math.PI/180, houghThreshold);
			lineArrayCW = makePoints(temp2);
			
			for(Point p : lineArrayCW)
				transformLineCW(p, cw.width());
		}
		
		if(ImagePreprocessor.pictureOutput && raw != null)
		{
			drawLines(ImagePreprocessor.outputFolder + "hough.png", lineArray, raw);
		}
		
		List<Point> toothpicks = new ArrayList<Point>();
		Point[] allLines = new Point[lineArray.length + lineArrayCW.length];
		System.arraycopy(lineArray, 0, allLines, 0, lineArray.length);
		System.arraycopy(lineArrayCW, 0, allLines, lineArray.length, lineArrayCW.length);
		
		double maxRho = Math.max(preProcessedImage.width(), preProcessedImage.height()) * Math.sqrt(2);
		count = HierarchicalClustering.countClusters(allLines, splitThreshold, maxRho, Math.PI, toothpicks);
		
		Point[] toothpickArray = new Point[toothpicks.size()];
		toothpicks.toArray(toothpickArray);

		if(ImagePreprocessor.pictureOutput && raw != null)
		{
			drawLines(ImagePreprocessor.outputFolder + "toothpickLines.png", toothpickArray, raw);
		}
		
		if(lines != null)
		{
			lines.addAll(toothpicks);
		}
		
		return count;
	}
	
	
	static void drawLines(String name, Point[] lines, Mat raw)
	{
		Mat out = new Mat();
		raw.copyTo(out);
		
		for(int i=0; i<lines.length; i++)
		{
			double rho = lines[i].x;
			double theta = lines[i].y;
			double a = Math.cos(theta), b = Math.sin(theta);
			double x0 = a*rho, y0 = b*rho;
			Point start = new Point(x0 + 5000 * -b, y0 + 5000 * a);
			Point end = new Point(x0 - 5000 * -b, y0 - 5000 * a);
			Imgproc.line(out, start, end, new Scalar(0,0,0), 1);
		}
		
		Imgcodecs.imwrite(name, out);
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
