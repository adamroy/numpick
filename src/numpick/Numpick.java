package numpick;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

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
	static boolean pictureOutput = false;
	static String outputFolder = "processedPictures/";
	static String suffix = "";
	
	static ImageProcessor gray = new ImageProcessor()
	{
		@Override
		public Mat process(Mat src)
		{
			Mat gray = new Mat();
			Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
			if(pictureOutput)
				Imgcodecs.imwrite(outputFolder+"gray"+suffix+".png", gray);
			return gray;
		}
	};
	
	static ImageProcessor eqHist = new ImageProcessor()
	{
		@Override
		public Mat process(Mat src)
		{
			Mat eqHist = new Mat();
			Imgproc.equalizeHist(src, eqHist);
			if(pictureOutput)
				Imgcodecs.imwrite(outputFolder+"eqHist"+suffix+".png", eqHist);
			return eqHist;
		}
	};
	
	static ImageProcessor blur = new ImageProcessor()
	{
		@Override
		public Mat process(Mat src)
		{
			Mat blur = new Mat();
			Imgproc.blur(src, blur, new Size(3, 3));
			if(pictureOutput) 
				Imgcodecs.imwrite(outputFolder+"blur"+suffix+".png", blur);
			return blur;
		}
	};
	
	static ImageProcessor canny = new ImageProcessor()
	{
		@Override
		public Mat process(Mat src)
		{
			int lowThreshold = 50;
			Mat canny = new Mat();
			Imgproc.Canny(src, canny, lowThreshold, lowThreshold*3);
			if(pictureOutput) 
				Imgcodecs.imwrite(outputFolder+"canny"+suffix+".png", canny);
			return canny;
		}
	};

	public static void main(String[] args)
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		
		Parameter houghThreshold = new Parameter("houghThreshold", 50, 150, 2);
		Parameter splitThreshold = new Parameter("splitThreshold", 0.005, 0.03, 0.001);
		
		Parameter[] params = ParameterizedGeneticAlgorithm.run(20, 1.0, new Evaluator()
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
				
				try
				{
					BufferedReader reader = new BufferedReader(new FileReader("unityPictures/toothpickCounts.txt"));
					int correctCount = 0;
					int totalCount = 0;
					
					for(int i=0; i<10; i++)
					{
						String line = reader.readLine();
						String filename = line.substring(0, 40);
						int actualToothpicks = Integer.parseInt(line.substring(41));
						int estimatedToothpicks = countToothpicks("unityPictures/" + filename, houghThreshold, splitThreshold);
						
						if(Math.abs(actualToothpicks - estimatedToothpicks) < 2)
							correctCount++;
						totalCount++;
					}
					
					reader.close();
					
					double fitness = (double)correctCount/totalCount; 
					System.out.printf("Fitness: %.2f\n", fitness);
					return fitness;
				}
				catch (Exception e)
				{
					e.printStackTrace();
				}
				
				return 0;
			}
		}, houghThreshold, splitThreshold);
		
		for(int i=0; i<params.length; i++)
		{
			System.out.println(params[i]);
		}
		
		// System.out.println(countToothpicks("pictures/unityToothpicks_23.png", 150));
	}
	
	
	public static int countToothpicks(String filename, int houghThreshold, double splitThreshold)
	{
		Mat raw = Imgcodecs.imread(filename);				
		ImageProcess preProcessor = new ImageProcess(gray, blur, canny);
		Mat preProcessedImage = preProcessor.process(raw);
		
		Mat lines = new Mat();
		Mat hough = raw.clone();
		boolean prob = false;
		
		int count = 0;
		
		if(!prob)
		{
			boolean customHough = false;
			boolean dumpLines = false;
			
			if(customHough)
			{	
				Point[] lineArray = HoughParallelLines.run(preProcessedImage, 1, Math.PI/180, 1, 20, 150, 10);
				
				double maxRho = Math.max(preProcessedImage.width(), preProcessedImage.height()) * Math.sqrt(2);
				count = HieracrchicalClustering.countClusters(lineArray, splitThreshold, maxRho, Math.PI);
				
				/*
				int[][] accumulatorArray = accumulatorRef.get();
				Mat accumulator = new Mat(accumulatorArray.length, accumulatorArray[0].length, CvType.CV_8U);
				{
					byte[] pixel = new byte[1];
					int thetaDim = accumulatorArray.length;
					int rhoDim = accumulatorArray[0].length;
					for(int j=0; j<thetaDim; j++)
						for(int i=0; i<rhoDim; i++)
						{
							pixel[0] = (byte)accumulatorArray[j][i];
							accumulator.put(j, i, pixel);
						}
					
					accumulator = eqHist.process(accumulator);
					Imgcodecs.imwrite(outputFolder+"accumulator"+suffix+".png", accumulator);
				}
				*/
				
				for(int i=0; i<lineArray.length; i++)
				{
				      Point vec = lineArray[i];
				      double rho = vec.x, theta = vec.y;
				      if(dumpLines)
				    	  System.out.printf("(rho = %.2f, theta = %.2f)\n", rho, theta);
				      double a = Math.cos(theta), b = Math.sin(theta);
				      double x0 = a*rho, y0 = b*rho;
				      Point start = new Point(x0 + 5000 * -b, y0 + 5000 * a);
				      Point end = new Point(x0 - 5000 * -b, y0 - 5000 * a);
				      Imgproc.line(hough, start, end, new Scalar(0,0,0), 1);
				}
			}
			else
			{
				Imgproc.HoughLines(preProcessedImage, lines, 1, Math.PI/180, houghThreshold);
				
				double maxRho = Math.max(preProcessedImage.width(), preProcessedImage.height()) * Math.sqrt(2);
				count = HieracrchicalClustering.countClusters(makePoints(lines), 0.015, maxRho, Math.PI);
				
				for(int i=0; i<lines.rows(); i++)
				{
				      double[] vec = lines.get(i, 0);
				      double rho = vec[0], theta = vec[1];
				      if(dumpLines)
				    	  System.out.printf("(rho=%.2f, theta=%.2f)\n", rho, theta);
				      double a = Math.cos(theta), b = Math.sin(theta);
				      double x0 = a*rho, y0 = b*rho;
				      Point start = new Point(x0 + 5000 * -b, y0 + 5000 * a);
				      Point end = new Point(x0 - 5000 * -b, y0 - 5000 * a);
				      Imgproc.line(hough, start, end, new Scalar(0,0,0), 1);
				}
				
			}
		}
		else 
		{
			Imgproc.HoughLinesP(preProcessedImage, lines, 1, Math.PI/180, 50, 10, 10);
			
			for (int x = 0; x < lines.rows(); x++) 
			{
			      double[] vec = lines.get(x, 0);
			      double x1 = vec[0], 
						 y1 = vec[1],
						 x2 = vec[2],
						 y2 = vec[3];
			      Point start = new Point(x1, y1);
			      Point end = new Point(x2, y2);
			
			      Imgproc.line(hough, start, end, new Scalar(0,0,0), 3);
			}
		}
		
		if(pictureOutput)
			Imgcodecs.imwrite(outputFolder+"lines"+suffix+".png", hough);
		
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
