package numpick;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Range;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class OpenCVTest
{
	static String outputFolder = "processedPictures/";
	static String suffix = "";
	
	
	static ImageProcessor gray = new ImageProcessor()
	{
		@Override
		public Mat process(Mat src)
		{
			Mat gray = new Mat();
			Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
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
			Imgcodecs.imwrite(outputFolder+"canny"+suffix+".png", canny);
			return canny;
		}
	};
	
	public static void main(String[] args)
	{
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		//Mat raw = Imgcodecs.imread("pictures/lineTest.png");
		Mat raw = Imgcodecs.imread("pictures/toothpicks.jpg");
				
		ImageProcess preProcessor = new ImageProcess(gray, blur, canny);
		Mat preProcessedImage = preProcessor.process(raw);
		
		Mat lines = new Mat();
		Mat hough = raw.clone();
		boolean prob = false;
		
		if(!prob)
		{
			boolean customHough = true;
			if(customHough)
			{	
				AtomicReference<int[][]> accumulatorRef = new AtomicReference<>();
				double[][] lineArray = HoughLines(preProcessedImage, 1, Math.PI/360, 0, 0, 100, 3, accumulatorRef);
				
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
				
				for(int i=0; i<lineArray.length; i++)
				{
				      double[] vec = lineArray[i];
				      double rho = vec[0], theta = vec[1];
				      double a = Math.cos(theta), b = Math.sin(theta);
				      double x0 = a*rho, y0 = b*rho;
				      Point start = new Point(x0 + 5000 * -b, y0 + 5000 * a);
				      Point end = new Point(x0 - 5000 * -b, y0 - 5000 * a);
				      Imgproc.line(hough, start, end, new Scalar(0,0,0), 1);
				}
			}
			else
			{
				Imgproc.HoughLines(preProcessedImage, lines, 1, Math.PI/180, 100);
				
				for(int i=0; i<lines.rows(); i++)
				{
				      double[] vec = lines.get(i, 0);
				      double rho = vec[0], theta = vec[1];
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
		
		
		
		//Mat all = combineMats(raw, gray, blur, canny, hough);
		Imgcodecs.imwrite(outputFolder+"lines"+suffix+".png", hough);
	}
	
	static Mat combineMats(Mat ...mats)
	{	
		int n = mats.length;
		int row = mats[0].rows();
		int col = mats[0].cols();
		
		Mat out = new Mat(row * n, col, CvType.CV_16UC3);
		
		for(int i=0; i<n; i++)
		{
			Range range = new Range(i*row, (i+1)*row);
			out.rowRange(range).setTo(mats[i]);
		}
		
		return out;
	}
	
	static double[][] HoughLines(Mat image, double deltaRho, double deltaTheta, double width, double maxWidth, int threshold, int maximaRadius, AtomicReference<int[][]> accumulatorOut)
	{
		int maxDim = (int)(Math.max(image.width(), image.height())/deltaRho);
		int accumWidth = (int)(maxDim * Math.sqrt(2)) * 2;
		int accumHeight = (int)(Math.PI/deltaTheta);
		int[][] accumulator = new int[accumHeight][accumWidth];
		
		// Run the accumulator on all non-zero pixels
		byte[] pixel = new byte[1];		
		for(int j=0; j<image.rows(); j++)
		{
			for(int i=0; i<image.cols(); i++)
			{
				int num = image.get(i, j, pixel);
				if(num > 0 )
				{
					int val = (int)pixel[0] & 0xFF;
					if(val > 0)
					{
						for(double theta=0; theta<Math.PI; theta+=deltaTheta)
						{
							double x = i,
									y = j,
									dx = Math.cos(Math.PI - theta) * 10,
									dy = Math.sin(Math.PI - theta) * 10,
									xp = x - dx,
									yp = y - dy,
									rho = originToLineDistance(x, y, xp, yp);
							
							int thetaIndex = (int)Math.floor(theta / deltaTheta);
							int rhoIndex = (int)Math.floor((rho)/deltaRho) + accumWidth/2;
							accumulator[thetaIndex][rhoIndex] += 1;
						}
					}
				}
			}
		}
		
		// Extract the rho's and theta's from the accumulator
		List<Double> rhoList = new ArrayList<>();
		List<Double> thetaList = new ArrayList<>();
		for(int j=0; j<accumHeight; j++)
		{
			for(int i=0; i<accumWidth; i++)
			{
				int value = accumulator[j][i];
				if(value > threshold)
				{
					// Check to see if this point is a local maxima, as if it is not then the other point should be the line
					// Prevents lines segment doubles
					// r is the radius to check for local maxima
					boolean localMaxima = true;
					int r = maximaRadius;
					for(int dx=-r; dx<=r; dx++)
					{
						for(int dy=-r; dy<=r; dy++)
						{
							if(j+dy>=0 && j+dy<accumHeight && i+dx>=0 && i+dx<accumWidth && !(dx==0 && dy==0))
							{
								if(accumulator[j+dy][i+dx] > value)
									localMaxima = false;
							}
						}
					}
					
					if(localMaxima)
					{
						rhoList.add((i-accumWidth/2)*deltaRho);
						thetaList.add(j*deltaTheta);
					}
				}
			}
		}
		
		// Package as a double array for easy consumption
		double[][] lines = new double[rhoList.size()][2];
		System.out.println(rhoList.size());
		for(int i=0; i<rhoList.size(); i++)
		{
			double rho = rhoList.get(i);
			double theta = thetaList.get(i);
			lines[i][0] = rho;
			lines[i][1] = theta;
			
			System.out.printf("(rho = %.2f, theta = %.2f)\n", rho, theta);
		}

		// Return accumulator if asked for
		if(accumulatorOut != null)
			accumulatorOut.set(accumulator);
		return lines;
	}
	
	// https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
	static double originToLineDistance(double x1, double y1, double x2 ,double y2)
	{
		double dx = x2-x1, dy = y2-y1;
		double numer = x2*y1 - y2*x1;
		double denom = Math.sqrt(dx*dx + dy*dy);		
		return numer / denom;
	}
}

