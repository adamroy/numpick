package numpick;

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
		Mat raw = Imgcodecs.imread("pictures/unityToothpicks_34.png");
		// Mat raw = Imgcodecs.imread("pictures/lineTest.png");
		// Mat raw = Imgcodecs.imread("pictures/toothpicks.jpg");
				
		ImageProcess preProcessor = new ImageProcess(gray, blur, canny);
		Mat preProcessedImage = preProcessor.process(raw);
		
		Mat lines = new Mat();
		Mat hough = raw.clone();
		boolean prob = false;
		
		if(!prob)
		{
			boolean customHough = false;
			boolean dumpLines = false;
			
			if(customHough)
			{	
				Point[] lineArray = HoughParallelLines.run(preProcessedImage, 1, Math.PI/180, 1, 20, 150, 10);
				
				double maxRho = Math.max(preProcessedImage.width(), preProcessedImage.height()) * Math.sqrt(2);
				int clusters = HierarchicalClustering.countClusters(lineArray, 0.015, maxRho, Math.PI);
				System.out.println("Clusters: " + clusters);
				
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
				Imgproc.HoughLines(preProcessedImage, lines, 1, Math.PI/180, 75);
				
				double maxRho = Math.max(preProcessedImage.width(), preProcessedImage.height()) * Math.sqrt(2);
				int clusters = HierarchicalClustering.countClusters(makePoints(lines), 0.015, maxRho, Math.PI);
				System.out.println("Clusters: " + clusters);
				
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

