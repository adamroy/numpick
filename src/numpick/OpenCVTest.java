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
	public static void main(String[] args)
	{
		String outputFolder = "processedPictures/";
		String suffix = "";
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat raw = Imgcodecs.imread("pictures/20151104_155628.jpg");
		
		Mat gray = new Mat();
		Imgproc.cvtColor(raw, gray, Imgproc.COLOR_BGR2GRAY);
		Imgcodecs.imwrite(outputFolder+"gray"+suffix+".png", gray);
		
		Mat blur = new Mat();
		Imgproc.blur(gray, blur, new Size(3, 3));
		Imgcodecs.imwrite(outputFolder+"blur"+suffix+".png", blur);
		
		int lowThreshold = 50;
		Mat canny = new Mat();
		Imgproc.Canny(blur, canny, lowThreshold, lowThreshold*3);
		Imgcodecs.imwrite(outputFolder+"canny"+suffix+".png", canny);
		
		Mat lines = new Mat();
		Mat hough = gray.clone();
		boolean prob = false;
		
		if(!prob)
		{
			Imgproc.HoughLines(canny, lines, 1, Math.PI/180, 100);
			System.out.printf("Hough lines:\n%s", lines.dump());
			
			for (int x = 0; x < Math.min(lines.rows(), 100); x++) 
			{
			      double[] vec = lines.get(x, 0);
			      double rho = vec[0], theta = vec[1];
			      double a = Math.cos(theta), b = Math.sin(theta);
			      double x0 = a*rho, y0 = b*rho;
			      Point start = new Point(x0 + 5000 * -b, y0 + 5000 * a);
			      Point end = new Point(x0 - 5000 * -b, y0 - 5000 * a);
			      Imgproc.line(hough, start, end, new Scalar(0,0,0), 3);
			}
		}
		else 
		{
			Imgproc.HoughLinesP(canny, lines, 1, Math.PI/180, 50, 10, 10);
			
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
		
		HoughLines(canny, 1, Math.PI/180, 0, 0, 100);
		
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
	
	static double[][] HoughLines(Mat image, double deltaRho, double deltaTheta, double width, double maxWidth, int threshold)
	{
		int maxDim = Math.max(image.width(), image.height());
		int[][] accumulator = new int[(int)(Math.PI/deltaTheta+1)][(int)(maxDim * Math.sqrt(2))];
		
		System.out.printf("Width: %d,  Height: %d\n", image.cols(), image.rows());
		
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
									xp = x + dx,
									yp = y + dy,
									rho = originToLineDistance(x, y, xp, yp);
							
							int thetaIndex = (int)Math.round(theta / deltaTheta);
							int rhoIndex = (int)Math.round(rho/deltaRho);
							accumulator[thetaIndex][rhoIndex] += 1;
						}
					}
				}
			}
		}
		
		int lineCount = 0;
		for(int j=0; j<accumulator.length; j++)
		{
			for(int i=0; i<accumulator[j].length; i++)
			{
				if(accumulator[j][i] > threshold)
				{
					lineCount++;
					System.out.printf("(theta = %f, rho = %.2f)\n", j*deltaTheta, i*deltaRho);
				}
			}
		}
		
		int index = 0;
		double[][] lines = new double[lineCount][2];
		for(int j=0; j<accumulator.length; j++)
		{
			for(int i=0; i<accumulator[j].length; i++)
			{
				if(accumulator[j][i] > threshold)
				{
					lines[index][0] = j*deltaTheta;
					lines[index][1] = i*deltaRho;
				    index++;
				}
			}
		}
		
		return lines;
	}
	
	// https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
	static double originToLineDistance(double x1, double y1, double x2 ,double y2)
	{
		double dx = x2-x1, dy = y2-y1;
		double numer = Math.abs(x2*y1 - y2*x1);
		double denom = Math.sqrt(dx*dx + dy*dy);		
		return numer / denom;
	}
}

