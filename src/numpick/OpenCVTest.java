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
		String suffix = "_2";
		
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		Mat raw = Imgcodecs.imread("toothpicks.jpg");
		
		Mat gray = new Mat();
		Imgproc.cvtColor(raw, gray, Imgproc.COLOR_BGR2GRAY);
		Imgcodecs.imwrite("gray"+suffix+".png", gray);
		
		Mat blur = new Mat();
		Imgproc.blur(gray, blur, new Size(3, 3));
		Imgcodecs.imwrite("blur"+suffix+".png", blur);
		
		int lowThreshold = 50;
		Mat canny = new Mat();
		Imgproc.Canny(blur, canny, lowThreshold, lowThreshold*3);
		Imgcodecs.imwrite("canny"+suffix+".png", canny);
		
		Mat lines = new Mat();
		Mat hough = gray.clone();
		boolean prob = true;
		
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
			      Point start = new Point(x0 + 1000 * -b, y0 + 1000 * a);
			      Point end = new Point(x0 - 1000 * -b, y0 - 1000 * a);
			      Imgproc.line(hough, start, end, new Scalar(0,0,0), 1);
			}
		}
		else 
		{
			Imgproc.HoughLinesP(canny, lines, 1, Math.PI/180, 100, 10, 10);
			
			for (int x = 0; x < lines.rows(); x++) 
			{
			      double[] vec = lines.get(x, 0);
			      double x1 = vec[0], 
						 y1 = vec[1],
						 x2 = vec[2],
						 y2 = vec[3];
			      Point start = new Point(x1, y1);
			      Point end = new Point(x2, y2);
			
			      Imgproc.line(hough, start, end, new Scalar(0,0,0), 1);
			}
		}
		
		Mat all = combineMats(raw, gray, blur, canny, hough);
		Imgcodecs.imwrite("result"+suffix+".png", all);
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
}

