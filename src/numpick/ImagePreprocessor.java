package numpick;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImagePreprocessor
{
	static boolean pictureOutput = true;
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
	
	static ImageProcessor resize = new ImageProcessor()
	{
		@Override
		public Mat process(Mat src)
		{
			int size = 700;
			// Resize so larger dimension is size
			int dim = Math.max(src.width(), src.height());
			
			if(dim < size) 
				return src;
			
			float scale = (float)size/dim;
			int width = (int)(scale * src.width());
			int height = (int)(scale * src.height());
			
			Mat dst = new Mat();
			Imgproc.resize(src, dst, new Size(width, height));
			if(pictureOutput) 
				Imgcodecs.imwrite(outputFolder+"resize"+suffix+".png", dst);
			return dst;
		}
	};
	
	public static Mat processImage(String filename)
	{
		try
		{
			Mat raw = Imgcodecs.imread(filename);
			if(pictureOutput) 
			{
				Imgcodecs.imwrite(outputFolder+"raw"+suffix+".png", raw);
				System.out.println(filename);
			}
			ImageProcess preProcessor = new ImageProcess(resize, gray, blur, canny);
			Mat preProcessedImage = preProcessor.process(raw);
			return preProcessedImage;
		}
		catch(Exception e)
		{
			return null;
		}
	}
}
