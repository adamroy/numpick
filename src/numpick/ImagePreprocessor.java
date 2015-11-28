package numpick;

import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ImagePreprocessor
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
	
	public static Mat processImage(String filename)
	{
		
		Mat raw = Imgcodecs.imread(filename);				
		ImageProcess preProcessor = new ImageProcess(gray, blur, canny);
		Mat preProcessedImage = preProcessor.process(raw);
		return preProcessedImage;
	}
}
