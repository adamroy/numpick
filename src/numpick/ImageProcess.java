package numpick;

import org.opencv.core.Mat;

public class ImageProcess implements ImageProcessor
{
	ImageProcessor[] processors;
	
	public ImageProcess(ImageProcessor ...imageProcessors)
	{
		this.processors = imageProcessors;
	}
	
	@Override
	public Mat process(Mat src)
	{
		Mat m = src;
		
		for(int i=0; i<processors.length; i++)
			m = processors[i].process(m);
		
		return m;
	}

}
