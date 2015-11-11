package numpick;

import org.opencv.core.Mat;

public interface ImageProcessor
{
	Mat process(Mat src);
}
