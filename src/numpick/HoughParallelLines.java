package numpick;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Mat;
import org.opencv.core.Point;

public class HoughParallelLines
{
	public static Point[] run(Mat image, double deltaRho, double deltaTheta,
			double deltaWidth, double maxWidth, int threshold, int maximaRadius)
	{
		// Dim 1, height: theta
		// Dim 2, width: rho
		// Dim 3, depth: width

		int maxDim = (int) ((Math.max(image.width(), image.height()) / deltaRho) + maxWidth);
		int accumWidth = (int) (maxDim * Math.sqrt(2)) * 2;
		int accumHeight = (int) (Math.PI / deltaTheta);
		int accumDepth = (int) (maxWidth / deltaWidth) + 1;
		int[][][] accumulator = new int[accumHeight][accumWidth][accumDepth];

		// Use to keep track of most likely width
		int[] widthCount = new int[accumDepth];

		// Run the accumulator on all non-zero pixels
		byte[] pixel = new byte[1];
		for (int j = 0; j < image.rows(); j++)
		{
			for (int i = 0; i < image.cols(); i++)
			{
				int num = image.get(i, j, pixel);
				if (num > 0)
				{
					int val = (int) pixel[0] & 0xFF;
					if (val > 0)
					{
						for (double theta = 0; theta < Math.PI; theta += deltaTheta)
						{
							// Computing the line through this pixel
							double x = i, y = j, dx = Math.cos(Math.PI - theta) * 10, dy = Math
									.sin(Math.PI - theta) * 10, xp = x - dx, yp = y
									- dy, rho = originToLineDistance(x, y, xp,
									yp);

							int thetaIndex = (int) Math.floor(theta / deltaTheta);

							for (double width = -maxWidth; width <= maxWidth; width += deltaWidth)
							{
								// The new rho in the center of the parallel lines
								double rhoPrime = rho + width / 2;

								int rhoIndex = (int) Math.floor((rhoPrime)
										/ deltaRho)
										+ accumWidth / 2;
								int widthIndex = (int) Math.floor(Math
										.abs(width) / deltaWidth);
								accumulator[thetaIndex][rhoIndex][widthIndex] += 1;
								widthCount[widthIndex] += 1;
							}
						}
					}
				}
			}
		}

		int maxWidthIndex = 0;
		for (int k = 0; k < accumDepth; k++)
			if (widthCount[k] > widthCount[maxWidthIndex])
				maxWidthIndex = k;

		// Extract the rho's and theta's from the accumulator
		List<Double> rhoList = new ArrayList<>();
		List<Double> thetaList = new ArrayList<>();
		for (int j = 0; j < accumHeight; j++)
		{
			for (int i = 0; i < accumWidth; i++)
			{
				for (int k = 0; k < accumDepth; k++)
				{
					int value = accumulator[j][i][k];
					if (value > threshold)
					{
						// Check to see if this point is a local maxima, as if
						// it is not then the other point should be the line
						// Prevents lines segment doubles
						// r is the radius to check for local maxima
						boolean localMaxima = true;
						int r = maximaRadius;
						for (int dx = -r; dx <= r; dx++)
						{
							for (int dy = -r; dy <= r; dy++)
							{
								// for(int dz=-r*5; dz<=r*5; dz++)
								{
									if (j + dy >= 0 && j + dy < accumHeight
											&& i + dx >= 0
											&& i + dx < accumWidth &&
											// k+dz>=0 && k+dz<accumDepth &&
											!(dx == 0 && dy == 0))
									{
										if (accumulator[j + dy][i + dx][k] > value)
											localMaxima = false;
									}
								}
							}
						}

						if (localMaxima)
						{
							rhoList.add((i - accumWidth / 2) * deltaRho);
							thetaList.add(j * deltaTheta);
						}
					}
				}
			}
		}

		// Package as a point array for easy consumption
		Point[] lines = new Point[rhoList.size()];
		for (int i = 0; i < rhoList.size(); i++)
		{
			double rho = rhoList.get(i);
			double theta = thetaList.get(i);
			lines[i] = new Point(rho, theta);
		}

		// Return accumulator if asked for
		// if(accumulatorOut != null)
		// accumulatorOut.set(accumulator);
		return lines;
	}

	// https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
	private static double originToLineDistance(double x1, double y1, double x2, double y2)
	{
		double dx = x2 - x1, dy = y2 - y1;
		double numer = x2 * y1 - y2 * x1;
		double denom = Math.sqrt(dx * dx + dy * dy);
		return numer / denom;
	}
}
