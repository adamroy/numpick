package numpick;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Point;

public class HieracrchicalClustering
{
	private class Cluster
	{
		private List<Point> lines = new ArrayList<Point>();
		
		public Cluster(Point p)
		{
			lines.add(p);
		}
		
		public Cluster(Cluster one, Cluster two)
		{
			lines.addAll(one.lines);
			lines.addAll(two.lines);
		}
		
		public double distance(Cluster other)
		{
			if(this == other)
				return Double.POSITIVE_INFINITY;
			
			double minDistance = Double.POSITIVE_INFINITY;
			for(Point p1 : this.lines)
			{
				for(Point p2 : other.lines)
				{
					double d = HieracrchicalClustering.distance(p1, p2); 
					if(d < minDistance)
						minDistance = d;
				}
			}
			
			return minDistance;
		}
	}
	
	/**
	 * Agglomerative hierarchical cluster
	 * https://en.wikipedia.org/wiki/Hierarchical_clustering
	 * https://en.wikipedia.org/wiki/Single-linkage_clustering
	 *  
	 * @param lines The lines to cluster, in (rho, theta) form
	 * @param maxDistanceInCluster maximum euclidean distance between elements in a cluster in normalized parameter space
	 * @param maxRho largest value rho can take
	 * @param maxTheta largest value theta can take
	 * @return the estimated number of distinct clusters
	 */
	public int countClusters(Point[] lines, double maxDistanceInCluster, double maxRho, double maxTheta)
	{
		// Construct initial clusters containing one point
		List<Cluster> clusters = new ArrayList<HieracrchicalClustering.Cluster>();
		for(int i=0; i<lines.length; i++)
		{
			Point normalizedPoint = new Point(lines[i].x / maxRho, lines[i].y / maxTheta);
			clusters.add(new Cluster(normalizedPoint));
		}
		
		while(true)
		{
			int n = clusters.size();
			double minimumDistance = Double.POSITIVE_INFINITY;
			int mi = -1, mj = -1;
			
			for(int i=0; i<n; i++)
			{
				for(int j=i; j<n; j++)
				{
					Cluster ci = clusters.get(i);
					Cluster cj = clusters.get(j);
					double distance =  ci.distance(cj);
					
					if(distance < minimumDistance)
					{
						minimumDistance = distance;
						mi = i;
						mj = j;
					}
				}
			}
			
			if(minimumDistance != Double.POSITIVE_INFINITY && minimumDistance > maxDistanceInCluster)
			{
				return clusters.size();
			}
			
			if(minimumDistance != Double.POSITIVE_INFINITY && mi != -1 && mj != -1)
			{
				// Remove j first because it is necessarily larger
				Cluster cj = clusters.remove(mj);
				Cluster ci = clusters.remove(mi);
				clusters.add(new Cluster(cj, ci));
			}
			
			if(clusters.size() == 1)
			{
				return 1;
			}
		}
	}
	
	
	private static double distance(Point p1, Point p2)
	{
		if(p1 == p2) 
			return Double.POSITIVE_INFINITY;
		double dx = p1.x - p2.x;
		double dy = p1.y - p2.y;
		return Math.sqrt(dx*dx + dy*dy);
	}
}
