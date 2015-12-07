package numpick;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Point;
import org.opencv.core.Point3;

public class HierarchicalClustering
{
	private static class Cluster
	{
		private List<Point> lines = new ArrayList<Point>();
		private Point centroid;
		
		public Cluster(Point p)
		{
			lines.add(p);
			centroid = p.clone();
		}
		
		public Cluster(Cluster one, Cluster two)
		{
			lines.addAll(one.lines);
			lines.addAll(two.lines);
			
			double rhoSum = 0, thetaSum = 0;
			for(Point p : lines)
			{
				rhoSum += p.x;
				thetaSum += p.y;
			}
			
			/*double oneRatio = (double)one.lines.size()/lines.size();
			double twoRatio = (double)two.lines.size()/lines.size();
			centroid = new Point(one.centroid.x *  oneRatio + two.centroid.x * twoRatio, one.centroid.y *  oneRatio + two.centroid.y * twoRatio);
			*/
			
			centroid = new Point(rhoSum / lines.size(), thetaSum / lines.size());
		}
		
		public double distance(Cluster other)
		{
			if(this == other)
				return Double.POSITIVE_INFINITY;
			
			return HierarchicalClustering.distance(this.centroid, other.centroid);
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
	public static int countClusters(Point[] lines, double maxDistanceInCluster, double maxRho, double maxTheta)
	{
		return countClusters(lines, maxDistanceInCluster, maxRho, maxTheta, null);
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
	 * @param finalLines an optional list if you want the lines back
	 * @return the estimated number of distinct clusters
	 */
	public static int countClusters(Point[] lines, double maxDistanceInCluster, double maxRho, double maxTheta, List<Point> finalLines)
	{
		if(lines.length == 0)
			return 0;
		if(lines.length == 1)
		{
			if(finalLines != null)
				finalLines.add(lines[0]);
			return 1;
		}
		
		// Construct initial clusters containing one point
		List<Cluster> clusters = new ArrayList<HierarchicalClustering.Cluster>();
		for(int i=0; i<lines.length; i++)
		{
			Point normalizedPoint = new Point(lines[i].x / maxRho, lines[i].y / maxTheta);
			clusters.add(new Cluster(normalizedPoint));
		}

		int distIndex = 0;
		
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
				int size = clusters.size();
				copyClustersToList(clusters, finalLines, maxRho, maxTheta);
				return size;
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
				copyClustersToList(clusters, finalLines, maxRho, maxTheta);
				return 1;
			}
		}
	}
	
	private static void copyClustersToList(List<Cluster> clusters, List<Point> lines, double maxRho, double maxTheta)
	{
		if(lines == null)
			return;
		
		for(Cluster c : clusters)
		{
			lines.add(new Point(c.centroid.x * maxRho, c.centroid.y * maxTheta));
		}
	}
	
	
	private static double distance(Point p1, Point p2)
	{
		if(p1 == p2) 
			return Double.POSITIVE_INFINITY;
		double dx = p1.x - p2.x;
		double dy = p1.y - p2.y;
		return Math.sqrt(dx*dx + dy*dy);
		// return Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y);
	}
}
