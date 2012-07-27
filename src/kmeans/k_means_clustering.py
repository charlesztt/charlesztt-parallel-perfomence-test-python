import numpy
import scipy.linalg
import pp
import multiprocessing as mp

def distance_calculation(centroids, one_data_x):
    centroids_size=centroids.shape;
    dists=numpy.zeros((centroids_size[1],1));
    for n in range(0, centroids_size[1]):
        dists[n,0]=scipy.linalg.norm(centroids[:,n]-one_data_x);
    return dists[:,0]

def get_distances(sub_data, centroids, cluster_K):
    sub_data_size=sub_data.shape
    dists=numpy.zeros((cluster_K,sub_data_size[1]))
    for n in range(0,sub_data_size[1]):
        dists[:,n]=distance_calculation(centroids,sub_data[:,n]);
    return dists

def kmeans(data_x,cluster_K,labels):
    data_size=data_x.shape
    centroids=numpy.zeros((data_size[0],cluster_K));
    cores=mp.cpu_count();
    step=data_size[1]/cores+1;
    job_server=pp.Server(ncpus=cores);

    for n in range(0,cluster_K):
        for m in range(0,data_size[0]):
            centroids[m,n]=data_x[m,numpy.random.randint(0,data_size[1])];
    
    convergence_diff=scipy.linalg.norm(centroids);
    new_centroids=numpy.zeros(centroids.shape);
    
    while (convergence_diff>0):
        dists=numpy.zeros((cluster_K, data_size[1]));
        jobs=[];
	#I want to make it parallelized.
        for n in range(0,cores):
            start_point=step*n;
            end_point=min([data_size[1],step*(n+1)]);
            jobs.append(job_server.submit(get_distances,(data_x[:,start_point:end_point],centroids,cluster_K),(distance_calculation,),('numpy','scipy.linalg')))

        for n in range(0,cores):
            start_point=step*n;
            end_point=min([data_size[1],step*(n+1)]);
            dists[:,start_point:end_point]=jobs[n]();
	#stop parallelized here
        labels[0]=dists.argmin(0)
        for n in range(0,cluster_K):
            if numpy.size(numpy.where(labels[0]==n))>0:
                new_centroids[:,n]=data_x[:,numpy.where(labels[0]==n)[0]].mean(1);
            else:
                a=numpy.random.randint(0,data_size[1]);
                new_centroids[:,n]=data_x[:,a].copy();
                labels[0][a]=n;
        convergence_diff=scipy.linalg.norm(new_centroids-centroids);
        centroids=new_centroids.copy();
        
if __name__=='__main__':
    data_set=numpy.random.random_sample([128,10000])
    labels=[numpy.zeros((1,10000))]
    kmeans(data_set, 2, labels)
