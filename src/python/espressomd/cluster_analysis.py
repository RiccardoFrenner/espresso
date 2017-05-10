from .script_interface import ScriptInterfaceHelper,script_interface_register
from .particle_data import ParticleHandle, ParticleSlice


@script_interface_register
class Cluster(ScriptInterfaceHelper):
    """Class representing a cluster of particles
    
    Methods:
    particle_ids() : Returns list of particle ids in the cluster
    particles(): Returns an instance of ParticleSlice containing the paritlces in the cluster
    size(): Returns the number of particles in the cluster
    """
    _so_name = "ClusterAnalysis::Cluster";
    _so_bind_methods = ("particle_ids","size")

    def particles(self):
        return ParticleSlice(self.particle_ids())



@script_interface_register
class ClusterStructure(ScriptInterfaceHelper):
    """Cluster structure of a simulation system, and access to cluster anaylsis

    Attributes and methods:
    ----------------------
    pair_criterion: Instance of PairCriterion or derived classes
        Criterion to decide whether two particles are neighbors.
    
    clusters: behaves like a read-only dictionary
        Access to individual clusters in the cluster structure either via
        cluster[i], wher i is a (non-consecutive) integer cluster id 
        or via iteration:
        for pair in clusters:
        where pair contains the numeric id and the corresponding cluster object.

    cluster_ids()
        List of all cluster ids of the clusters in the structure

    cid_for_particle(p): where p is an instace of ParticleHandle or aparticle id
        returns the id of the clsuter to which the particle belongs,
        or None if it does not belong to a cluster.

    clear():
        Clears the cluster structure

    run_for_all_pairs: 
        Runs the clsuter analysis, considering all pairs of particles in the system

    run_for_bonded_particles():
        Runts the cluster analysis, considering only pairs of particles connected 
        by a pairwise bonded potential
    """
    _so_name="ClusterAnalysis::ClusterStructure"
    _so_bind_methods = ("run_for_bonded_particles","run_for_all_pairs","clear","cluster_ids")

    def __init__(self,*args,**kwargs):
        super(type(self),self).__init__(*args,**kwargs)
        self._clusters=Clusters(self)
    
    def cid_for_particle(self,p):
        """Returns cluster id for the particle (passed as ParticleHandle or particle id)"""
        if isinstance(p,ParticleHandle):
            return self.call_method("cid_for_particle",pid=p.id)
        if isinstance(p,int):
            return self.call_method("cid_for_particle",pid=p)
        else:
            raise TypeError("The particle has to be passed as instance of Particle handle or as an integer particle id")
            

    @property 
    def clusters(self):
        return self._clusters

class Clusters:
   """Access to the clusters in the cluster structure. Behaves roughly like a dict"""

   def __init__(self,cluster_structure):
       self.cluster_structure=cluster_structure


   def __getitem__(self,cluster_id):
       return self.cluster_structure.call_method("get_cluster",id=cluster_id)

   def __iter__(self):
       for cid in self.cluster_structure.cluster_ids():
           yield (cid,self.cluster_structure.call_method("get_cluster",id=cid))
           
   def __len__(self):
        return self.cluster_structure.call_method("n_clusters")
