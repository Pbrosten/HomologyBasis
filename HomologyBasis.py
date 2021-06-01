#Package for persistent homology computation using homology basis
import numpy as np
from itertools import combinations, chain
from scipy.spatial.distance import cdist
from scipy.io import loadmat
from matplotlib import collections  as mc
import miniball
import math
#import matplotlib.pylabs as pl
import matplotlib.pyplot as plt
class SimplexTree(object):
    """ Construct a valid filtered simplicial complex with a structure mimicking GUDHI's SimplexTree.
    
    
    Attributes
    ----------
    simplices : list
            Simplices records the vertice representation of all simplices included in the simplicial complex.
    
    filtrations : dictionary
            Records every simplex and its filtration value for easy access. The filtration values ca be altered once added.
            

            
    Examples
    --------
    >>> import homology_basis_package import as hbp
    >>> st=hbp.SimplexTree
    >>> st.insert([0,1])
    True
    >>> st.insert([1,2],filtration=0.4)
    True
    >>> print(st.get_filtration())
    [([0],0.0),
     ([1],0.0),
     ([0,1],0.0),
     ([2],0.4),
     ([0,2],0.4)]
    >>> print(st.get_skeleton(0))
    [([0],0.0),([1],0.0),([2],0.4)]
    >>> print(st.persistence())
    [(0,(0,inf))]
    
    """
    def __init__(self):
        """ Contruct object"""
        self.simplices=list()
        #self.filtrations={():-np.inf}
        self.filtrations=dict()
        
    def insert(self,sigma,filtration=0.0):
        """Given an n-simplex, this function adds it and all faces to the simplicial complex with a given filtration. The filtration
        value of prexisting face is only altered if the it's value is incongruous with a valid simplicial complex (the face has a 
        higher filtration value than the n-simplex).
        
        
        
        Parameters
        ----------
        sigma : array-like of length n+1 unique integers
                This represents an n-simplex with the vetrex representation. Each unique integer corresponds to one of the n+1 
                vertices defining the boundary of the simplex.
        
        filtration : float , default=0.0
                Filtration gives the corresponding filtration value for the new n-simplex. 
        
        Return
        ------
        bool
            If sigma was successfully added to the complex return True.
            If sigma was already included in the complex return False.
        
        Note: The order of the unique integers in sigma does not matter as all simplices are added using increasing order to
        the simplicial complex.
        """
        f_value=float(filtration)
        t_sigma=tuple(sorted(sigma))
        if t_sigma in self.filtrations.keys():
            self.filtrations.update({t_sigma:f_value})
            return False
        
        for k in range(1, len(sigma)):
            for face in combinations(sigma, k):
                if (face not in self.filtrations.keys()) or (self.filtrations[face]>f_value):
                    self.simplices.append(face)
                    self.filtrations[face]=f_value
        self.filtrations[t_sigma]=f_value
        return True
        
    def get_filtration(self):
        """Pull from the simplicial complex a list of all pairs of simplices and their filtration values ordered by both degree 
        and inclusion.
        
        Returns
        -------
        filtered_complex : list 
            This list is formatted as stated above.
        """
        #pull complex from filtrations dictionary and sort by filtration value
        smplx_tree=list(self.filtrations.items())
        smplx_tree.sort(key=lambda tup: tup[1])
        #convert tuple construction into list construction
        smplx_tree_list=list([])
        for smplx in smplx_tree:
            smplx_tree_list.append((list(smplx[0]),smplx[1]))
        return smplx_tree_list
    
    def get_skeleton(self,n):
        """Produce the n-skelton, a list of all pairs of simplices and their filtration values ordered by both degree and inclusion;
        such that all simplices of degree n+1 are excluded.
        
        Parameters
        ----------
        n : integer
            This determines which skelton will be produced. Note that any integer given that is less than 0 will return the 
            0-skeleton.
        
        Results
        -------
        n_skeleton : list
            This list is contains pairs of simplices and filtration values ordered by filtration value and inclusion up to
            dimension n.
        """
        n_skeleton=list()
        our_complex=self.get_filtration()
        if n < 0:
            n=0
        #pickout all added simplices under or equal in dimension to n
        for smplx in our_complex:
            if len(smplx[0])<=n+1:
                n_skeleton.append(smplx)
        return n_skeleton
    
    def dimension(self):
        """ Return the dimension of the simplicial complex.
        
        Returns
        -------
        n : integer
            This is the length of the largest simplex included in the simplicial complex minus 1
        """
        def dim(x):
            return len(x)
        #sort simplices by increasing dimension
        sorted_complex=dict(sorted(self.filtrations.items(),key=dim))
        our_complex=list(sorted_complex.keys())
        #return dimension of larges simplex
        #print(our_complex)
        return len(our_complex[-1])-1
    
    def num_vertices(self):
        return len(self.get_skeleton(0))
    
    def num_simplices(self):
        return len(self.simplices)
    
    def find(self,sigma):
        """ Determines whether the given simplex is inluded in the complex.
        
        Parameters
        ----------
        sigma : array-like of length n+1 unique integers
            Sigma represents the n-simplex made up of the vertices listed.
            
        
        Returns
        -------
        bool
            If the simplex is included, return True.
            If the simplex is not inculded, return False.
        """
        if tuple(sorted(sigma)) in self.filtrations.keys():
            return True
        else:
            return False
        
    def filtration(self,sigma):
        return self.filtrations.get(tuple(sorted(sigma)))
    
    def assign_filtration(self,sigma,filtration):
        """ Given a simplex already included in the simplicial complex, change its filtration value. Note this can cause the complex 
        to no longer have a valid filtration. This must be corrected before trying to compute persistence.
        
        Parameters
        ----------
        sigma : array-like of length n+1 unique integers
                This represents an n-simplex with the vetrex representation. Each unique integer corresponds to one of the n+1 
                vertices defining the boundary of the simplex.
        
        filtration : float , default=0.0
                Filtration gives the corresponding filtration value for the new n-simplex.
        
        Returns
        -------
        bool or None
            If the simplex exists in the simplicial complex then return nothing, else return False.
        """
        if self.find(sigma)==True:
            self.filtrations.update({tuple(sorted(sigma)):filtration})
        else:
            return False
        
    def make_filtration_non_decreasing(self):
        """ Force the current filtration on the simplicial complex to have a valid filtration.
        
        This is done by enforcing that any time a face has a larger filtration value than its coface, the coface has its filtration
        value increased to match that of the face.
        
        Returns
        -------
        bool
            If the filtration was already valid, return False.
            If alterations were required, return True.
        """
        change=False
        for sigma in self.simplices:
            if len(sigma)==1:
                continue
            sigma_filtration=self.filtrations.get(sigma)
            for face_sigma in combinations(sigma,len(sigma)-1):
                face_filtration=self.filtrations.get(face_sigma)
                if face_filtration>sigma_filtration:
                    sigma_filtration=face_filtration
                    change=True
            self.filtrations.update({sigma:sigma_filtration})
        return change
    
    def homology_basis(self,coeff_field=11):
        """ Construct the homology basis for the current simplicial complex.
        
        Results
        -------
        phom : dictionary({'v': list},
                          {'basis': dictionary},
                          {'transpose': dictionary},
                          {'length': integer}, 
                          {'index_table': dictionary},
                          {'persistence': dictionary}, 
                          {'keys': set},
                          {'coeff_field': prime number},
                          {'inversion_function': function},
                          {'simplices': dictionary})
                Returns a data structure containing a list used for persistence reduction, a dictionary containing...
        """
        self.make_filtration_non_decreasing()
        our_complex=self.get_filtration()
        ordered_complex=list()
        for sigma in our_complex:
            ordered_complex.append(tuple(sigma[0]))
        phom=filtered_homology_basis(np.array(ordered_complex),max_face_card=self.dimension()+1,coeff_field=coeff_field)
        return phom
    
    def filtration_function(self):
        def filt_fct(sigma):
            return self.filtrations[sigma]
        return filt_fct
    
    def persistent_homology(self,coeff_field=11,min_persistence=0.0,persistence_dim_max=False):
        """ Compute persistent homology on the current simplicial complex.
        
        Parameters
        ----------
        coeff_field : prime number, default=11
                Defines the coefficient field that homology will be computed in (Z mod (coeff_field)).
            
        min_persistence : float, default=0.0
                Sets a minimum length for the persistence intervals that are returned.
               
        persistence_dim_max : bool, default=False
                 When true, the persistent homology for the maximal dimension in the complex is computed.
                 
        Results
        -------
        persistence_list : list of ndarrays
                One array for each dimension containing birth- and
                death values.
        """
        filt_fct=self.filtration_function()
        phom=self.homology_basis()
        p_pairs=persistence_pairs(phom,filt_fct,max_dimension=self.dimension()+1)
        pers_pairs=[]
        for dim in p_pairs.keys():
            pers_pairs.append(p_pairs[dim])
        return pers_pairs
    
    def persistence(self,min_persistence=0.0,max_dim=2,persistence_dim_max=False):
        """ Compute persistent homology on the current simplicial complex.
        
        Parameters
        ----------
        coeff_field : prime number, default=11
                Defines the coefficient field that homology will be computed in (Z mod (coeff_field)).
            
        min_persistence : float, default=0.0
                Sets a minimum length for the persistence intervals that are returned.
               
        persistence_dim_max : bool, default=False
                 When true, the persistent homology for the maximal dimension in the complex is computed.
                 
        Results
        -------
        persistence_list : list of pairs (dimension,(birth time, death time))
                This list gives the dimension paired with the birth and death times of all the desired persistent features 
                in the current simplicial complex.
        """
        filt_fct=self.filtration_function()
        phom=self.homology_basis()
        p_pairs=persistence_pairs(phom,filt_fct,max_dim)
        persistence_list=list()
        for n in p_pairs:
            for pair in p_pairs[n]:
                if (pair[1]-pair[0])>=min_persistence:
                    persistence_list.append((n,tuple(pair)))

        return persistence_list
    
def cech_filtration_function(X):
    def fct(face):
        if len(face) == 1:
            return 0
        else:
            vertex_face=[X[vertex] for vertex in face]
            mb=miniball.Miniball(vertex_face).squared_radius()
            return math.sqrt(mb)
    return fct

def rips_filtration_function(X):
    dists = cdist(X, X)
    def fct(face):
        if len(face) == 1:
            return 0
        elif len(face) == 2:
            return dists[tuple(face)]/2
        else:
            face = list(face)
            return np.max(dists[face, :][:, face])/2
    return fct

def simplicial_complex_from_maximal_faces(
    maximal_faces,
    max_dimension):
    simplicial_complex = []
    for k in range(1, max_dimension + 2):
        simplices = set()
        for face in maximal_faces:
            simplices.update(combinations(face, k))
        simplicial_complex.append(list(simplices))
    return simplicial_complex


def dictionary_filtration_function(
    simplicial_complex,
    filtration_value_function):
    filtration_dictionary = dict()
    for simplices in simplicial_complex:
        filtration_dictionary.update(dict((face, filtration_value_function(face)) for face in simplices))
    filtration_dictionary[()] = -np.inf
    def filt_fct(face):
        return filtration_dictionary[face]
    return filt_fct

def degreewise_ordered_complex(
    simplicial_complex,
    filtration_value_function):
    simplices = np.array(tuple(chain(*simplicial_complex)))
    filtration_values = np.array([filtration_value_function(face) for face in simplices])
    return simplices[np.argsort(filtration_values, kind='stable')]

def reduction(homology_basis, w, wlowv, v, support_v):
    if not w:
        return {tau: (-wlowv * val) % homology_basis['coeff_field'] for tau, val in v.items()} 
        
    v_multiplied = {tau: -wlowv * val for tau, val in v.items()}
    red = w.copy()
    red.update(v_multiplied)
    for tau in support_v.intersection(w.keys()):
        redtau = (w.get(tau, 0) + v_multiplied.get(tau,0)) % homology_basis['coeff_field']
        if redtau == 0:
            red.pop(tau)
        else:
            red[tau] = redtau
    return red 

def kill_persistence(homology_basis, v, simplex):
    homology_basis['v'].append(v)
    support_v = set(v) 
    lowv = homology_basis['simplices'][max(homology_basis['index_table'][face] for face in support_v)]
    support_v.remove(lowv)
    inversevlowv = homology_basis['inversion_function'](v[lowv])
    v.pop(lowv)
    v = {tau: (val * inversevlowv) %                                                               
         homology_basis['coeff_field'] for tau, val in v.items()}
    homology_basis['persistence'][lowv] = simplex

    for column, val in homology_basis['transpose'].pop(lowv):
        column_without_lowv = dict(homology_basis['basis'][column])
        column_without_lowv.pop(lowv)
        reduced_column = reduction(
            homology_basis,
            column_without_lowv,
            val, 
            v, 
            support_v)

        homology_basis['basis'][column] = frozenset(reduced_column.items())
        for row in support_v.intersection(column_without_lowv):
            homology_basis['transpose'][row].remove((column, column_without_lowv[row]))
        
        if not homology_basis['basis'][column]:
            homology_basis['basis'].pop(column)
            homology_basis['keys'].remove(column)
            homology_basis['index_table'].pop(column)
        else:
            for row in support_v.intersection(reduced_column):
                homology_basis['transpose'][row].add((column, reduced_column[row]))

def contract_homology_basis(
    homology_basis,
    simplex):
    
    boundary = tuple(combinations(simplex, len(simplex) - 1))
    v = dict()
    for face in set.intersection(set(boundary), homology_basis['keys']):
        if not homology_basis['basis'].get(face):
            homology_basis['basis'][face] = frozenset(((face, 1),))
            homology_basis['transpose'][face] = set(((face, 1),))
        for h_face, h_value in homology_basis['basis'][face]:            
            v_value = (v.pop(h_face, 0) +
                       (-1)**boundary.index(face) * h_value) % homology_basis['coeff_field']
            if v_value > 0: 
                v[h_face] = v_value
        
    if v:
        kill_persistence(homology_basis, v, simplex)
        return True
    else:
        return False

def extend_homology_basis(
    homology_basis,
    simplex):
        
        
    if not contract_homology_basis(homology_basis, simplex):
        
        homology_basis['length'] += 1
        homology_basis['keys'].add(simplex)
        homology_basis['index_table'][simplex] = homology_basis['length']
        homology_basis['simplices'][homology_basis['length']] = simplex

def field_inversion(coeff_field):
    def inverse(a, n):
        for x in range(n):
            if (a * x) % n == 1:
                return x
    inversion_table = [
        inverse(a + 1, coeff_field
               ) for a in range(coeff_field - 1)]
    def inversion_fct(a):
        return inversion_table[a-1]
    return inversion_fct

def filtered_homology_basis(ordered_complex, max_face_card, coeff_field=11):
    homology_basis = dict((
        ('v', list()),
        ('basis', dict()),
        ('transpose', dict()),
        ('length', 0), 
        ('index_table', dict()),
        ('persistence', dict()), 
        ('keys', set()), 
        #('row_support', dict()),
        ('coeff_field', coeff_field),
        ('inversion_function', field_inversion(coeff_field)),
        ('simplices', dict())
    ))
    
    for face in ordered_complex:
        if len(face) < max_face_card:
            extend_homology_basis(homology_basis, face)
        else:
            contract_homology_basis(homology_basis, face)
    return homology_basis

def persistence_pairs(
    homology_basis,
    filtration_function,
    max_dimension):
    persistent_homology_pairs = dict()
 
    for birth in homology_basis['keys'].difference(homology_basis['persistence'].keys()):
        pairs = persistent_homology_pairs.get(
            len(birth) - 1, list())
        pairs.append((filtration_function(birth),np.inf))
        persistent_homology_pairs[len(birth) - 1] = pairs
        
    for birth, death in homology_basis['persistence'].items():
        pairs = persistent_homology_pairs.get(
            len(birth) - 1, list())
        pairs.append((filtration_function(birth), filtration_function(death)))
        persistent_homology_pairs[len(birth) - 1] = pairs
    persistent_homology_pairs = {dim: np.array(pairs) for dim, pairs in persistent_homology_pairs.items()}
    persistent_homology_pairs = {dim: pairs[pairs[:,0] + 5*np.finfo(float).eps < pairs[:,1]]
                                 for dim, pairs in persistent_homology_pairs.items()}
    persistent_homology_pairs = {dim: pairs[np.argsort(
        pairs[:,0] - pairs[:,1])] for dim, pairs in persistent_homology_pairs.items()}
    persistent_homology_pairs = {dim: pairs for dim, pairs in persistent_homology_pairs.items() if len(pairs) > 0}
    for dim in range(max_dimension):
        persistent_homology_pairs.setdefault(dim, np.empty((0,2))) 
    return persistent_homology_pairs

def pers(sigma):
    return sigma[1]-sigma[0]

def barcodes(pers_pairs):
    barcodes=[]
    max_time=0
    bars=0
    for Tau in pers_pairs.values():
        bars+=len(Tau)
        for tau in Tau:
            if tau[1]>max_time and tau[1]!=np.inf:
                max_time=tau[1]
            if tau[0]>max_time:
                max_time=tau[0]
    max_time=max_time*1.01
    for dim in pers_pairs:
        pers_pairs[dim]=sorted(pers_pairs[dim], key=pers)
        for i,x in list(enumerate(pers_pairs[0]))[::-1]:
            if x[1]==np.inf:
                pers_pairs[dim][i][1]=max_time
                continue
            else:
                break

    fig,ax=plt.subplots(nrows=1,ncols=1,figsize=(6,8))
    ax.axvline(x=max_time,color='k',alpha=0.75,linestyle='--',label='∞')

    ax.axes.get_yaxis().set_visible(False)
    k=0
    step=1/bars
    colors=['tab:blue','tab:orange','tab:green']
    for i in range(len(pers_pairs)):
        for j,sigma in enumerate(pers_pairs[i]):
            if j==0:
                ax.plot([sigma[0],sigma[1]],[k,k],color=colors[i%3],label=f"$H_{i}$")
            ax.plot([sigma[0],sigma[1]],[k,k],color=colors[i%3])
            k+=step
    ax.set_title('Barcode Diagram')
    ax.legend(loc='lower right')

def plot_persistence(pers_pairs):
    max_time=0
    for Tau in pers_pairs.values():
        for tau in Tau:
            if tau[1]>max_time and tau[1]!=np.inf:
                max_time=tau[1]
    infinity=max_time*1.01
    fig,ax=plt.subplots()
    plt.plot([-max_time*0.05,max_time*1.05],[-max_time*0.05,max_time*1.05],color='k',zorder=0)
    plt.axhline(y=infinity,color='k',linestyle='--',label='∞',zorder=0)
    for i in range(len(pers_pairs)):
        X=[]
        Y=[]
        for sigma in pers_pairs[i]:
            if sigma[1]==np.inf:
                X.append(sigma[0])
                Y.append(infinity)
                continue
            X.append(sigma[0])
            Y.append(sigma[1])  
        ax.scatter(X,Y,s=100,alpha=0.4,label=f"$H_{i}$",zorder=1)
    ax.axis([-max_time*0.05,max_time*1.05,-max_time*0.05,max_time*1.05])
    ax.set_ylabel('Death')
    ax.set_xlabel('Birth')
    ax.set_title('Persistence Diagram')
    ax.legend(loc='lower right')