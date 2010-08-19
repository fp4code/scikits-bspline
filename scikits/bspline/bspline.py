# -*- coding: utf-8 -*-
"""
(C) 2010-07-29 FP, CNRS-LPN
Portage de 2010-07-29/matlab
"""

import numpy as np
import copy


"""
import numpy as np
import matplotlib.pyplot as plt
v_knots = np.array([1,2,3,4,5,6,7,7,7,7,7,8,9,10,11,12,13,14,15,16,17,18,19,20])*1.0
execfile('bspline.py'); b = BSpline(v_knots,5)
bd = b.compute_der()
cm = bd.coefs[5,:,:]
cp = bd.coefs[6,:,:]
for ii in range(6):
    print (ii, np.poly1d(cm[ii])(0), np.poly1d(cm[ii])(1), np.poly1d(cp[ii])(0), np.poly1d(cp[ii])(1))
#




b.plot()
plt.show()
bd = b.compute_der()
bd.plot()
"""


def assert_bspline_is_good(b):
    """
    2008-11-15 (FP, CNRS-LPN) Matlab code
    2010-07-30 (FP, CNRS-LPN) Python/Numpy porting

    Test is the bspline {b} is good.

                breaks vecteur des points de contrôle
                 coefs coefficients des polynômes
                 index
                    mb tableau des points de contrôle multiples
            splinesegs tableau pseudoliste des segments sur lequels reposent les splines

    Une structure b-spline repose sur {N} segments,
    """
    N = b.coefs.shape[0]
    """
    délimités par une suite strictement croissante de {N + 1} points,
    formant le vecteur {b.breaks} des points de contrôle.
    """
    if b.breaks.shape != (N+1,):
        raise(AssertionError('programming error'))
    #
    if np.any(np.diff(b.breaks) <= 0):
        raise(AssertionError('programming error'))
    #

    """
    Si un point de contrôle {i} a une multiplicité {m},
    le tableau {b.mb} contient la colonne {[i;m]}.
    
    Sur chaque segment, un b-spline est défini par un polynôme de degré {D},
    donc par {D+1} coefficients.
    """
    D = b.coefs.shape[2] - 1

    """
    Chaque b-spline est étalé sur au maximum {Q} segments.
    Le nombre de b-splines non nuls sur un segment donné est également égal à {Q}.
    """
    Q = b.coefs.shape[1]

    """
    Il y a au total {Nspl} b-splines
    """
    Nspl                          = b.splinesegs.shape[0]
    """

    Les coefficients des polynômes tiennent dans le tableau {b.coefs}
    de coefficients de dimensions {(N,Q,D+1)}.
    
    Pour un b-spline standard, {Q == D+1},
    mais pour un b-spline intégré ou dérivé, {D} sera plus grand ou plus petit.
    """

    """
    Le nombre total de points de contrôle, multiplicité incluse
    est égal au nombre {b.number_of_splines} de b-splines
    plus {number_of_spline_segments}.
    """
    if b.mb.size == 0:
        mumu = 0
    else:
        mumu = np.sum(b.mb[1,:]-1)
    #
    if N + 1 + mumu != Nspl + Q:
        raise(AssertionError('programming error'))
    #

    """    
    Le tableau des index {b.index} a pour dimensions les
    deux premières dimensions du tableau de coefficients {b.coefs} :
    """
    if b.index.shape != (N,Q):
        raise(AssertionError('programming error'))
    #

    """
    Sur chaque segment, il y a {Q} b-spline différents qui s'étalent. L'identification
    du b-spline est obtenue par le tableau {b.index}
    Comme pour {b.coefs}, le premier indice de {b.index} correspond à un segment.
    Comme pour {b.coefs}, le second indice de {b.index} correspond à un de {Q} b-spline
      ayant des valeurs non nulles sur ce segment.
    Le tableau {b.index} contient l'indice du noeud de départ de ce b-spline.
    Une valeur {-1} correspond à l'absence de b-spline.
    """
    if b.index.shape != (N,Q):
        raise(AssertionError('programming error'))
    #
    if np.any(b.index < -1):
        raise(AssertionError('programming error'))
    #
    if np.any(b.index >= Nspl):
        raise(AssertionError('programming error'))
    #

    """
    index_by_spline = np.zeros((Q,Nspl),np.int)
    index_by_spline[b.reverse_index[0,:]] = b.index[b.reverse_index[1,:]]
    is_null = index_by_spline != np.tile(np.arange(Nspl),(Q,1))
    if max(max(is_null & (index_by_spline ~= 0))) ~= 0
    """
    """
    Chaque colonne de {b.reverse_index} correspond à un segment d'un b-spline
    
    Supposons que cette colonne contienne (i,j) :
    
    i est un indice linéaire dans un tableau à "Nspl" colonnes ;
    dans ce tableau, la colonne correspond au segment relatif au b-spline ;
    ainsi, pour {Nspl==16},
    i=19 correspond au 3ème segment du 2ème b-spline
    
    j est un indice linéaire dans un tableau à {N} colonnes ;
    c'est l'indice de {b.index} et de {b.coefs[:,,]}

    nnnn représente b.breaksMultiplicity
    
    nn
         1   2= 2,1 | 14   1= 1,1 | 27          | 40          | 53 
    n               |             |             |             | 
         2   3= 3,1 | 15  18= 2,2 | 28  17= 1,2 | 41          | 54
    n               |             |             |             | 
         3   4= 4,1 | 16  19= 3,2 | 29  34= 2,3 | 42  33= 1,3 | 55
    n               |             |             |             | 
         4   5= 5,1 | 17  20= 4,2 | 30  35= 3,3 | 43  50= 2,4 | 56  49= 1,4
    n               |             |             |             | 
         5   6= 6,1 | 18  21= 5,2 | 31  36= 4,3 | 44  51= 3,4 | 57  66= 2,5 
    nnnn            |             |             |             | 
         6  10=10,1 | 19   9= 9,1 | 32   8= 8,1 | 45   7= 7,1 | 58  22= 6,2
    n               |             |             |             | 
         7  11=11,1 | 20  26=10,2 | 33  25= 9,2 | 46  24= 8,2 | 59  23= 7,2
    n               |             |             |             | 
         8  12=12,1 | 21  27=11,2 | 34  42=10,3 | 47  41= 9,3 | 60  40= 8,3
    n               |             |             |             | 
         9  13=13,1 | 22  28=12,2 | 35  43=11,3 | 48  58=10,4 | 61  57= 9,4
    n               |             |             |             | 
        10  14=14,1 | 23  29=13,2 | 36  44=12,3 | 49  59=11,4 | 62  74=10,5
    nnnn            |             |             |             | 
        11          | 24          | 37  16=16,1 | 50  15=15,1 | 63  30=14,2
    n               |             |             |             | 
        12          | 25          | 38          | 51  32=16,2 | 64  31=15,2
    n               |             |             |             | 
        13          | 26          | 39          | 52          | 65  48=16,3
    n
    """

def proto_m():
    """
    proto_m() retourne une matrice tutorielle
    """
    m = np.array([[[[11, 12, 13, 14],
                    [21, 22, 23, 24],
                    [31, 32, 33, 34]],
                   [[111, 112, 113, 114],
                    [121, 122, 123, 124],
                    [131, 132, 133, 134]]],
                  [[[511, 512, 513, 514],
                    [521, 522, 523, 524],
                    [531, 532, 533, 534]],
                   [[611, 612, 613, 614],
                    [621, 622, 623, 624],
                    [631, 632, 633, 634]]]])
    return m

def mat_shear_h(m,fill=0):
    """2010-07-29 (FP, CNRS-LPN)
    mat_shear_h(m) computes an horizontal shear of matrix m over two last indices 
      11 12 13 14 -> 11 12 13 24  0  0 via 11 12 13 14 0  0  0
      21 22 23 24     0 21 22 23 34  0     21 22 23 23 0  0  0
      31 32 33 34     0  0 31 22 23 34     31 32 33 34 0  0  0
    """
    sm = m.shape
    (oth,(nli,nco)) = (sm[0:-2],sm[-2:])
    noth = np.prod(oth)
    mv = m.view().reshape((noth,nli,nco))
    if fill == 0:
        ms = np.zeros((noth,nli,nli+nco),dtype=m.dtype)
    #
    else:
        ms = np.ones((noth,nli,nli+nco),dtype=m.dtype)*fill
    #
    ms[:,:,0:nco] = mv
    ms = ms.reshape((noth,nli*(nli+nco),))
    ms = ms[:,:-nli]
    ms = ms.reshape(sm[0:-1]+(nli+nco-1,))
    return ms

def mat_shear_v(m,fill=0):
    """
    mat_shear_v(m) computes a vertical shear of matrix m over two last indices
    2010-07-30 (FP, CNRS-LPN)
     11 12 13 14 -> 11  0  0  0
     21 22 23 24    21 12  0  0                                 0
     31 32 33 34    31 22 13  0
                     0 32 23 14
                     0  0 33 24
                     0  0  0 34
    """
    sm = m.shape
    (oth,(nli,nco)) = (sm[0:-2],sm[-2:])
    noth = np.prod(oth)
    mv = m.view().reshape((noth,nli,nco))
    mv = mv.transpose((0,2,1))
    if fill == 0:
        ms = np.zeros((noth,nco,nli+nco),dtype=m.dtype)
    #
    else:
        ms = np.ones((noth,nco,nli+nco),dtype=m.dtype)*fill
    #
    ms[:,:,0:nli] = mv
    ms = ms.reshape((noth,nco*(nli+nco),))
    ms = ms[:,:-nco]
    ms = ms.reshape((noth,nco,nli+nco-1))
    ms = ms.transpose((0,2,1))
    ms = ms.reshape(sm[0:-2]+(nli+nco-1,nco))
    return ms
#

def bspline_compute_stage1(v_knots, maxSplineDegree):
    """
    Cox - De Boor 1D b-spline coefficients computation
 
    2008-11-13 (FP, CNRS-LPN) spline_compute_stage1
    2008-11-15 (FP, CNRS-LPN) renamed bspline_compute_stage1
    2010-07-29 (FP, CNRS-LPN) Python/Numpy porting

    bspline_compute_stage1(v_knots, maxSplineDegree)
      Computes b-spline coefficients of degree up to maxSplineDegree
    v_knots is the vector of knots, size N+1.
    Returns a list B of size maxSplineDegree+1
    B[d] is a (N-D,D+1,D+1) matrix
    B[d][i_s,i_p,:] is a vector of polynom coefficients, ordered from x^d to x^0
      i_s is the b-spline index
      i_p is the polynom index inside b-spline i_s
    The vector v_knots should be ordered. In case of multiple points,
    the coefficient of polynoms defined on null-length segments may contain inf or nan.
    """
    #
    if len(v_knots.shape) != 1:
        raise(TypeError("v_knots argument should be a vector"))
    #
    ftype = v_knots.dtype
    if not issubclass(ftype.type,np.floating):
        raise(TypeError("v_knots argument should be floating"))
    #
    N = v_knots.size - 1
    B = [np.ones((N,1,1))]
    oldsettings = np.seterr(divide = 'ignore', invalid = 'ignore')
    for Q in range(1,maxSplineDegree+1):
        B.append(np.zeros((N-Q,Q+1,Q+1)))
        #
        m = np.tile(v_knots[Q:-1]-v_knots[:-Q-1],(Q,Q,1)).T
        B[Q][:,:-1,:-1] = B[Q][:,:-1,:-1] + B[Q-1][:-1,:,:] / m
        #
        m = np.tile(v_knots[Q+1:] - v_knots[1:N+1-Q],(Q,Q,1)).T
        B[Q][:,1:,:-1] = B[Q][:,1:,:-1] - B[Q-1][1:,:,:] / m
        # i+k
        ipk = np.tile(np.arange(N-Q),(Q+1,1)) + np.tile(np.arange(Q+1),(N-Q,1)).T
        #
        vka = v_knots[ipk[:-1,:]]
        vkb = np.tile(v_knots[:-Q-1],(Q,1))
        vkd = np.tile(v_knots[Q:-1] - v_knots[:-Q-1],(Q,1))
        m = np.tile((vka-vkb)/vkd,(Q,1,1)).T
        B[Q][:,:-1,1:] = B[Q][:,:-1,1:] + B[Q-1][:-1,:,:] * m
        #
        vka = np.tile(v_knots[Q+1:],(Q,1))
        vkb = v_knots[ipk[1:,:]]
        vkd = np.tile(v_knots[Q+1:] - v_knots[1:-Q],(Q,1))
        m = np.tile((vka - vkb) / vkd,(Q,1,1)).T
        B[Q][:,1:,1:] = B[Q][:,1:,1:] + B[Q-1][1:,:,:] * m
    # for Q
    np.seterr(divide = oldsettings['divide'], invalid = oldsettings['invalid'])
    return B
#

class BSpline:
    def __init__(self, v_knots, D):
        """
        2008-11-13 (FP, CNRS-LPN) initial Matlab code spline_compute_params
        2008-11-15 (FP, CNRS-LPN) renamed bspline_compute_params
        2008-11-15 (FP, CNRS-LPN) added .ind1 and .ind2
        2010-07-29 (FP, CNRS-LPN) Python/Numpy code
        spline_compute(v_knots, q) the b-spline coeffs of order D
        v_knots is a column vector of size N+1
        Returns b
        b.coefs(i,k,l)
        i = 1:N-D ; (i - k + 1) is the b-spline index
        k = 1:D+1 l is the polynom index inside b-spline i
        l = 1:D+1 ; D+1-l is the monom power 
        """
        
        """
        v_knots = np.array([0,1,2,3,3,4,5,6,6,6,7,8.])
        D = 2
        """

        # == checks ==
        if not isinstance(D, int):
            raise(TypeError("Argument D should be an int"))
        #
        if not isinstance(v_knots, np.ndarray):
            raise(TypeError("Argument v_knots should be a numpy.ndarray object"))
        #
        if len(v_knots.shape) != 1:
            raise(TypeError("Argument v_knots should be a vector"))
        #
        if not issubclass(v_knots.dtype.type,np.floating):
            raise(TypeError("Argument v_knots should be floating"))
        #

        """
        == Computes raw polynomials coefficients ==

        === coefs_raw_by_bspline ===
        coefs_raw_by_bspline is a (N-D,D+1,D+1) matrix
        coefs_raw_by_bspline[i_spl,i_p,:] is a vector of polynom coefficients,
          ordered from x^d to x^0
        i_spl is the b-spline index
        i_p is the polynom index inside b-spline i_s

        i_spl a i_p   segment->
                        0   1   2   3   4   5  coefs_raw_by_bspline     
            spline  0  a00 a01 a02               a00 a01 a02
              |     1      a10 a11 a12           a10 a11 a12
              v     2          a20 a21 a22       a20 a21 a22
                    3              a30 a31 a32   a30 a31 a33
        
        The vector v_knots should be ordered. In case of multiple points,
        the coefficient of polynoms defined on null-length segments may contain inf or nan.

        === coefs_raw ===
        coefs_raw is a (N,D+1,D+1) matrix
        coefs_raw[i_seg,i_p,:] is a vector of polynom coefficients,
          ordered from x^d to x^0
        i_seg is the segment index
        i_p is the polynom index inside b-spline matchine i_seg
          i_spl = i_seg - i_p

        i_seg b i_p   segment_raw ->
                        0   1   2   3   4   5     coefs_raw 
            spline  0  a00 a01 a02               a00        
              |     1      a10 a11 a12           a10 a01    
              v     2          a20 a21 a22       a20 a11 a02   
                    3              a30 a31 a32   a30 a21 a12
                                                     a31 a22
                                                         a32
         
        === index_raw ===
        index of b-spline                         index_raw
                                                  0  -1  -1
                                                  1   0  -1
                                                  2   1   0
                                                  3   2   1
                                                 -1   3   2
                                                 -1  -1   3

        == null segments elimination ==
        suppose that segment 3 is null

        === coefs, index ===
        i_seg b i_p
             segment->  0   1   2   3   4             
         segment_raw->  0   1   2   4   5         coefs       index
            spline  0  a00 a01 a02             a00           0 -1 -1
              |     1      a10 a11             a10 a01       1  0 -1
              v     2          a20 a22         a20 a11 a02   2  1  0
                    3              a31 a32         a31 a22  -1  3  2
                                                       a32  -1 -1  3

        === segment_raw ===
        indice de segment raw à partir de l'indice de segment = vecteur [0 1 2 4 5]
        pas utile

        === splinesegs ===
        étalement des splines
        0:[0 1 2] 1:[1 2] 2:[2 3] 3:[3 4]
          

        """
        coefs_raw_by_bspline = bspline_compute_stage1(v_knots,D)[-1]
        coefs_raw = mat_shear_h(coefs_raw_by_bspline.transpose(2,1,0)).transpose(2,1,0)
        del(coefs_raw_by_bspline)

        N = v_knots.size-1
        Nspl = N-D
        # b-spline indexing, a row corresponds to a segment
        mt = np.tile(np.arange(Nspl),(D+1,1))
        index_raw = mat_shear_h(mt,-1).T
        # indices of not null segments 
        not_null_segs = np.nonzero(np.diff(v_knots))[0]
        # b-splines main elements
        self.Nspl = Nspl
        self.coefs = coefs_raw[not_null_segs,:,:]
        self.index = index_raw[not_null_segs,:]
        self.breaks = np.concatenate((v_knots[not_null_segs],v_knots[-1:]))

        # == computing multiplicity ==
        v_dnull = np.diff(v_knots) == 0
        v_dnull = np.concatenate((np.array([False]),v_dnull,np.array([False])))
        v_raise = np.nonzero(np.logical_and(v_dnull[:-1] == False, v_dnull[1:] == True))[0]
        v_fall  = np.nonzero(np.logical_and(v_dnull[:-1] == True, v_dnull[1:] == False))[0]
        del(v_dnull)
        v_multiplicity = v_fall - v_raise + 1
        v_index = v_raise.copy() ; v_index[1:] -= v_multiplicity[:-1]
        self.mb = np.vstack((v_index, v_multiplicity))
        del(v_index,v_multiplicity)

        # == computing splinesegs ==
        index = self.index
        splinesegs = np.empty((Nspl,1+D+1),np.int)
        splinesegs[:,0]=0
        splinesegs[:,1:]=-1
        for iq in range(D+1):
            segs = np.nonzero(index[:,iq]>=0)
            splines = index[segs,iq]
            splinesegs[splines,0] += 1
            splinesegs[splines,splinesegs[splines,0]] = segs
        #
        self.splinesegs = splinesegs
        # optional control
        assert_bspline_is_good(self)
    #
    def segments(self,ispl):
        """
        returns the segments on which spline ispl is not null
        """
        nsegs = self.splinesegs[ispl,0]
        return self.splinesegs[ispl,1:nsegs+1,]
    # 
    def extends(self,ispl):
        """
        returns the extends of spline ispl
        """
        segs = self.segments(ispl)
        if segs.size == 0:
            return (False,0,0)
        #
        else:
            return (True,b.breaks[segs[0]],b.breaks[1+segs[-1]])
        #
    #
    def compute_der(self):
        """
        Computes the derivative of a b-spline
        
        2008-11-15 (FP, CNRS-LPN) initial Matlab code bspline_compute_der
        2010-08-02 (FP, CNRS-LPN) Python code
        """
        
        (N,Q,Dp1) = self.coefs.shape
        D = Dp1-1
        b_d = copy.copy(self)
        if D > 0:
            b_d.coefs = self.coefs[:,:,:D]*np.tile(np.arange(D,0,-1),(N,Q,1))
        #
        else:
            b_d.coefs = np.zeros_like(self.coefs)
        #
        assert_bspline_is_good(b_d);
        return b_d
    #
    def compute_int(self):
        """
        % 2008-11-15 (FP, CNRS-LPN) bspline_compute_int
        % computes the integral
        """
        (N,Q,Dp1) = self.coefs.shape
        b_i = copy.copy(self)
        coefs = np.empty((N,Q,Dp1+1),self.coefs.dtype)
        coefs[:,:,:-1] = self.coefs / np.tile(np.arange(Dp1,0,-1),(N,Q,1))
        m_xi = np.tile(np.diff(self.breaks),(Q,1)).T
        s = coefs[:,:,0]
        for d in range(1,Dp1):
            s = s*m_xi + coefs[:,:,d]
        #
        s = s*m_xi
        # s contains now the integral of each polynom
        index = b_i.index
        splinesegs = b_i.splinesegs
        nspl = splinesegs.shape[0]
        # intégrale en cours de chaque spline
        cumint = np.zeros(nspl,coefs.dtype)
        # boucle sur les morceaux
        for iq in range(Q):
            vindex = index[:,iq]
            good = vindex >= 0 
            splgood = vindex[good]
            coefs[good,iq,-1] = cumint[splgood]
            cumint[splgood] += s[good,iq]
        #
        b_i.coefs = coefs
        return b_i
    #
    """"
    x^2/2, -x^2+x+1/2, x^2 - x + 1/2
    x^3/6, -x^3/3 + x^2/2 + x/2, x^3/6 - x^2/2 + x/2
    1/6, 4/6, 1/6
    0, 1/6, 5/6
    """

    def compute_at(self, v_x, from_side='right'):
        """
        % 2008-11-13 (FP, CNRS-LPN) initial Matlab code, spline_compute_at
        % 2008-11-15 (FP, CNRS-LPN) renamed bspline_compute_at
        % 2008-11-15 (FP, CNRS-LPN) modified in order to use der and int bsplines
        % 2010-08-02 (FP, CNRS-LPN) Python/Numpy porting
        % structure contains
        % Nombres entiers :
        % .Nspl -> nombre de splines "Nspl"
        % Tableau contenant "Nspl" lignes
        % .extends -> chaque ligne formée de 2 ou 3 éléments :
        %             x_min et x_max du spline,
        %             et éventuellement la valeur à droite de l'extension
        % Tableau contenant "nb" lignes
        % .breaks  -> vecteur des noeuds
        % Tableaux contenant "nx" lignes
        % .x                 -> vecteur des abscisses
        % .iseg              -> vecteur des index d'intervalle inter-breaks
        %                       0 si abscisse < breaks(1)
        %                       length(breaks) si abscisse > breaks(end)
        % .coefs
        % .index
        """

        breaks = self.breaks
        coefs = self.coefs
        index = self.index
        """
        TEST
        coefs = b.coefs
        breaks = b.breaks
        """
        (v_iseg, v_xi) = put_in_seg(from_side, breaks, v_x)
        
        Nx = v_iseg.size
        Q = coefs.shape[1]
        D = coefs.shape[2]-1

        out_of_bounds = np.logical_or(v_xi < 0,v_iseg == breaks.size-1)
        # arbitrary values
        v_xi[out_of_bounds] = 0
        v_iseg[out_of_bounds] = 0
        
        values = coefs[v_iseg,:,0].T
        #
        # values[iseg,ispl] is the higher polynom coefficient of sbline ispl on segment iseg 
        for ip in range(1,D+1):
            values = values*v_xi + coefs[v_iseg,:,ip].T
        #
        values[:,out_of_bounds] = 0
    
        # indice total
        # indice de ligne, indice de colonne
        # v_iseg
        # index(v_iseg(indice_de_ligne), indice de colonne)
    
        splines = index[v_iseg,:].T.copy()

        return (values,splines)
    #
    def plot(self):
        nspl = self.splinesegs.shape[0]
        for ispl in range(nspl):
            (ok,x1,x2) = self.extends(ispl)
            if ok == True:
                x = np.linspace(x1,x2,100,endpoint=False)
                (ya,inds) = self.compute_at(x,'right')
                (yae,indse) =  self.compute_at(np.array([x2]),'left')
                good = inds == ispl
                goode = indse == ispl
                plt.plot(np.append(x,x2),np.append(ya[good],yae[goode]))
            #
        #
        x = np.linspace(self.breaks[0],self.breaks[-1],100,endpoint=False)
        y = np.sum(self.compute_at(x,'right')[0],0)
        ye = np.sum(self.compute_at(np.array([self.breaks[-1]]),'left')[0],0)
        plt.plot(np.append(x,self.breaks[-1]),np.append(y,ye))
    #
#


def put_in_seg(from_side, breaks, v_x):
    """
    (v_iseg, v_xi) = put_in_seg(from_side, breaks, v_x)

    2008-11-13 (FP, CNRS-LPN) initial matlab code
    2008-11-21 (FP, CNRS-LPN) better on-bounds or out-of-bounds x's handling
    2010-08-02 (FP, CNRS-LPN) Python/Numpy porting

    breaks is a vector of breaks, which should be strictly monotonically increasing
    v_x is a vector of x, which should be monotonically increasing

    returns two fresh vectors (v_iseg, v_xi)

    v_iseg[i] is the index of segment containing v_x[i]
    v_xi[i] is the offset of point x inside the segment
    If an x is on a break, we put it in {from_side} segment
      with xi = 0 if from_side == 'right'
      with xi = segment length if from_side == 'left'
    """

    if not isinstance(breaks, np.ndarray) or \
            not issubclass(breaks.dtype.type,np.floating) or \
            (np.diff(breaks)<=0).any():
        raise(TypeError("Argument breaks should be a strictly increasing floating ndarray"))
    #
    if not isinstance(v_x, np.ndarray) or \
            not issubclass(v_x.dtype.type,np.floating) or \
            (np.diff(v_x)<0).any():
        raise(TypeError("Argument v_x should be an increasing floating ndarray"))
    #
    if v_x[0] < breaks[0]:
        raise(TypeError("Argument v_x should not have value less than breaks[0]"))
    #
    if v_x[-1] > breaks[-1]:
        raise(TypeError("Argument v_x should not have value greater than breaks[-1]"))
    #
    if from_side == 'right':
        if v_x[-1] == breaks[-1]:
            raise(TypeError("Argument v_x should not have value equal to breaks[-1], if from_side == 'right'"))
        #
        v_iseg = breaks.searchsorted(v_x,'right')-1
    elif from_side == 'left':
        if v_x[0] == breaks[0]:
            raise(TypeError("Argument v_x should not have value equal to breaks[0], if from_side == 'left'"))
        #
        v_iseg = breaks.searchsorted(v_x,'left')-1
    #
    else:
        raise(TypeError("Argument 'from_side' should be 'left' or 'right'"))
    #
    v_xi = v_x - breaks[v_iseg]
    return (v_iseg, v_xi)
#
    
