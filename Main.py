#Import Packages
import tensorflow as tf
import tensorflow_probability as tfp
#from tensorflow_probability import experimental
tfd = tfp.distributions
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import healpy as hp
#import pandas as pd
import camb 
from camb import model, initialpower
import glob
import pylab as plty
from PIL import Image
from healpy.sphtfunc import Alm
import time 
import corner
#import seaborn as sns
import scipy.stats as st
from mpl_toolkits.mplot3d import Axes3D
import os
import sys


#%%
#Use CAMB to generate a power spectrum
#Use CAMB to generate a power spectrum
def call_CAMB_map(_parameters, _lmax): #lmax above 2551 makes no difference?
    '''
    parameters = [H0, ombh2, omch2, mnu, omk, tau]  = [Hubble Const, Baryon density, DM density, 
    Sum 3 neutrino masses/eV, Curvature parameter (Omega kappa), Reionisation optical depth]
    '''
    if _lmax <= 2551: #can only find power spectrum for lmax <= 2551 since that is the maximum value of the data.
        pars = camb.CAMBparams()
        pars.set_cosmology(H0 = _parameters[0], ombh2 = _parameters[1], omch2 = _parameters[2], mnu = _parameters[3],
                   omk = _parameters[4], tau = _parameters[5])  #Inputs the given cosmological parameters.
        pars.InitPower.set_params(As=2e-9, ns=0.965, r=0)
        
        pars.set_for_lmax(_lmax, lens_potential_accuracy=0) #input the given lmax value
        
        results = camb.get_results(pars)
        powers =results.get_cmb_power_spectra(pars, CMB_unit='muK') #returns the power spectrum in units muK.
        
        totCL=powers['total'] #returns the total (averaged) power spectrum - including lensed, unlensed power spectra 
        _DL = totCL[:,0] 
        
        #unlensedCL=powers['unlensed_scalar'] #returns the unlensed scalar power spectrum
        #_DL = unlensedCL[:,0] # 
    
        _l = np.arange(len(_DL)) #not sure this CL is actually CL but is actually DL
        _CL = []
        for i in range(_lmax): #also limits the length of power spectrum to the requested length
            if i == 0:
                _CL.append(_DL[i]) #since unsure what CL value is for this DL
        
            else:
                _CL.append(_DL[i]/(_l[i]*(_l[i] + 1)))
        
        _CL = np.array(_CL)    
    
        return _CL 
    
    else: #prints error if lmax is too large.
        print('lmax value is larger than the available data.')
        
        
#%%
#Plots a given power spectrum 
def plotpwrspctrm(_cls):
    _l = np.arange(len(_cls))
    plt.plot(_l, _l * (_l + 1) * _cls)
    plt.xlabel("$\l$")
    plt.ylabel("$\l(\l+1)C_{\l}$")
    plt.grid()
    plt.title("Power Spectrum")
    
#%%
#Plots a map in the mollview projection 
def mollviewmap(_map):
    hp.mollview(_map, title="Map displayed in the Molleview projection", cmap = None)
    hp.graticule()
    
#%%
#Adds random noise to each pixel on a map given a variance 
def noisemapfunc(_map,_var):
    _noisevec = np.random.normal(0,_var,len(_map)) #A vector of the noise applied to each pixel
    _newmap = [x + y for x, y in zip(_map, _noisevec)]
    _newmap, _noisevec = np.array(_newmap), np.array(_noisevec)
    return [_newmap, _noisevec] #returns an array consisiting of [map with added noise, array of the added noise]

#%%
#cls --> something
#cls --> something
def cltoalm(_cls, _NSIDE, _lmax): #doesn't work (isnt currently being used)
    _alms = []
    _count = 0
    for l in range(_lmax): 
        if _cls[l] > 0:
            _alms.append(np.complex(np.random.normal(0,_cls[l]),0))        #set m=0, which is real
        else:
            _alms.append(np.complex(0,0))
        
        for m in range(l+1): #set positive m's
            if _cls[l] > 0 and _cls[m] > 0:
                _alms.append(np.complex(np.random.normal(0,0.5*_cls[l]),np.random.normal(0,0.5*_cls[m])))
            if _cls[l] > 0 and _cls[m] <= 0:
                _alms.append(np.complex(np.random.normal(0,0.5*_cls[l]),0))
            if _cls[l] <= 0 and _cls[m] > 0:
                _alms.append(np.complex(0,np.random.normal(0,0.5*_cls[m])))
            else:
                _alms.append(np.complex(0,0))
    
    return _alms   

def hpcltoalm(_cls, _NSIDE, _lmax): #Healpy generate alms given cls
    return hp.synalm(_cls, _lmax, new = True)

def cltomap(_cls, _NSIDE, _lmax): #doesn't work (isnt currently being used)
    _alm = cltoalm(_cls, _NSIDE, _lmax)
    return almtomap(_alm, _NSIDE, _lmax)

def hpcltomap(_cls, _NSIDE, _lmax):   #Healpy generate a map given a power spectrum
    return hp.synfast(_cls, _NSIDE, _lmax, new=True) 

#%%
#map --> something
def maptocl(_map): #does this manually - doesn't work (isnt currently being used)
    return

def hpmaptocl(_map, _NSIDE, _lmax): #Generate a power spectrum given cls
    return hp.anafast(_map, lmax = _lmax - 1)    #lmax = 3NSIDE by default

def maptoalm(_map): #does this manually - doesn't work (isnt currently being used)
    _omegp = (4*np.pi)/len(_map)
    _lmax = int(np.sqrt(len(_map)*(3/4)))
    _NSIDE = int(_lmax/3)
    _alm = []
    for l in range(_lmax):
        for m in range(l+1):
            _TpYlm = []
            for i in range(len(_map)):
                _TpYlm.append(_map[i]*np.conjugate(sphharm(m, l, i, _NSIDE)))
                    
            _alm.append(_omegp*sum(_TpYlm))
    
    return np.array(_alm)


def hpmaptoalm(_map, _lmax): #Healpy generate alms from map. 
    return hp.map2alm(_map, _lmax-1)

#alm --> something
def almtocl(_alm, lmax): #alm --> cl using alms in my ordering (different to healpys).
    _l = np.arange(lmax)
    _scaling = 1 / ((2*_l + 1))
    count = 0
    _new = []
    _cl = []
    for l in range(lmax):
        _new.append([])
        for m in range(l):
            if m == 0:
                _new[l].append(np.absolute(_alm[count])**2)
                count = count + 1
                
            if m > 0:
                _new[l].append(2*np.absolute(_alm[count])**2)
                count = count + 1
              
    for i in range(len(_new)):
        _cl.append(_scaling[i] * sum(_new[i]))
    
    return _cl

def hpalmtocl(_alms, _lmax): #Healpy estimates the power spectrum from the cls.
    return hp.alm2cl(_alms, lmax = _lmax-1)

def almtomap(_alm, _NSIDE, _lmax):# alm --> map using alms in my ordering (different to healpys).    #used in psi
    _map = []
    _Npix = 12*(_NSIDE)**2

    for i in range(_Npix):
        _sum = []
        _count = 0
        for l in np.arange(0,_lmax):
            for m in np.arange(0,l+1):
                if m == 0:
                    _sum.append(_alm[_count]*sphharm(m,l,i, _NSIDE))
                    _count = _count + 1
                else:
                    _sum.append(2*(np.real(_alm[_count])*np.real(sphharm(m,l,i, _NSIDE)) -
                                   np.imag(_alm[_count])*np.imag(sphharm(m,l,i, _NSIDE))))
                    _count = _count + 1
        _map.append(sum(_sum))

    return np.real(_map)
        

def almtomap_tf(_alm, _NSIDE, _lmax): #alm --> map for tensorflow using alms in my ordering (different to healpys).
    _map = tf.constant([])
    for i in range(12*(_NSIDE)**2):
        _sum = tf.constant([])
        _count = 0
        for l in range(_lmax):
            for m in range(l+1):
                if m==0:
                    _sum = tf.concat((_sum,[_alm[_count]*sphharm(m,l,i, _NSIDE)]), axis = 0)
                    _count = _count + 1
                else:
                    _sum = tf.concat((_sum,[2*((np.real(_alm[_count]))*(np.real(sphharm(m,l,i, _NSIDE)))-
                                               np.imag(_alm[_count])*np.imag(sphharm(m,l,i, _NSIDE)))]), axis = 0)
                    _count = _count + 1
        _map = tf.concat((_map,[sum(_sum)]), axis = 0)
    return tf.convert_to_tensor(_map)


def almtomap_tf2(_alm,_NSIDE, _lmax):
    _map = tf.Variable([])
    _ralm = tf.math.real(_alm) 
    _ialm = tf.math.imag(_alm) 
    _rsph = tf.math.real(_sph) 
    _isph = tf.math.imag(_sph) 
    _map = tf.Variable(np.array([]))
    for i in range(12*(_NSIDE)**2):
        _count = 0
        _term1 = tf.Variable(0.0,dtype = np.float64)
        for l in range(_lmax):
            for m in range(l+1):
                if m==0:
                    tf.compat.v1.assign_add(_term1, _ralm[_count]*_rsph[i][_count])
                    _count = _count + 1
                else:
                    tf.compat.v1.assign_add(_term1,2*(_ralm[_count]*_rsph[i][_count] - 
                                                                  _ialm[_count]*_isph[i][_count]),0.0)
                    _count = _count + 1

        _map = tf.concat((_map, [_term1]), axis = 0)
    _map = tf.dtypes.cast(_map, np.float64)
    return _map

def almtomap_tf3(_alm,_NSIDE, _lmax):  #used in psitf
    _ones = np.ones(len(_alm), dtype = np.complex128)
    _count = 0
    for l in range(_lmax):
        for m in range(l+1):     
            if m == 0:
                _ones[_count] = np.complex(0.5,0)
            _count = _count + 1
    _ones = tf.convert_to_tensor(_ones)  
    _alm = _ones*_alm
    _ralm = tf.math.real(_alm) 
    _ialm = tf.math.imag(_alm) 
    _rsph = tf.math.real(_sph) 
    _isph = tf.math.imag(_sph) 

    _map1 = tf.linalg.matvec(_rsph,_ralm)
    _map2 = tf.linalg.matvec(_isph,_ialm)
    _map = 2*(_map1 - _map2)
    return _map


def hpalmtomap(_alms, _NSIDE, _lmax):
    return hp.alm2map(_alms, _NSIDE ,_lmax-1)


#%%
#healpy smoothing for the map and the alms
def hpmapsmooth(_map, _lmax): #smooths a given map with a gaussian beam smoother.
    return _map #hp.smoothing(_map, lmax = _lmax)


def hpalmsmooth(_alms): #smooths a given set of alms with a gaussian beam smoother.
    return hp.smoothalm(_alms, fwhm = 0.0)

#splits/rejoins the alms into real/imaginary parts so that they can be optimised with scipy.optimize.minimize()
def singulartosplitalm(_alm):
    _realalm, _imagalm = _alm.real, _alm.imag
    return [_realalm, _imagalm]
    

def splittosingularalm(_realalm, _imagalm):
    _alm = []
    _ralmcount = 0
    _ialmcount = 0
    for l in range(lmax):
        for m in range(l+1):
            if m == 0 or m == 1:
                _alm.append(complex(_realalm[_ralmcount], 0))
                _ralmcount = _ralmcount + 1
            else:
                _alm.append(complex(_realalm[_ralmcount], _imagalm[_ialmcount]))
                _ralmcount = _ralmcount + 1
                _ialmcount = _ialmcount + 1
          
    return _alm


def splittosingularalm_tf(_realalm, _imagalm): #takes the real and imaginary parts of the alms and creates a tensor
    _zero = tf.zeros(1, dtype = np.float64)
    _count = 0
    for l in range(lmax): #pads zeros to to lmax = 0 values 
        for m in range(l + 1):
            if m == 0 or m == 1: 
                if l == 0:
                    _imagalm = tf.concat([_zero,_imagalm], axis = 0)
                else:
                    _front = _imagalm[:_count]
                    _back = _imagalm[_count:]
                    _term = tf.concat([_zero, _back] , axis = 0)
                    _imagalm = tf.concat([_front, _term], axis = 0)
            _count = _count + 1
    return tf.complex(_realalm,_imagalm)


#%%
#Retrieves the spherical harmonics for a given, l, m and pixel number
def sphharm(m, l, _pixno, _NSIDE):
    _theta, _phi = hp.pix2ang(nside=_NSIDE, ipix=_pixno)
    return sp.special.sph_harm(m, l, _phi, _theta)

#%%
#Changes the ordering of the alms from healpy to mine or vice versa
def almmotho(_moalm, _lmax):
    '''changing the alm ordering from my ordering to healpys'''
    _hoalm = []
    _count4 = []
    _count5 = 0
    for i in np.arange(2,_lmax+2):
        _count4.append(_count5)
        _count5=_count5+i
    for i in range(_lmax):
        _count1 = 0 
        _count2 = np.arange(0,_lmax,1)
        _count3 = np.arange(_lmax,0,-1)
        for j in np.arange(i+1,_lmax+1):
            _hoalm.append(_moalm[_count1+_count4[i]])
            _count1 = _count1 + j
    return np.array(_hoalm)


def almhotmo(_hoalm, _lmax):
    '''changing the alm ordering from healpys ordering to mine'''
    _moalm = np.zeros(sum(np.arange(_lmax+1)), dtype = complex)
    _count4 = []
    _count5 = 0
    for i in np.arange(2,_lmax+2):
        _count4.append(_count5)
        _count5 = _count5+i
    _count1 = 0
    for i in range(_lmax):
        _count2 = 0    
        for j in np.arange(i+1,_lmax+1):
            _moalm[_count2 + _count4[i]] = _hoalm[_count1]
            _count1 = _count1 + 1
            _count2 = _count2 + j        
    return np.array(_moalm)

#%%
def multtensor(_lmax,_lenalm):
    _shape = np.zeros([_lmax,_lenalm]) #matrix for the calculation of the psi in psi_tf
    _count = 0
    for i in range(_lmax):
        for j in np.arange(0,i+1):
            if j == 0:
                _shape[i][_count] = 1.0
                _count = _count + 1
            else:
                _shape[i][_count] = 2.0
                _count = _count + 1
    return tf.convert_to_tensor(_shape, dtype = np.float64)


#%%
#negative log of the posterior, psi.
def psi(_params, _map, _lmax, _NSIDE, _Ninv): #unnormalised log probability
    _lncl, _realalm, _imagalm = [0,0], [], []
    for i in range(len_cl-2):
        _lncl.append(_params[i])
    for i in range(len_ralm):
        _realalm.append(_params[i + len_cl-2])
    for i in range(len_ialm-(2*lmax-1)):
        _imagalm.append(_params[i + len_cl-2 + len_ralm])

    _d = _map
    _a = splittosingularalm(_realalm, _imagalm)
    _Ya = hpalmtomap(almmotho(_a,_lmax), _NSIDE, _lmax)
    _BYa =  _Ya #mapsmooth(_Ya, _lmax)
    
    _elem, _term1, _term2, _psi1 ,_psi2, _psi3 = [], [], [], [], [], []
    _sum = 0
    
    for i in range(len(_d)):
        _elem.append(_d[i] - _BYa[i])
        _psi1.append(0.5*(_elem[i]**2)*_Ninv[i]) #first term in the taylor paper 
    
    _l = np.arange(lmax)
    for i in range(len(_lncl)):
        _psi2.append((_l[i] + 0.5)*(_lncl[i])) #second term in the taylor paper 

    _a = np.absolute(np.array(_a))**2
    _as = np.matmul(shape.numpy(),_a)
    _psi3 = 0.5*_as/np.exp(np.array(_lncl)) #third term in the taylor paper 

    _psi = sum(_psi1) + sum(_psi2) + sum(_psi3) 
    print('psi =',_psi)
    return _psi


#negative log of the posterior, psi.def psi(_params, _map, _lmax, _NSIDE, _Ninv): #unnormalised log probability    _lncl, _realalm, _imagalm = [0,0], [], []    for i in range(len_cl-2):        _lncl.append(_params[i])    for i in range(len_ralm):        _realalm.append(_params[i + len_cl-2])    for i in range(len_ialm-(2*lmax-1)):        _imagalm.append(_params[i + len_cl-2 + len_ralm])     _d = _map    _a = splittosingularalm(_realalm, _imagalm)    _Ya = hpalmtomap(almmotho(_a,_lmax), _NSIDE, _lmax)    _BYa =  _Ya #mapsmooth(_Ya, _lmax)        _elem, _term1, _term2, _psi1 ,_psi2, _psi3 = [], [], [], [], [], []    _sum = 0        for i in range(len(_d)):        _elem.append(_d[i] - _BYa[i])        _psi1.append(0.5*(_elem[i]**2)*_Ninv[i]) #first term in the taylor paper         _l = np.arange(lmax)    for i in range(len(_lncl)):        _psi2.append((_l[i] + 0.5)*(_lncl[i])) #second term in the taylor paper      _a = np.absolute(np.array(_a))**2    _as = np.matmul(shape.numpy(),_a)    _psi3 = 0.5*_as/np.exp(np.array(_lncl)) #third term in the taylor paper      _psi = sum(_psi1) + sum(_psi2) + sum(_psi3)     print('psi =',_psi)    return _psi
def psi_tf(_params):
    _map, _lmax, _NSIDE, _Ninv = noisemap_tf, lmax, NSIDE, Ninv
    _lnclstart = tf.zeros(2, np.float64)
    _lncl = tf.concat([_lnclstart,_params[:(len_cl - 2)]], axis = 0)
    _realalm = _params[len_cl - 2:(len_ralm + len_cl - 2)]
    _imagalm = _params[(len_ralm + len_cl - 2):]
    
    _d = _map
    _a = splittosingularalm_tf(_realalm, _imagalm)
    _Ya = almtomap_tf3(_a, _NSIDE, _lmax)
    _BYa =  _Ya #mapsmooth(_Ya, _lmax)
    
    _elem = _d - _BYa
    _psi1 = 0.5*(_elem**2)*_Ninv #first term in the taylor paper 
    
    _l = tf.range(_lmax, dtype = np.float64)
    _psi2 = (_l+0.5)*_lncl #second term in the taylor paper 
    
    _a = tf.math.abs(_a)**2
    _as = tf.linalg.matvec(shape,_a)
    _psi3 = 0.5*_as/tf.math.exp(_lncl) #third term in the taylor paper 
        
    _psi = tf.reduce_sum(_psi1) + tf.reduce_sum(_psi2) + tf.reduce_sum(_psi3) 
    #print(_psi)
    __psi_record.append(_psi)
    #print('psi1',tf.reduce_sum(_psi1),'psi2',tf.reduce_sum(_psi2),'psi3',tf.reduce_sum(_psi3))
    return _psi

#%%
#Run the normal hmc sampler
def run_chain_hmc(initial_state, num_results = 25, num_burnin_steps=0): 
    '''Returns the desired walks through parameter space for a fixed step size.'''
    return tfp.mcmc.sample_chain(num_results=num_results, num_burnin_steps=num_burnin_steps, 
                               current_state=initial_state, kernel=hmc_kernel, trace_fn=lambda current_state,
                               kernel_results: kernel_results)

#Run the nut sampler chain
def run_chain_nut(initial_state, num_results=1000, num_burnin_steps=0): 
    '''Returns the desired walks through parameter space for a fixed step size.'''
    return tfp.mcmc.sample_chain(num_results=num_results, num_burnin_steps=num_burnin_steps, 
                               current_state=initial_state, kernel=nut_kernel, trace_fn=lambda current_state,
                               kernel_results: kernel_results)
