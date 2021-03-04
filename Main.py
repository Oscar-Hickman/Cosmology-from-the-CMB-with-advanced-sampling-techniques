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
def cltoalm(_cls, _NSIDE, _lmax): #doesn't work (isnt currently being used)
    _alms = []
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
    return hp.synalm(_cls, _lmax - 1, new = True)

def cltomap(_cls, _NSIDE, _lmax): #doesn't work (isnt currently being used)
    _alm = cltoalm(_cls, _NSIDE, _lmax)
    return almtomap(_alm, _NSIDE, _lmax)

def hpcltomap(_cls, _NSIDE, _lmax):   #Healpy generate a map given a power spectrum
    return hp.synfast(_cls, _NSIDE, _lmax - 1, new=True) 


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


#%%
##alm --> something
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
        

def almtomap_tf(_alm,_NSIDE, _lmax, _sph):  #used in psitf
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


#%%
#splits/rejoins the alms into real/imaginary parts so that they can be optimised with scipy.optimize.minimize()
def singulartosplitalm(_alm):
    _realalm, _imagalm = _alm.real, _alm.imag
    return [_realalm, _imagalm]
    

def splittosingularalm(_realalm, _imagalm, lmax):
    _alm = []
    _ralmcount = 0
    _ialmcount = 0
    for l in range(lmax):
        for m in range(l+1):
            if l == 0 or l == 1:
                _alm.append(complex(0,0))
            else:  
                if m == 0 or m == 1:
                    _alm.append(complex(_realalm[_ralmcount], 0))
                    _ralmcount = _ralmcount + 1
                else:
                    _alm.append(complex(_realalm[_ralmcount], _imagalm[_ialmcount]))
                    _ralmcount = _ralmcount + 1
                    _ialmcount = _ialmcount + 1
          
    return _alm


def splittosingularalm_tf(_realalm, _imagalm, lmax): #takes the real and imaginary parts of the alms and creates a tensor
    _zero = tf.zeros(1, dtype = np.float64)
    _count = 0
    for i in range(3):
        _realalm = tf.concat([_zero,_realalm], axis = 0)
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


def alminit(_alms, _lmax):
    #pads zeros to the real l=0 and l=1 terms of the alms - in my ordering 
    _count = 0
    for l in range(_lmax):
        for m in range(l + 1):
            if l == 0 or l == 1:
                _alms[_count] = complex(0,0)
                _count = _count + 1
    _count = 0
    for l in range(_lmax):
        for m in range(l + 1):
            if m == 0 or m == 1:
                _alms[_count] = complex(np.real(_alms[_count]),0)
                _count = _count + 1
            else:
                _count = _count + 1
    return _alms


def hpalminit(_alms, _lmax):
    #pads zeros to the real l=0 and l=1 terms of the alms - in healpys ordering 
    _count = 0
    for l in range(_lmax):
        for m in range(l + 1):
            _count = _count + 1
            if _count == 1 or _count == 2 or _count == _lmax+1:
                _alms[_count - 1] = complex(0,0)
    _count = 0
    for l in range(2*_lmax - 1):
        _alms[_count] = complex(np.real(_alms[_count]),0)
        _count = _count + 1
    return _alms


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
#Run the normal hmc sampler
def run_chain_hmc(modelparams, initial_state,_step_size = 0.01, num_results = 1000, num_burnin_steps=0, _n_lfs = 2): 
    '''Returns the desired walks through parameter space for a fixed step size.'''
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(target_log_prob_fn=modelparams.psi_tf, step_size=_step_size, num_leapfrog_steps=_n_lfs)
    return tfp.mcmc.sample_chain(num_results=num_results, num_burnin_steps=num_burnin_steps, 
                               current_state=initial_state, kernel=hmc_kernel, trace_fn=lambda current_state,
                               kernel_results: kernel_results)
#Run the nut sampler chain
def run_chain_nut(modelparams, initial_state, _step_size, num_results=1000, num_burnin_steps=0, mtd = 10, med = 1000, u_lfs = 1, pi = 10): 
    '''Returns the desired walks through parameter space for a fixed step size.'''
    nut_kernel = tfp.mcmc.NoUTurnSampler(target_log_prob_fn=modelparams.psi_tf, step_size=_step_size, max_tree_depth=mtd, max_energy_diff=med,
                                         unrolled_leapfrog_steps=u_lfs, parallel_iterations=pi)
    return tfp.mcmc.sample_chain(num_results=num_results, num_burnin_steps=num_burnin_steps, 
                               current_state=initial_state, kernel=nut_kernel, trace_fn=lambda current_state,
                               kernel_results: kernel_results)

#%%
class Cosmology_Advanced_Sampling:
    '''
    '''
    def __init__(self, _lmax, _NSIDE, _noisesig):
        '''
        '''
        lcdm_parameters = np.array([67.74, 0.0486, 0.2589, 0.06, 0.0, 0.066]) #parameters for the ΛCDM model
        
        NPIX = 12*(_NSIDE**2)
        n = np.linspace(_noisesig,_noisesig,NPIX) #Array of stds for all the pixels
        Ninv = []
        for i in range(NPIX):
            Ninv.append(1/(n[i]**2)) #finds the inverse noise matrix
        lcdm_cls = call_CAMB_map(lcdm_parameters, _lmax) #power spectrum for the given parameters and lmax.
        notpad_lcdm_alms = hpcltoalm(lcdm_cls, _NSIDE, _lmax)
        pad_lcdm_alms = hpalminit(notpad_lcdm_alms, _lmax)
        pad_lcdm_map = hpalmtomap(pad_lcdm_alms, _NSIDE, _lmax)  #generates a map from the power spectrum
        pad_lcdm_map = hpmapsmooth(pad_lcdm_map, _NSIDE) #applies a gaussian beam smoother to the map
        notpad_prior_map = noisemapfunc(pad_lcdm_map,n[0])[0] #adds noise to the map
        notpad_prior_halms = hpmaptoalm(notpad_prior_map, _lmax) #noisey alms in my  healpys ordering
        pad_prior_halms = hpalminit(notpad_prior_halms,_lmax)
        pad_prior_map = hpalmtomap(pad_prior_halms, _NSIDE, _lmax)
        pad_prior_alms = almhotmo(pad_prior_halms, _lmax) #noisy alms in my ordering 
        pad_prior_cls = hpalmtocl(pad_prior_halms, _lmax) #noisy power spectrum
        
   
        _sph = []
        for i in range(int((_NSIDE**2)*12)):
            _sph.append([])
            _count = 0
            for l in range(_lmax):
                for m in range(l+1):
                    _theta, _phi = hp.pix2ang(nside=_NSIDE, ipix=i)
                    _sph[i].append(sp.special.sph_harm(m, l, _phi, _theta))
                    if l==0:    
                        _sph[i][_count] = np.complex(np.real(_sph[i][_count]),np.float64(0.0))
                    _count = _count + 1 
        _sph = tf.convert_to_tensor(_sph, dtype = np.complex128)
        shape = multtensor(_lmax,int(_lmax*(_lmax + 1)/2)) #A tensor for the spherical harmonics in the maptoalm_tf function
        r_alms_init = pad_prior_alms.real
        i_alms_init = pad_prior_alms.imag
        x0 = []
        
        _count = 0
        for i in range(_lmax - 2):
            if pad_prior_cls[i+2] > 0:
                x0.append(np.log(pad_prior_cls[i+2]))
            else:
                x0.append(0)
        _count = 0
        for l in range(_lmax):
            for m in range(l + 1):
                if l == 0 or l == 1:
                    _count = _count + 1
                else:
                    x0.append(r_alms_init[_count])
                    _count = _count + 1
        _count = 0
        for l in range(_lmax):
            for m in range(l + 1):
                if m == 0 or m == 1:
                    _count = _count + 1
                else:
                    x0.append(i_alms_init[_count])
                    _count = _count + 1
                 
        self.lmax = _lmax
        self.NSIDE = _NSIDE
        self.noisesig = _noisesig
        self.Ninv = Ninv
        self.NPIX = NPIX
        
        self.lcdm_cls = lcdm_cls
        self.lcdm_alms = pad_lcdm_alms
        self.lcdm_map = pad_lcdm_map
        self.prior_cls = pad_prior_cls
        self.prior_alms = pad_prior_alms
        self.prior_map = pad_prior_map
    
        self.shape = shape
        self.sph = _sph
        self.x0 = x0
    
    
    def lcdm_alms(self): #return the alms from the lambda cdm model
        return self.lcdm_alms
    def prior_alms(self): #return the prior alms
        return self.prior_alms
    def prior_parameters_tf(self): #return the prior for the log posterior
        return tf.convert_to_tensor(self.x0)
    
    
    def psi(self, _params): #unnormalised log probability
        '''
        Negative log of the posterior - 'psi'.
        '''
        _params = self.x0
        _lmax = self.lmax
        _NSIDE = self.NSIDE
        _map = self.prior_map
        _Ninv = self.Ninv
        _lncl, _realalm, _imagalm = [0,0], [], []
        for i in range(_lmax-2):
            _lncl.append(_params[i])
        for i in range(int(_lmax*(_lmax+1)/2) - 3):
            _realalm.append(_params[i + _lmax-2])
        for i in range(int(_lmax*(_lmax+1)/2)-(2*_lmax-1)):
            _imagalm.append(_params[i + _lmax-2 + int(_lmax*(_lmax+1)/2) - 3])
    
        _d = _map
        _a = splittosingularalm(_realalm, _imagalm, _lmax)
        _Ya = hpalmtomap(almmotho(_a,_lmax), _NSIDE, _lmax)
        _BYa =  _Ya #mapsmooth(_Ya, _lmax)
        
        _elem, _psi1 ,_psi2, _psi3 = [], [], [], []
        
        for i in range(len(_d)):
            _elem.append(_d[i] - _BYa[i])
            _psi1.append(0.5*(_elem[i]**2)*_Ninv[i]) #first term in the taylor paper 
        
        _l = np.arange(_lmax)
        for i in range(len(_lncl)):
            _psi2.append((_l[i] + 0.5)*(_lncl[i])) #second term in the taylor paper 
    
        _a = np.absolute(np.array(_a))**2
        _as = np.matmul(self.shape.numpy(),_a)
        _psi3 = 0.5*_as/np.exp(np.array(_lncl)) #third term in the taylor paper 
    
        _psi = sum(_psi1) + sum(_psi2) + sum(_psi3) 
        print('psi =',_psi)
        return _psi
    
    
    def psi_tf(self,_params):
        '''
        #negative log of the posterior - psi, in Tensorflow.
        '''
        _map, _lmax, _NSIDE, _Ninv = self.prior_map, self.lmax, self.NSIDE, self.Ninv
        _lnclstart = tf.zeros(2, np.float64)
        _lncl = tf.concat([_lnclstart,_params[:(_lmax - 2)]], axis = 0)
        _realalm = _params[_lmax - 2:(int(_lmax*(_lmax+1)/2) - 3 + _lmax - 2)]
        _imagalm = _params[(int(_lmax*(_lmax+1)/2) - 3 + _lmax - 2):]
        
        _d = _map
        _a = splittosingularalm_tf(_realalm, _imagalm, _lmax)
        _Ya = almtomap_tf(_a, _NSIDE, _lmax, self.sph)
        _BYa =  _Ya #mapsmooth(_Ya, _lmax)
        #print('a',_a)
        _elem = _d - _BYa
        _psi1 = 0.5*(_elem**2)*_Ninv #first term in the taylor paper 
        #print('d',_d)
        #print('Bya',_BYa)
        #print('abdif', abs(_d - _BYa))
        _l = tf.range(_lmax, dtype = np.float64)
        _psi2 = (_l+0.5)*_lncl #second term in the taylor paper 
        
        _a = tf.math.abs(_a)**2
        _as = tf.linalg.matvec(self.shape,_a)
        _psi3 = 0.5*_as/tf.math.exp(_lncl) #third term in the taylor paper 
            
        _psi = tf.reduce_sum(_psi1) + tf.reduce_sum(_psi2) + tf.reduce_sum(_psi3) 
        print('psi =',_psi.numpy())   
        print()
        #__psi_record1.append(tf.reduce_sum(_psi1).numpy())
        #__psi_record2.append(tf.reduce_sum(_psi2).numpy())
        #__psi_record3.append(tf.reduce_sum(_psi3).numpy())
        #__psi_record.append(_psi.numpy())
        #print('psi1',tf.reduce_sum(_psi1),'psi2',tf.reduce_sum(_psi2),'psi3',tf.reduce_sum(_psi3))
        return _psi

