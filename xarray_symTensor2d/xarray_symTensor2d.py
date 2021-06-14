import xarray as xr
import numpy as np


@xr.register_dataarray_accessor("sT")

class sT(object):
    '''
    This is a classe to work on 2d symetric Tensor of dim 3x3 in xarray environnement.
    
    Only 6 components are stored (exx,eyy,ezz,exy,exz,eyz)
    '''
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj 
    
    pass

#---------------------------------------Function-------------------------------------------
    def eqVonMises(self,lognorm=False):
        '''
        Compute the equivalent deforamtion of Von Mises (https://en.wikipedia.org/wiki/Infinitesimal_strain_theory)

        :return eqVM: (2./3.e^d_ij.e^d_ij)**0.5 with e^d=e-1/3*tr(e)*I
        :rtype eqVM: xr.DataArray
        '''

        deq=np.array(self._obj)
        tr=np.nanmean(np.array(self._obj[...,0:3]),axis=-1)

        deq[...,0]=deq[...,0]-tr
        deq[...,1]=deq[...,1]-tr
        deq[...,2]=deq[...,2]-tr

        deq=2/3*deq**2
        deq=np.nansum(deq,axis=-1)
        #deq[np.isnan(self._obj[...,0])]=np.nan
        
        if lognorm:
            med=np.nanmedian(deq,axis=(-1, -2))
            for i in range(len(self._obj.time)):
                deq[i,...]=np.log(deq[i,...]/med[i])
            
            
        xr_deq=xr.DataArray(deq,dims=self._obj.coords.dims[0:-1])

        return xr_deq

    
    def mean(self,axis='tyy'):
        '''
        Compute the average of one componant
        
        :param axis: txx,tyy,tzz,txy,txz,tyz
        :type axis: str
        :return mean: average of this component
        :rtype mean: xr.DataArray
        '''
        
        label=['txx','tyy','tzz','txy','txz','tyz']
        
        if axis in label:
            res=np.nanmean(np.array(self._obj[...,label.index(axis)]),axis=(-1, -2))
                       
            return xr.DataArray(res,dims=self._obj.coords.dims[0])
            
        else:
            print('Error: axis not defined please select one of thos axis : txx,tyy,tzz,txy,txz,tyz')
            return 
    