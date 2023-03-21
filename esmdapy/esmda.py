import pandas as pd
import numpy as np
import os 
import shutil 

from scipy.stats import truncnorm
import scipy.linalg as sla

from .utils import *
from .linalg import *


class esmda(object):
    def __init__(self,  **kwargs):
        # Set default parameters
        self.job_name = 'esmda_job'
        self.parameter_file_name = 'es_parameters.csv'
        self.observation_file_name = 'es_data.csv'
        self.data_file_name = 'es_data.csv'
        self.nEnsemble = 100 #the number of ensembles
        self.maxIter = 10  #the number of iterations
        self.inversion_type = 'svd'

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.parameter_data = pd.read_csv(self.parameter_file_name)
        self.observation_data = pd.read_csv(self.observation_file_name)

        # Initialise global arrays
        self.mLength = len(self.parameter_data)
        self.dLength = len(self.observation_data)
        self.mPrior = np.zeros([self.mLength, self.nEnsemble]) #Prior ensemble
        self.mPrior_untransformed = np.zeros([self.mLength, self.nEnsemble]) #Prior ensemble

        if os.path.exists(self.job_name):
            print('Deleting directory ' + self.job_name)
            shutil.rmtree(self.job_name)
        os.mkdir(self.job_name)
            
      


    def build_prior(self):
        """
        Generate an ensemble of logit transformed parameters by scaling truncnorm sampled parameters.
        add some more docstring
        """
        es_parameters = self.parameter_data.copy() 
        es_parameters['std'] = (es_parameters['upper'] - es_parameters['lower']) / es_parameters['width']
        es_parameters['a'] = (es_parameters['lower'] - es_parameters['mean']) / es_parameters['std']
        es_parameters['b'] = (es_parameters['upper'] - es_parameters['mean']) / es_parameters['std']
        print(es_parameters) 

        stdevM = es_parameters['std'].values
        param_mean = es_parameters['mean'].values
        a = es_parameters.a.values
        b = es_parameters.b.values

        # Generate prior parameter distribution in the parameter space
        for i in range(self.mLength):
            self.mPrior_untransformed[i,:] = truncnorm.rvs(a[i], b[i], loc=param_mean[i], scale=stdevM[i], size=self.nEnsemble, random_state=None)

        # Transform parameters to logit space
        for i in range(self.nEnsemble):
            scaled = scale_param(self.mPrior_untransformed[:,i], self.parameter_data['lower'].values, self.parameter_data['upper'].values)
            transed = logit(scaled)
            self.mPrior[:,i] = transed

        # Set the current parameter set for the 0th iteration
        self.mCurrent = self.mPrior 

    def report(self, iter, M, D):
        iteration_parameters = pd.DataFrame(M.T, columns=self.parameter_data.parameter.values)
        file_name = self.job_name + '_' + str(iter) + '_parameters.csv'
        iteration_parameters.T.to_csv(self.job_name + '/' + file_name)

        iteration_data = pd.DataFrame(D.T, columns=self.observation_data.name.values)
        iteration_data = iteration_data.T 
        file_name = self.job_name + '_' + str(iter) + '_data.csv'
        iteration_data.to_csv(self.job_name + '/' + file_name)

        print('Completed iteration ', iter)
        print('============================')
        print(' ')


    def run_esmda(self, fill_ensemble):
        alpha = self.maxIter
        for iter in range(self.maxIter+1):
    # Inverse tranform parameters
            M = np.zeros_like(self.mCurrent)
            for i in range(self.nEnsemble):
                scaled = inv_logit(self.mCurrent[:,i])
                M[:,i] = inverse_scale_param(scaled, self.parameter_data['lower'].values, self.parameter_data['upper'].values)

            #fill the data ensemble
            D = fill_ensemble(M, self.nEnsemble, self.mLength, self.dLength)

            self.report(iter, M, D) 
  
            # If we are at the final iteration, we have evaluated the model, we don't have to updaate the parameters
            # we can quit now
            if iter == self.maxIter :
                print('maxiter ', iter)
                return

            # calculate Cdd
            D_mean = D.mean(axis=1)
            del_D = np.zeros_like(D)

            for i in range(self.nEnsemble):
                del_D[:,i] = D[:,i] - D_mean

            Cdd = (del_D@del_D.T)/(self.nEnsemble-1)

            #Calculate Cmd
            M_mean = self.mCurrent.mean(axis=1)
            del_M = np.zeros_like(self.mCurrent)
            for i in range(self.nEnsemble):
                del_M[:,i] = self.mCurrent[:,i] - M_mean

            Cmd = del_M@del_D.T /(self.nEnsemble-1)

            # Perturb Observations
            Duc = np.zeros_like(D)
            for i in range(self.nEnsemble):
                Duc[:,i] = np.sqrt(alpha)*np.random.normal(0,1,self.dLength) * self.observation_data.noise.values+self.observation_data.value.values

            #calculate Cd

            Cd = np.zeros([self.dLength, self.dLength])
            for i in range(self.dLength):
                Cd[i,i] = self.observation_data.noise.values[i]**2

            Kinv = pseudo_inverse(del_D, alpha, Cd, self.nEnsemble, self.dLength, self.mLength, self.inversion_type)
           # K = Cdd + alpha*Cd
           # Kinv, svd_rank = sla.pinvh(K, return_rank=True)

    
            self.mCurrent = self.mCurrent+Cmd@Kinv@(Duc-D)

    def predictive_posterior(self, n):
        pred_post = pd.DataFrame()
        file_name = self.job_name + '_' + str(self.maxIter) + '_data.csv'
        posterior = pd.read_csv(self.job_name + '/' + file_name)
        del posterior[posterior.columns[0]]
        noise = self.observation_data.noise.values
        for i in posterior.columns:
            samples = pd.DataFrame()
            data = posterior[i].values
            samples[i] = data
            for j in range(n):
                label = str(i) + '_' + str(j+1)
        
                sample = np.random.normal(0,1,self.dLength) * self.observation_data.noise.values+data
                samples[label] = sample

            pred_post = pd.concat([pred_post, samples], axis=1)
        pred_post.set_index(self.observation_data.name, inplace=True)
        pred_post.to_csv(self.job_name + '/' + 'posterior_predictive.csv')
        