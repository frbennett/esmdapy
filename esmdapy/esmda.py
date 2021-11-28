import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import truncnorm
from .linalg import * 
from .utils import * 
import scipy.linalg as sla

# Start of class second commit

class esmda(object):
    def __init__(self,  **kwargs):
        
        self.parameter_file_name = 'es_parameters.csv'
        self.data_file_name = 'es_data.csv'
        self.nEnsemble = 100 #the number of ensembles
        self.maxIter = 10  #the number of iterations
        self.Error_Model = True,
        self.job_name = 'es_msda_run'
        self.adaptive = False
        self.standarderror = False
        self.initial_scale_factor = 1.0
        self.inverse_type = 'standard'
        self.set_rank = False
        self.rank = 1
        self.scale = True

        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def plot_data(self, **kwargs):
        self.upper_quant = 0.95
        self.lower_quant = 0.05
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
#        data_to_plot = pd.read_csv('final_data.1csv')
        data_to_plot = pd.read_csv(self.job_name + '_' + '_final_data.csv')
        observed_data = pd.read_csv(self.data_file_name)
        column_names = data_to_plot.columns.to_list()
        column_names[0] = 'name'
        data_to_plot.columns = column_names
        data_to_plot.set_index(data_to_plot.columns[0], inplace=True)
        data_to_plot['lower'] = data_to_plot.quantile(q=self.lower_quant, axis=1)
        data_to_plot['upper'] = data_to_plot.quantile(q=self.upper_quant, axis=1)
        data_to_plot['median'] = data_to_plot.quantile(q=0.5, axis=1) 

        fig, ax = plt.subplots(figsize=(15,8)) 

        ax.plot(data_to_plot.index, observed_data['value'], 'o')
        ax.plot(data_to_plot['median'], 'o')
        ax.fill_between(data_to_plot.index,data_to_plot['lower'], data_to_plot['upper'], alpha=0.2) 
            
                        
    def run_esmda(self, fill_Prior):
        print("Running job ", self.job_name)
        print("=============================================")
        print(" ")
        # read in parameters
        es_parameters = pd.read_csv(self.parameter_file_name)
        es_parameters['std'] = (es_parameters['upper'] - es_parameters['lower']) / es_parameters['width']
        es_parameters['a'] = (es_parameters['lower'] - es_parameters['mean']) / es_parameters['std']
        es_parameters['b'] = (es_parameters['upper'] - es_parameters['mean']) / es_parameters['std']
        mLength = len(es_parameters)  
        
        # read in data
        es_data = pd.read_csv(self.data_file_name)
        dLength = len(es_data) #the length of the data d
        stdevD = np.diag(es_data['noise'].values)

        #Constants
        maxIter = self.maxIter
        nEnsemble = self.nEnsemble
        Error_Model = self.Error_Model  
        #Declaring variables
        mInit = np.zeros(mLength) #Initial ensemble
        mAnswer = np.zeros(mLength) #True parameter values
        mPred = np.zeros([mLength, nEnsemble]) #Predicted ensemble
        mAverage = np.zeros(mLength)
        dAverage = np.zeros(dLength)

        sigma_last_iteration = np.zeros(dLength)
        previous_sigma_mean_dict = {} 
        E_tilde = np.zeros(mLength)
        Residuals = np.zeros([dLength, nEnsemble])  
        sp_matrix = np.zeros([dLength, dLength])

        OneN_e = np.ones(nEnsemble).reshape(1,nEnsemble)

        d_obs = es_data.value.values
        d_obs_matrix = np.zeros([dLength, nEnsemble])
        for i in range(nEnsemble):
            d_obs_matrix[:,i] = d_obs

        obsData = np.zeros([dLength, nEnsemble]) #Observed data --> true data + measurement noise
        dAnswer = np.zeros(dLength) #True data values
        z = np.zeros([dLength, nEnsemble])

        deltaM = np.zeros([mLength, nEnsemble])
        deltaD = np.zeros([dLength, nEnsemble])
        ddMD = np.zeros(nEnsemble)
        ddDD = np.zeros(nEnsemble)

        mLength = len(es_parameters)
        mPrior = np.zeros([mLength, nEnsemble]) #Prior ensemble
        dPrior = np.zeros([dLength, nEnsemble]) #Prior ensemble
        
        # Set the effective rank of the Kalman gain matrix
        if Error_Model:
            rank = nEnsemble*2
        else:
            rank = nEnsemble
        
        rank = min(rank, dLength)
        if self.set_rank:
            rank = self.rank
        print('rank ', rank)

        alpha_sum = 0

        # Construct the Parameter prior
        #Perturb the parameter using a truncated normal distribution
        stdevM = es_parameters['std'].values
        param_mean = es_parameters['mean'].values
        a = es_parameters.a.values
        b = es_parameters.b.values
        for i in range(mLength):
            mPrior[i,:] = truncnorm.rvs(a[i], b[i], loc=param_mean[i], scale=stdevM[i], size=nEnsemble, random_state=None)

        m=mPrior 
                
        #------------------------------------------MAIN LOOP STARTS HERE---------------------------------------------------------#
        the_as = []
        for p in range(maxIter):
            the_as.append((2**(maxIter-p)))
        
        print('Filling initial dPrior matrix')
        print('')
        print("=============================================")
        print(" ")
            
        for p in range(maxIter):
    
            dPrior = fill_Prior(mPrior,dLength, nEnsemble)
            if (p==0):
                d = dPrior 
    
            iteration_data = pd.DataFrame(dPrior.T, columns=es_data.name.values)
            data_median = iteration_data.median().values 
            data_mean = iteration_data.mean().values
            data_std = iteration_data.std().values
            lower_quant = iteration_data.quantile(0.025).values
            upper_quant = iteration_data.quantile(0.975).values
            iteration_data = iteration_data.T 
            iteration_data['mean'] = data_mean
            iteration_data['median'] = data_median 
            iteration_data['std'] = data_std 
            iteration_data['lower_quant'] = lower_quant
            iteration_data['upper_quant'] = upper_quant

            file_name = self.job_name + '_' + str(p) + '_data.csv'
            iteration_data.to_csv(file_name)
    
            iteration_parameters = pd.DataFrame(mPrior.T, columns=es_parameters.parameter.values)
            file_name = self.job_name + '_' + str(p) + '_parameters.csv'
            iteration_parameters.T.to_csv(file_name)
    
            #Choose value for alpha    
            Osum = 0
            for i in range(nEnsemble):
                Osum += (dPrior[:,i] - d_obs).T@np.linalg.inv(stdevD@stdevD)@(dPrior[:,i] - d_obs) *1/nEnsemble * 1/(2*dLength)
        #    alpha = min([aFactor * Osum, maxAlpha])
            alpha = Osum
            alpha = the_as[p]
            alpha = maxIter 
    
            if (p==(maxIter-1)):
                alpha = 1/(1-alpha_sum)
                
            
        
            alpha_sum += 1/alpha 

            print('')
            print("=============================================")
            print("===== Iteration ",p)
            print("===== Objective function ",Osum)
         #    print("===== Scaled Objective function ",aFactor * Osum)
            print("===== chosen alpha ",alpha)
            print("===== sum of 1/alpha ",alpha_sum)
            print("=============================================")
            print("")

            #Get data
    
            #Adding measurement noise to the true data   
            d_obs = es_data.value.values
            for i in range(dLength):
                obsData[i,:] = np.random.normal(d_obs[i], np.sqrt(alpha)*stdevD[i,i], nEnsemble)
            #Calculate Average and Covariance MD and Covariance DD

            Residuals = d_obs.reshape(dLength, 1)@OneN_e - dPrior 
            sigma_m = np.average(Residuals, axis=1)
            
            if self.standarderror:
                for i in range(dLength):
                    sigma_m[i] = sigma_m[i]/stdevD[i,i]
                                                 
            if (p==0):
                sigma_max = np.max(np.abs(Residuals), axis=1)
                residual_max_dict = {}
                residual_mean_dict = {}
                sp_dict = {}
                data_series_labels = es_data.series.unique()
                for i in data_series_labels:
                    residual_mean_dict[i] = []
                    residual_max_dict[i] = []
        
                for i in range(dLength):
                    residual_mean_dict[es_data.series[i]].append(sigma_m[i])
                    residual_max_dict[es_data.series[i]].append(sigma_max[i])
    
                for i in data_series_labels:
                    if (type(self.initial_scale_factor)==int) or (type(self.initial_scale_factor)==float):
                        scale_factor = self.initial_scale_factor

                    if (type(self.initial_scale_factor)==dict):
                        scale_factor = self.initial_scale_factor[i]    

                    sp_dict[i] = np.linalg.norm(residual_mean_dict[i])/np.linalg.norm(residual_max_dict[i]) * scale_factor 
                    previous_sigma_mean_dict[i] = np.linalg.norm(residual_mean_dict[i])
        
                for i in range(dLength):
                    sp_matrix[i,i] = sp_dict[es_data.series[i]]
                
                
         
            else:
                residual_mean_dict = {}
                sp_dict = {}
                data_series_labels = es_data.series.unique()
                for i in data_series_labels:
                    residual_mean_dict[i] = []
    
                for i in range(dLength):
                    residual_mean_dict[es_data.series[i]].append(sigma_m[i])
    
                for i in data_series_labels:
                    sp_dict[i] = np.linalg.norm(residual_mean_dict[i])/previous_sigma_mean_dict[i]
                    previous_sigma_mean_dict[i] = np.linalg.norm(residual_mean_dict[i])
                for i in range(dLength):
                    sp_matrix[i,i] = sp_dict[es_data.series[i]]
                    
            E_tilde = sp_matrix@Residuals
                                            
            print("Selected s_p = ", sp_dict)
            
        
             #Calculate Average and Covariance MD and Covariance DD

            mAverage = np.average(mPrior, axis=1).reshape(mLength,1)
            dAverage = np.average(dPrior, axis=1).reshape(dLength,1)
            E_tildeAverage = np.average(E_tilde, axis=1).reshape(dLength,1)
             
            deltaD = dPrior -dAverage
            deltaM = mPrior -mAverage 
            deltaE_tilde = E_tilde -E_tildeAverage

            covarianceMD = deltaM@deltaD.T / (nEnsemble - 1.)
            covarianceDD = deltaD@deltaD.T / (nEnsemble - 1.)
            covarianceEE = deltaE_tilde@deltaE_tilde.T / (nEnsemble-1.)
            ## My test
            test = covarianceDD + covarianceEE + alpha*stdevD@stdevD
            if np.allclose(test, test.T):
                print(' matrix is symmmetric')
            else: 
                print(' matrix is not symmmetric')
              
            
#            if Error_Model :
#                deltaM = covarianceMD@np.linalg.inv(covarianceDD + covarianceEE + alpha*stdevD@stdevD)@(obsData - dPrior - E_tilde)
#                deltaM = covarianceMD@linalg.tinv(covarianceDD + covarianceEE + alpha*stdevD@stdevD, rank, type=self.inverse_type)@(obsData - dPrior - E_tilde)
#            else :
#                deltaM = covarianceMD@np.linalg.inv(covarianceDD + alpha*stdevD@stdevD)@(obsData - dPrior)
#                deltaM = covarianceMD@linalg.tinv(covarianceDD + alpha*stdevD@stdevD, rank, type=self.inverse_type)@(obsData - dPrior)


                
            if Error_Model :
                km = covarianceDD + covarianceEE + alpha*stdevD@stdevD
            else :
                km = covarianceDD + alpha*stdevD@stdevD

            print('Shape of Kalman matrix ', np.shape(km))
            my_timer = Timer()
            print("Starting Kalman matrix decomposition")
            my_timer.start()
            if self.scale:
                print("Scaling Kalman matrix")
                inv_stdevD = np.diag(stdevD.diagonal()**-1)
                I_N_n = np.identity(dLength)
                C_tilde = inv_stdevD@km@inv_stdevD
                C_tilde_inv = tinv(C_tilde, rank, type=self.inverse_type)
                km_inv = inv_stdevD@C_tilde_inv@inv_stdevD
            else:
                km_inv= tinv(km, rank, type=self.inverse_type)

                
            my_timer.stop()   
            print("Completed Kalman matrix decomposition")

            if Error_Model:
                deltaM = covarianceMD@km_inv@(obsData - dPrior - E_tilde)
            else:
                deltaM = covarianceMD@km_inv@(obsData - dPrior)

            print('Shape of obsData ', np.shape(obsData))
            
            mPred = mPrior + deltaM
            #Check parmater bounds 
            for j in range(nEnsemble):
                for i in range(0,mLength):
                    if mPred[i,j] > es_parameters.upper.values[i] :
                        mPred[i,j] = es_parameters.upper.values[i]
                    if mPred[i,j] < es_parameters.lower.values[i] :
                        mPred[i,j] = es_parameters.lower.values[i] 
    
    
            #Update the prior parameter for next iteration
            mPrior = mPred
    
            print('')
            print("=============================================")
            print("===== Parameters predicted from iteration ",p)
            print("=============================================")
            print("")

            print('')
            for j in range(mLength):
                print(es_parameters.parameter.values[j],  mPrior[j,].mean(),mPrior[j,].min(),mPrior[j,].max(),mPrior[j,].std())
            print("=============================================")
            print('')
            
        iteration_data = pd.DataFrame(dPrior.T, columns=es_data.name.values)
        iteration_data = iteration_data.T 
        file_name = self.job_name + '_' + '_final_data.csv'
        iteration_data.to_csv(file_name)

        iteration_parameters = pd.DataFrame(mPrior.T, columns=es_parameters.parameter.values)
        file_name = self.job_name + '_' + '_final_parameters.csv'
        iteration_parameters.T.to_csv(file_name)
        np.savetxt
    

