# An implementation of the Kalman filter.
import numpy as np
from numpy.linalg import inv

class Kalman(object):
    def __init__(self,x0,covx0,z,covz,A,b,covupdate,M):
        """
        Kalman filter is an example of a Gaussian process where the future predicted probability distribution
        an application of Bayes' rule where the prior and likelihoods are just Gaussian on a Markov chain.
        2016-12-23

        Params:
        -------
        x0 (ndarray)
            Initial estimate of hidden state of system.
        covx0 (ndarray)
            Covariance of x0.
        z (ndarray)
            n_dim x n_samples. Observations of observable states.
        covz (ndarray)
            (n_dim,n_dim) Covariance of z.
        A (ndarray)
            Transition update. x_{t+1}=A*x_t+b
        b (ndarray)
            (n_dim,1) or (n_dim,n_samples). Constant offset in transition update.
        covupdate (ndarray)
            Covariance of error in update step.
        M (ndarray)
            Matrix to convert from hidden to observable state. z=M*x.
        """
        self.x0=x0
        self.covx0=covx0
        self.z=z
        self.covz=covz
        self.A=A
        if b.ndim==1 or b.shape[1]==1:
            self.b=np.tile(b,(1,z.shape[1]))
        else:
            self.b=b
        self.covupdate=covupdate
        self.M=M

        self.L=inv(covupdate)
        self.Q=inv(covz)

        # Should make sure that Qij=0 and Qii=1 for unobserved states.

    def filter(self,dt=0):
        """
        Standard Kalman filter. If dt>0, predict into the future. Store list of expected mu0 in correctedmux.
        Update estimate of state in the past using the correction step in the KF and use that to predict dt
        into the future.
        2016-12-23

        Params:
        -------
        dt (int=0)
            
        Values:
        -------
        predictedmux
        correctedmux 2016-12-23
        """
        if dt==0:
            T=self.z.shape[1]
            correctedmux=np.zeros((len(self.x0),T))  # result of prediction once accounting for obs

            # Calculate KF corrected estimate of state with data.
            Sigmax=self.covx0
            iSigmax=inv(Sigmax)
            mux=self.x0

            for i in xrange(T):
                R=self.M.T.dot(self.Q.dot(self.M.T))+iSigmax  # cov of state estimate
                mux=(inv(R).dot(iSigmax).dot(mux) + 
                     inv(R).dot(self.M.T).dot(self.Q).dot(self.z[:,i]))  # corrected state estimate
                correctedmux[:,i]=mux[:]
                
                # Calculate prior on next state.
                mux=self.A.dot(mux)+self.b[:,i]
                iSigmax=(inv(self.A.T).dot(inv( inv(self.A.T.dot(self.L).dot(self.A))+inv(R) )).dot(inv(self.A)))
        else:
            T=self.z.shape[1]
            correctedmux=np.zeros((len(self.x0),T+dt))  # result of prediction once accounting for obs
            correctedmux[:,:dt]=np.tile(self.x0[:,None],(1,dt))
            predictedmux=np.zeros((len(self.x0),T+dt))  # result of prediction once accounting for obs
            predictedmux[:,:dt]=np.tile(self.x0[:,None],(1,dt))
            SigmaX=np.zeros((T+1,len(self.x0),len(self.x0)))
            
            # Calculate KF corrected estimate of state with data.
            SigmaX[0]=self.covx0
            iSigmax=inv(SigmaX[0])
            mux=self.x0
            
            for i in xrange(T):
                R=self.M.T.dot(self.Q.dot(self.M.T))+iSigmax  # cov of state estimate
                mux=(inv(R).dot(iSigmax).dot(predictedmux[:,i]) + 
                     inv(R).dot(self.M.T).dot(self.Q).dot(self.z[:,i]))  # corrected state estimate
                correctedmux[:,i]=mux[:]
                
                # Calculate prior on next state.
                predictedmux[:,i+dt]=self.A.dot(mux)+self.b[:,i]
                iSigmax=(inv(self.A.T).dot(inv( inv(self.A.T.dot(self.L).dot(self.A))+inv(R) )).dot(inv(self.A)))
                SigmaX[i+1]=inv(iSigmax)
        return predictedmux,correctedmux,SigmaX
