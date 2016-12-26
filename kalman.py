# An implementation of the Kalman filter.
import numpy as np
from numpy.linalg import inv

class Kalman(object):
    def __init__(self,x0,covx0,z,covz,A,b,covupdate,M,dt=1):
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
        dt (float=1)
            Time scale for observations. This is important in unscented filter where we have to make local
            derivative estimates.
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
        self.dt=dt

        self.ndim=len(x0)
        # Should make sure that Qij=0 and Qii=1 for unobserved states.

    def filter(self,delay=0):
        """
        Standard Kalman filter. If delay>0, predict into the future. Store list of expected mu0 in correctedmux.
        Update estimate of state in the past using the correction step in the KF and use that to predict delay
        into the future.
        2016-12-23

        Params:
        -------
        delay (int=0)
            
        Values:
        -------
        predictedmux
        correctedmux 2016-12-23
        """
        if delay==0:
            T=self.z.shape[1]
            correctedmux=np.zeros((len(self.x0),T))  # result of prediction once accounting for obs
            predictedmux=np.zeros((self.ndim,T+1))
            SigmaX=np.zeros((T+1,self.ndim,self.ndim))

            # Calculate KF corrected estimate of state with data.
            SigmaX[0]=self.covx0
            predictedmux[:,0]=self.x0
            iSigmax=inv(SigmaX[0])
            
            for i in xrange(T):
                R=self.M.T.dot(self.Q.dot(self.M.T))+iSigmax  # cov of state estimate
                correctedmux[:,i]=(inv(R).dot(iSigmax).dot(predictedmux[:,i]) + 
                     inv(R).dot(self.M.T).dot(self.Q).dot(self.z[:,i]))  # corrected state estimate
                
                # Calculate prior on next state.
                predictedmux[:,i+1]=self.A.dot(correctedmux[:,i]) + self.b[:,i]
                iSigmax=(inv(self.A.T).dot(inv( inv(self.A.T.dot(self.L).dot(self.A))+inv(R) )).dot(inv(self.A)))
                SigmaX[i+1]=inv(iSigmax)
        else:
            T=self.z.shape[1]
            correctedmux=np.zeros((len(self.x0),T+delay))  # result of prediction once accounting for obs
            correctedmux[:,:delay]=np.tile(self.x0[:,None],(1,delay))
            predictedmux=np.zeros((len(self.x0),T+delay+1))  # result of prediction once accounting for obs
            predictedmux[:,:delay]=np.tile(self.x0[:,None],(1,delay))
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
                predictedmux[:,i+delay+1]=self.A.dot(mux)+self.b[:,i]
                iSigmax=(inv(self.A.T).dot(inv( inv(self.A.T.dot(self.L).dot(self.A))+inv(R) )).dot(inv(self.A)))
                SigmaX[i+1]=inv(iSigmax)
        return predictedmux,correctedmux,SigmaX

    def unscented_filter(self,delay=0):
        """
        Kalman filter using local Jacobian approximation to model nonlinear dynamics.
        2016-12-26

        Params:
        -------
        delay (int=0)
            
        Values:
        -------
        predictedmux
        correctedmux 2016-12-23
        """
        if delay==0:
            T=self.z.shape[1]
            correctedmux=np.zeros((self.ndim,T))  # result of prediction once accounting for obs
            predictedmux=np.zeros((self.ndim,T))
            SigmaX=np.zeros((T+1,self.ndim,self.ndim))
            
            # Calculate KF corrected estimate of state with data.
            SigmaX[0]=self.covx0
            predictedmux[:,0]=self.x0
            
            for i in xrange(T):
                iSigmax=inv(SigmaX[i])
                R=self.M.T.dot(self.Q.dot(self.M.T))+iSigmax  # cov of state estimate
                mux=(inv(R).dot(iSigmax).dot(predictedmux[:,i]) + 
                     inv(R).dot(self.M.T).dot(self.Q).dot(self.z[:,i]))  # corrected state estimate
                correctedmux[:,i]=mux[:]
                
                # Calculate prior on next state.
                spline=self.der(correctedmux[:,i-3:i+1])*self.dt
                predictedmux[:,i]=self.A.dot(mux) + self.b[:,i] + spline
                iSigmax=(inv(self.A.T).dot(inv( inv(self.A.T.dot(self.L).dot(self.A))+inv(R) )).dot(inv(self.A)))
                SigmaX[i+1]=inv(iSigmax)
        else:
            T=self.z.shape[1]
            correctedmux=np.zeros((len(self.x0),T+delay))  # result of prediction once accounting for obs
            correctedmux[:,:delay]=np.tile(self.x0[:,None],(1,delay))
            predictedmux=np.zeros((len(self.x0),T+delay))  # result of prediction once accounting for obs
            predictedmux[:,:delay]=np.tile(self.x0[:,None],(1,delay))
            SigmaX=np.zeros((T+1,self.ndim,self.ndim))
            
            # Calculate KF corrected estimate of state with data.
            SigmaX[0]=self.covx0
            iSigmax=inv(SigmaX[0])
            predictedmux[:,0]=self.x0
            
            for i in xrange(T):
                R=self.M.T.dot(self.Q.dot(self.M.T))+iSigmax  # cov of state estimate
                mux=(inv(R).dot(iSigmax).dot(predictedmux[:,i]) + 
                     inv(R).dot(self.M.T).dot(self.Q).dot(self.z[:,i]))  # corrected state estimate
                correctedmux[:,i]=mux[:]
                
                # Calculate prior on next state.
                predictedmux[:,i+delay]=self.A.dot(mux)+self.b[:,i]
                iSigmax=(inv(self.A.T).dot(inv( inv(self.A.T.dot(self.L).dot(self.A))+inv(R) )).dot(inv(self.A)))
                SigmaX[i+1]=inv(iSigmax)
        return predictedmux,correctedmux,SigmaX

    def der(self,x,n=1):
        if x.shape[1]<3:
            return 0.
        return [(x[:,-1]-x[:,-3])/(2*self.dt),
               ][n-1]
