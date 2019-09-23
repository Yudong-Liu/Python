"""
This file is used for Interest Rate Swap: data generation and risk measure
We generate zero rate, float payment rate
We calculate the Historical VaR and True VaR
"""
import numpy as np
from numba import jit

class IRS:
    def __init__(self, a, b, sigma, r, corr,freq,maturity):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.freq=freq
        self.maturity=maturity
        self.r = r # initial risk factor values (dtype: pd.Series)
        self.corr = corr
        std = np.sqrt(self.sigma**2 * r * 1.0/252.0)
        self.corr_mat = np.mat(([1.0,self.corr],[self.corr,1.0]))
        self.cov =  np.diag(std).dot(self.corr_mat).dot(np.diag(std))
        del std

    def CIR_zero_bond(self,a,b,sigma,T,t,r,getYield=False):
        """
        Function to get the discounted price of the zero coupon bond
        under CIR model. 
        
        This is a closed-form formula, and is retrieved from
        https://en.wikipedia.org/wiki/Cox%E2%80%93Ingersoll%E2%80%93Ross_model#Bond_pricing
        ------------------
        Parameters:
        factor: factor name
        T, t: time expiration, one year is unit 1
        r: current instantaneous interest rate 
        ------------------
        Return: discounted price of zero coupon bond
        Type: scalar or list depending on type amd size of r
        """

        h=np.sqrt(a**2+ 2 * sigma**2)
        A=(2*h*np.exp((a+h)*(T-t)/2)/(2*h+(a+h)*(np.exp((T-t)*h)-1)))**(2*a*b/sigma**2)
        B=2*(np.exp((T-t)*h)-1)/(2*h+(a+h)*(np.exp((T-t)*h)-1))
        P=A*np.exp(-B*r)
        # all are formulae copied from source. Nothing to say
        if getYield:
            return -np.log(P) / (T-t)
        else:
            return P

    @jit
    def simpath(self,t,N):
        """
        Simulate risk factors in t days for N times under CIR stochastic process.
        -----------------
        Parameters:
        N: no of simulations
        r: spot rates of risk factors. (type: pd.Series, with name of rate as index)
        t: no of days ahead
        ----------------
        Return: N possible spot rates in 5 days (type: np.ndarray, size = N)
        The first column is the zero rate and the second column is the float payment rate.
        """
        t=t-1
        ret=[]
        dt= 1.0 / 252.0 /N # one year is time unit of 1
        ret.append(self.r)
        vol = self.sigma*np.sqrt(ret[0] * dt)
        cov_mat = self.cov
        for i in range(t*N):
            mu = self.a * (self.b - ret[i]) * dt
            drt = np.random.multivariate_normal(mu.tolist(), cov_mat.tolist())        
            ret.append(ret[i]+drt)
            vol = self.sigma*np.sqrt(ret[i] * dt)
            cov_mat = np.diag(vol).dot(self.corr_mat).dot(np.diag(vol))
        return (ret[::N])

    def IRS_FRA(self,float_rate,fixed_rate,discount,principal=100):
        """
        Calculate IRS price using FRA pricing method.
        --------------------
        Parameters:
        float_rate: payment rate from float part; type: float array
        fixed_rate: payment rate from fixed part; type: float array
        r: spot price of instantaneous interest rate; type: float
        discount: discount factors tenor; type: float array
        ------------------
        Return: swap price for the day (type: float)
        """
        CF_fixed=fixed_rate*principal  # list of fixed payments 
        CF_float=float_rate*principal   
        PV_fixed=(CF_fixed*discount).sum(1)
        PV_float=(CF_float*discount).sum(1)
        irs_price=PV_fixed-PV_float # IRS price is difference of summed present value of fix interest and float interest
        return irs_price

    def interpolate(self,curve,t,maturity,freq):
        if len(curve.shape)==1:
            return self.interpolation(curve,t,maturity,freq)
        else:
            return np.apply_along_axis(self.interpolation,1,curve,t,maturity,freq)

    def forward_rate(self,curve,t,maturity,freq):
        times=int(maturity/freq)
        time=(np.repeat(freq,times))*np.array(list(range(1,times+1)))-t
        return np.diff(curve*time)/freq

    def interpolation(self,curve,t,maturity,freq):
    #Y=a+bx+e ~ linear interpolation
        time=np.array([1/12,0.25,0.5,1,2,3,5,7,10,20,30])
        n=len(time)       
        paytimes=int(maturity/freq)
        timeindex=np.array([freq*i-t for i in range(1,paytimes+1)])
        newcurve=np.zeros(paytimes)
        for i in range(0,paytimes):
            #find the location of tenor series [curve[index],curve[index+1])
            index=-1#means t<curve[0]
            for j in range(0,n):
                if timeindex[i]>=time[j] and timeindex[i]<time[j+1]:
                    index=j
                    break
                elif timeindex[i]>=time[-1]:
                    index=n-1
                    break
            if index>=0 & index<n-1:
                x= curve[index]+(timeindex[i]-time[index])*(curve[index+1]-curve[index])/(time[index+1]-time[index])           
                newcurve[i]=x#(np.power(np.exp(x*timeindex[i]*0.01),1.0/timeindex[i])-1)*100
            elif index<0:
                x= curve[0]+(timeindex[i]-time[0])*(curve[0]/time[0])
                newcurve[i]=x#(np.power(np.exp(x*timeindex[i]*0.01),1.0/timeindex[i])-1)*100
            else:
                x= curve[-1]+(timeindex[i]-time[-1])*(curve[-1]/time[-1])
                newcurve[i]=x#(np.power(np.exp(x*timeindex[i]*0.01),1.0/timeindex[i])-1)*100
        return newcurve

    @jit
    def ewma(self,ret,lamb=0.97):
        """
        input the return of the risk factors, output rescale return
        """
        n=len(ret)
        vol=np.zeros(n+1)
        vol[0]=np.sqrt(ret[0]**2)    
        for i in range(1,n+1):
            vol[i]=np.sqrt(lamb*vol[i-1]**2+(1-lamb)*ret[i-1]**2)
        fvol=vol[n]
        nret=ret/vol[:n]*fvol
        #nret=np.apply_along_axis(rolling_sum,0,nret,5)
        return(nret)    

    def scenario_value(self,risk_factor,lamb=0.97):
        """
        Input: risk factors,type float array or matrix
        Output: scenario value of the risk factors by rescaling
        """
        #risk_factor=pd.DataFrame(risk_factor)
        log_rf=np.log(risk_factor)
        #ret=np.apply_along_axis(np.diff,0,log_rf,1)
        ret=log_rf[5:]-log_rf[:-5]
        #ret=np.apply_along_axis(ndiff,0,log_rf,5)###modified mark!!!!
        scen_ret=np.apply_along_axis(self.ewma,0,ret,lamb)
        scenario_risk_factor=risk_factor[-1]*np.exp(scen_ret)
        return (scenario_risk_factor)

    def realized_PnL(self,data):

        payment_times=int(self.maturity/self.freq)
        terms = [1.0/12, 3.0/12, 6.0/12, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
        self.time=np.repeat(self.freq,payment_times)*np.array(list(range(1,payment_times+1)))

        ######discount curve######
        discount_rate=data[:,0]
        self.discount_rate_curve_all=np.array(list(map(lambda r: [self.CIR_zero_bond(self.a[0],self.b[0],self.sigma[0],t,0,r,True) for t in terms],discount_rate)))

        ######float curve######
        float_rate=data[:,1]
        self.float_rate_curve_all=np.array(list(map(lambda r: [self.CIR_zero_bond(self.a[1],self.b[1],self.sigma[1],t,0,r,True) for t in terms],float_rate)))

        ############Realized Pnl and current price############
        ######interpolate discount rate curve######

        discount_factors=np.array(list(map(lambda r: [self.CIR_zero_bond(self.a[0],self.b[0],self.sigma[0],t,0,r) for t in (self.time)],discount_rate)))
        #fixed
        self.fixed_rate=(np.array([0.026 for terms in range(payment_times)]).reshape(1,payment_times))*self.freq

        ######The factors at current day######
        self.latest_pay_rate=self.float_rate_curve_all[:,1]#float_rate_curve_all[:,1] is the 3M libor rate
        current_float_curve=np.array(list(map(lambda r: [self.CIR_zero_bond(self.a[1],self.b[1],self.sigma[1],t,0,r,True) for t in (self.time)],float_rate)))
        current_float_foward=self.forward_rate(current_float_curve,0,self.maturity,self.freq)
        current_float_rate=np.column_stack((self.latest_pay_rate,current_float_foward)) 
        current_float_rate=np.exp(current_float_rate*self.freq)-1
        
        current_discount=np.array(list(map(lambda r: [self.CIR_zero_bond(self.a[0],self.b[0],self.sigma[0],t,0,r) for t in (self.time-5/252)],discount_rate)))   

        ###### Realized Part ( the factors after 5 days from current day) ######
        interpolate_float_curve=np.array(list(map(lambda r: [self.CIR_zero_bond(self.a[1],self.b[1],self.sigma[1],t,0,r,True) for t in (self.time-5/252)],float_rate)))
        forward_float=self.forward_rate(interpolate_float_curve,0,self.maturity,self.freq)
        float_pay_rate=np.column_stack((np.array(np.hstack((np.zeros(5),self.latest_pay_rate[5:]))),forward_float))
        #interpolate_float_curve[:,0]=np.array(np.hstack((np.zeros(5),latest_pay_rate[5:])))
        float_pay_rate=np.exp(float_pay_rate*self.freq)-1
        realized_price=self.IRS_FRA(float_pay_rate,self.fixed_rate,discount_factors)[5:]

        self.current_price=self.IRS_FRA(current_float_rate,self.fixed_rate,current_discount) [:-5]

        self.realized_PnL=realized_price-self.current_price      
        return(self.realized_PnL)

    def HVaR(self,win=1500,end_index=2515,throw=200):
         
        payment_times=int(self.maturity/self.freq)
        pnl_HVaR=list()
        HVaR=np.zeros(end_index)
        for i in range(win,end_index):
            discount_rate_curve=self.discount_rate_curve_all[(i-win):i+1]
            float_rate_curve=self.float_rate_curve_all[(i-win):i+1]
            ######scenario factors######
            
            ######interpolate discount rate curve######
            scenario_discount_rate=self.scenario_value(discount_rate_curve,lamb=0.97)[throw:]
            interpolate_discount_curve=self.interpolate(scenario_discount_rate,5/252.0,self.maturity,self.freq)
            time=np.repeat(self.freq,payment_times)*np.array(list(range(1,payment_times+1)))
            discount_factors=np.exp(-interpolate_discount_curve*(time-5/252.0))    
         
            ######interpolate float rate curve######
            scenario_float_rate=self.scenario_value(float_rate_curve,lamb=0.97)[throw:]
            
            float_pay_rate=self.interpolate(scenario_float_rate,5/252.0,self.maturity,self.freq)         
            
            forward_float=self.forward_rate(float_pay_rate,5/252.0,self.maturity,self.freq)
            float_pay_rate=np.column_stack((np.repeat(self.latest_pay_rate[i],len(float_pay_rate)),forward_float))
            float_pay_rate=np.exp(float_pay_rate*self.freq)-1
            ###########################
            
            ######Historical VaR######    
            scenario_price=self.IRS_FRA(float_pay_rate,self.fixed_rate,discount_factors)
            pnl_HVaR.append(list(scenario_price-self.current_price[i])) 
            HVaR[i]=np.percentile(np.array(pnl_HVaR[i-win]),1)
            #print('HVaR: Day ',i,' finished')
        print('HVaR Calculation finished')
        return(HVaR[win:end_index])

    @jit
    def simulation(self,r,t):
        """
        Risk factors simulation for true VaR calculation
        Using known parameters to simulate all possible risk factors 5 days later as the analytical results
        """
        cov_mat=self.cov.copy()
        dt= 1.0 / 252 # one year is time unit of 1
        ret=r
        vol = self.sigma*np.sqrt(ret * dt)
        cov_mat = np.diag(vol).dot(self.corr_mat).dot(np.diag(vol))
        for i in range(t):
            mu = self.a * (self.b - ret) * dt
            drt = np.random.multivariate_normal(mu.tolist(), cov_mat.tolist())        
            ret=ret+drt
            vol = self.sigma*np.sqrt(ret * dt)
            cov_mat = np.diag(vol).dot(self.corr_mat).dot(np.diag(vol))
        return (ret)

    def true_VaR(self,data,win=1500,end_index=2515,N=10000):
        pnl_trueVaR=list()
        tVaR=np.zeros(end_index)

        for i in range(win,end_index):
            f=lambda r: [self.simulation(r,5) for _ in range(N)]
            np.random.seed(i)#keep specific seeds so that we can replicate the results
            sim_rate=np.array(f(data[i,:]))
            true_discount_rate=sim_rate[:,0]
            true_float_rate=sim_rate[:,1]
            
            discount_factors=np.array(list(map(lambda r: [self.CIR_zero_bond(self.a[0],self.b[0],self.sigma[0],t-5/252.0,0,r) for t in self.time],true_discount_rate)))   
            float_rate_curve=np.array(list(map(lambda r: [self.CIR_zero_bond(self.a[1],self.b[1],self.sigma[1],t-5/252.0,0,r,getYield=True) for t in self.time],true_float_rate)))
            ######scenario factors######
            
            ######interpolate float rate curve######
            forward_float=self.forward_rate(float_rate_curve,5.0/252,self.maturity,self.freq)
            float_pay_rate=np.column_stack((np.repeat(self.latest_pay_rate[i],N),forward_float))
            float_pay_rate=np.exp(float_pay_rate*self.freq)-1
            ###########################
                
            scenario_price=self.IRS_FRA(float_pay_rate,self.fixed_rate,discount_factors)
            pnl_trueVaR.append(list(scenario_price-self.current_price[i]))
            tVaR[i]=np.percentile(np.array(pnl_trueVaR[i-win]),1)
        print('True VaR Calculation finished')
        return(tVaR[win:end_index])

































