# The following code is partly copied, extended on build on https://github.com/lawrennd/mlai/
# based on commit: bb4b776a21ec17001bf20ee6406324217f902944
# expand it to the different basis funcitons in the source. 
from math import gamma
import numpy as np
from src.nullspace import nullspace_correction
from src.nullspace import plot_nullspace_correction

class Basis():
    """Basis function class
    """
    def __init__(self, function, number, **kwargs):
        self.arguments=kwargs
        self.number=number
        self.function=function
        self.Phi_vals = None
        self.X = None
        self.x = None

    def Phi(self, x):
        """Create basis vector phi
        
        Parameters
        ----------
        x : ndarray 
            1D array of x values where the function should be evaluated
        """
        self.Phi_vals = self.function(x, num_basis=self.number, **self.arguments)
        self.x = x
        return self

    def construct_X_data(self, basis_weights):
        """Constructs a data matrix based on

        Parameters
        ----------
        basis_weights: ndarray
            2D array of rows equal to desired observations in the data matrix, cols equal num_basis
        """
        if self.number != basis_weights.shape[1]:
            raise ValueError("Number of basis weights per observation must equal the number defined for this object!")
        self.X = np.zeros((basis_weights.shape[0], self.Phi_vals.shape[0]))
        for i in range(basis_weights.shape[0]):
            # Itereate trhough the rows. 
            self.X[i, :] = np.dot(self.Phi_vals, basis_weights[i, :].T)
        return self

    def add_wgn(self, snr):
        """Add white Gaussian noise to measurements

        Parameters
        ----------
        snr: float
            signal to noise ration for gaussion noise that's added to the data 
        """
        # Add Gaussian noise to the measurements
        # Snippet below partly copied/adapted/inspired by: 
        # https://stackoverflow.com/questions/14058340/adding-noise-to-a-signal-in-python
        # Answer from Noel Evans, accessed: 18.05.2022, 15:37 CET
        # Calculate signal power and convert to dB 
        rows, columns = self.X.shape
        for i in range(rows):
            row_i = self.X[i, :]
            sig_avg_watts = np.mean(row_i**2)
            sig_avg_db = 10 * np.log10(sig_avg_watts)
            # Calculate noise according to [2] then convert to watts
            noise_avg_db = sig_avg_db - snr
            noise_avg_watts = 10 ** (noise_avg_db / 10)
            # Generate an sample of white noise
            mean_noise = 0
            noise = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), columns)
            # Noise up the original signal
            self.X[i, :] += noise
        return self
        
class SynMLData(Basis):
    """Synthethic Machine Learning Data class. this class inherits from basis (takes Phi and X from basis)
    This class can generate response variables (y).
    """
    def __init__(self, function, number, **kwargs):
        super().__init__(function, number, **kwargs)
        self.y = None       # Repsonse variable
        self.y_ = None
        self.X_std = None
        self.std = None
        self.weights = {}                 # Dictionary to store weights learned from mean subtracted data X_
        # self.weights_std = {}           # Dictionary to store weights learned from mean subtracted standardized data X_std
        # self.weights_std_retrans = {}   # Dictionary to store weights learned from mean subtracted standardized data X_std rescaled to match X_
        self.nullsp = {}

    def place_X_y(self, X, x, y):
        """Insert data via this class inc ase you don't want to use data generation via basis classes

        Parameters
        ----------
        X : ndarray
            2D numpy array of data 
        x : ndarray
            1D numpy array, representing the domain values small x corresponding to each column in X
        y : ndarray
            1D array of responses
        """
        self.X = X
        self.X_ = self.X - np.mean(self.X, axis=0)
        self.x = x
        self.y = y
        self.y_ = self.y - np.mean(self.y)
        return self

    def Phi(self, x):
        super().Phi(x)
        return self
    
    def construct_X_data(self, basis_weights):
        super().construct_X_data(basis_weights)
        return self

    def construct_y_data(self, response_trans):
        """Construct responsese
                
        Parameters
        ----------
        response_trans : callable
            function that transforms X to y
        """
        self.y = response_trans(self.X)
        return self
    
    def learn_weights(self, model, string_id):
        """Learn weights from data and sotre them in the dictionary. 

        Parameters
        ----------
        model : object
            sklearn model object with methods fit(), predict() (e.g. sklearn)
            for custom models you might want to write a wrapper
        string_id : str
            model description string
        """
        # We manually mean center the data here for this purpose
        self.X_ = self.X - np.mean(self.X, axis=0)
        self.y_ = self.y-self.y.mean()

        self.std = np.std(self.X_, axis=0)
        self.X_std = self.X_ / self.std

        reg = model.fit(self.X_, self.y_)
        coef = reg.coef_.reshape(-1)
        self.weights[string_id] = coef
        
        # For some synthethic data std might be 0 -> Standardization not possible!
        try:
            reg_std = model.fit(self.X_std, self.y_)
            coef_std = reg_std.coef_.reshape(-1)
            self.weights[string_id + ' std'] = coef_std
            self.weights[string_id + ' std retrans'] = coef_std/self.std
        except:
            self.weights[string_id + ' std'] = 'Undefined Std = 0 for some column'
            self.weights[string_id + ' std retrans'] = 'Undefined Std = 0 ofr some column'
        return self
    
    def nullspace_correction(
        self, w_alpha=None, w_alpha_name=None, w_beta=None, w_beta_name=None, std=False, 
        plot_results=False, save_plot=False, path_save='', file_name='', **kwargs):
        """Function that calls 'nullspace_correction allowing to shorten syntax and use SynMLData class.

        Parameters
        ----------
        w_alpha : ndarray
            Regression coefficients obtained with method alpha
        w_beta : ndarray
            Regression coefficients obtained with method beta
        data_ml_obj : obj 
            Object from SynMLData class
        std : bool, default=False
            Indicate whether Standardization(Z-Scoring) shall be performed. 
            If yes, use X_std of the data_ml_object, if not use X.
        plot_results : bool, default=False
            Indicate whether to plot results
        save_results : bool, default=False
            indicate whether to save plot as pdf
        path_save : str, default=''
            path for the case save resutls is true
        fig_name : str, defautl=''
            figure file name

        Returns
        -------
        self

        depending on whether plot_results is true
        fig : object
            matplotlib figure object
        ax : object 
            matplotlib axis objects
        """

        if (w_alpha is None) & ('key_alpha' not in kwargs):
            NameError('Either w_alpha is passed directly or key for learned coefficients must be passed')

        if (w_beta is None) & ('key_beta' not in kwargs):
            NameError('Either w_alpha is passed directly or key for learned coefficients must be passed')
    
        self.nullsp['w_beta_name'] = w_beta_name
        self.nullsp['w_alpha_name'] = w_alpha_name

        # To keep things simple, standardized weights also included here! 
        # In case std==False this is not efficient, because the standardized coef. aren't used. 
        # But it's easier to understand and less verbose. 
        if w_alpha is None:
            self.nullsp['w_alpha'] = self.weights[kwargs.get('key_alpha')]
            self.nullsp['w_alpha_std'] = self.weights[kwargs.get('key_alpha') + ' std']
        else:
            self.nullsp['w_alpha'] = w_alpha
            self.nullsp['w_alpha_std'] = w_alpha * self.std

        if w_beta is None:
            self.nullsp['w_beta'] = self.weights[kwargs.get('key_beta')]
            self.nullsp['w_beta_std'] = self.weights[kwargs.get('key_beta') + ' std']
        else:
            self.nullsp['w_beta'] = w_beta
            self.nullsp['w_beta_std'] = w_beta * self.std

        if std: 
            X = self.X_std
            self.nullsp['w_alpha_name'] += ' std'
            self.nullsp['w_beta_name'] += ' std'
            key_alpha = 'w_alpha_std'
            key_beta = 'w_beta_std'
        else:
            X = self.X_
            key_alpha = 'w_alpha'
            key_beta = 'w_beta'
            
        x = self.x
        self.nullsp['info'] = ''

        if 'nb_gammas' in kwargs:
            nb_gammas = kwargs.get('nb_gammas')
        else:
            nb_gammas = 30

        y_ = self.y_

        # Simple approach to set gamma
        y_range = np.max(y_) - np.min(y_)
        min_exp = -5
        max_exp = np.floor(np.log10(int((10**2)*y_range)))

        gamma_vals = np.geomspace(10**(-5), (10**2)*y_range, nb_gammas)
        # gamma_vals  = np.logspace(min_exp, max_exp, nb_gammas)
        # gamma_vals = np.append(gamma_vals, [int((10**2)*y_range)])

        self.nullsp['v'], self.nullsp['v_'], self.nullsp['norm_'], self.nullsp['gamma'] = nullspace_correction(
            self.nullsp[key_alpha], self.nullsp[key_beta], X, x, gs=gamma_vals, comp_block=0, snig=0)
        
        if plot_results:
            fig, ax = self.plot_nullspace_correction(std=std)
            if save_plot:
                fig.savefig(path_save + file_name)
            return self, fig, ax 
        else:
            return self

    def plot_nullspace_correction(self, std=False, title=''):
        """Function that calls plot_nullspace_correction, uses basis object.

        Parameters
        ----------
        title : str 
            Suptitle of the resulting figure
        std : bool, default=False
            Indicate whether Standardization(Z-Scoring) shall be performed. 
            If yes, use X_std of the data_ml_object, if not use X.

        Returns
        -------
        fig : object
            matplotlib figure object
        ax : object 
            matplotlib axis objects
        """
        if std: 
            X = self.X_std
            w_alpha = self.nullsp['w_alpha_std']
            w_beta = self.nullsp['w_beta_std']
        else: 
            X = self.X_
            w_alpha = self.nullsp['w_alpha']
            w_beta = self.nullsp['w_beta']

        fig, ax = plot_nullspace_correction(
                w_alpha, w_beta, self.nullsp['v_'], self.nullsp['gamma'],
                X, self.x, name=title, coef_name_alpha=self.nullsp['w_alpha_name'], coef_name_beta=self.nullsp['w_beta_name'])
        return fig, ax
    

# Differen basis for functions.e
def polynomial(x, num_basis=4, data_limits=[-1., 1.]):
    """Polynomial basis"""
    centre = data_limits[0]/2. + data_limits[1]/2.
    span = data_limits[1] - data_limits[0]
    z = np.asarray(x, dtype=float) - centre
    z = 2*z/span
    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i:i+1] = z**i
    return Phi

def radial(x, num_basis=4, data_limits=[-1., 1.], width=None):
    """Radial basis constructed using exponentiated quadratic form."""
    if num_basis>1:
        centres=np.linspace(data_limits[0], data_limits[1], num_basis)
        if width is None:
            width = (centres[1]-centres[0])/2.
    else:
        centres = np.asarray([data_limits[0]/2. + data_limits[1]/2.])
        if width is None:
            width = (data_limits[1]-data_limits[0])/2.
    
    Phi = np.zeros((x.shape[0], num_basis))
    for i in range(num_basis):
        Phi[:, i:i+1] = np.exp(-0.5*((np.asarray(x, dtype=float)-centres[i])/width)**2)
    return Phi

def fourier(x, num_basis=4, data_limits=[-1., 1.]):
    """Fourier basis"""
    tau = 2*np.pi
    span = float(data_limits[1]-data_limits[0])
    Phi = np.ones((x.shape[0], num_basis))
    for i in range(1, num_basis):
        if i % 2:
            Phi[:, i:i+1] = np.sin((i+1)*tau*np.asarray(x, dtype=float))
        else:
            Phi[:, i:i+1] = np.cos((i+1)*tau*np.asarray(x, dtype=float))
    return Phi

def relu(x, num_basis=4, data_limits=[-1., 1.], gain=None):
    """Rectified linear units basis"""
    if num_basis>2:
        centres=np.linspace(data_limits[0], data_limits[1], num_basis)[:-1]
    elif num_basis==2:
        centres = np.asarray([data_limits[0]/2. + data_limits[1]/2.])
    else:
        centres = []
    if num_basis < 3:
        basis_gap = (data_limits[1]-data_limits[0])
    else:
        basis_gap = (data_limits[1]-data_limits[0])/(num_basis-2)
    if gain is None:
        gain = np.ones(num_basis-1)/basis_gap
    Phi = np.zeros((x.shape[0], num_basis))
    # Create the bias
    Phi[:, 0] = 1.0
    for i in range(1, num_basis):
        Phi[:, i:i+1] = gain[i-1]*(np.asarray(x, dtype=float)>centres[i-1])*(np.asarray(x, dtype=float)-centres[i-1])
    return Phi

def hyperbolic_tangent(x, num_basis=4, data_limits=[-1., 1.], gain=None):
    """Hyperbolic tangents"""
    if num_basis>2:
        centres=np.linspace(data_limits[0], data_limits[1], num_basis-1)
        width = (centres[1]-centres[0])/2.
    elif num_basis==2:
        centres = np.asarray([data_limits[0]/2. + data_limits[1]/2.])
        width = (data_limits[1]-data_limits[0])/2.
    else:
        centres = []
        width = None
    if gain is None and width is not None:
        gain = np.ones(num_basis-1)/width
    Phi = np.zeros((x.shape[0], num_basis))
    # Create the bias
    Phi[:, 0] = 1.0
    for i in range(1, num_basis):
        Phi[:, i:i+1] = np.tanh(gain[i-1]*(np.asarray(x, dtype=float)-centres[i-1]))
    return Phi