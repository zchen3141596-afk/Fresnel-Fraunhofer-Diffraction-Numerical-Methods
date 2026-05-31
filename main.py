# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 00:01:33 2025

@author: Chen Zhouhan 

Purpose: 
    Compute Fresnel and Fraunhofer diffraction patterns for rectangular and 
    circular apertures using numerical methods and analyse the limitations of 
    the methods used. 
    
Methods: 
    - Interpreting the Fresnel complex integral of the electric field of 
      diffraction in real and imaginary parts. 
    - Solving the integral directly using dblquad integration of real and 
      imaginary parts separetely and then computing the intensity function. 
    - Estimating the integral using Monte Carlo method for circular apertures. 
    - Error propagation for intensity in every method. 
Notes: 
    The present programme is used to analyse Fresnel (near-field) and 
    Fraunhofer (far-field) diffraction patterns and to evaluate convergence 
    properties of dblquad and Monte Carlo integration methods. 
""" 

#============================================================================== 
#########################   Library importation   ############################# 
#============================================================================== 

import numpy as np 
import matplotlib.pyplot as plt 
from scipy.integrate import dblquad 
from time import time 
from numba import njit 

#NOTE: 
    #This programme uses Numba's Just-In-Time compilation (@njit) to 
    #accelerate Fresnel intensity computation functions. 
    
    #To use this library correctly, please first install the package by 
    #executing "conda install numba" in Anaconda Prompt. 
    
    #If Numba is not available, the code will still run correctly without 
    #(@njit). Simply remove or comment out. 



#============================================================================== 
#####################   Fundamental physical constants   ###################### 
#============================================================================== 

c = 3.00e8  #m s^-1, speed of light in the free space 
epsilon_0 = 8.854187812e-12  #A^2 s^4 kg^-1 m^-3, permittivity of free space 



#============================================================================== 
#########################   Function definition   #############################
#============================================================================== 

#------------------------------------------------------------------------------ 
#Parameter values to set up the menu 
#------------------------------------------------------------------------------ 

#General parameter values choosing 
def general_parameters (_lambda, _z): 
    """
    Function to enter the general parameters of the simulation. 

    Parameters
    ----------
    _lambda : floating point, wavelength of wave source (m). 
    _k : floating point, wavenumber of wave source (m). 
    _z : floating point, the distance between aperture and screen (m). 
    ------- 
    Returns: the floating points _wavelength, _k and _screen_distance entered by the user. 
    """
    _lambda = float(input("Enter the wavelength of the wave source (suggested range: 0.5e-6m - 1.0e-6m): ")) 
    _k = 2 * np.pi / _lambda  #Compute the wavenumber 
    _z = float(input("Enter the distance of the aperture to the screen (suggested values: 0.005m for Fresnel or 0.05m for Fraunhofer): ")) 
    
    return _lambda, _k, _z 


#Rectangular parameter values choosing 
def rectangular_parameters (_xp, _yp): 
    """
    Function to enter and calculate the parameters of the rectangular aperture. 

    Parameters
    ----------
    _xp : floating point, the absolute distance (end-to-end) of the x-direction width. 
    _yp : floating point, the absolute distance (end-to-end) of the y-direction width. 
    ------- 
    Returns: the x-limits and y-limits of the aperture - i.e. the integration limits of the rectangular aperture. 
    """
    _xp = float(input("Enter the aperture width in the x-direction (suggested values: 2.0e-4m for Fresnel or 2.0e-5m for Fraunhofer): ")) 
    _xpmin = - _xp / 2 
    _xpmax = _xp / 2
    
    _yp = float(input("Enter the aperture width in the y-direction (suggested values: 2.0e-4m for Fresnel or 2.0e-5m for Fraunhofer): ")) 
    _ypmin = - _yp / 2 
    _ypmax = _yp / 2 
    
    return _xpmin, _xpmax, _ypmin, _ypmax 


#Circular parameter values choosing 
def circular_parameters(_R): 
    """
    Function to enter and calculate the parameters of the circular aperture 

    Parameters
    ----------
    _R : floating point, the radius of the circular aperture chosen by the user. 
    ------- 
    Returns: floating point, the radius of the circular aperture. 
    """ 
    _R = float(input("Enter the radius of the circular aperture (suggested values: 1.0e-4m for Fresnel or 1.0e-5m for Fraunhofer): ")) 
    
    _xpmin = _ypmin = - _R 
    _xpmax = _ypmax = _R 
    
    return _R, _xpmin, _xpmax, _ypmin, _ypmax 


#------------------------------------------------------------------------------ 
#Fresnel diffraction calculation with different methods 
#------------------------------------------------------------------------------ 

#Real part of electric field of Fresnel diffraction 
@njit 
def Fresnel_2D_real (_yp, _xp, _y, _x, _k, _z): 
    """
    Function to compute the real part of the electric field of Fresnel diffraction in 2D. 

    Parameters
    ----------
    _yp : y-coordinates of a point in the aperture. 
    _xp : x-coordinates of a point in the aperture. 
    _y : independent variable, y-coordinate of a point on the screen. 
    _x : independent variable, x-coordinate of a point on the screen. 
    _k : floating point, wavenumber of the function. 
    _z : floating point, distance between aperture and screen. 
    -------
    Returns: real part of the electric field function of Fresnel diffraction in 2D. 
    """ 
    real_E = np.cos(_k / (2 * _z) * ((_x - _xp)**2 + (_y - _yp)**2)) 
    
    return real_E 


#Imaginary part of electric field of Fresnel diffraction 
@njit 
def Fresnel_2D_imag (_yp, _xp, _y, _x, _k, _z): 
    """
    Function to compute the imaginary part of the electric field of Fresnel diffraction. 
    
    Parameters
    ----------
    _yp : y-coordinates of a point in the aperture. 
    _xp : x-coordinate of a point in the aperture. 
    _y : independent variable, y-coordinate of a point on the screen. 
    _x : independent variable, x-coordinate of a point on the screen. 
    _k : floating point, wavenumber of the function. 
    _z : floating point, distance between aperture and screen. 
    -------
    Returns: imaginary part of the electric field function of Fresnel diffraction. 
    """ 
    imag_E = np.sin(_k / (2 * _z) * ((_x - _xp)**2 + (_y - _yp)**2)) 
    
    return imag_E 


#Intensity of Fresnel diffraction pattern 
def Fresnel_2D_intensity (_xpmin, _xpmax, _ypmin, _ypmax, _y, _x, _k, _z, _epsabs, _epsrel): 
    """
    Function to compute the intensity of Fresnel diffraction in 2D and its errors. 

    Parameters
    ----------
    _xpmin : floating point, lower limit of integration in the x-coordinate. 
    _xpmax : floating point, upper limit of integration in the x-coordinate. 
    _ypmin : floating point, lower limit of integration in the y-coordinate. 
    _ypmax : floating point, upper limit of integration in the y-coordinate. 
    _y : independent variable, y-coordinate of a point on the screen. 
    _x : independent variable, x-coordinate of a point on the screen. 
    _k : floating point, wavenumber of the function (m^-1). 
    _z : floating point, distance between aperture and screen (m). 
    _c : floating point, the speed of light in vacuum (m s^(-1)). 
    _epsilon_0 : floating point, the permittivity of free space (m kg s^-2 A^-2). 
    ------- 
    Returns: intensity of Fresnel diffraction pattern with respective errors. 
    """
    Fresnel_real, real_error = dblquad(Fresnel_2D_real, _xpmin, _xpmax, _ypmin, _ypmax, args = (_y,_x,_k,_z), epsabs = _epsabs, epsrel = _epsrel) 
    Fresnel_imag, imag_error = dblquad(Fresnel_2D_imag, _xpmin, _xpmax, _ypmin, _ypmax, args = (_y,_x,_k,_z), epsabs = _epsabs, epsrel = _epsrel) 
    constant = _k / (2 * np.pi * _z) 
    Fresnel_E = constant * (Fresnel_real + 1.0j * Fresnel_imag)
    Fresnel_I = epsilon_0 * c * np.abs(Fresnel_E)**2
    error_I = epsilon_0 * c * np.sqrt(((2 * Fresnel_real * real_error)**2 + (2 * Fresnel_imag * imag_error)**2))
    
    return Fresnel_I, error_I 


#Solving Fresnel diffraction with Monte Carlo method 
@njit 
def Fresnel_2D_montecarlo (_y, _x, _k, _z, _R, _N): 
    """
    Function to compute the Monte Carlo method. 

    Parameters
    ----------
    _y : independent variable, x-coordinate of a point on the screen. 
    _x : independent variable, y-coordinate of a point on the screen. 
    _k : floating point, wavenumber of the wave source (m^-1). 
    _z : floating point, distance between aperture and screen (m). 
    _R : floating point, radius of the circular aperture (m). 
    _N : integer point, sampling count of Monte Carlo method. 
    ------- 
    Returns: intensity of diffraction pattern with respective errors by Monte Carlo. 
    """ 
    _xp = np.random.uniform(-_R, _R, _N) 
    _yp = np.random.uniform(-_R, _R, _N) 
    
    #Establishing the circular integration limits 
    circle = _xp**2 + _yp**2 <= _R**2 
    _xp = _xp[circle] 
    _yp = _yp[circle] 
    
    area = np.pi * _R**2  #Area of circle 
    
    #Real part calculation with respective errors 
    real_values = Fresnel_2D_real(_yp, _xp, _y, _x, _k, _z) 
    real_mean = np.mean(real_values) 
    real_mean_sq = np.mean(real_values**2) 
    E_real = area * real_mean 
    E_real_error = area * np.sqrt(abs(real_mean_sq - real_mean**2) / _N) 
    
    #Imaginary part calculation with respective errors 
    imag_values = Fresnel_2D_imag(_yp, _xp, _y, _x, _k, _z) 
    imag_mean = np.mean(imag_values) 
    imag_mean_sq = np.mean(imag_values**2) 
    E_imag = area * imag_mean 
    E_imag_error = area * np.sqrt(abs(imag_mean_sq - imag_mean**2) / _N) 
    
    intensity_montecarlo = epsilon_0 * c * (E_real**2 + E_imag**2) 
    error_montecarlo = epsilon_0 * c * np.sqrt(((2 * E_real * E_real_error)**2 + (2 * E_imag * E_imag_error)**2)) 
    
    return intensity_montecarlo, error_montecarlo 


#------------------------------------------------------------------------------ 
#Integration limits of the circular aperture 
#------------------------------------------------------------------------------ 

#Lower limit of integration in the y-coordinates 
def yp1func (_xp): 
    """ 
    Function to compute the y-coordinates lower integration limit in a circle of radius R. 

    Parameters
    ---------- 
    _xp : indepedent variable, x-coordinate of a point on the screen. 
    ------- 
    Returns: lower integration limit in a circle of radius R. 
    """ 
    return (-np.sqrt(radius**2 - _xp**2)) 


#Upper limit of integration in the y-coordinates 
def yp2func (_xp): 
    """
    Function to compute the y-coordinates upper integration limit in a circle of radius R. 

    Parameters
    ---------- 
    _xp : independent variable, x-coordinate of a point on the screen. 
    ------- 
    Returns: upper integration limit in a circle of radius R. 
    """ 
    return (np.sqrt(radius**2 - _xp**2)) 


#------------------------------------------------------------------------------ 
#Screen coordinates calculation 
#------------------------------------------------------------------------------ 

def coordinates_calculation (_lambda, _xp, _yp, _z, _numpoints): 
    """
    Function to compute the x-y coordinates of the screen with Fresnel number. 

    Parameters
    ----------
    _lambda : floating point, wavelength of the wave source (m). 
    _xp : floating point, x-coordinate of the aperture width halved (m). 
    _yp : floating point, y-coordinate of the aperture width halved (m). 
    _z : floating point, distance between aperture and screen (m). 
    _numpoints : integer point, number of points to compute. 
    -------
    Returns: _xvals : NumPy array, interval of x-coordinates of the screen (m). 
             _yvals : NumPy array, interval of y-coordinates of the screen (m). 
    """ 
    #Fresnel number calculation 
    F = (_xp * _yp) / (_lambda * _z) 
    
    #Screen limits for a Fresnel diffraction 
    if F >= 1: 
        _xvals = np.linspace(-0.0002, 0.0002, _numpoints) 
        _yvals = np.linspace(-0.0002, 0.0002, _numpoints) 
        
    #Screen limits for a Fraunhofer diffraction 
    elif F < 1: 
        _xvals = np.linspace(-0.005, 0.005, _numpoints) 
        _yvals = np.linspace(-0.005, 0.005, _numpoints) 
    
    return _xvals, _yvals 


#------------------------------------------------------------------------------ 
#Convergence test 
#------------------------------------------------------------------------------ 

#Convergence test for dblquad method 
def convergence_test_dblquad (_xpmin, _xpmax, _ypmin, _ypmax, _x, _y, _k, _z, _tolerances): 
    """
    Function to compute the convergence test for dblquad method. 

    Parameters
    ---------- 
    _xpmin : floating point, lower limit of integration in the x-coordinate. 
    _xpmax : floating point, upper limit of integration in the x-coordinate. 
    _ypmin : floating point, lower limit of integration in the y-coordinate. 
    _ypmax : floating point, upper limit of integration in the y-coordinate. 
    _yp : y-coordinates of a point in the aperture. 
    _xp : x-coordinate of a point in the aperture. 
    _y : independent variable, y-coordinate of a point on the screen. 
    _x : independent variable, x-coordinate of a point on the screen. 
    _k : floating point, wavenumber of the function. 
    _z : floating point, distance between aperture and screen. 
    ------- 
    Returns: numerical error for different tolerances in dblquad method. 
    """ 
    reference_I = np.zeros(len(_x)) 
    
    #Loop to compute intensity at every point on screen 
    for _i in range(len(_x)): 
        ref_I, _ = Fresnel_2D_intensity(_xpmin, _xpmax, _ypmin, _ypmax, _y, _x[_i], _k, _z, _epsabs = 1.0e-14, _epsrel = 1.0e-14) 
        reference_I [_i] = ref_I 
        
    max_errors = np.zeros(len(_tolerances)) 
    
    #First loop to compute intensity for each tolerance with index _j 
    for _j, _tol in enumerate(_tolerances): 
        tolerance_I = np.zeros(len(_x)) 
        #Second loop to compute intensity at every point on screen. 
        for _i in range (len(_x)): 
            tolerance_I [_i], _ = Fresnel_2D_intensity(_xpmin, _xpmax, _ypmin, _ypmax, _y, _x[_i], _k, _z, _epsabs = _tol, _epsrel = _tol) 
        max_errors [_j] = np.max(np.abs(tolerance_I - reference_I)) 
        
    return max_errors 


#Convergence test for Monte Carlo method 
def convergence_test_montecarlo (_xpmin, _xpmax, _ypmin, _ypmax, _x, _y, _k, _z, _R, _Nvalues): 
    """
    Function to compute the convergence test for Monte Carlo method. 

    Parameters
    ---------- 
    _xpmin : floating point, lower limit of integration in the x-coordinate. 
    _xpmax : floating point, upper limit of integration in the x-coordinate. 
    _ypmin : floating point, lower limit of integration in the y-coordinate. 
    _ypmax : floating point, upper limit of integration in the y-coordinate. 
    _yp : y-coordinates of a point in the aperture. 
    _xp : x-coordinate of a point in the aperture. 
    _y : independent variable, y-coordinate of a point on the screen. 
    _x : independent variable, x-coordinate of a point on the screen. 
    _k : floating point, wavenumber of the function. 
    _z : floating point, distance between aperture and screen. 
    ------- 
    Returns: numerical error for different tolerances in Monte Carlo method. 
    """ 
    reference_I = np.zeros((len(_x), len(_y))) 
    
    #Loop to compute intensity at every point on screen 
    for _i in range(len(_x)): 
        x = _x [_i] 
        for _j in range(len(_y)): 
            y = _y[_j] 
            ref_I, _ = Fresnel_2D_intensity(_xpmin, _xpmax, _ypmin, _ypmax, y, x, _k, _z, _epsabs = 1.0e-14, _epsrel = 1.0e-14)  
            reference_I [_i][_j] = ref_I 
    
    max_errors = np.zeros(len(_Nvalues)) 
    
    #First loop to compute intensity for each sampling numbers with index _k 
    for _k, _N in enumerate(_Nvalues): 
        tolerance_I = np.zeros((len(_x), len(_y))) 
        #Second loop to compute the intensity at every point x on screen 
        for _i in range(len(_x)): 
            x = _x [_i] 
            #Third loop to compute the intensity at every point y on screen
            for _j in range(len(_y)): 
                y = _y [_j] 
                tolerance_I [_i] [_j], _ = Fresnel_2D_montecarlo(y, x, _k, _z, _R, _N)  
            max_errors [_k] = np.max(np.abs(tolerance_I - reference_I)) 
        
    return max_errors 


#------------------------------------------------------------------------------ 
#Plotting results 
#------------------------------------------------------------------------------ 

#1D plotting 
def plotting_1D (_x, _intensity, _title): 
    """
    Function to plot 1D results with respective titles. 

    Parameters
    ----------
    _x : independent variable, x-coordinates of a point on the screen. 
    _intensity : dependent variable, intensity magnitude of the wave at each position. 
    _title : string point, title of the graph plotted. 
    ------- 
    Returns: None 
    """
    plt.figure() 
    plt.plot(_x, _intensity) 
    plt.ticklabel_format(style = 'sci', axis = 'x', scilimits = (-3, 3), useOffset = False) 
    plt.grid(True) 
    plt.xlabel("x-position (m)") 
    plt.ylabel("Intensity (W m^-2)") 
    plt.title(_title) 


#2D plotting 
def plotting_2D (_intensity, _extents, _title): 
    """
    Function to plot 2D results with respective titles. 

    Parameters
    ----------
    _intensity : dependent variable, intensity magnitude of the wave at each position. 
    _extents : tutple of the extensions of the 2D plot. 
    _title : string point, title of the graph plotted. 
    ------- 
    Returns: None 
    """
    plt.figure() 
    plt.imshow(_intensity, cmap = "nipy_spectral_r", extent = _extents, origin = 'lower') 
    plt.colorbar(label = "Intensity (W m^-2)") 
    plt.xlabel("x-position (m)") 
    plt.ylabel("y-position (m)") 
    plt.title(_title) 


#Convergence test plotting 
def plotting_convergence (_tolerances, _errors, _xlabel, _title) : 
    """
    Function to plot logarithmical scale of convergence test. 
    
    Parameters
    ----------
    _values : independent variable, values of tolerances considered. 
    _errors : dependent variable, error of given tolerances in comparison with reference. 
    _title : string point, title of the graph plotted. 
    ------- 
    Returns: None 
    """
    plt.figure() 
    plt.semilogx(_tolerances, _errors, marker = 'o') 
    #plt.ylim(-1.0e-17, 1.0e-17) 
    plt.ticklabel_format(style = 'sci', useOffset = False, axis = 'y') 
    plt.gca().invert_xaxis() 
    plt.grid(True) 
    plt.xlabel(_xlabel) 
    plt.ylabel("Absolute error") 
    plt.title(_title) 



#============================================================================== 
######################   Main part of the code   ##############################
#============================================================================== 

#Menu variable initialisation 
MyInput = '0' 
wavelength = '0' 
wavenumber = '0' 
screen_distance = '0' 
x_aperture_width = '0' 
y_aperture_width = '0' 
radius = '0' 
Nsample = '0' 


#Menu to choose the part of exercise to be run 
while MyInput != 'q': 
    
    #Description of the task covered in each part 
    print("\n") 
    print("'1' to simulate the 1D diffraction from 2D rectangular aperture") 
    print("'2' to simulate the 2D diffraction from 2D rectangular aperture") 
    print("'3' to simulate the 2D diffraction from 2D circular aperture") 
    print("'4' to simulate the 2D diffraction from 2D circular aperture using Monte Carlo method") 
    print("'q' to quit from the programme \n") 
    
    MyInput = input("Enter a choice '1', '2', '3', '4' or 'q': ") 
    print("You entered the choice: ", MyInput) 
    
    
#------------------------------------------------------------------------------ 
#Part (1): 1D diffraction from 2D rectangular aperture 
#------------------------------------------------------------------------------    
    if MyInput == '1': 
        print("You have chosen part (1): 1D diffraction from 2D rectangular aperture") 
        
        #Parameters choosing 
        wavelength, wavenumber, screen_distance = general_parameters(wavelength, screen_distance) 
        xpmin, xpmax, ypmin, ypmax = rectangular_parameters(x_aperture_width, y_aperture_width) 
        
        #Compute the screen coordinates 
        numpoints = 200 
        xvals, _ = coordinates_calculation(wavelength, xpmax, ypmax, screen_distance, numpoints) 
        yvals = 0.0  #yvals is a scalar in 1D 
        
        #Define NumPy arrays to store values 
        intensity_1D = np.zeros(numpoints) 
        error_intensity_1D = np.zeros(numpoints) 
        
        #Tolerance choice for dblquad function to compute intensity 
        epsabs = 1.0e-10 
        epsrel = 1.0e-10 
        
        initial = time() 
        #Loop to compute the intensity at each point x 
        for i in range(numpoints): 
            intensity, error_intensity = Fresnel_2D_intensity(xpmin, xpmax, ypmin, ypmax, yvals, xvals[i], wavenumber, screen_distance, epsabs, epsrel) 
            intensity_1D [i] = intensity 
            error_intensity_1D [i] = error_intensity 
        final = time() 
        
        #Plotting the diffraction pattern 
        title_pattern = "1D diffraction pattern" 
        plotting_1D (xvals, intensity_1D, title_pattern) 
        
        #Plotting the error on intensity 
        title_error = "1D diffraction error" 
        plotting_1D (xvals, error_intensity_1D, title_error) 
        
        #Convergence test and visualisation 
        tolerances = [1.0e-2, 1.0e-4, 1.0e-6, 1.0e-8] 
        x_label = "epsabs = epsrel" 
        title_convergence = "Dblquad convergence test in 1D" 
        convergence_test = convergence_test_dblquad(xpmin, xpmax, ypmin, ypmax, xvals, yvals, wavenumber, screen_distance, tolerances) 
        plotting_convergence(tolerances, convergence_test, x_label, title_convergence) 
        
        plt.show() 
        
        print("Part (1) completed in: ", final - initial) 
        
        
#------------------------------------------------------------------------------ 
#Part (2): 2D diffraction from 2D rectangular aperture 
#------------------------------------------------------------------------------ 
    elif MyInput == '2': 
        print("You have chosen part (2): 2D diffraction from 2D rectangular aperture") 
        
        #Parameters choosing 
        wavelength, wavenumber, screen_distance = general_parameters(wavelength, screen_distance) 
        xpmin, xpmax, ypmin, ypmax = rectangular_parameters(x_aperture_width, y_aperture_width) 
        
        #Compute the screen coordinates 
        numpoints = 200 
        xvals, yvals = coordinates_calculation(wavelength, xpmax, ypmax, screen_distance, numpoints) 
        
        #Define the extensions of the 2D map 
        extents = (xvals.min(), xvals.max(), yvals.min(), yvals.max()) 
        
        #Define NumPy arrays to store values 
        intensity_2D_rectangular = np.zeros((numpoints, numpoints)) 
        error_intensity_2D_rectangular = np.zeros((numpoints, numpoints)) 
        
        #Tolerance choice for dblquad function to compute intensity 
        epsabs = 1.0e-10 
        epsrel = 1.0e-10 
        
        initial = time() 
        #First loop to compute the intensity at each point x 
        for i in range(numpoints): 
            x = xvals[i] 
            #Second loop to compute the intensity at each point y 
            for j in range(numpoints): 
                y = yvals[j] 
                intensity_2D_rectangular [i][j], error_intensity_2D_rectangular [i][j] = Fresnel_2D_intensity(xpmin, xpmax, ypmin, ypmax, y, x, wavenumber, screen_distance, epsabs, epsrel) 
        #The whole loop computes the intensity at every point (x, y) in the interval 
        final = time() 
        
        #Plotting the diffraction pattern 
        title_pattern = "2D diffraction pattern in rectangular aperture" 
        plotting_2D (intensity_2D_rectangular, extents, title_pattern) 
        
        #Plotting the error on intensity 
        title_error = "2D diffraction error in rectangular aperture" 
        plotting_2D (error_intensity_2D_rectangular, extents, title_error) 
        
        plt.show() 
        
        print("Part (2) completed in: ", final - initial) 
        
        
#------------------------------------------------------------------------------ 
#Part (3): 2D diffraction from 2D circular aperture 
#------------------------------------------------------------------------------ 
    elif MyInput == '3': 
        print("You have chosen part (3): 2D diffraction from 2D circular aperture") 
        
        #Parameters choosing 
        wavelength, wavenumber, screen_distance = general_parameters(wavelength, screen_distance) 
        radius, xpmin, xpmax, ypmin, ypmax = circular_parameters(radius) 
        
        #Compute the screen coordinates 
        numpoints = 200 
        xvals, yvals = coordinates_calculation(wavelength, xpmax, ypmax, screen_distance, numpoints) 
        
        #Define the extensions of the 2D map 
        extents = (xvals.min(), xvals.max(), yvals.min(), yvals.max()) 
        
        #Define NumPy arrays to store intensity values 
        intensity_2D_circular = np.zeros((numpoints, numpoints)) 
        error_intensity_2D_circular = np.zeros((numpoints, numpoints)) 
        
        #Tolerance choice for dblquad function to compute intensity 
        epsabs = 1.0e-10 
        epsrel = 1.0e-10 
        
        initial = time() 
        #First loop to compute the intensity at each point x 
        for i in range(numpoints): 
            x = xvals[i] 
            #Second loop to compute the intensity at each point y 
            for j in range(numpoints): 
                y = yvals[j] 
                intensity_2D_circular [i][j], error_intensity_2D_circular [i][j] = Fresnel_2D_intensity(-radius, radius, yp1func, yp2func, y, x, wavenumber, screen_distance, epsabs, epsrel) 
        #The whole loop computes the intensity at every point (x, y) in the interval 
        final = time() 
        
        #Plotting the diffraction pattern 
        title_pattern = "2D diffraction pattern in circular aperture" 
        plotting_2D (intensity_2D_circular, extents, title_pattern) 
        
        #Plotting the error on intensity 
        title_error = "2D diffraction error in circular aperture" 
        plotting_2D (error_intensity_2D_circular, extents, title_error) 
        
        plt.show() 
        
        print("Part (3) completed in: ", final - initial) 
        
        
#------------------------------------------------------------------------------ 
#Part (4): 2D diffraction from 2D circular aperture by Monte Carlo method 
#------------------------------------------------------------------------------ 
    elif MyInput == '4': 
        print("You have chosen part (4): 2D diffraction from 2D circular aperture using Monte Carlo method") 
        
        #Parameters choosing 
        wavelength, wavenumber, screen_distance = general_parameters(wavelength, screen_distance) 
        radius, xpmin, xpmax, ypmin, ypmax = circular_parameters(radius) 
        Nsample = int(input("Enter the number of trials for the Monte Carlo method: ")) 
        
        #Compute the screen coordinates 
        numpoints = 200 
        xvals, yvals = coordinates_calculation(wavelength, xpmax, ypmax, screen_distance, numpoints) 
        
        #Define the extensions of the 2D map 
        extents = (xvals.min(), xvals.max(), yvals.min(), yvals.max()) 
        
        #Define NumPy arrays to store intensity values 
        intensity_2D_montecarlo = np.zeros((numpoints, numpoints)) 
        error_intensity_2D_montecarlo = np.zeros((numpoints, numpoints)) 
        
        initial = time() 
        #First loop to compute intensity at each point x 
        for i in range(numpoints): 
            x = xvals [i] 
            #Second loop to compute intensity at each point y 
            for j in range(numpoints): 
                y = yvals [j] 
                intensity_2D_montecarlo [i][j], error_intensity_2D_montecarlo [i][j] = Fresnel_2D_montecarlo(y, x, wavenumber, screen_distance, radius, Nsample) 
        #The whole loop computes the intensity at every point (x, y) in the interval 
        final = time() 
        
        #Plotting the diffraction pattern 
        title_pattern = "2D diffraction pattern in circular aperture by Monte Carlo" 
        plotting_2D (intensity_2D_montecarlo, extents, title_pattern) 
        
        #Plotting the error on intensity 
        title_error = "2D diffraction error in circular aperture by Monte Carlo" 
        plotting_2D (error_intensity_2D_montecarlo, extents, title_error) 
        
        #Convergence test and visualisation 
        Nvalues = [100, 1000, 5000, 10000, 100000] 
        x_label = "Number of samples"
        title_convergence = "Monte Carlo convergence test in 2D" 
        convergence_test = convergence_test_montecarlo(xpmin, xpmax, ypmin, ypmax, xvals, yvals, wavenumber, screen_distance, radius, Nvalues) 
        plotting_convergence(Nvalues, convergence_test, x_label, title_convergence) 
        
        plt.show() 
        
        print("Part (4) completed in: ", final - initial) 
        
    elif MyInput != 'q': 
        print("This is not a valid choice") 
        
print("You have chosen to finish the - goodbye.") 


