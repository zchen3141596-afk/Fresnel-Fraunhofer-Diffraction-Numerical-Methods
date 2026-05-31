# Fresnel-Fraunhofer-Diffraction-Numerical-Methods 

Python simulations of Fresnel and Fraunhofer diffraction patterns using Gaussian quadrature and Monte Carlo integration methods.

The code supports a computational physics study comparing deterministic and stochastic numerical integration methods for simulating Fresnel and Fraunhofer diffraction patterns. The simulation consider one-dimensional and two-dimensional diffraction through renctangular and circular apertures. 

## Project overview 

In this project, scalar diffraction is modelled numerically using the Fresnel diffraction integral. The main aims is to compare two numerical integration approaches: 

1. Gaussian quadrature, implemented using scipy.integrate.dblquad. 
2. Monte Carlo integration, implemented using random sampling over the aperture region.


The methods are compared in terms of: 

- Simulated intensity distributions
- 1D and 2D diffraction patterns
- Fresnel and Fraunhofer regimes
- Rectangular and circular aperture geometries
- Numerical uncertainty / error maps
- Convergence behavoiur
- Computational runtime

The repository is intended to make the numerical implementation reproducible without overloading the main paper with code-level details. 


## Physical model 

The simulations are based on the Fresnel diffraction integral for the complex electric field 

E(x,y,z)=\frac{e^{ikz}}{i\lambda z} \iint_A E(x',y') \exp\left[ \frac{ik}{2z}\left((x-x')^2+(y-y')^2\right) \right] dx' dy'

The intensity distribution is calculated from the squared magnitude of the electric field, 

I(x,y,z)=\epsilon_0 c |E(x,y,z)|^2 

The Fresnel number is used to distinguish the near-field and far-field regimes: 

F=\frac{r^2}{\lambda z} 

Larger values of F coorrespond to Fresnel diffraction, while smaller values correspond to the fraunhofer limit. 


## Numerical methods 

### Gaussian quadrature 

Gaussian quadrature approximates an integral using weighted function evaluations at selected quadrature nodes. In two dimensions, the approximation can be written as

\iint_A f(x,y)\,dx\,dy \approx \sum_{i=1}^{n}\sum_{j=1}^{m} w_i v_j f(x_i,y_j) 


### Monte Carlo integration 

Monte Carlo integration estimates the aperture integral by randomly sampling points inside the integration region. For an aperture of area (A), the integral is approximated by 

\int_A f(\mathbf{r})\,dA \approx \frac{A}{N}\sum_{i=1}^{N} f(\mathbf{r}_i) 



## Physical assumptions 

The simulations use an ideal scalar diffraction model. The following assumptions are made: 

- incident light is monochromatic
- aperture boundaries are perfectly sharp
- aperture plane and observation screen are parallel
- incident light propagates perpendicular to the aperture plane
- reflection, absorption, scattering and aperture imperfections are neglected
- experimental noise and detector response are not included


## Installation 

Clone the repository: 

git clone https://github.com/zchen3141596-afk/Fresnel-Fraunhofer-Diffraction-Numerical-Methods.gitcd Fresnel-Fraunhofer-Diffraction-Numerical-Methods

Install the required dependencies: 

pip install -r requirements.txt 


## Dependencies 

The project uses: 
 - numpy
 - scipy
 - matplotlib
 - numba


## Reproducibility notes 

Runtime values depend on hardware, Python version, and the numerical backend used by SciPy and NumPy. Monte Carlo results may vary slightly between runs because of random sampling. For exact reproducibility, set a fixed random seed in the Monte Carlo functions. 

## Citation 

If you use this repository, please cite: 

ian quadrature and Monte Carlo integration methods", Student Research Journal, University of Bristol, 2026.



