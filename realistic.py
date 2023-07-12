import numpy as np
from scipy import interpolate, integrate
from scipy.optimize import fsolve 

####################################
#              MODULE              #
####################################
class RealisticModel:
    def __init__(self):
        self.kB_SI = np.longdouble(8.617333262e-05)                # [eV/K]                
        self.Clight_SI = np.longdouble(299792458.0)                # [m/s]  
        self.hbar_SI = np.longdouble(6.582119569e-16)              # [eV.s]
        self.qe_SI = np.longdouble(1.602176634e-19)                # [C]
        
    def reflectivity(sel, nn, kappa):
        """Calculate reflectivity from refractive index and extinction coefficient"""
        num = (nn - 1.)**2 + kappa**2
        den = (nn + 1.)**2 + kappa**2
        return np.divide(num, den)
    
    def absorptivity(self, alpha, RR, dd):
        """Calculate absorptivity"""
        return (1. - RR)*(1. - np.exp(-alpha*dd))
    
    def photon_flux(self, egap, temp, mu, omega, alpha, refractive, reflectivity, thickness):
        """Calculate photon flux [1/m^2 s]"""
        const = np.longdouble(1.)/((self.Clight_SI**2) * (np.pi**2) * (self.hbar_SI**3))
        aa = self.absorptivity(alpha, reflectivity, thickness)
        idx = np.where(omega >= egap)[0]
        En = omega[idx]
        aa = aa[idx]
        nn = refractive[idx]
        num = aa * (nn**2) * (En**2)
        den = np.exp((En - mu) / (self.kB_SI * temp)) - 1.
        integrand = np.divide(num, den)    
        integral = integrate.simps(integrand, En)
        return const*integral
    
    def radiative_heat_flux(self, egap, temp, mu, omega, alpha, refractive, reflectivity, thickness):
        """Calculate radiative heat flux [W/m^2]"""
        const = np.longdouble(1.)/((self.Clight_SI**2) * (np.pi**2) * (self.hbar_SI**3))
        aa = self.absorptivity(alpha, reflectivity, thickness)
        idx = np.where(omega >= egap)[0]
        En = omega[idx]
        aa = aa[idx]
        nn = refractive[idx]
        num = aa * (nn**2) * (En**3)
        den = np.exp((En - mu) / (self.kB_SI * temp)) - 1.
        integrand = np.divide(num, den)    
        integral = integrate.simps(integrand, En)
        return self.qe_SI*const*integral
    
    def current_density(self, egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness):
        """Calculate current density [A/m^2]"""
        FluxTa = self.photon_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
        FluxTc = self.photon_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
        return self.qe_SI*(FluxTa-FluxTc)
    
    def current_density_NR(self, egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness, lambdaNR):
        """Calculate current density r.w.t nonradiative generation ratio[A/m^2]"""
        FluxTa = self.photon_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
        FluxTc = self.photon_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
        return self.qe_SI*( (FluxTa/(1.0 - lambdaNR)) - FluxTc )
    
    def power_density(self, egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness):
        """Calculate power density [W/m^2]"""
        FluxTa = self.photon_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
        FluxTc = self.photon_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
        return self.qe_SI*mu*(FluxTa-FluxTc)
    
    def power_density_NR(self, egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness, lambdaNR):
        FluxTa = self.photon_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
        FluxTc = self.photon_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
        return self.qe_SI*mu*( (FluxTa/(1.0 - lambdaNR)) - FluxTc )
    
    def efficiency_ideal(self, egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness):
        """Calculate efficiency ideal case"""
        PD = self.power_density(egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTa = self.radiative_heat_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTc = self.radiative_heat_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
        return np.divide(PD, (PD+HeatFluxTc-HeatFluxTa))
    
    def find_mu(self, egap, temp_cell, temp_amb, omega, alpha, refractive, reflectivity, thickness):
        """Find the value of mu that makes the current density zero using fsolve."""
        def current_density_func(mu):
            FluxTa = self.photon_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
            FluxTc = self.photon_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
            return self.qe_SI*(FluxTa-FluxTc)    
        return fsolve(current_density_func, np.longdouble(0.0))[0]

    def find_mu_NR(self, egap, temp_cell, temp_amb, omega, alpha, refractive, reflectivity, thickness, lambdaNR):
        """Find the value of mu that makes the current density zero using fsolve."""
        def current_density_NR_func(mu):
            FluxTa = self.photon_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
            FluxTc = self.photon_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
            return self.qe_SI*( (FluxTa/(1.0 - lambdaNR)) - FluxTc )  
        return fsolve(current_density_NR_func, np.longdouble(0.0))[0]
    
    def radiative_heat_flux_below_egap(self, egap, temp, omega, alpha, refractive, reflectivity, thickness):
        const = np.longdouble(1.)/((self.Clight_SI**2) * (np.pi**2) * (self.hbar_SI**3))
        # Calculate absorptivity
        aa = self.absorptivity(alpha, reflectivity, thickness)
        # filter data omega <= omega_g
        idx = np.where(omega <= egap)[0]
        En = omega[idx]
        aa = aa[idx]
        nn = refractive[idx]
        # Calculate integrand
        with np.errstate(divide='ignore', invalid='ignore'):
            num = aa * (nn**2) * (En**3)
            den = np.exp((En) / (self.kB_SI * temp)) - np.longdouble(1.)
            integrand = np.divide(num, den, dtype=np.longdouble)    
            integrand[~np.isfinite(integrand)] = np.longdouble(0)
        # Calculate integral using Simpson's rule
        integral = integrate.simps(integrand, En)
        return self.qe_SI*const*integral
    
    def efficiency_with_subbandgap_losses(self, egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness):
        PD = self.power_density(egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTaAboveEgap = self.radiative_heat_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTcAboveEgap = self.radiative_heat_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTaBelowEgap = self.radiative_heat_flux_below_egap(egap, temp_amb, omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTcBelowEgap = self.radiative_heat_flux_below_egap(egap, temp_cell, omega, alpha, refractive, reflectivity, thickness)
        return np.divide(PD, (PD + HeatFluxTcAboveEgap + HeatFluxTcBelowEgap - HeatFluxTaAboveEgap - HeatFluxTaBelowEgap), dtype=np.longdouble)

    def heat_conduction_losses(self, U_const, temp_cell, temp_amb):
        return U_const*(temp_cell-temp_amb)

    def efficiency_with_heat_losses(self, egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness, U_const):
        PD = self.power_density(egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTaAboveEgap = self.radiative_heat_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTcAboveEgap = self.radiative_heat_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
        Ec = self.heat_conduction_losses(U_const, temp_cell, temp_amb)
        return np.divide(PD, (PD + HeatFluxTcAboveEgap + Ec - HeatFluxTaAboveEgap), dtype=np.longdouble)

    def efficiency_with_Subbandgap_Heat(self, egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness, U_const):
        PD = self.power_density(egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTaAboveEgap = self.radiative_heat_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTcAboveEgap = self.radiative_heat_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTaBelowEgap = self.radiative_heat_flux_below_egap(egap, temp_amb, omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTcBelowEgap = self.radiative_heat_flux_below_egap(egap, temp_cell, omega, alpha, refractive, reflectivity, thickness)
        Ec = self.heat_conduction_losses(U_const, temp_cell, temp_amb)
        return np.divide(PD, (PD + HeatFluxTcAboveEgap + HeatFluxTcBelowEgap + Ec - HeatFluxTaAboveEgap - HeatFluxTaBelowEgap), dtype=np.longdouble)

    def efficiency_with_NR_losses(self, egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness, lambdaNR):
        PD = self.power_density_NR(egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness, lambdaNR)
        HeatFluxTa = self.radiative_heat_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTc = self.radiative_heat_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
        return np.divide(PD, (PD+HeatFluxTc-HeatFluxTa))

    def efficiency_all_losses(self, egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness, U_const, lambdaNR):
        PD = self.power_density_NR(egap, temp_cell, temp_amb, mu, omega, alpha, refractive, reflectivity, thickness, lambdaNR)
        HeatFluxTaAboveEgap = self.radiative_heat_flux(egap, temp_amb, np.longdouble(0.0), omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTcAboveEgap = self.radiative_heat_flux(egap, temp_cell, mu, omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTaBelowEgap = self.radiative_heat_flux_below_egap(egap, temp_amb, omega, alpha, refractive, reflectivity, thickness)
        HeatFluxTcBelowEgap = self.radiative_heat_flux_below_egap(egap, temp_cell, omega, alpha, refractive, reflectivity, thickness)
        Ec = self.heat_conduction_losses(U_const, temp_cell, temp_amb)
        return np.divide(PD, (PD + HeatFluxTcAboveEgap + HeatFluxTcBelowEgap + Ec - HeatFluxTaAboveEgap - HeatFluxTaBelowEgap), dtype=np.longdouble)

cellTRRealistic = RealisticModel()
