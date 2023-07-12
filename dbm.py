# numerical module
import numpy as np
import scipy.integrate as integrate
from scipy.optimize import fsolve

####################################
#              MODULE              #
####################################
class DBMModel:
    egapMax = np.longdouble(15.0)  # [eV]
    
    def __init__(self):
        self.c = 299792458.0            # [m/s]
        self.h = 4.135667696e-15        # [eV.s]
        self.e = 1.602176634e-19        # [C]
        self.kB = 8.617333262e-05       # [eV/K]
        
    # photon flux
    def photon_flux(self, temp, mu, egap):
        const = (2. * np.pi) / (self.c ** 2 * self.h ** 3)
        integrand = lambda E: (E ** 2) / (np.exp((E - mu) / (self.kB * temp)) - 1. )
        integral, _ = integrate.quad(integrand, egap, self.egapMax)
        return const * integral
    
    # current density in ideal case
    def current_density(self, mu, temp_cell, temp_amb, egap):
        flux_Ta = self.photon_flux(temp_amb, 0.0, egap)
        flux_Tc = self.photon_flux(temp_cell, mu, egap)
        return self.e * (flux_Ta - flux_Tc)
    
    # current density introduce nonradiative generation ratio \lambda
    def current_density_NR(self, mu, temp_cell, temp_amb, egap, lambdaNR):
        flux_Ta = self.photon_flux(temp_amb, 0.0, egap)
        flux_Tc = self.photon_flux(temp_cell, mu, egap)
        return self.e*( (flux_Ta/(1.0 - lambdaNR)) - flux_Tc )
    
    # radiative heat flux density above bandgap
    def radiative_energy_flux(self, temp, mu, egap):
        const = (2. * np.pi) / (self.c ** 2 * self.h ** 3)
        integrand = lambda E: (E ** 3) / (np.exp((E - mu) / (self.kB * temp)) - 1.0 ) 
        integral, _ = integrate.quad(integrand, egap, self.egapMax)
        return self.e * const * integral
    
    # radiative heat flux density below bandgap
    def radiative_heat_flux_below_egap(self, temp, egap):
        const = (2. * np.pi) / (self.c ** 2 * self.h ** 3)
        integrand = lambda E: (E ** 3) / (np.exp((E) / (self.kB * temp)) - 1.0 )
        integral, _ = integrate.quad(integrand, 0.0, egap)
        return self.e * const * integral
    
    # power density ideal case 
    def power_density(self, temp_cell, temp_amb, mu, egap):
        flux_Ta = self.photon_flux(temp_amb, 0.0, egap)
        flux_Tc = self.photon_flux(temp_cell, mu, egap)
        return self.e * mu * (flux_Ta - flux_Tc)
    
    # power density introduce nonradiative generation ratio
    def power_density_NR(self, temp_cell, temp_amb, mu, egap, lambdaNR):
        flux_Ta = self.photon_flux(temp_amb, 0.0, egap)
        flux_Tc = self.photon_flux(temp_cell, mu, egap)
        return self.e * mu * ( (flux_Ta/(1.0 - lambdaNR)) - flux_Tc )
    
    # determine \mu ideal case
    def find_mu(self, temp_cell, temp_amb, egap):
        mu = fsolve(self.current_density, 0.0, args=(temp_cell, temp_amb, egap))
        return mu[0]
    
    # determine \mu introduce nonradiative generation ratio
    def find_mu_NR(self, temp_cell, temp_amb, egap, lambdaNR):
        mu = fsolve(self.current_density_NR, 0.0, args=(temp_cell, temp_amb, egap, lambdaNR))
        return mu[0]
    
    # heat conduction loss
    def heat_conduction_losses(self, U_const, temp_cell, temp_amb):
        return U_const*(temp_cell-temp_amb)
    
    # Carnot efficiency as upper limit
    def carnot_efficiency(self, temp_cell, temp_amb):
        return 1.0 - (temp_amb / temp_cell)
    
    # efficiency ideal
    def efficiency_ideal(self, temp_cell, temp_amb, mu, egap):
        Erad = self.radiative_energy_flux(temp_cell, mu, egap)
        Eabs = self.radiative_energy_flux(temp_amb, 0.0, egap)
        PD = self.power_density(temp_cell, temp_amb, mu, egap)
        return np.divide(PD, (PD + Erad - Eabs))     
    
    # efficiency introduce sub-bandgap loss only
    def efficiency_with_subbandgap_losses(self, temp_cell, temp_amb, mu, egap):        
        PD = self.power_density(temp_cell, temp_amb, mu, egap)
        EabsAboveEgap = self.radiative_energy_flux(temp_amb, 0.0, egap)
        EradAboveEgap = self.radiative_energy_flux(temp_cell, mu, egap)
        EabsBelowEgap = self.radiative_heat_flux_below_egap(temp_amb, egap)
        EradBelowEgap = self.radiative_heat_flux_below_egap(temp_cell, egap)
        return np.divide(PD, (PD + EradAboveEgap+ EradBelowEgap - EabsAboveEgap -EabsBelowEgap))
    
    # efficiency introduce heat loss only
    def efficiency_with_heat_losses(self, temp_cell, temp_amb, mu, egap, U_const):
        PD = self.power_density(temp_cell, temp_amb, mu, egap)
        EabsAboveEgap = self.radiative_energy_flux(temp_amb, 0.0, egap)
        EradAboveEgap = self.radiative_energy_flux(temp_cell, mu, egap)
        Ec = self.heat_conduction_losses(U_const, temp_cell, temp_amb)
        return np.divide(PD, (PD+ EradAboveEgap + Ec - EabsAboveEgap))
    
    # efficiency with sub-bandgap and heat losses
    def efficiency_with_Subbandgap_Heat(self, temp_cell, temp_amb, mu, egap, U_const):
        PD = self.power_density(temp_cell, temp_amb, mu, egap)
        EabsAboveEgap = self.radiative_energy_flux(temp_amb, 0.0, egap)
        EradAboveEgap = self.radiative_energy_flux(temp_cell, mu, egap)
        EabsBelowEgap = self.radiative_heat_flux_below_egap(temp_amb, egap)
        EradBelowEgap = self.radiative_heat_flux_below_egap(temp_cell, egap)
        Ec = self.heat_conduction_losses(U_const, temp_cell, temp_amb)
        return np.divide(PD, (PD + EradAboveEgap+ EradBelowEgap + Ec - EabsAboveEgap -EabsBelowEgap))
    
    # efficiency introduce nonradiative generation ratio only
    def efficiency_with_NR_losses(self, temp_cell, temp_amb, mu, egap, lambdaNR):
        PD = self.power_density_NR(temp_cell, temp_amb, mu, egap, lambdaNR)
        Erad = self.radiative_energy_flux(temp_cell, mu, egap)
        Eabs = self.radiative_energy_flux(temp_amb, 0.0, egap)
        return np.divide(PD, (PD + Erad - Eabs))
    
    # efficiency with all losses
    def efficiency_all_losses(self, temp_cell, temp_amb, mu, egap, U_const, lambdaNR):
        PD = self.power_density_NR(temp_cell, temp_amb, mu, egap, lambdaNR)
        EabsAboveEgap = self.radiative_energy_flux(temp_amb, 0.0, egap)
        EradAboveEgap = self.radiative_energy_flux(temp_cell, mu, egap)
        EabsBelowEgap = self.radiative_heat_flux_below_egap(temp_amb, egap)
        EradBelowEgap = self.radiative_heat_flux_below_egap(temp_cell, egap)
        Ec = self.heat_conduction_losses(U_const, temp_cell, temp_amb)
        return np.divide(PD, (PD + EradAboveEgap+ EradBelowEgap + Ec - EabsAboveEgap -EabsBelowEgap))    
    
cellTRDBM = DBMModel()
