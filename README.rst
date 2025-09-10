VacancyFormationEnergyRelaxationVolume Test Driver

SUMMARY
  
  This test driver calculates the vacancy formation energy and relaxation volume at zero temperature and zero pressure.
  
  It works for all prototypes-generating all symmetry unique vacancies.
  Vacancy energies are calculated by considering the vacancy moving to a known reference reservoir crystal structure and including the chemical potential associated with this. 
  Reference reservoirs are element specific to the removed atom and follow those used by `CHIPS-FF <https://github.com/usnistgov/chipsff/blob/main/chipsff/chemical_potentials.json>`_.
  Chemical potentials are model specific and taken from the reference-elemental-energy property.

  For structures that are not stable upon the creation of a monovacancy, it reports the relaxation result in stdout.
  For structures that are stable, it reports in results.edn the value of vacancy formation energy and relaxation volume.
  
METHOD
  
  The calculation consists of two steps for each unique Wyckoff site monovacancy:
  
  1. Calculate the vacancy formation energy and relaxation volume corresponding to three different sizes of supercell.
  The minimum size is determined by the smallest number n, which makes a n*n*n unit cell with at least 216 atoms.
  The other two sizes are the two numbers that follows the minimum size.
  
  The Calculation of each size starts from constructing the periodic supercell, then take out one atom from the supercell.
  Then it does the relaxation of positions within the supercell and the relaxation of the cell vectors alternatively, until converge.
  It will stop when it finds the volume decreased significantly or the potential energy decreased significantly, which indicates the crystal collapse.
  
  2. Extrapolate from these three results to get the value corresponding to an infinitely large supercell.
  
  The extrapolation is based on the elastic theory, which gives the dependencies of vacancy formation energy and relaxation volume on r are 1/r^3.
  Therefore, we use the two larger size to fit f(x) = a0 + a1/r^3, and obtain a0, which we use as the value
  
  It also estimate the uncertainty, based on two other fits, taking whichever gives larger difference to the value obtained above, and use that difference as the uncertainty.
  One is fitting with all the three sizes to the function above.
  The other one includes a2/r^4 term, and also using all the three size
