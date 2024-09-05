VacancyFormationEnergyRelaxationVolume Test Driver

SUMMARY
  
  This test driver calculates the vacancy formation energy and relaxation volume at zero temperature and zero pressure.
  
  It works for all prototypes-generating all symmetry unique vacancies.
  For single-element systems the reservoir is defined as the system itself, while for multi-element systems the reservoirs are defined to be the lowest energy reference structures obtained from the `reference elemental ground state test <WIP>`_. 
  For structures that are not stable upon the creation of a monovacancy, it reports the relaxation result in stdout.
  For structures that are stable, it reports in results.edn, not only the value of vacancy formation energy and relaxation volume, but also the uncertainty range
  
METHOD
  
  The calculation consists of two steps:
  
  1. Calculate the vacancy formation energy and relaxation volume corresponding to three different sizes of supercell.
  The minimum size is 3*3*3 cubic unit cells for diamond structure (27 * 8 = 216 atoms).
  For other structures the minimum size is determined by the smallest number n, which makes n*n*n conventional unit cells has at least the same number of atoms as the above diamond supercell.
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
