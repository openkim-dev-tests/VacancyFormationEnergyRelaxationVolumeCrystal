from kim_tools import SingleCrystalTestDriver
from kim_tools.aflow_util.core import get_atom_indices_for_each_wyckoff_orb
import numpy as np
from ase.geometry.cell import cellpar_to_cell
from ase.optimize import FIRE
from scipy.optimize import fmin
import sys
import math
from collections import OrderedDict

KEY_SOURCE_VALUE = 'source-value'
KEY_SOURCE_UNIT = 'source-unit'
KEY_SOURCE_UNCERT = 'source-std-uncert-value'

def V(value, unit = '', uncert = ''):
    # Generate OrderedDict for JSON Dump
    res = OrderedDict([
        (KEY_SOURCE_VALUE, value),
    ])
    if unit != '':
        res.update(OrderedDict([
            (KEY_SOURCE_UNIT, unit),
        ]))
    if uncert != '':
        res.update(OrderedDict([
            (KEY_SOURCE_UNCERT, uncert)
        ]))
    return res

# Parameters for Production
FIRE_MAX_STEPS = 1000
FIRE_UNCERT_STEPS = 20
FIRE_TOL = 1e-3 # absolute
FMIN_FTOL = 1e-6 # relative
FMIN_XTOL = 1e-10 # relative
VFE_TOL = 1e-5 # absolute
MAX_LOOPS = 20
CELL_SIZE_MIN = 3
CELL_SIZE_MAX = 5
COLLAPSE_CRITERIA_VOLUME = 0.1
COLLAPSE_CRITERIA_ENERGY = 0.1
DYNAMIC_CELL_SIZE = True # Increase Cell Size According to lattice structure
EPS = 1e-3

# Extrapolation Parameters
FITS_CNT = [2, 3, 3, 3, 3] # Number of data points used for each fitting
FITS_ORDERS = [
    [0, 3],
    [0, 3],
    [0, 3, 4],
    [0, 3, 5],
    [0, 3, 6],
] # Number of orders included in each fitting
# Fit Results Used (Corresponding to the above)
FITS_VFE_VALUE = 0 # Vacancy Formation Energy
FITS_VFE_UNCERT = [1, 2]
FITS_VRV_VALUE = 0 # Vacancy Relaxation Volume
FITS_VRV_UNCERT = [1, 2]

# Strings for Output
UNIT_ENERGY = 'eV'
UNIT_LENGTH = 'angstrom'
UNIT_ANGLE = 'degree'
UNIT_PRESSURE = 'GPa'
UNIT_VOLUME = UNIT_LENGTH + '^3'

# Disclaimer Uncertainty
DISCLAIMER_THRESHOLD_PERCENT = 1.
DISCLAIMER = f"value has an estimated uncertainty greater than {DISCLAIMER_THRESHOLD_PERCENT}%."

class TestDriver(SingleCrystalTestDriver):
    def _calculate(self, reservoir_info=None, **kwargs):
        self.atoms = self._get_atoms()
        if reservoir_info is None:
            ele = set(self.atoms.get_chemical_symbols())
            reservoir_info = {}
            for e in ele:
                reservoir_info[e] = [{"binding-potential-energy-per-atom":{"source-value": 0}, "prototype-label": {"source-value": 'Not provided'}}]
        self.reservoir_info = reservoir_info 
        prototype_label  = self._SingleCrystalTestDriver__nominal_crystal_structure_npt['prototype-label']['source-value']
        self.prototype_label = prototype_label
        self.equivalent_atoms = get_atom_indices_for_each_wyckoff_orb(prototype_label)
        # store intermediate results for effective vacancy
        self.effective_results = []
        self.bulk_energies = {}

        if DYNAMIC_CELL_SIZE == True:
            numAtoms = self.atoms.get_number_of_atoms()
            factor = math.pow(8 / numAtoms, 0.333)
            self.cell_size_min = int(math.ceil(factor * CELL_SIZE_MIN))
            self.cell_size_max = self.cell_size_min + 2
            print('CELL_SIZE_MIN:', self.cell_size_min)
            print('CELL_SIZE_MAX:', self.cell_size_max)
            print('Smallest System Size:', numAtoms * self.cell_size_min**3)
            print('Largest System Size:', numAtoms * self.cell_size_max**3)

        results = []
        for wkof in self.equivalent_atoms:
            idx = wkof['indices'][0] 
            results.append(self.getResults(idx, len(wkof['indices'])))
        organized_props, uncertainty = self.organize_properties(results)
        for k,v in organized_props.items():
            disclaimer = None
            for u in uncertainty[k]["percent"]:
                if u > DISCLAIMER_THRESHOLD_PERCENT: 
                    disclaimer = f"{uncertainty[k]['prop']} " + DISCLAIMER
                    print ("DISCLAIMER:" + disclaimer)
            self._add_property_instance_and_common_crystal_genome_keys(k,
                                                                   write_stress=False, write_temp=False, disclaimer=disclaimer)
            for k2,v2 in v.items():
                if 'source-unit' in v2:
                    if k2 == uncertainty[k]["prop"]:
                        self._add_key_to_current_property_instance(k2, v2['source-value'], v2['source-unit'], {KEY_SOURCE_UNCERT: uncertainty[k]['values']})
                    else:
                        self._add_key_to_current_property_instance(k2, v2['source-value'], v2['source-unit'])
                else:
                    self._add_key_to_current_property_instance(k2, v2['source-value'])
        
        # combine effective results, extrapolate and write property
        bulkEnergies = [v for k,v in self.bulk_energies.items()]
        sizes =  [k for k,v in self.bulk_energies.items()]
        effectiveUnrelaxedList = []
        effectiveRelaxedList = []
        for idx,be in enumerate(bulkEnergies): 
            effectiveUnrelaxed = 0
            effectiveRelaxed = 0
            for effRes in self.effective_results:
                effectiveUnrelaxed += (effRes['multiplicity']/len(self.atoms) * effRes['unrelaxed'][idx])# - (len(self.atoms) * sizes[idx]**3 - 1) / (len(self.atoms) * sizes[idx]**3) * be
                effectiveRelaxed += (effRes['multiplicity']/len(self.atoms) * effRes['relaxed'][idx])# - (len(self.atoms) * sizes[idx]**3 - 1) / (len(self.atoms) * sizes[idx]**3) * be
            effectiveUnrelaxed -= (len(self.atoms) * sizes[idx]**3 - 1) / (len(self.atoms) * sizes[idx]**3) * be
            effectiveRelaxed -= (len(self.atoms) * sizes[idx]**3 - 1) / (len(self.atoms) * sizes[idx]**3) * be
            effectiveUnrelaxedList.append(effectiveUnrelaxed)
            effectiveRelaxedList.append(effectiveRelaxed)
        uev, rev, ueu, reu  = self.extrapolate(effectiveUnrelaxedList, effectiveRelaxedList)
        disclaimer = None
        if ueu / uev *100 > DISCLAIMER_THRESHOLD_PERCENT:
            disclaimer = "unrelaxed-effective-formation-potential-energy " + DISCLAIMER
            print ("DISCLAIMER: "+ disclaimer)
        self._add_property_instance_and_common_crystal_genome_keys(
            "effective-vacancy-unrelaxed-formation-potential-energy-crystal",
            write_stress=False, 
            write_temp=False,
            disclaimer=disclaimer
        )
        self._add_key_to_current_property_instance("unrelaxed-effective-formation-potential-energy", uev, UNIT_ENERGY, {KEY_SOURCE_UNCERT: ueu})

        disclaimer = None
        if reu / rev *100 > DISCLAIMER_THRESHOLD_PERCENT:
            disclaimer = "relaxed-effective-formation-potential-energy " + DISCLAIMER
            print ("DISCLAIMER: "+ disclaimer)
        self._add_property_instance_and_common_crystal_genome_keys(
            "effective-vacancy-relaxed-formation-potential-energy-crystal", 
            write_stress=False, 
            write_temp=False,
            disclaimer=disclaimer
        )
        self._add_key_to_current_property_instance("relaxed-effective-formation-potential-energy", rev, UNIT_ENERGY, {KEY_SOURCE_UNCERT: reu})
        
        
    def _createSupercell(self, size):
        atoms = self.atoms.copy()
        atoms.set_calculator(self._calc)
        atoms *= (size, size, size)
        return atoms

    def _cellVector2Cell(self, cellVector):
        cell = cellpar_to_cell(cellVector)
        return cell
    
    def _getVFE(self, cellVector, atoms):
        newCell = self._cellVector2Cell(cellVector)
        atoms.set_cell(newCell, scale_atoms = True)
        enAtomsWithVacancy = atoms.get_potential_energy()
        return enAtomsWithVacancy
        
    def _getResultsForSize(self, size, idx):
        # Setup Environment
        unrelaxedCell = self.atoms.get_cell() * size
        atoms = self._createSupercell(size)
        unrelaxedCellVector = atoms.get_cell_lengths_and_angles() 
        numAtoms = atoms.get_number_of_atoms()
        enAtoms = atoms.get_potential_energy()
        self.bulk_energies[size] = enAtoms
        unrelaxedCellEnergy = enAtoms
        unrelaxedCellVolume = np.abs(np.linalg.det(unrelaxedCell))
        print('\nSupercell Size:\n', size)
        print('Unrelaxed Cell:\n', unrelaxedCell)
        print('Unrelaxed Cell Vector:\n', unrelaxedCellVector)
        print('Unrelaxed Cell Energy:\n', unrelaxedCellEnergy)

        # Create Vacancy 
        del atoms[idx]
        enAtomsWithVacancy = atoms.get_potential_energy()

        print('Energy of Unrelaxed Cell With Vacancy:\n', enAtomsWithVacancy)
        enVacancyUnrelaxed = enAtomsWithVacancy - enAtoms + self.chemical_potential

        # Self Consistent Relaxation
        #enVacancy = 0
        effectiveRelaxed = 0

        relaxedCellVector = unrelaxedCellVector
        loop = 0
        while 1:
            # Position Relaxation
            print('==========')
            print('Loop:', loop)
            print('Position Relaxation...')
            dyn = FIRE(atoms)
            dyn.run(fmax = FIRE_TOL, steps = FIRE_MAX_STEPS)
            numSteps = dyn.get_number_of_steps()
            if numSteps >= FIRE_MAX_STEPS:
                print('WARNING: Max number of steps exceeded. Structure may be unstable.')
                # sys.exit(0)
            print('Relaxation Completed. Steps:', numSteps)

            # Cell Size Relaxation
            print('Cell Size Relaxation...')
            tmpCellVector, tmpEnVacancy = fmin(
                self._getVFE,
                relaxedCellVector,
                args = (atoms,),
                ftol = FMIN_FTOL,
                xtol = FMIN_XTOL,
                full_output = True,
            )[:2]

            # Convergence Requirement Satisfied
            if abs(tmpEnVacancy - effectiveRelaxed) < VFE_TOL and dyn.get_number_of_steps() < 1:
                dyn.run(fmax = FIRE_TOL * EPS, steps = FIRE_UNCERT_STEPS)
                tmpCellVector, tmpEnVacancy = fmin(
                    self._getVFE,
                    relaxedCellVector,
                    args = (atoms,),
                    ftol = FMIN_FTOL * EPS,
                    xtol = FMIN_XTOL * EPS,
                    full_output = True,
                )[:2]
                self.VFEUncert = np.abs(tmpEnVacancy - effectiveRelaxed)
                enVacancy = tmpEnVacancy - enAtoms + self.chemical_potential
                oldVolume = np.linalg.det(self._cellVector2Cell(relaxedCellVector))
                newVolume = np.linalg.det(self._cellVector2Cell(tmpCellVector.tolist()))
                self.VRVUncert = np.abs(newVolume - oldVolume)
                relaxedCellVector = tmpCellVector.tolist()
                break

            # Evf = Ev - E0 + mu, where mu is chemical potential of removed element
            enVacancy = tmpEnVacancy - enAtoms + self.chemical_potential
            print ('enVacancy', enVacancy)
            relaxedCellVector = tmpCellVector.tolist()
            # for effective
            effectiveRelaxed = tmpEnVacancy 

            # Check Loop Limit
            loop += 1
            if loop > MAX_LOOPS:
                print('Loops Limit Exceeded. Structure Unstable.')
                sys.exit(0)

            # Output Temporary Result
            relaxedCell = self._cellVector2Cell(relaxedCellVector)
            relaxedCellVolume = np.abs(np.linalg.det(relaxedCell))
            relaxationVolume = unrelaxedCellVolume - relaxedCellVolume
            print('Current VFE:', enVacancy)
            print('Energy of Supercell:', enAtoms)
            print('Unrelaxed Cell Volume:', unrelaxedCellVolume)
            print('Current Relaxed Cell Volume:', relaxedCellVolume)
            print('Current Relaxation Volume:', relaxationVolume)
            print('Current Cell:\n', np.array(self._cellVector2Cell(relaxedCellVector)))

            # Determine Collapse
            if np.abs(relaxationVolume) > COLLAPSE_CRITERIA_VOLUME * unrelaxedCellVolume:
                print('System Collapsed. Volume significantly changed.')
                sys.exit(0)
            if np.abs(enVacancy) > COLLAPSE_CRITERIA_ENERGY * np.abs(enAtoms):
                print('System Collapsed. System Energy significantly changed.')
                sys.exit(0)


        # Print Summary
        print('---------------')
        print('Calculation Completed.')
        print('Number Of Atoms in Supercell:', numAtoms)
        print('Vacancy Formation Energy (relaxed):', enVacancy)
        print('Vacancy Formation Energy (unrelaxed):', enVacancyUnrelaxed)
        print('Unrelaxed Cell Volume:', unrelaxedCellVolume)
        print('Relaxed Cell Volume:', relaxedCellVolume)
        print('Relaxation Volume:', relaxationVolume)
        print('Relaxed Cell Vector:\n', relaxedCellVector)
        print('Unrelaxed Cell Vector:\n', unrelaxedCellVector)
        print('Relaxed Cell:\n', np.array(self._cellVector2Cell(relaxedCellVector)))
        print('Unrelaxed Cell:\n', np.array(self._cellVector2Cell(unrelaxedCellVector)))

        return enVacancyUnrelaxed, relaxedCellVector, enVacancy, relaxationVolume, enAtomsWithVacancy, effectiveRelaxed

    def _getFit(self, xdata, ydata, orders):
        # Polynomial Fitting with Specific Orders
        A = []
        print('\nFit with Size:', xdata)
        print('Orders:', orders)
        for order in orders:
            A.append(np.power(xdata * 1.0, -order))
        A = np.vstack(A).T
        print('Matrix A (Ax = y):\n', A)
        print('Data for Fitting:', ydata)
        res = np.linalg.lstsq(A, ydata, rcond=None)
        print('Fitting Results:', res)
        return res[0]

    def _getValueUncert(self, valueFitId, uncertFitIds, systematicUncert, maxSizeId, dataSource):
        # Get sourceValue and sourceUncert use only certain size and fits
        # Get source value
        valueFitCnt = FITS_CNT[valueFitId]
        sourceValue = dataSource[valueFitId][maxSizeId - valueFitCnt + 1]

        # Get source uncertainty (statistical)
        sourceUncert = 0
        for uncertFitId in uncertFitIds:
            uncertFitCnt = FITS_CNT[uncertFitId]
            uncertValue = dataSource[uncertFitId][maxSizeId - uncertFitCnt + 1]
            sourceUncert = max([abs(uncertValue - sourceValue), sourceUncert])

        # Include systematic error, assuming independent of statistical errors
        sourceUncert = math.sqrt(sourceUncert**2 + systematicUncert**2)
        return sourceValue, sourceUncert

    def getResults(self, idx, mult):
        # grab chemical potential
        # add back isolated atom energy
        if len(set(self.atoms.get_chemical_symbols())) == 1:
            print ("Single element crystal detected-using self as reservoir.")
            self.chemical_potential = self.atoms.get_potential_energy()/len(self.atoms) 
            self.single_element = True
        else:
            self.chemical_potential = self.reservoir_info[self.atoms[idx].symbol][0]["binding-potential-energy-per-atom"]["source-value"] + self.get_isolated_energy_per_atom(self.atoms[idx].symbol) 
            self.single_element = False
        print ('Chemical Potential', self.chemical_potential)


        unitBulk = self.atoms

        # Calculate VFE and VRV for Each Size
        sizes = []
        unrelaxedformationEnergyBySize = []
        formationEnergyBySize = []
        relaxationVolumeBySize = []
        unrelaxedEffectiveBySize = []
        relaxedEffectiveBySize = []
        print('\n[Calculation]')
        for size in range(self.cell_size_min, self.cell_size_max + 1):
            unrelaxedFormationEnergy, relaxedCellVector, relaxedFormationEnergy, relaxationVolume, unrelaxedEffective, relaxedEffective = self._getResultsForSize(size, idx)
            sizes.append(size)
            unrelaxedformationEnergyBySize.append(unrelaxedFormationEnergy)
            formationEnergyBySize.append(relaxedFormationEnergy)
            relaxationVolumeBySize.append(relaxationVolume)
            unrelaxedEffectiveBySize.append(unrelaxedEffective)
            relaxedEffectiveBySize.append(relaxedEffective)

        print('\n[Calculation Results Summary]')
        print('Sizes:', sizes)
        print('Unrelaxed Formation Energy By Size:\n', unrelaxedformationEnergyBySize)
        print('Formation Energy By Size:\n', formationEnergyBySize)
        print('Relaxation Volume By Size:\n', relaxationVolumeBySize)
        self.effective_results.append({'multiplicity': mult, 'unrelaxed': unrelaxedEffectiveBySize, "relaxed": relaxedEffectiveBySize})
        self.sizes = sizes

        # Extrapolate for VFE and VRV of Infinite Size
        print('\n[Extrapolation]')
        naSizes = np.array(sizes)
        naUnrelaxedFormationEnergyBySize = np.array(unrelaxedformationEnergyBySize)
        naFormationEnergyBySize = np.array(formationEnergyBySize)
        naRelaxationVolumeBySize = np.array(relaxationVolumeBySize)
        unrelaxedformationEnergyFitsBySize = []
        formationEnergyFitsBySize = []
        relaxationVolumeFitsBySize = []
        for i in range(0, len(FITS_CNT)):
            cnt = FITS_CNT[i] # Num of Data Points Used
            orders = FITS_ORDERS[i] # Orders Included
            print('Fitting with', cnt, 'points, including orders', orders)
            unrelaxedformationEnergyFits = []
            formationEnergyFits = []
            relaxationVolumeFits = []
            for j in range(0, len(sizes) - cnt + 1):
                print('Fit with data beginning', j)
                xdata = naSizes[j:(j + cnt)]
                unrelaxedformationEnergyFits.append(self._getFit(
                    xdata,
                    naUnrelaxedFormationEnergyBySize[j:(j + cnt)],
                    orders
                )[0])
                formationEnergyFits.append(self._getFit(
                    xdata,
                    naFormationEnergyBySize[j:(j + cnt)],
                    orders
                )[0])
                relaxationVolumeFits.append(self._getFit(
                    xdata,
                    naRelaxationVolumeBySize[j:(j + cnt)],
                    orders
                )[0])
            unrelaxedformationEnergyFitsBySize.append(unrelaxedformationEnergyFits)
            formationEnergyFitsBySize.append(formationEnergyFits)
            relaxationVolumeFitsBySize.append(relaxationVolumeFits)

        # Output Fitting Results
        print('\n[Fitting Results Summary]')
        print('Sizes:', sizes)
        print('Data Points Used:', FITS_CNT)
        print('Orders Included:\n', FITS_ORDERS)
        print('Unrelaxed Formation Energy Fits By Size:\n', unrelaxedformationEnergyFitsBySize)
        print('Formation Energy Fits By Size:\n', formationEnergyFitsBySize)
        print('Relaxation Volume Fits By Size:\n', relaxationVolumeFitsBySize)

        # Obtain Extrapolated Value and Uncertainty
        unrelaxedformationEnergy, unrelaxedformationEnergyUncert = self._getValueUncert(
            FITS_VFE_VALUE,
            FITS_VFE_UNCERT,
            # FMIN_FTOL * formationEnergyBySize[-1],
            self.VFEUncert,
            2,
            unrelaxedformationEnergyFitsBySize,
        )
        formationEnergy, formationEnergyUncert = self._getValueUncert(
            FITS_VFE_VALUE,
            FITS_VFE_UNCERT,
            # FMIN_FTOL * formationEnergyBySize[-1],
            self.VFEUncert,
            2,
            formationEnergyFitsBySize,
        )
        relaxationVolume, relaxationVolumeUncert = self._getValueUncert(
            FITS_VRV_VALUE,
            FITS_VRV_UNCERT,
            # FMIN_XTOL * (self.latticeConsts[0] * CELL_SIZE_MAX)**3,
            self.VRVUncert,
            2,
            relaxationVolumeFitsBySize,
        )

        # Construct Results Dictionary
        unrelaxedformationEnergyResult = OrderedDict([
            ('unrelaxed-formation-potential-energy', V(unrelaxedformationEnergy, UNIT_ENERGY, unrelaxedformationEnergyUncert)),
        ])
        formationEnergyResult = OrderedDict([
            ('relaxed-formation-potential-energy', V(formationEnergy, UNIT_ENERGY, formationEnergyUncert)),
        ])
        relaxationVolumeResult = OrderedDict([
            ('relaxation-volume', V(relaxationVolume, UNIT_VOLUME, relaxationVolumeUncert)),
        ])

        results = {"monovacancy-unrelaxed-formation-potential-energy-crystal": unrelaxedformationEnergyResult, 
                   "monovacancy-relaxed-formation-potential-energy-crystal": formationEnergyResult, 
                   "monovacancy-relaxation-volume-crystal": relaxationVolumeResult}
        return results

    def extrapolate(self, unrelaxed, relaxed):
        naSizes = np.array(self.sizes)
        naUnrelaxed = np.array(unrelaxed)
        naRelaxed = np.array(relaxed)
        unrelaxedEffectiveFormationEnergyFitsBySize = []
        effectiveFormationEnergyFitsBySize = []
        for i in range(0, len(FITS_CNT)):
            cnt = FITS_CNT[i] # Num of Data Points Used
            orders = FITS_ORDERS[i] # Orders Included
            print('Fitting with', cnt, 'points, including orders', orders)
            unrelaxedEffectiveFormationEnergyFits = []
            relaxedEffectiveFormationEnergyFits = []
            for j in range(0, len(self.sizes) - cnt + 1):
                print('Fit with data beginning', j)
                xdata = naSizes[j:(j + cnt)]
                unrelaxedEffectiveFormationEnergyFits.append(self._getFit(
                    xdata,
                    naUnrelaxed[j:(j + cnt)],
                    orders
                )[0])
                relaxedEffectiveFormationEnergyFits.append(self._getFit(
                    xdata,
                    naRelaxed[j:(j + cnt)],
                    orders
                )[0])
            unrelaxedEffectiveFormationEnergyFitsBySize.append(unrelaxedEffectiveFormationEnergyFits)
            effectiveFormationEnergyFitsBySize.append(relaxedEffectiveFormationEnergyFits)

        # Output Fitting Results
        print('\n[Fitting Results Summary]')
        print('Sizes:', self.sizes)
        print('Data Points Used:', FITS_CNT)
        print('Orders Included:\n', FITS_ORDERS)
        print('Unrelaxed Effective Formation Energy Fits By Size:\n', unrelaxedEffectiveFormationEnergyFitsBySize)
        print('Effective Formation Energy Fits By Size:\n', effectiveFormationEnergyFitsBySize)

        # Obtain Extrapolated Value and Uncertainty
        unrelaxedEffectiveFormationEnergy, unrelaxedEffectiveFormationEnergyUncert = self._getValueUncert(
            FITS_VFE_VALUE,
            FITS_VFE_UNCERT,
            # FMIN_FTOL * formationEnergyBySize[-1],
            self.VFEUncert,
            2,
            unrelaxedEffectiveFormationEnergyFitsBySize,
        )
        relaxedEffectiveFormationEnergy, relaxedEffectiveFormationEnergyUncert = self._getValueUncert(
            FITS_VFE_VALUE,
            FITS_VFE_UNCERT,
            # FMIN_FTOL * formationEnergyBySize[-1],
            self.VFEUncert,
            2,
            effectiveFormationEnergyFitsBySize,
        )
        return unrelaxedEffectiveFormationEnergy, relaxedEffectiveFormationEnergy, unrelaxedEffectiveFormationEnergyUncert, relaxedEffectiveFormationEnergyUncert
    def organize_properties(self, results):
        uncertainty = {}
        organized_props = {}
        for r in results:
            for k,v in r.items():
                if k not in organized_props:
                    organized_props[k] = {}
                if k not in uncertainty:
                    uncertainty[k] = {"values": [], "percent": []}
                for k2,v2 in v.items():
                    if k2 not in organized_props[k]:
                        organized_props[k][k2] = {}
                        organized_props[k][k2]['source-value'] = [v2['source-value']]
                    else:
                        organized_props[k][k2]['source-value'].append(v2['source-value'])
                uncertainty[k]['values'].append(v2[KEY_SOURCE_UNCERT])
                uncertainty[k]['percent'].append(v2[KEY_SOURCE_UNCERT] / v2['source-value'] * 100.)
                uncertainty[k]['prop'] = k2
                organized_props[k][k2]['source-unit'] = v2['source-unit'] # must all be same
       
        # get reservoir and host info
        res_info = {}
        host_info = {}
        for idx,i in enumerate(self.equivalent_atoms):
            ele = self.atoms.get_chemical_symbols()[i['indices'][0]]
            if self.single_element:
                res_info[ele] = {
                    'chemical_potential': self.chemical_potential,
                    'prototype_label': self.prototype_label
                }
            else:
                res_info[ele] = {
                    'chemical_potential': self.reservoir_info[ele][0]["binding-potential-energy-per-atom"]["source-value"],
                    'prototype_label': self.reservoir_info[ele][0]["prototype-label"]["source-value"]
                }
            host_info[idx] = {
                'species': ele,
                'coord': self.atoms.get_scaled_positions()[i['indices'][0]], 
                'letter': i['letter']
            }
        for k,v in organized_props.items():
            if k != 'monovacancy-relaxation-volume-crystal': # add reservoir info
                organized_props[k].setdefault('reservoir-chemical-potential', {})['source-value'] = [v['chemical_potential'] for k,v in res_info.items()]
                organized_props[k].setdefault('reservoir-chemical-potential', {})['source-unit'] = UNIT_ENERGY
                organized_props[k].setdefault('reservoir-prototype-label', {})['source-value'] =   [v['prototype_label'] for k,v in res_info.items()]
                # set host info
            organized_props[k].setdefault('vacancy-wyckoff-coordinates', {})['source-value'] = [v['coord'] for k,v in host_info.items()]
            organized_props[k].setdefault('vacancy-wyckoff-species', {})['source-value'] = [v['species'] for k,v in host_info.items()]
            organized_props[k].setdefault('vacancy-wyckoff-letter', {})['source-value'] = [v['letter'] for k,v in host_info.items()]
            organized_props[k].setdefault('host-primitive-cell', {})['source-value'] = self.atoms.get_cell()[:,:]
            organized_props[k].setdefault('host-primitive-cell', {})['source-unit'] = UNIT_LENGTH

        return organized_props, uncertainty

    def _resolve_dependencies(self, material, **kwargs):
        import kimvv
        print("Resolving dependencies...")
        # relax structure
        ecs_test = kimvv.EquilibriumCrystalStructure(self._calc)
        ecs_results = ecs_test(material)
        for result in ecs_results:
            if result["property-id"].endswith("crystal-structure-npt"):
                material_relaxed = result
                break
        # get reservoir info
        gse_test = kimvv.GroundStateCrystalStructure(self._calc)
        reservoir_info = {}
        for ele in result['stoichiometric-species']['source-value']:
            results = gse_test(ele)
            reservoir_info[ele] = results
        kwargs['reservoir_info'] = reservoir_info
        return material_relaxed, kwargs    

if __name__ == "__main__":
    from ase.build import bulk
    from kim_tools import query_crystal_structures
    kim_model_name = "EAM_IMD_SchopfBrommerFrigan_2012_AlMnPd__MO_878712978062_003"
    #kim_model_name = "EAM_Dynamo_AcklandTichyVitek_1987_Ni__MO_977363131043_005"
    list_of_queried_structures = query_crystal_structures(
        kim_model_name=kim_model_name,
        stoichiometric_species=['Al', 'Pd'],
        prototype_label="A2B_cF12_225_c_a",
    )
    print (type(list_of_queried_structures),list_of_queried_structures)
    test = TestDriver(kim_model_name)
    test(list_of_queried_structures[0])
    #test(bulk('Ni'))
    test.write_property_instances_to_file()
