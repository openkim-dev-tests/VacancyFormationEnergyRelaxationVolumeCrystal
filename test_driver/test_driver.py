from kim_tools import SingleCrystalTestDriver
from kim_tools.aflow_util.core import get_atom_indices_for_each_wyckoff_orb
import numpy as np
from ase.geometry.cell import cellpar_to_cell
from ase.optimize import FIRE
from scipy.optimize import fmin
import sys
import math
from collections import OrderedDict
import kimvv

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
KEY_SOURCE_VALUE = 'source-value'
KEY_SOURCE_UNIT = 'source-unit'
KEY_SOURCE_UNCERT = 'source-std-uncert-value'
UNIT_ENERGY = 'eV'
UNIT_LENGTH = 'angstrom'
UNIT_ANGLE = 'degree'
UNIT_PRESSURE = 'GPa'
UNIT_VOLUME = UNIT_LENGTH + '^3'

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
        self.equivalent_atoms = get_atom_indices_for_each_wyckoff_orb(prototype_label)

        if DYNAMIC_CELL_SIZE == True:
            numAtoms = self.atoms.get_number_of_atoms()
            factor = math.pow(8 / numAtoms, 0.333)
            global CELL_SIZE_MIN, CELL_SIZE_MAX
            CELL_SIZE_MIN = int(math.ceil(factor * CELL_SIZE_MIN))
            CELL_SIZE_MAX = CELL_SIZE_MIN + 2
            print('CELL_SIZE_MIN:', CELL_SIZE_MIN)
            print('CELL_SIZE_MAX:', CELL_SIZE_MAX)
            print('Smallest System Size:', numAtoms * CELL_SIZE_MIN**3)
            print('Largest System Size:', numAtoms * CELL_SIZE_MAX**3)

        results = []
        for wkof in self.equivalent_atoms:
            idx = wkof['indices'][0] 
            results.append(self.getResults(idx))
        organized_props = self.organize_properties(results)
        for k,v in organized_props.items():
        
            self._add_property_instance_and_common_crystal_genome_keys(k,
                                                                   write_stress=True, write_temp=True)
            for k2,v2 in v.items():
                if 'source-unit' in v2:
                    self._add_key_to_current_property_instance(k2, v2['source-value'], v2['source-unit'])
                else:
                    self._add_key_to_current_property_instance(k2, v2['source-value'])

    def _createSupercell(self, size):
        atoms = self.atoms.copy()
        atoms.set_calculator(self._calc)
        atoms *= (size, size, size)
        return atoms

    def _cellVector2Cell(self, cellVector):
        cell = cellpar_to_cell(cellVector)
        return cell
    
    # Evf = Ev - E0 + mu, where mu is chemical potential of removed element
    def _getVFE(self, cellVector, atoms, enAtoms, numAtoms):
        newCell = self._cellVector2Cell(cellVector)
        atoms.set_cell(newCell, scale_atoms = True)
        enAtomsWithVacancy = atoms.get_potential_energy()
        enVacancy = enAtomsWithVacancy - enAtoms + self.chemical_potential
        return enVacancy

    def _getResultsForSize(self, size, idx):
        # Setup Environment
        unrelaxedCell = self.atoms.get_cell() * size
        atoms = self._createSupercell(size)
        unrelaxedCellVector = atoms.get_cell_lengths_and_angles() 
        numAtoms = atoms.get_number_of_atoms()
        enAtoms = atoms.get_potential_energy()
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
        enVacancy = 0

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
                args = (atoms, enAtoms, numAtoms),
                ftol = FMIN_FTOL,
                xtol = FMIN_XTOL,
                full_output = True,
            )[:2]

            # Convergence Requirement Satisfied
            if abs(tmpEnVacancy - enVacancy) < VFE_TOL and dyn.get_number_of_steps() < 1:
                dyn.run(fmax = FIRE_TOL * EPS, steps = FIRE_UNCERT_STEPS)
                tmpCellVector, tmpEnVacancy = fmin(
                    self._getVFE,
                    relaxedCellVector,
                    args = (atoms, enAtoms, numAtoms),
                    ftol = FMIN_FTOL * EPS,
                    xtol = FMIN_XTOL * EPS,
                    full_output = True,
                )[:2]
                self.VFEUncert = np.abs(tmpEnVacancy - enVacancy)
                enVacancy = tmpEnVacancy
                oldVolume = np.linalg.det(self._cellVector2Cell(relaxedCellVector))
                newVolume = np.linalg.det(self._cellVector2Cell(tmpCellVector.tolist()))
                self.VRVUncert = np.abs(newVolume - oldVolume)
                relaxedCellVector = tmpCellVector.tolist()
                break

            enVacancy = tmpEnVacancy
            relaxedCellVector = tmpCellVector.tolist()

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

        return enVacancyUnrelaxed, relaxedCellVector, enVacancy, relaxationVolume

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

    def getResults(self, idx):
        # grab chemical potential
        # add back isolated atom energy
        self.chemical_potential = self.reservoir_info[self.atoms[idx].symbol][0]["binding-potential-energy-per-atom"]["source-value"] + self.get_isolated_energy_per_atom(self.atoms[idx].symbol) 
        print ('Chemical Potential', self.chemical_potential)


        unitBulk = self.atoms

        # Calculate VFE and VRV for Each Size
        sizes = []
        unrelaxedformationEnergyBySize = []
        formationEnergyBySize = []
        relaxationVolumeBySize = []
        print('\n[Calculation]')
        for size in range(CELL_SIZE_MIN, CELL_SIZE_MAX + 1):
            unrelaxedFormationEnergy, relaxedCellVector, relaxedFormationEnergy, relaxationVolume = self._getResultsForSize(size, idx)
            sizes.append(size)
            unrelaxedformationEnergyBySize.append(unrelaxedFormationEnergy)
            formationEnergyBySize.append(relaxedFormationEnergy)
            relaxationVolumeBySize.append(relaxationVolume)

        print('\n[Calculation Results Summary]')
        print('Sizes:', sizes)
        print('Unrelaxed Formation Energy By Size:\n', unrelaxedformationEnergyBySize)
        print('Formation Energy By Size:\n', formationEnergyBySize)
        print('Relaxation Volume By Size:\n', relaxationVolumeBySize)

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

        results = {"monovacancy-unrelaxed-formation-potential-energy-crystal-npt": unrelaxedformationEnergyResult, 
                   "monovacancy-relaxed-formation-potential-energy-crystal-npt": formationEnergyResult, 
                   "monovacancy-relaxation-volume-crystal-npt": relaxationVolumeResult}
        return results
    
    def organize_properties(self, results):
        organized_props = {}
        for r in results:
            for k,v in r.items():
                if k not in organized_props:
                    organized_props[k] = {}
                for k2,v2 in v.items():
                    if k2 not in organized_props[k]:
                        organized_props[k][k2] = {}
                        organized_props[k][k2]['source-value'] = [v2['source-value']]
                    else:
                        organized_props[k][k2]['source-value'].append(v2['source-value'])
                organized_props[k][k2]['source-unit'] = v2['source-unit'] # must all be same
                # TODO: look at uncertainty later
       
        # get reservoir and host info
        res_info = {}
        host_info = {}
        for idx,i in enumerate(self.equivalent_atoms):
            ele = self.atoms.get_chemical_symbols()[i['indices'][0]]
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
            if k != 'monovacancy-relaxation-volume-crystal-npt': # add reservoir info
                organized_props[k].setdefault('reservoir-chemical-potential', {})['source-value'] = [v['chemical_potential'] for k,v in res_info.items()]
                organized_props[k].setdefault('reservoir-chemical-potential', {})['source-unit'] = UNIT_ENERGY
                organized_props[k].setdefault('reservoir-prototype-label', {})['source-value'] =   [v['prototype_label'] for k,v in res_info.items()]
                # set host info
            organized_props[k].setdefault('vacancy-wyckoff-coordinates', {})['source-value'] = [v['coord'] for k,v in host_info.items()]
            organized_props[k].setdefault('vacancy-wyckoff-species', {})['source-value'] = [v['species'] for k,v in host_info.items()]
            organized_props[k].setdefault('vacancy-wyckoff-letter', {})['source-value'] = [v['letter'] for k,v in host_info.items()]
            organized_props[k].setdefault('host-primitive-cell', {})['source-value'] = self.atoms.get_cell()[:,:]
            organized_props[k].setdefault('host-primitive-cell', {})['source-unit'] = UNIT_LENGTH

        return organized_props

    def _resolve_dependencies(self, material, **kwargs):
        print("Resolving dependencies...")
        # relax structure
        ecs_test = kimvv.EquilibriumCrystalStructure(self._calc)
        ecs_results = ecs_test(material)
        for result in ecs_results:
            if result["property-id"].endswith("crystal-structure-npt"):
                material_relaxed = result
                break
        # get reservoir info
        gse_test = kimvv.GroundStateEnergy(self._calc)
        reservoir_info = {}
        for ele in set(atoms.get_chemical_symbols()):
            gse_test(ele)
            # get results and populate
            result = [gse_test.property_instance]
            reservoir_info[ele] = results
        kwargs['reservoir_info'] = reservoir_info
        return material_relaxed, kwargs    

