from kim_tools import KIMTestDriver, CrystalGenomeTestDriver, aflow_util, KIMTestDriverError
from ase import Atoms
from typing import Any, Optional, List, Union, Dict, IO
import numpy as np

from ase.build import bulk
from ase.optimize import FIRE
from ase.spacegroup import get_basis

from kim_query import raw_query

import os
from scipy.optimize import fmin
import sys
import re
import json
import math
from collections import OrderedDict

from helper_functions import V
import string


# TODO: Check how many of these I actually  use
# Parameters for Production
FIRE_LOG = 'fire.log'
FIRE_MAX_STEPS = 50
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

# Parameters for Debugging
#FIRE_MAX_STEPS = 200
#FIRE_TOL = 1e-3 # absolute
#FMIN_FTOL = 1e-3 # relative
#FMIN_XTOL = 1e-5 # relative

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
SPACE_GROUPS = {
    'fcc': 'Fm-3m',
    'bcc': 'Im-3m',
    'sc': 'Pm-3m',
    'diamond': 'Fd-3m',
    'hcp': 'P63/mmc',
}
WYCKOFF_CODES = {
    'fcc': ['4a'],
    'bcc': ['2a'],
    'sc': ['1a'],
    'diamond': ['8a'],
    'hcp': ['2d'],
}
WYCKOFF_SITES = {
    'fcc': [[0.0, 0.0, 0.0]],
    'bcc': [[0.0, 0.0, 0.0]],
    'sc': [[0.0, 0.0, 0.0]],
    'diamond': [[0.0, 0.0, 0.0]],
    'hcp': [[2.0 / 3.0, 1.0 / 3.0, 0.25]],
}


class TestDriver(CrystalGenomeTestDriver):

    def _calculate(self, **kwargs):
        if len(np.unique(self.atoms.get_atomic_numbers()))>1:
            self.unary = False
        else:
            self.unary = True
        # symmetry stuff    
        sg_kinds = self.atoms.arrays['spacegroup_kinds']
        self.chemical_symbols = self.atoms.get_chemical_symbols() 
        sorted_symbols = sorted(np.unique(self.chemical_symbols))
        self.letters = {}
        for it, i in enumerate(sorted_symbols):
            self.letters[str(i)] = string.ascii_lowercase[it]
        self.atoms.info['basis'] = get_basis(self.atoms,self.atoms.info['spacegroup'])
        self.atoms.info['sg_symbol'] = self.atoms.info['spacegroup'].symbol.replace(' ','')
        _, self.unique_idxs, self.multiplicities = np.unique(sg_kinds,return_index=True, return_counts=True)

        for i in range(len(self.unique_idxs)):
            res = self.getResults(i)
            for k,r in res.items():
            # TODO: set up property instances
                self._add_property_instance_and_common_crystal_genome_keys(k,
                                                                   write_stress=False, write_temp=False)
                for k2,v in r.items():
                    if isinstance(v,OrderedDict):
                        if 'source-unit' in v:
                            self._add_key_to_current_property_instance(k2, v['source-value'], v['source-unit'])
                        else:
                            self._add_key_to_current_property_instance(k2, v['source-value'])
                    else:
                        self._add_key_to_current_property_instance(k2, v)


    # First 3 functions could be moved into utility 
    def _createSupercell(self, size):
        atoms = self.atoms.copy()
        atoms.set_calculator(self._calc)
        atoms *= (size, size, size)
        return atoms

    def _cellVector2Cell(self, cellVector):
        # Reconstruct cell From cellVector
        cell = [
            [cellVector[0], 0, 0],
            [cellVector[1], cellVector[2], 0],
            [cellVector[3], cellVector[4], cellVector[5]]
        ]
        return cell

    def _cell2CellVector(self, cell):
        # Extract cellVector From cell
        # For reducing degree of freedom during relaxation
        cellVector = [
            cell[0, 0],
            cell[1, 0],
            cell[1, 1],
            cell[2, 0],
            cell[2, 1],
            cell[2, 2],
        ]
        return cellVector
    # TODO: remove numAtoms and replace with chemical potential
    # Evf = Ev - E0 + mu, where mu is chemical potential of removed element
    # query OpenKIM for lowest energy/atom structure to find mu
    # TODO: Investigate if leaving unary system  as is, i.e, Evf = Ev - (N-1)/N*E0
    def _getVFE(self, cellVector, atoms, enAtoms, numAtoms):
        newCell = self._cellVector2Cell(cellVector)
        atoms.set_cell(newCell, scale_atoms = True)
        enAtomsWithVacancy = atoms.get_potential_energy()
        if not self.unary:
            enVacancy = enAtomsWithVacancy - enAtoms + self.chemical_potential
        else:
            enVacancy = enAtomsWithVacancy - enAtoms * (numAtoms - 1) / numAtoms
        return enVacancy

    def _getResultsForSize(self, size, idx):
        # Setup Environment
        unrelaxedCell = self.atoms.get_cell() * size
        unrelaxedCellVector = self._cell2CellVector(unrelaxedCell)
        atoms = self._createSupercell(size)
        numAtoms = atoms.get_number_of_atoms()
        enAtoms = atoms.get_potential_energy()
        unrelaxedCellEnergy = enAtoms
        unrelaxedCellVolume = np.abs(np.linalg.det(unrelaxedCell))
        print('\nSupercell Size:\n', size)
        print('Unrelaxed Cell:\n', unrelaxedCell)
        print('Unrelaxed Cell Vector:\n', unrelaxedCellVector)
        print('Unrelaxed Cell Energy:\n', unrelaxedCellEnergy)

        # Create Vacancy 
        print (atoms[self.unique_idxs[idx]])
        del atoms[self.unique_idxs[idx]]
        enAtomsWithVacancy = atoms.get_potential_energy()

        print('Energy of Unrelaxed Cell With Vacancy:\n', enAtomsWithVacancy)
        if not self.unary:
            enVacancyUnrelaxed = enAtomsWithVacancy - enAtoms + self.chemical_potential
        else:
            enVacancyUnrelaxed = enAtomsWithVacancy - enAtoms * (numAtoms - 1) / numAtoms

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
            # dyn = FIRE(atoms, logfile = FIRE_LOG)
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

    # Possibly move 2 below functions to utility
    def _getUnitVector(self, vec):
        return vec / np.linalg.norm(vec)

    def _getAngle(self, vec1, vec2):
        # Get acute angle between two vectors in degrees (always between 0 - 90)
        vec1Unit = self._getUnitVector(vec1)
        vec2Unit = self._getUnitVector(vec2)
        angle = np.arccos(np.dot(vec1Unit, vec2Unit))
        if np.isnan(angle):
            return 0.0
        angle = angle * 180.0 / np.pi
        # if angle < 0:
            # return 180.0 + angle
        return angle

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

        #TODO: Populate reservoir information if necessary
        #grab chemical potential
        if not self.unary:
            query_result = raw_query(
                query={
                    "meta.type": "tr",
                    # TODO: change below query
                    "property-id": "tag:staff@noreply.openkim.org,2023-02-21:property/binding-energy-crystal",
                    "meta.subject.extended-id": self.model_name,
                    "stoichiometric-species.source-value":{
                        "$size": 1,
                        "$all": [self.atoms[self.unique_idxs][idx].symbol]
                    },
                },
                fields={
                    "binding-potential-energy-per-atom": 1,
                    "short-name.source-value":1,
                    },
                database="data", limit=0, sort=[["binding-potential-energy-per-atom", 1]])
            self.chemical_potential = query_result[0]["binding-potential-energy-per-atom"]["source-value"] 
            print (query_result[0])
            print ('Chemical Potential', self.chemical_potential)


        unitBulk = self.atoms
        unitCell = unitBulk.get_cell()

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

        # Eric->Keeping below original comments in case they are useful in the future
        # Data for skipping computation when debugging extrapolation and output
        # sizes = [3, 4, 5, 6, 7, 8, 9]
        # formationEnergyBySize = [
                # 0.6721479768766585 ,
                # 0.67372899358906579,
                # 0.67440913973746319,
                # 0.6747228089247983 ,
                # 0.67488432759455463,
                # 0.6749755557248136 ,
                # 0.67503091578691965,
        # ]
        # relaxationVolumeBySize = [
                # 8.2664887840680876,
                # 8.2145358736270282,
                # 8.2008345712674782,
                # 8.1943833508903481,
                # 8.1916426682910242,
                # 8.1898981954873307,
                # 8.1889297673697001,
        # ]

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
        # hack to keep same ordering 
        hostInfo = OrderedDict([
            ('host-cauchy-stress', V([0, 0, 0, 0, 0, 0], UNIT_PRESSURE)),
            ('host-removed-atom', V(idx)),
        ])

        # TODO: Check what this is
        reservoirInfo = OrderedDict([
            ('reservoir-cohesive-potential-energy', V(-unitBulk.get_potential_energy()/unitBulk.get_global_number_of_atoms(), UNIT_ENERGY)),
        ])
        
        if self.short_name is not None:
            hostInfo.update({'host-short-name': V(self.short_name)})


        hostInfo.update(OrderedDict([
            ('host-a', V(np.linalg.norm(unitCell[0]), UNIT_LENGTH)),
            ('host-b', V(np.linalg.norm(unitCell[1]), UNIT_LENGTH)),
            ('host-c', V(np.linalg.norm(unitCell[2]), UNIT_LENGTH)),
            ('host-alpha', V(self._getAngle(unitCell[1], unitCell[2]), UNIT_ANGLE)),
            ('host-beta', V(self._getAngle(unitCell[2], unitCell[0]), UNIT_ANGLE)),
            ('host-gamma', V(self._getAngle(unitCell[0], unitCell[1]), UNIT_ANGLE)),
            ('host-space-group', V(self.atoms.info['sg_symbol'])),
            ('host-wyckoff-multiplicity-and-letter', V([str(self.multiplicities[idx])+self.letters[self.chemical_symbols[idx]]])),
            ('host-wyckoff-coordinates', V([self.atoms.info['basis'][idx]])),
            ('host-wyckoff-species', V([self.chemical_symbols[idx]])),
        ]))

        # TODO: Probably remove reservoir info related to wyckoff stuff. We assume chemical potential is E/n. 
        if self.unary:
            if self.short_name is not None:
                reservoirInfo.update(OrderedDict([
                    ('reservoir-short-name', V(self.short_name))
                ]))
            reservoirInfo.update(OrderedDict([
                ('reservoir-cauchy-stress', V([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], UNIT_PRESSURE)),
                ('reservoir-a', V(np.linalg.norm(unitCell[0]), UNIT_LENGTH)),
                ('reservoir-b', V(np.linalg.norm(unitCell[1]), UNIT_LENGTH)),
                ('reservoir-c', V(np.linalg.norm(unitCell[2]), UNIT_LENGTH)),
                ('reservoir-alpha', V(self._getAngle(unitCell[1], unitCell[2]), UNIT_ANGLE)),
                ('reservoir-beta', V(self._getAngle(unitCell[2], unitCell[0]), UNIT_ANGLE)),
                ('reservoir-gamma', V(self._getAngle(unitCell[0], unitCell[1]), UNIT_ANGLE)),
                ('reservoir-space-group', V(self.atoms.info['sg_symbol'])),
                #('reservoir-wyckoff-multiplicity-and-letter', V([str(self.multiplicities[idx])+self.letters[self.chemical_symbols[idx]]])),
                #('reservoir-wyckoff-coordinates', V([self.atoms.info['basis'][idx]])),
                #('reservoir-wyckoff-species', V([self.chemical_symbols[idx]])),
            ]))
        else: 
            # TODO: Put in proper reservoir info
            pass
        unrelaxedformationEnergyResult.update(hostInfo)
        unrelaxedformationEnergyResult.update(reservoirInfo)
        formationEnergyResult.update(hostInfo)
        formationEnergyResult.update(reservoirInfo)
        relaxationVolumeResult.update(hostInfo)

        results = {"monovacancy-neutral-unrelaxed-formation-potential-energy-crystal-npt": unrelaxedformationEnergyResult, 
                   "monovacancy-neutral-relaxed-formation-potential-energy-crystal-npt": formationEnergyResult, 
                   "monovacancy-neutral-relaxation-volume-crystal-npt": relaxationVolumeResult}
        return results

if __name__ == "__main__":
    from ase.build import bulk
    test = TestDriver('EAM_Dynamo_ZhouWadleyJohnson_2001_Al__MO_049243498555_000')
    atoms = bulk('Al','fcc',a=4.04)
    #test(atoms)
    d = {"stoichiometric_species": ["Al"], "prototype_label": "A_cF4_225_a", "parameter_values_angstrom": [4.081654928624631], "rebuild_atoms": False}
    test(**d) 
    test.write_property_instances_to_file()
