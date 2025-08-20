from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

from tudatpy.interface import spice
from tudatpy.numerical_simulation import environment_setup, propagation_setup, environment
from tudatpy import numerical_simulation
from tudatpy.util import result2array

@dataclass
class Scenario:
    start_epoch: float
    end_epoch: float
    step: float = 10.0
    target: str = "Triton"
    center: str = "Neptune"

@dataclass
class NeptuneSH:
    # Fully-normalized C̄20, C̄40 (from Jacobson 2009)
    Cbar20: float
    Cbar40: float
    #ref_radius_m: float

@dataclass
class TudatOrbitRunner:
    # Minimal kernel list you already have locally
    kernel_paths: List[str]
    # Bodies we want in the env
    bodies_to_create: List[str] = field(default_factory=lambda: ["Sun", "Neptune", "Triton"])
    # Global frame (Neptune-centered for Triton propagation)
    global_origin: str = "Neptune"
    global_orientation: str = "J2000"

    # Acceleration toggles
    use_neptune_point_mass: bool = True
    use_sun_point_mass: bool = True
    use_neptune_sh: bool = True
    neptune_sh: Optional[NeptuneSH] = None
    
    # (Optional) dependent variables
    save_keplerian: bool = True
    save_total_acc: bool = True
    #save_rsw_frame: bool = True
    # internal
    _bodies: Optional[environment.SystemOfBodies] = None

    def load_kernels(self):
        spice.load_standard_kernels()
        for k in self.kernel_paths:
            spice.load_kernel(k)

    def _make_body_settings(self) -> environment_setup.BodyListSettings:
        body_settings = environment_setup.get_default_body_settings(
            self.bodies_to_create,
            self.global_origin,
            self.global_orientation
        )
        # Inject Neptune SH if requested
        if self.use_neptune_sh and self.neptune_sh is not None:
            ref_radius_m = spice.get_body_properties("Neptune", "RADII", 3)[0]*1e3  # meters
            mu_N = spice.get_body_gravitational_parameter("Neptune")
            lmax, mmax = 4, 0
            Cbar = np.zeros((lmax+1, lmax+1))
            Sbar = np.zeros_like(Cbar)
            Cbar[2,0] = self.neptune_sh.Cbar20
            Cbar[4,0] = self.neptune_sh.Cbar40
            body_settings.get("Neptune").gravity_field_settings = environment_setup.gravity_field.spherical_harmonic(
                gravitational_parameter=mu_N,
                reference_radius=ref_radius_m,
                normalized_cosine_coefficients=Cbar,
                normalized_sine_coefficients=Sbar,
                associated_reference_frame="IAU_Neptune"
            )
        return body_settings

    def build_environment(self):
        self._bodies = environment_setup.create_system_of_bodies(self._make_body_settings())

    def build_acceleration_models(self, scenario: Scenario):
        assert self._bodies is not None

        acts: Dict[str, List] = {}

        for body_name in self.bodies_to_create:
            if body_name == scenario.target:
                continue  # no self-acceleration
            if self._bodies.get(body_name) is None:
                continue  # not created

            # Special case: Neptune
            if body_name == "Neptune":
                if self.use_neptune_sh and self.neptune_sh is not None:
                    acts.setdefault("Neptune", []).append(
                        propagation_setup.acceleration.spherical_harmonic_gravity(4, 0)
                    )
                elif self.use_neptune_point_mass:
                    acts.setdefault("Neptune", []).append(
                        propagation_setup.acceleration.point_mass_gravity()
                    )
                continue

            # Special case: Sun
            if body_name == "Sun" and self.use_sun_point_mass:
                acts.setdefault("Sun", []).append(
                    propagation_setup.acceleration.point_mass_gravity()
                )
                continue

            # Default for "most" other bodies: point-mass gravity
            acts.setdefault(body_name, []).append(
                propagation_setup.acceleration.point_mass_gravity()
            )

        acceleration_settings = { scenario.target: acts }

        return propagation_setup.create_acceleration_models(
            self._bodies,
            acceleration_settings,
            [scenario.target],
            [scenario.center],
        )


    def initial_state_from_spice(self, scenario: Scenario) -> np.ndarray:
        return spice.get_body_cartesian_state_at_epoch(
            target_body_name=scenario.target,
            observer_body_name=scenario.center,
            reference_frame_name=self.global_orientation,
            aberration_corrections="none",
            ephemeris_time=scenario.start_epoch
        )

    def dependent_vars(self, scenario: Scenario):
        dv = []
        if self.save_total_acc:
            dv.append(propagation_setup.dependent_variable.total_acceleration(scenario.target))
        if self.save_keplerian:
            dv.append(propagation_setup.dependent_variable.keplerian_state(scenario.target, scenario.center))
       # if self.save_rsw_frame:
       #    dv.append(propagation_setup.dependent_variable.rsw_to_inertial_rotation_matrix(scenario.target, scenario.center))
        
        return dv

    def run(self, scenario: Scenario):
        self.load_kernels()
        self.build_environment()

        acceleration_models = self.build_acceleration_models(scenario)
        initial_state = self.initial_state_from_spice(scenario)

        integ = propagation_setup.integrator.runge_kutta_fixed_step(
            scenario.step, coefficient_set=propagation_setup.integrator.CoefficientSets.rk_4
        )
        term = propagation_setup.propagator.time_termination(scenario.end_epoch)

        prop = propagation_setup.propagator.translational(
            [scenario.center],
            acceleration_models,
            [scenario.target],
            initial_state,
            scenario.start_epoch,
            integ,
            term,
            output_variables=self.dependent_vars(scenario)
        )

        sim = numerical_simulation.create_dynamics_simulator(self._bodies, prop)

        states = result2array(sim.propagation_results.state_history)
        deps   = result2array(sim.propagation_results.dependent_variable_history)
        return states, deps
