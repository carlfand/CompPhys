import numpy as np
from matplotlib import pyplot as plt


class Particle:
    def __init__(self, mass: float, charge: float, position : np.ndarray, velocity : np.ndarray):
        self.r = position
        self.v = velocity
        self.m = mass
        self.q = charge

    def get_r(self) -> np.ndarray:
        return self.r

    def utdate_position(self, delta_r: np.ndarray):
        self.r += delta_r

    def utdate_velocity(self, delta_v: np.ndarray):
        self.v += delta_v


class PenningTrap:
    """Assumes a B-field pointing in the z-direction of magnitude B_0 and a E-field described by the electric
    potential V = V_0(t) / (2 d^2) * (2 z^2 - x^2 - y^2)."""
    def  __init__(self, B_0, V_0, d, *particles):
        # TODO: sensible units
        self.k_e = 1  # Coloumb's constant. Setting it to 1
        self.B_0 = B_0
        self.V_0 = V_0
        self.d_square = d**2
        self.V_by_d_sq = V_0 / self.d_square
        self.particles = [*particles]  # Container for particle objects.
        self.N_particles = len(particles)
        self.rs = np.array([part.r for part in particles])    # create Nx3 D array of positions of all the particles
        self.vs = np.array([part.v for part in particles])    # create Nx3 D array of velocities of all the particles
        self.t = 0
        self.delta_rs = np.zeros((self.N_particles, 3))     # Needed for intermediate steps in RK4
        self.delta_vs = np.zeros((self.N_particles, 3))     # Needed for intermediate steps in RK4

    def __repr__(self):
        pos_str = "Particle positions (x, y, z):\n{}".format(self.rs)
        vel_str = "\nParticle velocities (v_x, v_y, v_z): \n{}".format(self.vs)
        return pos_str

    def add_particle(self, part: Particle):
        """Adds a particle object to the list of particles and its position and velocity to self.rs and self.vs."""
        self.particles.append(part)
        self.N_particles += 1
        self.rs = np.array([*self.rs, part.r])    # Appending r to self.rs
        self.vs = np.array([*self.vs, part.v])    # Appending v to self.vs
        self.delta_rs = np.zeros((self.N_particles, 3))
        self.delta_vs = np.zeros((self.N_particles, 3))

    def remove_particle_n(self, n: int):
        """Removes particle at index n."""
        del self.particles[n]   # Delete particle at place n
        self.rs = np.array([self.rs[i] for i in range(self.N_particles) if i != n])   # Delete position at index n
        self.vs = np.array([self.vs[i] for i in range(self.N_particles) if i != n])  # Delete position at index n
        self.N_particles -= 1
        self.delta_rs = np.zeros((self.N_particles, 3))
        self.delta_vs = np.zeros((self.N_particles, 3))

    def external_forces(self) -> np.ndarray:
        """Calculates the contribution of external forces divided by the particle mass.
        I.e. returns the acceleration.
        Note that it takes into consideration a contribution from self.delta_rs and self.delta_vs."""
        ext_a = np.zeros((self.N_particles, 3))  # The external contribution to the acceleration
        for i, particle in enumerate(self.particles):
            q_by_m = particle.q / particle.m
            omega_0 = self.B_0 * q_by_m
            omega_z_square = 2 * self.V_by_d_sq * q_by_m
            x, y, z = particle.r + self.delta_rs[i]
            v_x, v_y, _ = particle.v + self.delta_vs[i]
            ext_a[i, :] = np.array([omega_0 * v_y + 1/2 * omega_z_square * x, - omega_0 * v_x + 1 / 2 * omega_z_square * y,
                           -omega_z_square * z])
        return ext_a

    def part_int(self, n):
        """Returns the sum of the forces of all the other particles on particle number n,
        divided by the particle mass m_i."""
        q_by_m, r_n = self.particles[n].q / self.particles[n].m, self.particles[n].r + self.delta_rs[n]
        force = np.zeros(3)
        for i, particle in enumerate(self.particles):
            if i == n:
                continue
            else:
                delta_r = r_n - (self.particles[i].r + self.delta_rs[i])
                delta_r_norm = np.linalg.norm(delta_r)
                force += self.k_e * q_by_m * self.particles[i].q * delta_r / delta_r_norm**3
        return force

    def internal_forces(self):
        """Calls part_int for every particle. I.e. returns an array of acceleration vectors"""
        int_a = np.zeros((self.N_particles, 3))
        for i in range(self.N_particles):
            int_a[i, :] = self.part_int(i)
        return int_a
        # return np.array([self.part_int(i) for i in range(self.N_particles)])

    def f(self):
        """Calculates the derivative of we need in the propagator. Returns vs (Nx3 array of velocities)
        and as (Nx3 array of accelerations as calculated by self.external_forces and self.internal_forces)."""
        as_internal = self.internal_forces()
        print(as_internal)
        as_external = self.external_forces()
        print(as_external)
        return np.array([self.vs + self.delta_vs, as_internal + as_external])

    def rk4(self, h):
        k1 = h * self.f()
        self.delta_rs = k1[0] / 2   # delta_rs and delta_vs are taken into account
        self.delta_vs = k1[1] / 2   # when vi calculate f
        self.t += h / 2
        k2 = h * self.f()
        self.delta_rs = k2[0] / 2
        self.delta_vs = k2[1] / 2
        k3 = h * self.f()
        self.delta_rs = k3[0]
        self.delta_vs = k3[1]
        self.t += h / 2
        k4 = h * self.f()
        self.delta_rs = np.zeros((self.N_particles, 3))     # Resetting delta_rs
        self.delta_vs = np.zeros((self.N_particles, 3))     # and delta vs
        return np.array([self.rs + 1 / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
                         self.vs + 1 / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])])

    def euler(self, h):
        rs_add, vs_add = h * self.f()
        return np.array([self.rs + rs_add, self.vs + vs_add])

    def propagate_steps(self, n, h, rk4=True):
        rs_prop = np.zeros((n, self.N_particles, 3))
        for i in range(n):
            # print("Step {}: {}".format(i, self))
            rs_prop[i, :, :] = self.rs
            if rk4:
                self.rs, self.vs = self.rk4(h)
            else:
                self.rs, self.vs = self.euler(h)
            for j, part in enumerate(self.particles):
                part.r = self.rs[j]
                part.v = self.vs[j]
        return rs_prop




r = np.array([1.0, 0.0, 0.5])
epsilon = 0.01
r_2 = r + epsilon
v = np.array([0.0, 0.5, 0.0])
r_3 = np.array([-1.0, 0.0, -0.5])
v_3 = np.array([0.0, -0.5, 0.0])
m, q = 2.0, 1.0
p_0 = Particle(m, q, r, v)
p_1 = Particle(m, q, r_3, v_3)
p_2 = Particle(m, q, 1/2 * r, v_3)

V_0, B_0, d = 2.5, 1.0, 10.0
trap_0 = PenningTrap(B_0, V_0, d, p_0)
trap_1 = PenningTrap(B_0, V_0, d, p_0)
trap_1.add_particle(p_1)
trap_1.add_particle(p_2)

# print(trap_0)

# trap_0.add_particle(p_1)

# trap_0.propagate_steps(50, 0.1, rk4=False)
n = 1000
rs_prop = trap_1.propagate_steps(n, 0.05, rk4=True)

def plot_positions(arr):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))
    for i in range(len(rs_prop[0, :, 0])):
        ax1.plot(arr[:, i, 0], arr[:, i, 1])
        ax2.plot(np.linspace(0, 1, n), arr[:, i, 2])
    plt.show()

plot_positions(rs_prop)

def analytic_solution(t: np.ndarray, r_0: np.ndarray, v_0: np.ndarray) -> np.ndarray:
    xs = 0

