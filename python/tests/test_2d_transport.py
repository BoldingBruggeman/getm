import sys
import argparse

import numpy
import matplotlib.pyplot

import pygetm
import pygetm.domain

def check_range(name, values, rtol=1e-12, atol=1e-12, target_value=None):
    print('  %s... ' % name, end='', flush=True)
    absrange = values.max() - values.min()
    mean = values.mean()
    crit = rtol * abs(mean) + atol
    valid = absrange < crit
    result = 'OK' if valid else 'FAILED because field is not homogeneous'
    relrange = 0 if absrange == 0 else absrange / abs(mean)
    if valid and target_value is not None:
        valid = abs(mean - target_value) < rtol * abs(target_value) + atol
        if not valid:
            result = 'FAILED to match target value %.6g' % target_value
    print('%s (mean %.6g, spread %.6g, relative spread %.6g)' % (result, mean, absrange, relrange))
    if not valid:
        plot('%s.png' % name, values)
    return valid

def plot(name, values):
    fig, ax = matplotlib.pyplot.subplots()
    pc = ax.pcolormesh(values)
    fig.colorbar(pc)
    fig.savefig(name, dpi=300)

def test(name, periodic_x=False, periodic_y=False, tau_x=0., tau_y=0., timestep=10., ntime=360, apply_bottom_friction=False):
    print('%s, tau_x = %s, tau_y = %s...' % (name, tau_x, tau_y), flush=True)

    # Set up rectangular domain (all points unmasked)
    domain = pygetm.domain.Domain.create_cartesian(500.*numpy.arange(100), 500.*numpy.arange(30), 1, f=0, H=50, periodic_x=periodic_x, periodic_y=periodic_y)
    sim = pygetm.Simulation(domain, runtype=1, advection_scheme=1, apply_bottom_friction=apply_bottom_friction)

    # Idealized surface forcing
    tausx, tausx_ = domain.U.array(fill=tau_x)
    tausy, tausy_ = domain.V.array(fill=tau_y)
    sp, sp_ = domain.T.array(fill=0.)

    dist_U = domain.distribute(sim.U_)
    dist_V = domain.distribute(sim.V_)
    dist_U.update_halos()
    dist_V.update_halos()
    for istep in range(ntime):
        sim.update_surface_pressure_gradient(domain.T.z_, sp_)
        sim.uv_momentum_2d(timestep, tausx_, tausy_, sim.dpdx_, sim.dpdy_)
        dist_U.update_halos()
        dist_V.update_halos()
        sim.update_sealevel(timestep, sim.U_, sim.V_)
        sim.update_depth()

    rho0 = 1025.
    success = check_range('U', sim.U, target_value=None if apply_bottom_friction else 3600 * tau_x / rho0)
    success = check_range('V', sim.V, target_value=None if apply_bottom_friction else 3600 * tau_y / rho0) and success
    success = check_range('z', domain.T.z, target_value=0) and success
    return success

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply_bottom_friction', action='store_true')
    args = parser.parse_args()

    success = test('Periodic in x', periodic_x=True, tau_x=0.01, apply_bottom_friction=args.apply_bottom_friction)
    success = test('Periodic in x', periodic_x=True, tau_x=-0.01, apply_bottom_friction=args.apply_bottom_friction) and success
    success = test('Periodic in y', periodic_y=True, tau_y=0.01, apply_bottom_friction=args.apply_bottom_friction) and success
    success = test('Periodic in y', periodic_y=True, tau_y=-0.01, apply_bottom_friction=args.apply_bottom_friction) and success
    success = test('Periodic in x and y', periodic_x=True, periodic_y=True, tau_x=0.01, tau_y=0.01, apply_bottom_friction=args.apply_bottom_friction) and success
    success = test('Periodic in x and y', periodic_x=True, periodic_y=True, tau_x=0.01, tau_y=-0.01, apply_bottom_friction=args.apply_bottom_friction) and success
    success = test('Periodic in x and y', periodic_x=True, periodic_y=True, tau_x=-0.01, tau_y=0.01, apply_bottom_friction=args.apply_bottom_friction) and success
    success = test('Periodic in x and y', periodic_x=True, periodic_y=True, tau_x=-0.01, tau_y=-0.01, apply_bottom_friction=args.apply_bottom_friction) and success
    if not success:
        sys.exit(1)