import numpy as np
import copy
from numbalsoda import lsoda_sig, solve_ivp, lsoda
from numba import cfunc
import numba as nb
import pyqg_jax
import jax
import jax.numpy as jnp
from functools import partial
import warnings
import multiprocessing as mp
import functools

class Model:
      def __init__(self, modelparams: dict, dt: float):
            """Initialize a model object, tailored to a specific model
            Parameters
            ----------
            modelparams : dict
                A dictionary containing all the configurable parameters
                that go into the model's formulation. For example, for
                Lorenz63, the modelparams would be:
                modelparams = {'s': 10, 'r': 28, 'b': 2.666666666666666}
            dt : float
                The timestep provided by the dapyr experiment on which 
                step over. A single "step" in model time will be dt.
                For example, for Lorenz63 the dt would be 0.01.

            Other attributes can be stored in the inherited model. 
            In the LorenzModel case, I have a variable storing the number
            of CPUs to include in the multiprocessing pool for batch forecasts,
            and a function pointer pointing to the integration function in
            order to do functions.
            """            
            self.dt = dt
            self.modelparams = modelparams

      def forecast_rollout(self, x: np.ndarray, steps: int) -> tuple[np.ndarray, int]:
            """Integrate the model foreward, return all previous timesteps as well

            Parameters
            ----------
            x : np.ndarray
                A 1-D numpy array of size (Nx,), where Nx is the model state size when flattened
            steps : int
                The number of dt timesteps to forecast out to.

            Returns
            -------
            tuple[np.ndarray, int]
                np.ndarray: A 2-D numpy array of floats representing the model state at each model timestep.
                The array should be of dimensions (Nx, steps+1), where Nx is the flattened model state,
                and steps+1 is the number of time steps requests.
                The 0th index of the array should be the initial state (x) inputted into the function.

                int: An integer flagging if model integration failed for any reason.
                0 means no errors occured.
                Any other integer means model failure.
            """            
            pass

      def forecast(self, x: np.ndarray, steps: int)-> tuple[np.ndarray, int]:
            """Integrate the model forward in time by steps*dt

            Parameters
            ----------
            x : np.ndarray
                A 1-D numpy array of size (Nx,), where Nx is the model state size when flattened
            steps : int
                The number of dt timesteps to forecast out to.

            Returns
            -------
            tuple[np.ndarray, int]
                np.ndarray: A 1-D numpy array of size (Nx,), where Nx is the model state size when flattened.
                This state is the final state at time dt*steps

                int: An integer flagging if model integration failed for any reason.
                0 means no errors occured.
                Any other integer means model failure.
            """            
            sol, model_error = self.forecast_rollout(x, steps)
            return sol[:, -1], model_error
      

      def forecast_batch(self, x_ens: np.ndarray, steps: int)-> tuple[np.ndarray, np.ndarray]: 
            """Runs a forecast forward in time on a batch of Ne models states. 

            Parameters
            ----------
            x_ens : np.ndarray
                A 2-D numpy array of size (Nx,Ne), where Nx is the model state size when flattened,
                and Ne is the number of ensemble members to forecast
            steps : int
                The number of dt timesteps to forecast out to.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                np.ndarray: A 2-D numpy array of size (Nx,Ne), where Nx is the model state size when flattened,
                and Ne is the number of ensemble members to forecast. 
                The array represents the final model state of all ensemble members
                at time dt*steps

                np.ndarray: A 1-D numpy array of integers, representing
                the model integration flag for EACH ensemble member.

                All zeros: Every ensemble member integration completed successfully
            """            
            sol, model_errors = self.forecast_batch_rollout(x_ens, steps)
            return sol[:, -1, :], model_errors

      def forecast_batch_rollout(self, x_ens: np.ndarray, steps: int)-> tuple[np.ndarray, np.ndarray]:
            """_summary_

            Parameters
            ----------
            x_ens : np.ndarray
                A 2-D numpy array of size (Nx,Ne), where Nx is the model state size when flattened,
                and Ne is the number of ensemble members to forecast
            steps : int
                The number of dt timesteps to forecast out to.

            Returns
            -------
            tuple[np.ndarray, np.ndarray]
                np.ndarray: A 3-D numpy array of size (Nx, steps + 1, Ne), 
                where Nx is the model state size when flattened,
                steps + 1 is the number of timesteps between the initial
                state and the final state, and Ne is the number of 
                ensemble members to forecast.

                np.ndarray: A 1-D numpy array of integers, representing
                the model integration flag for EACH ensemble member.

                All zeros: Every ensemble member integration completed successfully
            """            
            pass
# Here's an example of me trying to inherit the Model class, then
# Implement my own forecast versions as needed
class LorenzModel(Model):
      def __init__(self, modelparams: dict, dt, numprocs: int = None):
            # Calling super() here invokes the basic initialize method
            # of the Model Class, so modelparams and dt will load
            super().__init__(modelparams, dt)
            # I added additional attribute here that's needed for integration
            # Lorenz Models
            self.funcptr = ''
            self.numprocs = numprocs if numprocs is not None else mp.cpu_count()
            # The pool is created lazily rather than
            # here, since subclasses set self.funcptr after calling this
            # __init__. Forking a pool before funcptr exists gives workers a
            # copy of memory that never had the JIT-compiled function in it,
            # which segfaults the worker and hangs pool.map forever.
            self._pool = None
            self.Nx = modelparams['Nx']

      @property
      def pool(self):
            if self._pool is None:
                  self._pool = mp.get_context('fork').Pool(self.numprocs)
            return self._pool

      def set_numprocs(self, numprocs: int):
            """Resize the pool, restarting it if one is already running."""
            self.close_pool()
            self.numprocs = numprocs

      def close_pool(self):
            if self._pool is not None:
                  self._pool.close()
                  self._pool.join()
                  self._pool = None

      def __del__(self):
            if getattr(self, '_pool', None) is not None:
                  self._pool.terminate()

      @staticmethod
      def _integrate(x, steps, dt, funcptr):
            model_error = 0
            tspan = np.array([0, dt*steps])
            teval = np.linspace(0, tspan[1], steps + 1)
            usol = copy.deepcopy(x)
            #Try with a Runga Kutta Method first
            sol = solve_ivp(funcptr, tspan, usol, teval, rtol = 1e-9, atol = 1e-30)
            tmp, success = sol.y, sol.success
            # There are points when L63 changes attractor when the problem becomes stiff
            # If so, retry with a stiff LSODA solver
            if not success or np.allclose(tmp[-1, :], 0.0):
                  tmp, success = lsoda(funcptr, usol, teval, rtol = 1e-9, atol = 1e-30)
            if not success or np.allclose(tmp[-1, :], 0.0):
                  model_error = 1
            return tmp.T, model_error

      def forecast_rollout(self, x, steps, **kwargs):
            funcptr = kwargs.get('funcptr', self.funcptr)
            dt = kwargs.get('dt', self.dt)
            return self._integrate(x, steps, dt, funcptr)

      def forecast(self, x, steps, **kwargs):
            tmp, model_error = self.forecast_rollout(x, steps, **kwargs)
            return tmp[:, -1], model_error


      def forecast_batch_rollout(self, x_ens, steps, **kwargs):
            Nx, Ne = x_ens.shape
            dt = kwargs.get('dt', self.dt)
            funcptr = self.funcptr
            pfunc = partial(self._integrate, steps = steps, dt = dt, funcptr = funcptr)
            pool_results  = self.pool.map(pfunc, [x_ens[:, i] for i in range(Ne)])
            x_fore = np.stack([x for x, _ in pool_results], axis = -1)
            model_errors = np.array([y for _, y in pool_results])
            return x_fore, model_errors

      def forecast_batch(self, x_ens, steps, **kwargs):
            sols, model_errors = self.forecast_batch_rollout(x_ens, steps, **kwargs)
            return sols[:, -1, :], model_errors

      def init_condition(self, rng, Ne) -> tuple[np.ndarray, np.ndarray]:
            xt_0 = 3*np.sin(np.arange(self.Nx)/(6*2*np.pi))

            xf_0 = xt_0[:, np.newaxis] + 1*rng.standard_normal((self.Nx, Ne))

            return xf_0, xt_0



class Lorenz63(LorenzModel):
      def __init__(self, modelparams, dt, numprocs: int = None):
            super().__init__(modelparams, dt, numprocs)
            rhs = self.make_rhs_l63(modelparams)
            self.funcptr = rhs.address

      # In order to integrate using numbalsoda, we need to declare a
      # C-compiled function for outputting dx/dt
      # This will let us integrate much faster.
      # This Cfunc depends on the lorenz model, so I'm storing it
      # as a staticmethod in each Lorenz Model Class
      # I'm them saving the pointer to the function as a model attribute
      # in the __init__ method.
      # All the rest of the methods are inherited from the LorenzModel Class
      @staticmethod
      def make_rhs_l63(kwargs):
            s = kwargs['s']
            r = kwargs['r']
            b = kwargs['b']      
            @cfunc(lsoda_sig)
            def rhs(t, u, du, p):
                  du[0] = s*(u[1]-u[0])
                  du[1] = u[0]*(r-u[2]) - u[1]
                  du[2] = u[0]*u[1] - b*u[2]
            return rhs
      def __str__(self):
            return 'Lorenz 63 Model\nNx: 3\ndt: {}'.format(self.dt)
      
class Lorenz96(LorenzModel):
      def __init__(self, modelparams, dt, numprocs: int = None):
            super().__init__(modelparams, dt, numprocs)
            rhs = self.make_rhs_l96(modelparams)
            self.funcptr = rhs.address
      @staticmethod
      def make_rhs_l96(kwargs):
            F = kwargs['F']
            Nx = 40
            @cfunc(lsoda_sig)
            def rhs(t, u, du, p):
                  u_ = nb.carray(u, (Nx,))
                  tmp = (np.roll(u_, -1) - np.roll(u_, 2))*np.roll(u_, 1) - u_ + F
                  for i in range(Nx):
                        du[i] = tmp[i]
            return rhs
      def __str__(self):
            return 'Lorenz 96 Model\nNx: 40\ndt: {}'.format(self.dt)

class Lorenz05(LorenzModel):
      def __init__(self, modelparams, dt, numprocs: int = None):
            super().__init__(modelparams, dt, numprocs)
            rhs = self.make_rhs_l05(modelparams)
            self.funcptr = rhs.address
      @staticmethod
      def make_rhs_l05(kwargs):
            K = int(kwargs['l05_K'])
            I = int(kwargs['l05_I'])
            b = kwargs['l05_b']
            c = kwargs['l05_c']
            F = kwargs['l05_F']
            Nx = 480
            K = np.round(K)
            I = np.round(I)
            alpha = (3*I**2 + 3)/(2*I**3 + 4*I)
            beta = (2*I**2+1)/(I**4 + 2*I**2)
            @cfunc(lsoda_sig)
            def rhs(t, z, dz, p):
                  z_ = nb.carray(z, (Nx,))
                  z0 = np.concatenate((z_, z_, z_))
                  i = np.arange(-(I-1), I, dtype=np.int64)
                  if I == 1:
                        x0 = z0
                  else:
                        x0 = np.empty((Nx,))
                        for m in range(Nx):
                              n = Nx + m
                              x0[m] = np.sum((alpha - beta*np.abs(i))*z0[n+i]) + (alpha - beta*np.abs(-I))*z0[n-I]/2 + (alpha - beta*np.abs(I))*z0[n+I]/2
                        y0 = z0[Nx:2*Nx] - x0
                  x0 = np.concatenate((x0, x0, x0))
                  if I > 1:
                        y0 = np.concatenate((y0, y0, y0))
                  w = np.empty((3*Nx))
                  J = int(np.floor(K/2))
                  j = np.arange(-(J-1), J, dtype = np.int64)
                  if K%2 == 0:
                        norm = 1/2
                  else:
                        norm = 1
                  j = np.arange(-(J-1), J, dtype=np.int64)
                  J = int(J)
                  for m in np.arange(Nx-2*K, 2*Nx+2*K):
                        w[m] = (np.sum(x0[m-j]) + (x0[m-J] + x0[m+J])*norm)/K
                  xx = np.empty((Nx,))
                  for m in range(Nx):
                        n = Nx + m 
                        xx[m] = -w[n-2*K]*w[n-K] + (np.sum(w[n-K+j]*x0[n+K+j]) + (w[n-K-J]*x0[n+K-J] + w[n-K+J]*x0[n+K+J])*norm)/K
                  i1 = Nx + np.arange(-2, Nx-2, dtype = np.int16)
                  i2 = Nx + np.arange(-1, Nx-1, dtype = np.int16)
                  i3 = Nx + np.arange(0, Nx, dtype = np.int16)
                  i4 = Nx + np.arange(1, Nx+1, dtype = np.int16)

                  if I>1:
                        yy = -y0[i1]*y0[i2] + y0[i2]*y0[i4]
                        yx = -y0[i1]*x0[i2] + y0[i2]*x0[i4]
                        tmp = xx + (b**2)*yy + c*yx - x0[i3] - b*y0[i3] + F
                  else:
                        tmp = xx - x0[i3] + F
                  for n in range(Nx):
                        dz[n] = tmp[n]
            return rhs

      def __str__(self):
            return 'Lorenz 05 Model\nNx: 480\ndt: {}'.format(self.dt)

class QGModel(Model):
      def __init__(self, modelparams: dict, dt: float):
            super().__init__(modelparams, dt)
            self.xt_model, self.xf_model, self.init_state = self.make_qg(modelparams, dt)
            self.original_shape = self.init_state.state.q.shape
            self.curr_state = self.init_state
            self.roll_out_state = jax.jit(
                  jax.vmap(self._roll_out_state, in_axes=(0, None)),
                  static_argnames="num_steps"
            )
            return

      def _roll_out_state(self, state, num_steps):

            def loop_fn(carry, _state):
                  current_state = carry
                  
                  next_state = self.xf_model.step_model(current_state)
            
                  return next_state, next_state
      
            final_state, _ = jax.lax.scan(
                  loop_fn, state, None, length=num_steps
            )
            
            return final_state      

      @functools.partial(jax.jit, static_argnames=["self", "steps"])
      def forecast_rollout(self, x: np.ndarray, steps: int) -> tuple[np.ndarray, int]:
            
            model_error = 0
            
            x_shaped = x.reshape(self.original_shape)
            x_shaped = jnp.array(x_shaped)
            x_shaped = x_shaped.astype(jnp.float32)

            model_state = self.curr_state.state.update(q=x_shaped)
            ab3_model_state = self.xt_model.initialize_stepper_state(model_state)
            x = ab3_model_state
            
            def loop_fn(carry, _x_shaped):
                  current_state = carry
                  
                  next_state = self.xt_model.step_model(current_state)
            
                  return next_state, next_state
      
            _final_carry, traj_steps = jax.lax.scan(
                  loop_fn, x, None, length=steps
                  )
            
            #Need to reshape final output to match required format
            tmp = traj_steps.state.q
            ntime, nlayers, nx, ny = tmp.shape
            tmp = tmp.reshape(ntime, nlayers * nx * ny).T
            
            return tmp, model_error
      
      def forecast(self, x: np.ndarray, steps: int)-> tuple[np.ndarray, int]:
            tmp, model_error = self.forecast_rollout(x, steps)
            return tmp[:, -1], model_error

      

      def forecast_batch(self, x_ens: np.ndarray, steps: int)-> tuple[np.ndarray, np.ndarray]: 

            Nx, Ne = x_ens.shape
            model_error = 0

            #Need code that transforms x_ens into a usable multistate format
            #Test this out in PyQG_test, then implement
            nlayers, nx, ny = self.original_shape
            x_ens_shaped = jnp.array(
                  x_ens.reshape(nlayers, nx, ny, Ne).transpose(3, 0, 1, 2)
            ).astype(jnp.float32)

            model_state = self.curr_states.state.update(q=x_ens_shaped)
            ab3_model_state = self.curr_states.update(state=model_state)
            x = ab3_model_state
            # .update(
            #       t=np.zeros(Ne, dtype=np.float32),
            #       tc=jnp.zeros(Ne, dtype=jnp.uint32),
            # )

            final_state = self.roll_out_state(x, steps)
            tmp = final_state.state.q.transpose(1, 2, 3, 0).reshape(Nx, Ne)
            
            return tmp, model_error

      def forecast_batch_rollout(self, x_ens: np.ndarray, steps: int)-> tuple[np.ndarray, np.ndarray]:
            # Nx, Ne = x_ens.shape

            x_ens_T = x_ens.T

            def single_forecast(x):
                  return self.forecast_rollout(x, steps=steps)

            x_fore, model_errors = jax.vmap(single_forecast)(x_ens_T)

            x_fore = jnp.transpose(x_fore, (1,2,0))
            
            return x_fore, model_errors

      #Should each model start from a different or the same init condition?
      #add in init_condition function
      def make_qg(self, kwargs, dt):
            nx        = kwargs['nx']
            ny        = kwargs['ny']
            L         = kwargs['L']
            W         = kwargs['W']
            rek       = kwargs['rek']
            filterfac = kwargs['filterfac']
            f         = kwargs['f']
            g         = kwargs['g']
            beta      = kwargs['beta']
            rd        = kwargs['rd']
            delta     = kwargs['delta']
            H1        = kwargs['H1']
            U1        = kwargs['U1']
            U2        = kwargs['U2']
            precision = kwargs['precision']

            if ny is None:
                  ny = nx
            
            self.Nx = nx*ny*2

            #Try with new values for rd, H1, delta, U1. I think it was set for ocean before owing to slow speeds
            base_model = pyqg_jax.qg_model.QGModel(nx=nx, ny=ny, L=L, W=W, rek=rek, filterfac=filterfac, f=f, g=g, beta=beta, rd=rd, delta=delta, H1=H1, U1=U1, U2=U2, precision=precision)

            stepper = pyqg_jax.steppers.AB3Stepper(dt=dt)

            xt_model = pyqg_jax.steppers.SteppedModel(
                  base_model, stepper
            )

            xf_model = pyqg_jax.steppers.SteppedModel(
                  base_model, stepper
            )
            
            init_state = xt_model.create_initial_state(
                  jax.random.key(0)
            )

            return xt_model, xf_model, init_state

      def init_condition(self, rng, Ne) -> tuple[np.ndarray, np.ndarray]:     

            init_rngs = jnp.broadcast_to(jax.random.key(0), (Ne,))
            self.init_states = jax.vmap(self.xf_model.create_initial_state)(init_rngs)
            xt_0 = self.init_state.state.q.reshape(self.Nx,)

            xf_0 = self.init_states.state.q.transpose(1, 2, 3, 0).reshape(self.Nx, Ne) + 1e-8*rng.standard_normal((self.Nx, Ne))

            self.curr_states = self.init_states  

            # xf_0 = xt_0[:, np.newaxis] + 1e-8*rng.standard_normal((self.Nx, Ne))

            return xf_0, xt_0