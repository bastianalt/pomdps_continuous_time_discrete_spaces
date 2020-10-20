import abc
import torch
from .misc import _assert_increasing, _handle_unused_kwargs
import collections
from .interp import _interp_evaluate
from .misc import _scaled_dot_product
import numpy as np
from bisect import bisect_left, bisect_right


class AdaptiveStepsizeODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, atol, rtol, dense_output=False, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.atol = atol
        self.rtol = rtol
        self.dense_output = dense_output
        self.interpolator = None

    def before_integrate(self, t):
        pass

    @abc.abstractmethod
    def advance(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        _assert_increasing(t)
        if self.dense_output:
            solution = []
        else:
            solution = [self.y0]
        t = t.to(self.y0[0].device, torch.float64)
        self.before_integrate(t)
        for i in range(1, len(t)):
            y = self.advance(t[i])
            solution.append(y)

        if self.dense_output:
            ts, interpolants = _solution2interpolants(solution)
            return OdeTorchSolution(ts, interpolants, interpolator=self.interpolator)
        else:
            return tuple(map(torch.stack, tuple(zip(*solution))))


class FixedGridODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, step_size=None, grid_constructor=None, dense_output=False, **unused_kwargs):
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('atol', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.dense_output = dense_output
        self.interpolator = '_linear_interp'

        if step_size is not None and grid_constructor is None:
            self.grid_constructor = self._grid_constructor_from_step_size(step_size)
        elif grid_constructor is None:
            self.grid_constructor = lambda f, y0, t: t
        else:
            raise ValueError("step_size and grid_constructor are exclusive arguments.")

    def _grid_constructor_from_step_size(self, step_size):

        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters).to(t) * step_size + start_time
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    @property
    @abc.abstractmethod
    def order(self):
        pass

    @abc.abstractmethod
    def step_func(self, func, t, dt, y):
        pass

    def integrate(self, t):
        _assert_increasing(t)
        t = t.type_as(self.y0[0])
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        time_grid = time_grid.to(self.y0[0])
        if self.dense_output:
            solution = []
        else:
            solution = [self.y0]
        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self.step_func(self.func, t0, t1 - t0, y0)
            y1 = tuple(y0_ + dy_ for y0_, dy_ in zip(y0, dy))

            while j < len(t) and t1 >= t[j]:
                if self.dense_output:
                    solution.append(_FixedGridState(t0, t1, y0, y1))
                else:
                    solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                j += 1
            y0 = y1
        if self.dense_output:
            ts, interpolants = _solution2interpolants([solution])
            return OdeTorchSolution(ts, interpolants, interpolator=self.interpolator)
        else:
            return tuple(map(torch.stack, tuple(zip(*solution))))

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
        slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
        return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))


class _FixedGridState(collections.namedtuple('_FixedGridState', 't0, t1, y0, y1')):
    """
    """


def _solution2interpolants(solution):
    t0 = None
    t1 = None
    ts = []
    interpolants = []
    for i in range(len(solution)):
        for state in solution[i]:
            # Dont add non interpolant states
            if state.t0 != state.t1:
                # If there is the same t0 state replace the last one
                if t0 == state.t0:
                    t1 = state.t1
                    ts[-1] = state.t0
                    interpolants[-1] = state
                # Else add the new state
                else:
                    t0 = state.t0
                    t1 = state.t1
                    ts.append(state.t0)
                    interpolants.append(state)
    ts.append(t1)
    ts = torch.stack(ts)
    return ts, interpolants


class OdeTorchSolution:
    def __init__(self, ts, interpolants, interpolator='_interp_evaluate'):
        self.interpolator = interpolator
        d = ts[1:] - ts[:-1]
        # The first case covers integration on zero segment.
        if not (torch.all(d > 0) or torch.all(d < 0)):
            raise ValueError("`ts` must be strictly increasing or decreasing.")

        self.n_segments = len(interpolants)
        if ts.shape != (self.n_segments + 1,):
            raise ValueError("Numbers of time stamps and interpolants "
                             "don't match.")

        self.ts = ts
        self.interpolants = interpolants
        if ts[-1] >= ts[0]:
            self.t_min = ts[0]
            self.t_max = ts[-1]
            self.ascending = True
            self.ts_sorted = ts.tolist()
        else:
            self.t_min = ts[-1]
            self.t_max = ts[0]
            self.ascending = False
            self.ts_sorted = ts[::-1].tolist()

    def __call__(self, t):
        # Here we preserve a certain symmetry that when t is in self.ts,
        # then we prioritize a segment with a lower index.
        t = t.squeeze()
        if self.ascending:
            ind = bisect_left(self.ts_sorted, t.tolist())
        else:
            ind = bisect_right(self.ts_sorted, t.tolist())

        segment = min(max(ind - 1, 0), self.n_segments - 1)
        if not self.ascending:
            segment = self.n_segments - segment

        if t < self.interpolants[segment].t0:
            t = self.interpolants[segment].t0

        if t > self.interpolants[segment].t1:
            t = self.interpolants[segment].t1

        if self.interpolator == '_linear_interp':
            return self._linear_interp(self.interpolants[segment].t0, self.interpolants[segment].t1,
                                       self.interpolants[segment].y0,
                                       self.interpolants[segment].y1, t)[0]
        elif self.interpolator == '_interp_evaluate':
            return _interp_evaluate(self.interpolants[segment].interp_coeff, self.interpolants[segment].t0,
                                    self.interpolants[segment].t1, t)[0]
        elif self.interpolator == '_interp_eval_tsit5':
            return self._interp_eval_tsit5(self.interpolants[segment].t0, self.interpolants[segment].t1,
                                           self.interpolants[segment].interp_coeff, t)[0]
        else:
            return None

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
        slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
        return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))

    def _interp_eval_tsit5(self, t0, t1, k, eval_t):
        dt = t1 - t0
        y0 = tuple(k_[0] for k_ in k)
        interp_coeff = self._interp_coeff_tsit5(t0, dt, eval_t)
        y_t = tuple(y0_ + _scaled_dot_product(dt, interp_coeff, k_) for y0_, k_ in zip(y0, k))
        return y_t

    @staticmethod
    def _interp_coeff_tsit5(t0, dt, eval_t):
        t = float((eval_t - t0) / dt)
        b1 = -1.0530884977290216 * t * (t - 1.3299890189751412) * (t ** 2 - 1.4364028541716351 * t + 0.7139816917074209)
        b2 = 0.1017 * t ** 2 * (t ** 2 - 2.1966568338249754 * t + 1.2949852507374631)
        b3 = 2.490627285651252793 * t ** 2 * (t ** 2 - 2.38535645472061657 * t + 1.57803468208092486)
        b4 = -16.54810288924490272 * (t - 1.21712927295533244) * (t - 0.61620406037800089) * t ** 2
        b5 = 47.37952196281928122 * (t - 1.203071208372362603) * (t - 0.658047292653547382) * t ** 2
        b6 = -34.87065786149660974 * (t - 1.2) * (t - 0.666666666666666667) * t ** 2
        b7 = 2.5 * (t - 1) * (t - 0.6) * t ** 2
        return [b1, b2, b3, b4, b5, b6, b7]
