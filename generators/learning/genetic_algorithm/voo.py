import numpy as np


class VOO:
    def __init__(self, domain, explr_p, sampling_mode, switch_counter, distance_fn=None):
        self.domain = domain
        self.dim_x = domain.shape[-1]
        self.explr_p = explr_p
        if distance_fn is None:
            self.distance_fn = lambda x, y: np.linalg.norm(x - y)

        self.switch_counter = np.inf
        self.sampling_mode = sampling_mode
        self.GAUSSIAN = False
        self.CENTERED_UNIFORM = False
        self.UNIFORM = False
        if sampling_mode == 'centered_uniform':
            self.CENTERED_UNIFORM = True
        elif sampling_mode == 'gaussian':
            self.GAUSSIAN = True
        elif sampling_mode.find('hybrid') != -1:
            self.UNIFORM = True
            self.switch_counter = switch_counter
        elif sampling_mode.find('uniform') != -1:
            self.UNIFORM = True
            self.switch_counter = switch_counter
        else:
            raise NotImplementedError

        self.UNIFORM_TOUCHING_BOUNDARY = False

    def optimize(self, obj_fcn, n_evals):
        evaled_x = []
        evaled_y = []

        for _ in range(n_evals):
            newx = self.sample_next_point(evaled_x, evaled_y)
            newy = obj_fcn(newx)
            evaled_x.append(newx)
            evaled_y.append(newy)

        max_idx = np.argmax(evaled_y)
        return evaled_x[max_idx], np.max(evaled_y)

    def sample_next_point(self, evaled_x, evaled_y):
        rnd = np.random.random()  # this should lie outside
        is_sample_from_best_v_region = (rnd < 1 - self.explr_p) and len(evaled_x) > 1
        if is_sample_from_best_v_region:
            x = self.sample_from_best_voronoi_region(evaled_x, evaled_y)
        else:
            x = self.sample_from_uniform()
        return x

    def sample_from_best_voronoi_region(self, evaled_x, evaled_y):
        best_dist = np.inf
        other_dists = np.array([-1])
        counter = 1

        best_evaled_x_idxs = np.argwhere(evaled_y == np.amax(evaled_y))
        best_evaled_x_idxs = best_evaled_x_idxs.reshape((len(best_evaled_x_idxs, )))
        best_evaled_x_idx = best_evaled_x_idxs[0]  # np.random.choice(best_evaled_x_idxs)
        best_evaled_x = evaled_x[best_evaled_x_idx]
        other_best_evaled_xs = evaled_x

        # todo perhaps this is reason why it performs so poorly
        curr_closest_dist = np.inf

        while np.any(best_dist > other_dists):
            if self.GAUSSIAN:
                possible_max = (self.domain[1] - best_evaled_x) / np.exp(counter)
                possible_min = (self.domain[0] - best_evaled_x) / np.exp(counter)
                possible_values = np.max(np.vstack([np.abs(possible_max), np.abs(possible_min)]), axis=0)
                new_x = np.random.normal(best_evaled_x, possible_values)
                while np.any(new_x > self.domain[1]) or np.any(new_x < self.domain[0]):
                    new_x = np.random.normal(best_evaled_x, possible_values)
            elif self.CENTERED_UNIFORM:
                dim_x = self.domain[1].shape[-1]
                possible_max = (self.domain[1] - best_evaled_x) / np.exp(counter)
                possible_min = (self.domain[0] - best_evaled_x) / np.exp(counter)

                possible_values = np.random.uniform(possible_min, possible_max, (dim_x,))
                new_x = best_evaled_x + possible_values
                while np.any(new_x > self.domain[1]) or np.any(new_x < self.domain[0]):
                    possible_values = np.random.uniform(possible_min, possible_max, (dim_x,))
                    new_x = best_evaled_x + possible_values
            elif self.UNIFORM:
                new_x = np.random.uniform(self.domain[0], self.domain[1])
                if counter > self.switch_counter:
                    if self.sampling_mode.find('hybrid') != -1:
                        if self.sampling_mode.find('gaussian'):
                            self.GAUSSIAN = True
                        else:
                            self.CENTERED_UNIFORM = True
                    else:
                        break
            else:
                raise NotImplementedError

            best_dist = self.distance_fn(new_x, best_evaled_x)
            other_dists = np.array([self.distance_fn(other, new_x) for other in other_best_evaled_xs])
            counter += 1
            if best_dist < curr_closest_dist:
                curr_closest_dist = best_dist
                curr_closest_pt = new_x

        return curr_closest_pt

    def sample_from_uniform(self):
        dim_parameters = self.domain.shape[-1]
        domain_min = self.domain[0]
        domain_max = self.domain[1]
        return np.random.uniform(domain_min, domain_max, (1, dim_parameters)).squeeze()
