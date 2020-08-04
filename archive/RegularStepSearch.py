from opentuner.search import technique

class RegularStepSearch(technique.SequentialSearchTechnique):
  """
  This is a special technique used simply for demonstrating the influence of a single parameter.
  Therefore, we assume ONE paramter only.
  """
  def main_generator(self):

    objective   = self.objective
    driver      = self.driver
    manipulator = self.manipulator

    # start at a random position
    init = driver.get_configuration(manipulator.random())

    assert(len(manipulator.parameters(init.data)) == 1)
    param = manipulator.parameters(init.data)[0]
    assert(param.is_primitive())
    param_min_value, param_max_value = param.legal_range(init.data)

    # initial step size is arbitrary
    step_size = 0.1

    current_cfg = manipulator.copy(init.data)
    param.set_value(current_cfg, param_min_value)
    current_cfg = driver.get_configuration(current_cfg)
    
    yield current_cfg

    while True:
      points = list()
      # get current value of param, scaled to be in range [0.0, 1.0]
      unit_value = param.get_unit_value(current_cfg.data)
      print unit_value
      points.append(current_cfg)

      if unit_value < 1.0:
        # produce new config with param set step_size higher
        up_cfg = manipulator.copy(current_cfg.data)
        param.set_unit_value(up_cfg, min(1.0, unit_value + step_size))
        up_cfg = driver.get_configuration(up_cfg)
        yield up_cfg
        points.append(up_cfg)

      #sort points by quality, best point will be points[0], worst is points[-1]
      # points.sort(cmp=objective.compare)

      current_cfg = points[-1]
      # if objective.lt(points[0], center):
      #   # we found a better point, move there
      #   center = points[0]
      # else:
      #   # no better point, shrink the pattern
      #   step_size /= 2.0

# register our new technique in global list
technique.register(RegularStepSearch())