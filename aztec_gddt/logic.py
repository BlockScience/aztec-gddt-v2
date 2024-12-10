from aztec_gddt.types import ModelParams, ModelState

def p_evolve_time(params: ModelParams, _2, _3, _4):
    return {'delta_days': params['timestep_in_days']}

def s_days_passed(_1, _2, _3,
                  state: ModelState,
                  signal):
    return ('days_passed', state['days_passed'] + signal['delta_days'])

def s_delta_days(_1, _2, _3, _4, signal):
    return ('delta_days', signal['delta_days'])
