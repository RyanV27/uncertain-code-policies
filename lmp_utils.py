import copy
import shapely
import numpy as np
from shapely.geometry import *
from shapely.affinity import *

from tabletop_config import lmp_tabletop_coords
from lmp_wrapper import LMP_wrapper
from lmp import LMP, LMPFGen

def setup_LMP(env, cfg_tabletop, model_name, tokenizer = None, model = None):
  # LMP env wrapper
  cfg_tabletop = copy.deepcopy(cfg_tabletop)
  cfg_tabletop['env'] = dict()
  cfg_tabletop['env']['init_objs'] = list(env.obj_name_to_id.keys())
  cfg_tabletop['env']['coords'] = lmp_tabletop_coords
  LMP_env = LMP_wrapper(env, cfg_tabletop)

  # creating APIs that the LMPs can interact with
  fixed_vars = {
      'np': np
  }
  fixed_vars.update({
      name: eval(name)
      for name in shapely.geometry.__all__ + shapely.affinity.__all__
  })
  variable_vars = {
      k: getattr(LMP_env, k)
      for k in [
          'get_bbox', 'get_obj_pos', 'get_color', 'is_obj_visible', 'denormalize_xy',
          'put_first_on_second', 'get_obj_names',
          'get_corner_name', 'get_side_name',
      ]
  }
  variable_vars['say'] = lambda msg: print(f'robot says: {msg}')

  # creating the function-generating LMP
  lmp_fgen = LMPFGen(cfg_tabletop['lmps']['fgen'], fixed_vars, variable_vars, 
                        model_name=model_name, tokenizer=tokenizer, model=model)

  # creating other low-level LMPs
  variable_vars.update({
      k: LMP(k, cfg_tabletop['lmps'][k], lmp_fgen, fixed_vars, variable_vars, tokenizer=tokenizer, model=model)
      for k in ['parse_obj_name', 'parse_position', 'parse_question', 'transform_shape_pts']
  })

  # creating the LMP that deals w/ high-level language commands
  lmp_tabletop_ui = LMP(
      'tabletop_ui', cfg_tabletop['lmps']['tabletop_ui'], lmp_fgen, fixed_vars, variable_vars, 
        tokenizer=tokenizer, model=model
   )

  return lmp_tabletop_ui