import re
import shapely
import ast
import astunparse
from time import sleep
from shapely.geometry import *
from shapely.affinity import *
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import TerminalFormatter

class LMP:

    def __init__(self, name, cfg, lmp_fgen, fixed_vars, variable_vars, tokenizer, model):
        self._name = name
        self._cfg = cfg

        self._base_prompt = self._cfg['prompt_text']
        
        self._stop_tokens = '|'.join(r'\n?' + re.escape(tok) for tok in list(self._cfg['stop']))

        self._lmp_fgen = lmp_fgen

        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ''

        self.tokenizer = tokenizer
        self.model = model

    def clear_exec_hist(self):
        self.exec_hist = ''

    def build_prompt(self, query, context=''):
        if len(self._variable_vars) > 0:
            variable_vars_imports_str = f"from utils import {', '.join(self._variable_vars.keys())}"
        else:
            variable_vars_imports_str = ''
        prompt = self._base_prompt.replace('{variable_vars_imports}', variable_vars_imports_str)

        if self._cfg['maintain_session']:
            prompt += f'\n{self.exec_hist}'

        if context != '':
            prompt += f'\n{context}'

        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f'\n{use_query}'

        return prompt, use_query

    def __call__(self, query, context='', **kwargs):
        prompt, use_query = self.build_prompt(query, context=context)
        print(f"\nCalling the LMP for prompt: {query}")
        while True:
            try:
                input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                input_size = input_ids["input_ids"].shape[1]
                
                output = self.model.generate(
                    **input_ids, 
                    cache_implementation="static", 
                    temperature=self._cfg['temperature'],
                    max_new_tokens=self._cfg['max_tokens'],
                    top_p=self._cfg['top_p'],
                    do_sample=self._cfg['do_sample'],
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                code_str = self.tokenizer.decode(output[0][input_size:], skip_special_tokens=True).strip()
                code_str = re.split(self._stop_tokens, code_str)[0]        # Removing the extra tokens as model.generate doesn't have a parameter for stop tokens
                break
            except Exception as e:
                print(f'Error running {self._cfg["engine"]} through Hugging Face:\n{e}')
                print('Retrying after 10s.')
                sleep(10)
                
        if self._cfg['include_context'] and context != '':
            to_exec = f'{context}\n{code_str}'
            to_log = f'{context}\n{use_query}\n{code_str}'
        else:
            to_exec = code_str
            to_log = f'{use_query}\n{to_exec}'

        to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
        print(f'LMP {self._name} exec:\n\n{to_log_pretty}\n')

        new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
        self._variable_vars.update(new_fs)

        gvars = merge_dicts([self._fixed_vars, self._variable_vars])
        lvars = kwargs

        if not self._cfg['debug_mode']:
            exec_safe(to_exec, gvars, lvars)

        self.exec_hist += f'\n{to_exec}'

        if self._cfg['maintain_session']:
            self._variable_vars.update(lvars)

        if self._cfg['has_return']:
            return lvars[self._cfg['return_val_name']]


class LMPFGen:

    def __init__(self, cfg, fixed_vars, variable_vars, model_name, tokenizer, model):
        self._cfg = cfg

        self._stop_tokens = '|'.join(r'\n?' + re.escape(tok) for tok in list(self._cfg['stop']))
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars

        self._base_prompt = self._cfg['prompt_text']
        self.tokenizer = tokenizer
        self.model = model
        
    def create_f_from_sig(self, f_name, f_sig, other_vars=None, fix_bugs=False, return_src=False):
        print(f'Creating function: {f_sig}')

        use_query = f'{self._cfg["query_prefix"]}{f_sig}{self._cfg["query_suffix"]}'
        prompt = f'{self._base_prompt}\n{use_query}'

        while True:
            try:
                input_ids = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                input_size = input_ids["input_ids"].shape[1]
                
                output = self.model.generate(
                    **input_ids, 
                    cache_implementation="static", 
                    temperature=self._cfg['temperature'],
                    max_new_tokens=self._cfg['max_tokens'],
                    top_p=self._cfg['top_p'],
                    do_sample=self._cfg['do_sample'],
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
                f_src = self.tokenizer.decode(output[0][input_size:], skip_special_tokens=True).strip()
                f_src = re.split(self._stop_tokens, f_src)[0]        # Removing the extra tokens as model.generate doesn't have a parameter for stop tokens
                break
            except Exception as e:
                print(f'Error running {self._cfg["engine"]} through Hugging Face:\n{e}')
                print('Retrying after 10s.')
                sleep(10)

        if fix_bugs:
            fix_bugs_prompt = f"""### Instruction: Fix the bug if there is one. Improve readability. Keep same inputs and outputs. Only small changes. No comments.
### Input:
{f_src}"""
            
            input_ids = self.tokenizer(fix_bugs_prompt, return_tensors="pt").to("cuda")
            input_size = input_ids["input_ids"].shape[1]
            
            output = self.model.generate(
                **input_ids, 
                cache_implementation="static", 
                temperature=0,
                top_p=1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            f_src = self.tokenizer.decode(output[0][input_size:], skip_special_tokens=True).strip()

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars = {}

        exec_safe(f_src, gvars, lvars)

        f = lvars[f_name]

        to_print = highlight(f'{use_query}\n{f_src}', PythonLexer(), TerminalFormatter())
        print(f'LMP FGEN created:\n\n{to_print}\n')

        if return_src:
            return f, f_src
        return f

    def create_new_fs_from_code(self, code_str, other_vars=None, fix_bugs=False, return_src=False):
        fs, f_assigns = {}, {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs = {}
        srcs = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(f_name, f_sig, new_fs, fix_bugs=fix_bugs, return_src=True)

                # recursively define child_fs in the function body if needed
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # redefine parent f so newly created child_fs are in scope
                    gvars = merge_dicts([self._fixed_vars, self._variable_vars, new_fs, other_vars])
                    lvars = {}

                    exec_safe(f_src, gvars, lvars)

                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs


class FunctionParser(ast.NodeTransformer):

    def __init__(self, fs, f_assigns):
      super().__init__()
      self._fs = fs
      self._f_assigns = f_assigns

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.value.func).strip()
            self._f_assigns[f_name] = assign_str
        return node


def var_exists(name, all_vars):
    try:
        eval(name, all_vars)
    except:
        exists = False
    else:
        exists = True
    return exists


def merge_dicts(dicts):
    return {
        k : v
        for d in dicts
        for k, v in d.items()
    }


def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    exec(code_str, custom_gvars, lvars)