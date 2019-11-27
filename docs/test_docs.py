import nbformat
import os
import pytest
import re
from nbconvert.preprocessors import ExecutePreprocessor
 
 
def run_notebook(notebook_path):
    nb_name, _ = os.path.splitext(os.path.basename(notebook_path))
    dirname = os.path.dirname(notebook_path)
 
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)
 
    proc = ExecutePreprocessor(timeout=600, kernel_name='python3')
    proc.allow_errors = True
 
    proc.preprocess(nb)
    output_path = os.path.join(dirname, '{}_all_output.ipynb'.format(nb_name))
 
    with open(output_path, mode='wt') as f:
        nbformat.write(nb, f)
 
    for num, cell in enumerate(nb.cells):
        if 'outputs' in cell:
            for output in cell['outputs']:
                if output.output_type == 'error':
                    return cell.execution_count, output.traceback
 
    return None
 
# 7-bit C1 ANSI sequences
def escape_ansi_control(error):
    ansi_escape = re.compile(r'''
        \x1B    # ESC
        [@-_]   # 7-bit C1 Fe
        [0-?]*  # Parameter bytes
        [ -/]*  # Intermediate bytes
        [@-~]   # Final byte
    ''', re.VERBOSE)
    sanitized = ""
    for line in error:
        sanitized += ansi_escape.sub('', line) + "\n"
    return sanitized

class TestDocs:
    def test_docs(self):
        filename = 'quickstart.ipynb'
        errors = run_notebook(filename)
        if errors is not None:
            pytest.fail("\nNotebook {}, Cell {} failed:\n{}".format(filename,errors[0], escape_ansi_control(errors[1]))) 
